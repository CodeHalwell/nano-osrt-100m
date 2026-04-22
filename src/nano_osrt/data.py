"""Streaming data pipeline for NanoOSRT pre-training.

Handles:
- Progressive seq_len (2048 → 4096 → 8192) across phases
- Multi-dataset weighted sampling within each phase
- Code + text mixing from the start
- Instruction format handling (messages column)
- Resilient streaming: connection drops and corrupt shards are caught
  and retried instead of killing the training run.
"""

import random
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


class TokenStream(IterableDataset):
    """Streaming token dataset with multi-dataset weighted sampling.

    Streams from multiple HuggingFace datasets simultaneously,
    sampling according to configured weights. Handles both plain text
    and instruction-format (messages column) datasets.

    Args:
        dataset_configs: List of dataset config dicts with hf_id, weight, etc.
        seq_len: Sequence length for this phase.
        tok_name: HuggingFace tokenizer identifier.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        seq_len: int,
        tok_name: str,
        seed: int,
    ) -> None:
        self.dataset_configs = dataset_configs
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        tok = AutoTokenizer.from_pretrained(self.tok_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(seed)

        # Load all dataset streams
        streams = []
        weights = []
        for ds_cfg in self.dataset_configs:
            print(f"[DataWorker] Connecting to {ds_cfg['hf_id']}...")
            # Per-config split (defaults to "train"). The eval loader
            # passes split="validation" when the dataset has one, or
            # uses skip=N on the train split to carve out a disjoint
            # subset. Previously the hardcoded "train" silently
            # measured a training shuffle as "eval".
            load_kwargs = {
                "split": ds_cfg.get("split", "train"),
                "streaming": True,
            }
            if ds_cfg.get("hf_config"):
                load_kwargs["name"] = ds_cfg["hf_config"]
            ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
            # Optional `skip` offset for held-out evaluation on datasets
            # without a validation split (e.g. FineWeb-Edu). With a
            # training shuffle buffer of 5k examples, a skip of 5M+
            # examples leaves effectively zero collision with any
            # realistic training run.
            skip_n = ds_cfg.get("skip", 0)
            if skip_n > 0:
                ds = ds.skip(skip_n)
            ds = ds.shuffle(buffer_size=5_000, seed=seed)
            streams.append(iter(ds))
            weights.append(ds_cfg["weight"])
            print(f"[DataWorker] Stream ready for {ds_cfg['hf_id']}")

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Token-weighted sampling: pick the stream whose observed token
        # fraction is furthest behind its configured target. This makes
        # a "60% FineWeb / 40% CodeParrot" config produce the actual
        # token mix regardless of per-stream example lengths — otherwise
        # code streams with longer examples would dominate. Starts
        # weight-random during the bootstrap phase (no tokens seen yet).
        tokens_seen: list[int] = [0] * len(streams)

        def _pick_stream() -> int:
            total = sum(tokens_seen)
            if total == 0:
                return rng.choices(range(len(streams)), weights=weights, k=1)[0]
            deficits = [
                weights[i] - tokens_seen[i] / total
                for i in range(len(streams))
            ]
            max_def = max(deficits)
            candidates = [
                i for i, d in enumerate(deficits) if d >= max_def - 1e-6
            ]
            return rng.choice(candidates)

        buffer: list[int] = []

        def _reconnect_stream(stream_idx: int) -> None:
            ds_cfg = self.dataset_configs[stream_idx]
            load_kwargs = {
                "split": ds_cfg.get("split", "train"),
                "streaming": True,
            }
            skip_n = ds_cfg.get("skip", 0)
            if ds_cfg.get("hf_config"):
                load_kwargs["name"] = ds_cfg["hf_config"]
            ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
            if skip_n > 0:
                ds = ds.skip(skip_n)
            ds = ds.shuffle(
                buffer_size=5_000,
                seed=seed + rng.randint(0, 100_000),
            )
            streams[stream_idx] = iter(ds)
            print(
                f"[DataWorker] Reconnected to {ds_cfg['hf_id']}",
                flush=True,
            )

        max_retries = 5

        while True:
            idx = _pick_stream()
            ds_cfg_i = self.dataset_configs[idx]
            ds_name = ds_cfg_i.get("name", ds_cfg_i["hf_id"])

            try:
                example = next(streams[idx])
            except StopIteration:
                _reconnect_stream(idx)
                try:
                    example = next(streams[idx])
                except StopIteration:
                    continue
            except Exception as exc:
                # Connection drops, corrupt shards, HTTP errors —
                # log, sleep, reconnect, and continue. A flaky remote
                # gzip shard should never kill a multi-hour Modal job.
                for attempt in range(1, max_retries + 1):
                    print(
                        f"[DataWorker] {ds_name}: {type(exc).__name__}: "
                        f"{exc} — reconnecting [{attempt}/{max_retries}]",
                        flush=True,
                    )
                    time.sleep(2 * attempt)
                    try:
                        _reconnect_stream(idx)
                        example = next(streams[idx])
                        break
                    except Exception as retry_exc:
                        exc = retry_exc
                else:
                    print(
                        f"[DataWorker] {ds_name}: giving up after "
                        f"{max_retries} retries, skipping batch",
                        flush=True,
                    )
                    continue

            # Extract text from example
            text = self._extract_text(example, tok)
            if not text or not text.strip():
                continue

            tokens = tok.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(tok.eos_token_id)
            # Record token count for the debt-based sampler. We count
            # real tokens but not the structural EOS so code streams
            # with many short examples aren't artificially inflated.
            tokens_seen[idx] += len(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

    def _extract_text(self, example: dict, tok) -> str:
        """Extract text from various dataset formats."""
        # Instruction format (messages column)
        if "messages" in example:
            try:
                return tok.apply_chat_template(example["messages"], tokenize=False)
            except Exception:
                parts: list[str] = []
                for m in example["messages"]:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}")
                    if role == "assistant":
                        parts.append(tok.eos_token)
                return "\n".join(parts)

        # Conversations format (OpenHermes, SlimOrca)
        if "conversations" in example:
            parts = []
            for m in example["conversations"]:
                role = m.get("from", m.get("role", "user"))
                value = m.get("value", m.get("content", ""))
                parts.append(f"{role}: {value}")
            return "\n".join(parts)

        # Code format (content column)
        if "content" in example:
            return example["content"]

        # Instruction/output format (Alpaca, Evol-Instruct-Code)
        if "instruction" in example and "output" in example:
            inp = example.get("input", "")
            if inp:
                return f"{example['instruction']}\n{inp}\n{example['output']}"
            return f"{example['instruction']}\n{example['output']}"

        # Plain text
        if "text" in example:
            return example["text"]

        return ""


def make_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    step_num: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for a training phase.

    Args:
        dataset_configs: List of dataset config dicts for this phase.
        seq_len: Sequence length for this phase.
        tokenizer_name: HuggingFace tokenizer identifier.
        batch_size: Micro-batch size.
        step_num: Current step (used to vary shuffle seed).

    Returns:
        DataLoader yielding (input_ids, labels) batches.
    """
    ds = TokenStream(
        dataset_configs, seq_len, tokenizer_name, seed=42 + step_num
    )
    # num_workers=2 offloads HF streaming + BPE tokenisation to two
    # background processes so the main training thread doesn't wait on
    # them. Each worker gets its own seed (TokenStream reads
    # worker_info.id at line 55 and offsets) so they don't produce
    # duplicate batches. persistent_workers keeps the processes across
    # phase transitions — new loaders still spawn fresh workers, but
    # within a phase we don't tear them down for every step's reload.
    # prefetch_factor=2 keeps a small queue of ready batches per worker.
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
