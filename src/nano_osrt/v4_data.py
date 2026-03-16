"""Streaming data pipeline for NanoOSRT v4 pre-training.

Handles:
- Progressive seq_len (2048 → 4096 → 8192) across phases
- Multi-dataset weighted sampling within each phase
- Code + text mixing from the start
- Instruction format handling (messages column)
"""

import itertools
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


class V4TokenStream(IterableDataset):
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
            load_kwargs = {"split": "train", "streaming": True}
            if ds_cfg.get("hf_config"):
                load_kwargs["name"] = ds_cfg["hf_config"]
            ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
            ds = ds.shuffle(buffer_size=5_000, seed=seed)
            streams.append(iter(ds))
            weights.append(ds_cfg["weight"])
            print(f"[DataWorker] Stream ready for {ds_cfg['hf_id']}")

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        buffer: list[int] = []

        while True:
            # Weighted random dataset selection
            idx = rng.choices(range(len(streams)), weights=weights, k=1)[0]

            try:
                example = next(streams[idx])
            except StopIteration:
                # Reload exhausted stream
                ds_cfg = self.dataset_configs[idx]
                load_kwargs = {"split": "train", "streaming": True}
                if ds_cfg.get("hf_config"):
                    load_kwargs["name"] = ds_cfg["hf_config"]
                ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
                ds = ds.shuffle(buffer_size=5_000, seed=seed + rng.randint(0, 10000))
                streams[idx] = iter(ds)
                try:
                    example = next(streams[idx])
                except StopIteration:
                    continue

            # Extract text from example
            text = self._extract_text(example, tok)
            if not text or not text.strip():
                continue

            tokens = tok.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(tok.eos_token_id)

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


def make_v4_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    step_num: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for a v4 training phase.

    Args:
        dataset_configs: List of dataset config dicts for this phase.
        seq_len: Sequence length for this phase.
        tokenizer_name: HuggingFace tokenizer identifier.
        batch_size: Micro-batch size.
        step_num: Current step (used to vary shuffle seed).

    Returns:
        DataLoader yielding (input_ids, labels) batches.
    """
    ds = V4TokenStream(
        dataset_configs, seq_len, tokenizer_name, seed=42 + step_num
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
