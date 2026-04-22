"""Streaming data pipeline for Modal deployment training."""

import itertools

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


class TokenStream(IterableDataset):
    """Streaming token dataset for HuggingFace datasets.

    Lazily tokenises text from a HuggingFace streaming dataset and yields
    fixed-length ``(input_ids, labels)`` pairs suitable for causal LM
    training.

    Handles both plain-text datasets (``text`` column) and instruction-
    tuning datasets (``messages`` column with chat-template formatting).

    Args:
        dataset_name: HuggingFace dataset identifier.
        seq_len: Sequence length for each yielded chunk.
        tok_name: HuggingFace tokenizer identifier.
        seed: Base random seed for shuffling.
    """

    def __init__(
        self,
        dataset_name: str,
        seq_len: int,
        tok_name: str,
        seed: int,
        dataset_config: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed
        self.dataset_config = dataset_config

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        tok = AutoTokenizer.from_pretrained(self.tok_name)
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id

        print(f"[DataWorker] Connecting to {self.dataset_name}...")
        load_kwargs = {"split": "train", "streaming": True}
        if self.dataset_config:
            load_kwargs["name"] = self.dataset_config
        ds = load_dataset(self.dataset_name, **load_kwargs)
        ds = ds.shuffle(buffer_size=5_000, seed=seed)
        print(f"[DataWorker] Stream ready for {self.dataset_name}")

        if worker_info is not None:
            try:
                ds = ds.shard(
                    num_shards=worker_info.num_workers, index=worker_info.id
                )
            except Exception:
                ds = itertools.islice(
                    ds, worker_info.id, None, worker_info.num_workers
                )

        buffer: list[int] = []
        for example in ds:
            # Handle Phase 3 instruction-tuning format (messages column)
            if "messages" in example:
                try:
                    text = tok.apply_chat_template(
                        example["messages"], tokenize=False
                    )
                except Exception:
                    # GPT-NeoX has no default chat template; manual fallback.
                    # EOS after assistant turns teaches the model to stop.
                    parts: list[str] = []
                    for m in example["messages"]:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        parts.append(f"{role}: {content}")
                        if role == "assistant":
                            parts.append(tok.eos_token)
                    text = "\n".join(parts)
            else:
                text = example.get("text", "")

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


def make_loader(
    dataset_name: str,
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    step_num: int,
    dataset_config: str | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming :class:`DataLoader` for a training phase.

    Args:
        dataset_name: HuggingFace dataset identifier.
        seq_len: Sequence length for each sample.
        tokenizer_name: HuggingFace tokenizer identifier.
        batch_size: Micro-batch size.
        step_num: Current training step (used to vary the shuffle seed).
        dataset_config: Optional dataset config name (e.g. 'all' for smoltalk).

    Returns:
        A :class:`DataLoader` yielding ``(input_ids, labels)`` batches.
    """
    ds = TokenStream(dataset_name, seq_len, tokenizer_name, seed=42 + step_num, dataset_config=dataset_config)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
