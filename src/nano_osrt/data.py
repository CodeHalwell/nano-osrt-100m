"""Data utilities for nano-osrt-100m."""

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset


class TokenDataset(Dataset):
    """Memory-mapped token dataset backed by a flat binary file of uint16 tokens.

    Expected format: a single binary file produced by pre-tokenising text with
    tiktoken and saving via ``np.array(tokens, dtype=np.uint16).tofile(path)``.
    """

    def __init__(self, data_path: str | Path, block_size: int) -> None:
        self.block_size = block_size
        data_path = Path(data_path)
        assert data_path.exists(), f"Data file not found: {data_path}"
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = torch.from_numpy(
            self.data[idx : idx + self.block_size + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class StreamingTokenDataset(IterableDataset):
    """Streaming dataset that yields random blocks from a memory-mapped file.

    Suitable for very large datasets where random-access indexing is costly.
    """

    def __init__(
        self,
        data_path: str | Path,
        block_size: int,
        seed: int = 42,
    ) -> None:
        self.block_size = block_size
        self.seed = seed
        data_path = Path(data_path)
        assert data_path.exists(), f"Data file not found: {data_path}"
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_chunks = len(self.data) - block_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id)
        while True:
            idx = rng.integers(0, self.n_chunks)
            chunk = torch.from_numpy(
                self.data[idx : idx + self.block_size + 1].astype(np.int64)
            )
            yield chunk[:-1], chunk[1:]


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample a random batch from a numpy memory-mapped array.

    Args:
        data: 1-D array of token ids (uint16 or int64).
        block_size: Context length.
        batch_size: Number of sequences per batch.
        device: Target device.

    Returns:
        (x, y) each of shape (batch_size, block_size).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(data[i : i + block_size].astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    if "cuda" in str(device):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def load_data_split(
    data_dir: str | Path,
    split: str,
) -> np.ndarray:
    """Load a pre-tokenised split (train/val) from *data_dir*.

    Expects files named ``train.bin`` and ``val.bin`` in *data_dir*.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.bin"
    assert path.exists(), (
        f"Data file '{path}' not found. "
        "Run your tokenisation script to generate train.bin / val.bin first."
    )
    return np.memmap(path, dtype=np.uint16, mode="r")
