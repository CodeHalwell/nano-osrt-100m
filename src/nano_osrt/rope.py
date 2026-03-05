"""Rotary Position Embedding (RoPE) utilities."""

import torch
from torch import Tensor


def compute_rope_freqs(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Pre-compute RoPE cosine and sine frequency tensors.

    Args:
        seq_len: Maximum sequence length.
        dim: Per-head dimension (must be even).
        theta: Rotation base frequency.
        device: Target device for the tensors.

    Returns:
        ``(cos, sin)`` each of shape ``(1, seq_len, 1, dim)`` for
        broadcasting with ``(B, S, heads, head_dim)`` tensors.
    """
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2]
            / dim
        )
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # Shape: (1, S, 1, head_dim) for broadcasting with (B, S, heads, head_dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2)
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to *x*.

    Args:
        x: Input tensor of shape ``(..., dim)`` where *dim* is even.
        cos: Cosine frequencies from :func:`compute_rope_freqs`.
        sin: Sine frequencies from :func:`compute_rope_freqs`.

    Returns:
        Tensor of the same shape as *x* with RoPE applied.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin
