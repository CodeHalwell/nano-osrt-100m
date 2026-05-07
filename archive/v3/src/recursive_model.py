"""Recursive-block NanoOSRT model for Modal deployment.

This module implements the production variant of the NanoOSRT architecture
that uses recursive weight sharing: 2 physical blocks are looped 6 times
to simulate 12 effective layers, with unique per-pass residual adapters to
give each virtual layer its own geometric identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nano_osrt.modal_config import ModalConfig
from nano_osrt.rope import apply_rope, compute_rope_freqs


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = int(dim * 8 / 3)
        hidden = 64 * ((hidden + 63) // 64)  # TC-align
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RecursiveBlock(nn.Module):
    """A single physical transformer block with RoPE and adapter support.

    Each forward pass accepts per-pass adapter matrices so that the same
    physical weights can produce different behaviour on each recursive loop.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_attn = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_ffn = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(
        self,
        x: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        B, S, D = x.shape

        # Per-pass low-rank residual adapter on activations.
        # NOT weight-LoRA (Hu et al.) — modulates hidden state directly.
        x_mod = x + adapter_scale * (x @ adapter_a @ adapter_b)

        # --- Causal Attention with RoPE ---
        h = self.norm_attn(x_mod)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.heads, self.head_dim)
        k = k.view(B, S, self.heads, self.head_dim)
        v = v.view(B, S, self.heads, self.head_dim)

        # RoPE applied before head transpose
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard causal SDPA — dispatches to FlashAttention-2 on H100.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Residual connects to x_mod (not x) so the adapter's geometric
        # shift survives into the FFN, not orphaned.
        x = x_mod + self.out_proj(attn_out)

        x = x + self.ffn(self.norm_ffn(x))
        return x


class RecursiveNanoOSRT(nn.Module):
    """Recursive-block NanoOSRT model (~104.5M parameters).

    Uses recursive weight sharing: *num_blocks* physical transformer blocks
    are looped *recursive_loops* times, giving ``num_blocks × recursive_loops``
    effective layers.  Unique per-pass low-rank adapters differentiate each
    virtual layer.

    Args:
        cfg: A :class:`ModalConfig` instance.
    """

    def __init__(self, cfg: ModalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Precomputed RoPE buffers (non-persistent = won't bloat checkpoints)
        cos, sin = compute_rope_freqs(cfg.seq_len, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [RecursiveBlock(cfg.dim, cfg.heads) for _ in range(cfg.num_blocks)]
        )

        total_pairs = cfg.num_blocks * cfg.recursive_loops
        self.adapters_a = nn.ParameterList(
            [
                nn.Parameter(torch.randn(cfg.dim, cfg.adapter_rank) * 0.01)
                for _ in range(total_pairs)
            ]
        )
        self.adapters_b = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(cfg.adapter_rank, cfg.dim))
                for _ in range(total_pairs)
            ]
        )
        self.adapter_scale = cfg.adapter_alpha / cfg.adapter_rank
        self.norm_loop = nn.RMSNorm(cfg.dim)
        self.norm_out = nn.RMSNorm(cfg.dim)

        # GPT-standard init: N(0, 0.02) prevents logit variance explosion
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass.

        Args:
            input_ids: Token indices of shape ``(B, S)``.

        Returns:
            ``(logits, loop_rms)`` where *logits* has shape
            ``(B, S, vocab_size)`` and *loop_rms* is a list of per-loop
            RMS values (as scalar tensors, no ``.item()`` graph breaks).
        """
        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        cos = self.rope_cos[:, :S, :, :]
        sin = self.rope_sin[:, :S, :, :]

        loop_rms: list[Tensor] = []

        for loop in range(self.cfg.recursive_loops):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.cfg.num_blocks + block_idx
                x = block(
                    x,
                    self.adapters_a[idx],
                    self.adapters_b[idx],
                    self.adapter_scale,
                    cos,
                    sin,
                )
            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < self.cfg.recursive_loops - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        logits = F.linear(x, self.embedding.weight)
        return logits, loop_rms
