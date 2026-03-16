"""NanoOSRT v4 — Recursive MoE Model.

3 physical blocks × 6 recursive loops = 18 effective layers.
Each block: causal attention + dense SwiGLU FFN + MoE FFN (parallel residual).
MoE: 1 shared expert (always active) + 11 routed experts (top-2).
Loop-aware router with learned loop embeddings.

HuggingFace-compatible from day one via PreTrainedModel.

Physical params: ~305M
Active params/token: ~155M
Effective params (recursive): ~930M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nano_osrt.v4_config import NanoOSRTv4Config


# ── RoPE ────────────────────────────────────────────────────────────────


def compute_rope_freqs(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Pre-compute RoPE cos/sin tensors. Shape: (1, seq_len, 1, dim)."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(2)
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin


# ── Expert FFN ──────────────────────────────────────────────────────────


class ExpertFFN(nn.Module):
    """Single expert: small SwiGLU network."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        hidden = 64 * ((hidden + 63) // 64)  # TC-align
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ── MoE Layer ───────────────────────────────────────────────────────────


class MoELayer(nn.Module):
    """Mixture of Experts with shared expert + top-k routing.

    Architecture:
        - 1 shared expert (always active, unconditional)
        - N routed experts (top-k selected per token)
        - Loop-aware router: receives hidden state + loop embedding

    Load balancing via auxiliary loss to prevent expert collapse.
    """

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__()
        self.num_routed = config.num_routed_experts
        self.top_k = config.top_k_experts
        self.num_loops = config.recursive_loops
        self.aux_loss_coeff = config.router_aux_loss_coeff

        # Shared expert (always active)
        self.shared_expert = ExpertFFN(config.dim, config.expert_hidden)

        # Routed experts
        self.experts = nn.ModuleList(
            [ExpertFFN(config.dim, config.expert_hidden) for _ in range(self.num_routed)]
        )

        # Router: projects hidden state to expert scores
        # Loop-aware: concatenates a learned loop embedding
        self.loop_embeddings = nn.Embedding(config.recursive_loops, config.dim)
        self.router = nn.Linear(config.dim * 2, self.num_routed, bias=False)

        # Store aux loss for training
        self.aux_loss: Tensor | None = None

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        """Forward pass through MoE.

        Args:
            x: Hidden states (B, S, dim).
            loop_idx: Current recursive loop index (0 to num_loops-1).

        Returns:
            Output tensor (B, S, dim).
        """
        B, S, D = x.shape

        # Shared expert (unconditional)
        shared_out = self.shared_expert(x)

        # Loop-aware routing
        loop_emb = self.loop_embeddings(
            torch.tensor(loop_idx, device=x.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, S, -1)  # (B, S, dim)

        router_input = torch.cat([x, loop_emb], dim=-1)  # (B, S, 2*dim)
        router_logits = self.router(router_input)  # (B, S, num_routed)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize top-k weights
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute auxiliary load balancing loss
        if self.training:
            self._compute_aux_loss(router_probs, top_k_indices, B, S)

        # Dispatch to selected experts
        # For small models, simple loop is fine (no expert parallelism needed)
        routed_out = torch.zeros_like(x)
        flat_x = x.reshape(-1, D)  # (B*S, D)
        flat_indices = top_k_indices.reshape(-1, self.top_k)  # (B*S, top_k)
        flat_weights = top_k_weights.reshape(-1, self.top_k)  # (B*S, top_k)

        for k in range(self.top_k):
            expert_indices = flat_indices[:, k]  # (B*S,)
            expert_weights = flat_weights[:, k]  # (B*S,)

            for expert_idx in range(self.num_routed):
                mask = expert_indices == expert_idx  # (B*S,)
                if mask.any():
                    expert_input = flat_x[mask]  # (num_tokens, D)
                    expert_output = self.experts[expert_idx](expert_input)
                    routed_out.view(-1, D)[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        return shared_out + routed_out

    def _compute_aux_loss(self, router_probs: Tensor, top_k_indices: Tensor, B: int, S: int) -> None:
        """Compute load balancing auxiliary loss (Switch Transformer style)."""
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(top_k_indices, self.num_routed).float()  # (B, S, top_k, num_routed)
        tokens_per_expert = one_hot.sum(dim=(0, 1, 2))  # (num_routed,)
        fraction_routed = tokens_per_expert / (B * S * self.top_k)

        # Average router probability per expert
        avg_prob = router_probs.mean(dim=(0, 1))  # (num_routed,)

        # Auxiliary loss: encourages uniform routing
        self.aux_loss = self.num_routed * (fraction_routed * avg_prob).sum()


# ── Dense FFN ───────────────────────────────────────────────────────────


class DenseSwiGLU(nn.Module):
    """Dense SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        hidden = 64 * ((hidden + 63) // 64)  # TC-align
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ── Recursive Block ─────────────────────────────────────────────────────


class RecursiveBlockV4(nn.Module):
    """Physical transformer block with attention + dense FFN + MoE FFN.

    Dense and MoE run as parallel residual paths:
        x = x + attn(norm(x))
        x = x + dense_ffn(norm(x)) + moe_ffn(norm(x), loop_idx)
    """

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim

        # Attention
        self.norm_attn = nn.RMSNorm(config.dim)
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

        # Dense FFN
        self.norm_dense = nn.RMSNorm(config.dim)
        self.ffn_dense = DenseSwiGLU(config.dim, config.dense_hidden)

        # MoE FFN (parallel with dense)
        self.norm_moe = nn.RMSNorm(config.dim)
        self.moe = MoELayer(config)

    def forward(
        self,
        x: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        loop_idx: int,
    ) -> Tensor:
        B, S, D = x.shape

        # Per-pass residual adapter (activation-space, not weight-LoRA)
        x_mod = x + adapter_scale * (x @ adapter_a @ adapter_b)

        # ── Causal Attention with RoPE ──
        h = self.norm_attn(x_mod)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.heads, self.head_dim)
        k = k.view(B, S, self.heads, self.head_dim)
        v = v.view(B, S, self.heads, self.head_dim)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        x = x_mod + self.out_proj(attn_out)

        # ── Parallel Dense + MoE FFN ──
        h_dense = self.ffn_dense(self.norm_dense(x))
        h_moe = self.moe(self.norm_moe(x), loop_idx)
        x = x + h_dense + h_moe

        return x


# ── Main Model ──────────────────────────────────────────────────────────


class NanoOSRTv4PreTrainedModel(PreTrainedModel):
    """Base class for NanoOSRT v4 models."""

    config_class = NanoOSRTv4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class NanoOSRTv4Model(NanoOSRTv4PreTrainedModel):
    """Core NanoOSRT v4 model (without LM head)."""

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # RoPE (non-persistent buffers)
        cos, sin = compute_rope_freqs(config.max_position_embeddings, config.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Physical blocks
        self.blocks = nn.ModuleList(
            [RecursiveBlockV4(config) for _ in range(config.num_blocks)]
        )

        # Per-pass adapters (num_blocks × recursive_loops pairs)
        total_pairs = config.num_blocks * config.recursive_loops
        self.adapters_a = nn.ParameterList(
            [nn.Parameter(torch.randn(config.dim, config.adapter_rank) * 0.01)
             for _ in range(total_pairs)]
        )
        self.adapters_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.adapter_rank, config.dim))
             for _ in range(total_pairs)]
        )
        self.adapter_scale = config.adapter_alpha / config.adapter_rank

        # Inter-loop normalization
        self.norm_loop = nn.RMSNorm(config.dim)
        self.norm_out = nn.RMSNorm(config.dim)

        self.post_init()

    def forward(self, input_ids: Tensor, **kwargs) -> tuple[Tensor, list[Tensor], Tensor]:
        """Forward pass.

        Returns:
            (hidden_states, loop_rms_list, total_aux_loss)
        """
        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        cos = self.rope_cos[:, :S, :, :]
        sin = self.rope_sin[:, :S, :, :]

        loop_rms: list[Tensor] = []
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for loop in range(self.config.recursive_loops):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.config.num_blocks + block_idx
                x = block(
                    x,
                    self.adapters_a[idx],
                    self.adapters_b[idx],
                    self.adapter_scale,
                    cos, sin,
                    loop_idx=loop,
                )
                # Accumulate MoE auxiliary loss
                if block.moe.aux_loss is not None:
                    total_aux_loss = total_aux_loss + block.moe.aux_loss

            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < self.config.recursive_loops - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        return x, loop_rms, total_aux_loss


class NanoOSRTv4ForCausalLM(NanoOSRTv4PreTrainedModel):
    """NanoOSRT v4 with causal language modeling head.

    HuggingFace-compatible: supports from_pretrained(), generate(), push_to_hub().
    """

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__(config)
        self.model = NanoOSRTv4Model(config)
        # Weight-tied LM head
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embedding = value

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden, loop_rms, aux_loss = self.model(input_ids)

        # Weight-tied LM head
        logits = F.linear(hidden, self.model.embedding.weight)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :self.config.real_vocab_size].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.real_vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add MoE auxiliary loss during training
            loss = loss + self.config.router_aux_loss_coeff * aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int | None = None,
        **kwargs,
    ) -> Tensor:
        """Generate tokens autoregressively with top-p/top-k sampling."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_position_embeddings:]
            outputs = self.forward(context)
            next_logits = outputs.logits[:, -1, :self.config.real_vocab_size].float()

            if temperature > 0:
                next_logits = next_logits / temperature

                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits.scatter_(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated
