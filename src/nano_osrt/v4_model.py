"""NanoOSRT v4 — Recursive MoE Model.

3 physical blocks × 6 recursive loops = 18 block applications.
Each block: causal attention (parallel adapter branch) + dense SwiGLU
FFN + MoE FFN with parallel residual gating (dense_gate=1.0, moe_gate=0.01
init — dense-first warmup).

MoE: 1 shared expert (always active) + 11 routed experts (top-2,
sigmoid gating). Loop-aware router adds a learned per-loop embedding
to the hidden state before projecting (additive, not concat).

HuggingFace-compatible from day one via PreTrainedModel.

Default config (32K vocab, dim=1536):
  Physical params      : ~306M
  Active / token (body): ~130M (shared expert + 2 of 11 routed + dense + attn)
  Block applications    : 18    (num_blocks × recursive_loops)

Note: "effective parameters" is a misleading narrative at this scale.
The model has 306M unique weights — it just runs them 6 times per forward
with distinct per-pass adapters and loop-conditioned routing. Think of it
as iterative refinement on a fixed parameter budget, not a 1:1 substitute
for a dense 1.8B model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nano_osrt.v4_config import NanoOSRTv4Config

# ── RoPE ────────────────────────────────────────────────────────────────


def compute_rope_freqs(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    scaling: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Pre-compute RoPE cos/sin tensors. Shape: (1, seq_len, 1, dim).

    Supports optional NTK-aware scaling for extending context beyond
    training length. Pass scaling={"type": "ntk", "factor": 4.0} in the
    model config to extend the effective max position by ~4x at
    inference time without retraining.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even dimension, got dim={dim}")

    # NTK-aware scaling: rescale theta so higher positions fit.
    # factor = desired_context / training_context.
    effective_theta = theta
    if scaling is not None:
        stype = scaling.get("type", "").lower()
        factor = float(scaling.get("factor", 1.0))
        if stype == "ntk" and factor > 1.0:
            effective_theta = theta * (factor ** (dim / (dim - 2)))

    freqs = 1.0 / (
        effective_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim
        )
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
        - Sigmoid gating (DeepSeek-V3 style) for independent expert scores
        - Batched scatter-gather dispatch (each expert runs once per forward)

    Load balancing via auxiliary loss + router z-loss to prevent expert collapse.
    """

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__()
        self.num_routed = config.num_routed_experts
        self.top_k = config.top_k_experts
        self.num_loops = config.recursive_loops
        self.z_loss_coeff = config.router_z_loss_coeff

        # Router noise schedule. Stored as a scalar tensor buffer so
        # torch.compile sees it as a graph input (no .item() graph break)
        # and the training loop can update it in-place each step to
        # implement the linear anneal from router_noise_std_init down to
        # router_noise_std_final. Setting it to 0.0 temporarily lets the
        # training loop run "clean" diagnostic forwards that measure
        # learned routing without the noise perturbation.
        self.register_buffer(
            "noise_std",
            torch.tensor(config.router_noise_std_init, dtype=torch.float32),
            persistent=False,
        )

        # Shared expert (always active)
        self.shared_expert = ExpertFFN(config.dim, config.expert_hidden)

        # Routed experts
        self.experts = nn.ModuleList(
            [ExpertFFN(config.dim, config.expert_hidden) for _ in range(self.num_routed)]
        )

        # Router: projects hidden state to expert scores.
        # Loop-aware routing: add a learned per-loop embedding to the hidden
        # state before the router projection. Addition (not concat) keeps
        # router parameters at (dim -> num_routed) instead of (2*dim -> num_routed),
        # halving router params with equivalent expressive power because a
        # linear layer over cat([x, e]) can always be re-expressed as the
        # sum of two linear layers over x and e.
        #
        # Loop embeddings get their own larger init_std
        # (config.loop_embedding_init_std, default 0.1) so that they
        # actually shift routing between loops at init — the default
        # 0.02 was too small compared to the x-signal std ~0.78, making
        # early routing loop-invariant.
        #
        # NOTE: HF PreTrainedModel.post_init() walks the module tree and
        # calls _init_weights on every nn.Embedding with
        # initializer_range (default 0.02), which would silently stomp
        # whatever we set here. We tag this embedding with
        # _osrt_init_std so the overridden _init_weights in
        # NanoOSRTv4PreTrainedModel uses config.loop_embedding_init_std
        # instead. Do NOT call nn.init.normal_ here — post_init will
        # overwrite it anyway; the tag is what actually takes effect.
        self.loop_embeddings = nn.Embedding(config.recursive_loops, config.dim)
        self.loop_embeddings._osrt_init_std = config.loop_embedding_init_std
        self.router = nn.Linear(config.dim, self.num_routed, bias=False)

        # Pre-register loop index tensors to avoid torch.tensor() in forward
        self.register_buffer(
            "loop_indices", torch.arange(config.recursive_loops), persistent=False
        )

        # Store losses for training (load-balance and z-loss kept separate)
        self.load_balance_loss: Tensor | None = None
        self.z_loss: Tensor | None = None

        # Per-loop telemetry — indexed [0..num_loops-1], overwritten on
        # each forward. Used by the training loop for per-loop W&B metrics.
        # Keep as plain tensors (no grad) so logging is free.
        #
        # router_prob_entropy: entropy of the normalised sigmoid probs. Can
        # be near-uniform even when top-k is collapsed — not a reliable
        # collapse detector by itself.
        #
        # assignment_entropy: entropy of the actual top-k token assignment
        # distribution. Uniform routing gives ~ln(num_routed)=2.40, full
        # top-2 collapse gives ~ln(2)=0.69. THIS is the reliable metric.
        self.last_router_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_assignment_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_expert_fraction: list[list[float]] = [
            [0.0] * config.num_routed_experts for _ in range(config.recursive_loops)
        ]

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        """Forward pass through MoE.

        Args:
            x: Hidden states (B, S, dim).
            loop_idx: Current recursive loop index (0 to num_loops-1).

        Returns:
            Output tensor (B, S, dim).
        """
        B, S, D = x.shape

        # Reset losses at start of every forward (prevents stale values in eval)
        self.load_balance_loss = None
        self.z_loss = None

        # Shared expert (unconditional)
        shared_out = self.shared_expert(x)

        # Loop-aware routing: add the learned per-loop embedding to x.
        # Broadcasts (dim,) -> (B, S, dim) without materialising an
        # expanded tensor, then router projects (dim -> num_routed).
        loop_emb = self.loop_embeddings(self.loop_indices[loop_idx])  # (dim,)
        router_input = x + loop_emb  # (B, S, dim)
        router_logits = self.router(router_input)  # (B, S, num_routed)

        # Training-time router noise (Switch-Transformer style). Breaks
        # deterministic top-k tie-locking at init when all sigmoid values
        # are near 0.5 and the same two indices would win every time.
        # noise_std is a scalar tensor buffer — zero means clean routing,
        # so no Python-level branching is required and torch.compile
        # stays happy. Training loop sets it to 0.0 temporarily during
        # diagnostic forwards to measure learned (non-random) routing.
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # Sigmoid gating: each expert gate is independent (DeepSeek-V3 style)
        router_probs = torch.sigmoid(router_logits)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalise the selected weights so their sum is 1.0 per token.
        # Without this, if all sigmoid values are small (e.g. early training
        # with low-magnitude logits) the routed MoE output magnitude is
        # also small and gradients into the router are suppressed. With
        # renormalisation, the routed output has a consistent scale
        # regardless of the absolute sigmoid magnitudes.
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Compute load balancing + z-loss (kept separate for independent scaling)
        if self.training:
            self._compute_losses(router_logits, router_probs, top_k_indices, B, S)
            self._record_telemetry(router_probs, top_k_indices, loop_idx)

        # Batched scatter-gather dispatch: each expert runs exactly once
        routed_out = self._dispatch_experts(
            x.reshape(-1, D), top_k_indices.reshape(-1, self.top_k),
            top_k_weights.reshape(-1, self.top_k), D,
        )

        return shared_out + routed_out.view(B, S, D)

    @torch._dynamo.disable
    def _dispatch_experts(
        self, flat_x: Tensor, flat_indices: Tensor, flat_weights: Tensor, D: int,
    ) -> Tensor:
        """Batch tokens by expert, run each expert once, scatter back.

        Instead of looping over top_k × num_experts (22 iterations), this
        sorts tokens by expert assignment and runs each expert on its full
        batch in a single call.

        @torch._dynamo.disable: per-expert batch size is data-dependent,
        which causes torch.compile to exceed its recompile_limit and fall
        back to full-graph eager. Keeping only this method in eager lets
        the rest of the model (attention, dense FFN, RoPE) stay compiled.
        """
        N = flat_x.shape[0]  # B*S

        # Expand for top-k: each token appears top_k times
        x_rep = flat_x.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, D)  # (N*top_k, D)
        idx_flat = flat_indices.reshape(-1)        # (N*top_k,)
        w_flat = flat_weights.reshape(-1, 1)       # (N*top_k, 1)

        # Sort by expert for batched execution
        sorted_order = idx_flat.argsort(stable=True)
        sorted_experts = idx_flat[sorted_order]
        sorted_x = x_rep[sorted_order]
        sorted_w = w_flat[sorted_order]

        # Find token counts per expert
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_routed)
        expert_counts_list = expert_counts.tolist()

        # Run each expert once on its batch of tokens
        sorted_out = torch.empty_like(sorted_x)
        offset = 0
        for eid in range(self.num_routed):
            count = expert_counts_list[eid]
            if count == 0:
                continue
            expert_in = sorted_x[offset : offset + count]
            sorted_out[offset : offset + count] = self.experts[eid](expert_in) * sorted_w[offset : offset + count]
            offset += count

        # Unsort and reduce across top-k
        result = torch.zeros_like(x_rep)
        result[sorted_order] = sorted_out
        # Sum over the top_k dimension
        result = result.view(N, self.top_k, D).sum(dim=1)
        return result

    def _compute_losses(
        self, router_logits: Tensor, router_probs: Tensor,
        top_k_indices: Tensor, B: int, S: int,
    ) -> None:
        """Compute load balancing loss and router z-loss (stored separately).

        Load balancing (Switch Transformer style): encourages uniform routing.
        Z-loss (ST-MoE / PaLM): penalises large router logits for stability.

        Losses are stored on separate attributes so the training loss can
        apply router_aux_loss_coeff and router_z_loss_coeff independently.
        """
        # Fraction of tokens routed to each expert — bincount avoids the
        # (B, S, top_k, num_routed) one-hot allocation which spikes memory
        # at seq_len 4096/8192. Cast is needed because bincount returns int.
        flat_indices = top_k_indices.reshape(-1)  # (B * S * top_k,)
        tokens_per_expert = torch.bincount(
            flat_indices, minlength=self.num_routed
        ).to(router_probs.dtype)
        fraction_routed = tokens_per_expert / (B * S * self.top_k)

        # Average router probability per expert (sigmoid — matches gating)
        avg_prob = router_probs.mean(dim=(0, 1))  # (num_routed,)

        # Load balancing loss (Switch Transformer style)
        self.load_balance_loss = self.num_routed * (fraction_routed * avg_prob).sum()

        # Router z-loss: penalise large logits to prevent overconfident routing
        # For sigmoid routing, penalise squared logits directly (large |logit|
        # pushes sigmoid toward 0 or 1, reducing routing flexibility).
        self.z_loss = router_logits.pow(2).mean()

    @torch.no_grad()
    def _record_telemetry(
        self, router_probs: Tensor, top_k_indices: Tensor, loop_idx: int,
    ) -> None:
        """Record per-loop routing telemetry.

        Called inside forward() during training only. Values live on CPU
        as plain Python lists so the training loop can log them cheaply.

        Two entropy metrics:
          - router_prob_entropy: entropy of the normalised sigmoid probs.
            CAN BE MISLEADING — near-uniform sigmoid can still produce
            deterministic top-k (the failure mode we saw at step 200 of
            the first sanity run).
          - assignment_entropy: entropy of the actual top-k token
            assignment distribution. Uniform routing over 11 experts
            gives ~ln(11) = 2.40; full top-2 collapse gives ~ln(2) = 0.69.
            THIS is the reliable collapse detector.
        """
        # 1. Router-prob entropy (kept for backwards compat + sanity)
        norm = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-8)
        prob_entropy = -(norm * (norm + 1e-8).log()).sum(dim=-1).mean()
        if 0 <= loop_idx < len(self.last_router_entropy):
            self.last_router_entropy[loop_idx] = prob_entropy.item()

        # 2. Assignment entropy + per-expert fractions
        flat = top_k_indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_routed).float()
        total = counts.sum().clamp_min(1)
        fractions_tensor = counts / total
        fractions = fractions_tensor.tolist()
        if 0 <= loop_idx < len(self.last_expert_fraction):
            self.last_expert_fraction[loop_idx] = fractions

        # Entropy over the actual assignment distribution (the real metric)
        assign_entropy = -(
            fractions_tensor * (fractions_tensor + 1e-8).log()
        ).sum()
        if 0 <= loop_idx < len(self.last_assignment_entropy):
            self.last_assignment_entropy[loop_idx] = assign_entropy.item()


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

        # Learnable gates for parallel dense + MoE residual scaling.
        # Init dense_gate=1.0, moe_gate=0.01 so the model starts as a
        # (nearly) pure dense network and gradually blends in the MoE path
        # as the router learns to route sensibly. This avoids destabilising
        # early training with a half-random MoE contribution before the
        # experts have specialised.
        self.dense_gate = nn.Parameter(torch.tensor(1.0))
        self.moe_gate = nn.Parameter(torch.tensor(0.01))

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

        # Per-pass residual adapter (activation-space, not weight-LoRA).
        # Treated as a strictly additive parallel branch so the main
        # residual stream stays clean — attention and FFN see the
        # unadapted x, which is safer early in training when B is
        # zero-initialised and the adapter direction is still noise.
        adapter_out = adapter_scale * (x @ adapter_a @ adapter_b)

        # ── Causal Attention with RoPE (reads x, not x_mod) ──
        h = self.norm_attn(x)
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

        # Attention residual + adapter (parallel branches into x)
        x = x + self.out_proj(attn_out) + adapter_out

        # ── Parallel Dense + MoE FFN with learnable scaling ──
        h_dense = self.ffn_dense(self.norm_dense(x))
        h_moe = self.moe(self.norm_moe(x), loop_idx)
        x = x + self.dense_gate * h_dense + self.moe_gate * h_moe

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
            # HF's post_init() walks the tree and calls _init_weights on
            # every nn.Embedding with initializer_range (default 0.02).
            # The MoE layer's loop_embeddings needs a LARGER init std
            # (config.loop_embedding_init_std, default 0.1) so routing
            # can actually differentiate across recursive loops at init.
            # We tag that specific embedding in MoELayer.__init__ with
            # a `_osrt_init_std` attribute so we use it here instead of
            # the generic initializer_range.
            custom_std = getattr(module, "_osrt_init_std", None)
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=custom_std if custom_std is not None else std,
            )


class NanoOSRTv4Model(NanoOSRTv4PreTrainedModel):
    """Core NanoOSRT v4 model (without LM head)."""

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # RoPE (non-persistent buffers). Optional NTK scaling extends
        # the effective context beyond max_position_embeddings at
        # inference time — set config.rope_scaling = {"type": "ntk",
        # "factor": 4.0} and bump max_position_embeddings accordingly.
        cos, sin = compute_rope_freqs(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            scaling=config.rope_scaling,
        )
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

        # Gradient checkpointing (disabled by default, enable for long sequences)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(self, input_ids: Tensor, **kwargs) -> tuple[Tensor, list[Tensor], Tensor, Tensor]:
        """Forward pass.

        Returns:
            (hidden_states, loop_rms_list, total_load_balance_loss, total_z_loss)
        """
        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        cos = self.rope_cos[:, :S, :, :]
        sin = self.rope_sin[:, :S, :, :]

        loop_rms: list[Tensor] = []
        total_lb_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)

        use_ckpt = self.gradient_checkpointing and self.training

        for loop in range(self.config.recursive_loops):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.config.num_blocks + block_idx
                adapter_a = self.adapters_a[idx]
                adapter_b = self.adapters_b[idx]
                if use_ckpt:
                    # Wrap in closure to capture non-tensor args (adapter_scale,
                    # loop_idx) — gradient_checkpoint only accepts tensor inputs.
                    def _block_fn(
                        _x, _a, _b, _cos, _sin,
                        _block=block, _scale=self.adapter_scale, _loop=loop,
                    ):
                        return _block(_x, _a, _b, _scale, _cos, _sin, _loop)

                    x = gradient_checkpoint(
                        _block_fn, x, adapter_a, adapter_b, cos, sin,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x, adapter_a, adapter_b,
                        self.adapter_scale, cos, sin,
                        loop_idx=loop,
                    )
                # Accumulate MoE losses (kept separate for independent scaling)
                if block.moe.load_balance_loss is not None:
                    total_lb_loss = total_lb_loss + block.moe.load_balance_loss
                if block.moe.z_loss is not None:
                    total_z_loss = total_z_loss + block.moe.z_loss

            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < self.config.recursive_loops - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        return x, loop_rms, total_lb_loss, total_z_loss


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
        hidden, loop_rms, lb_loss, z_loss = self.model(input_ids)

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
            # MoE losses are accumulated across all (num_blocks × recursive_loops)
            # effective MoE applications (18 for default v4). Normalise by that
            # count so the configured coefficient matches the per-layer weight
            # instead of being silently multiplied by 18.
            n_moe_layers = self.config.num_blocks * self.config.recursive_loops
            lb_norm = lb_loss / n_moe_layers
            z_norm = z_loss / n_moe_layers
            loss = (
                loss
                + self.config.router_aux_loss_coeff * lb_norm
                + self.config.router_z_loss_coeff * z_norm
            )

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
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumprobs - sorted_probs >= top_p
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
