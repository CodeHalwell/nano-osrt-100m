"""NanoOSRT v4 — Recursive MoE Model.

3 physical blocks × 6 recursive loops = 18 effective layers.
Each block: causal attention + dense SwiGLU FFN + MoE FFN (parallel residual).
MoE: 1 shared expert (always active) + 11 routed experts (top-2).
Loop-aware router with learned loop embeddings.

HuggingFace-compatible from day one via PreTrainedModel.

Physical params: ~356M (with 64K vocab)
Active params/token: ~180M
Effective params (recursive): ~2.1B
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
) -> tuple[Tensor, Tensor]:
    """Pre-compute RoPE cos/sin tensors. Shape: (1, seq_len, 1, dim)."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even dimension, got dim={dim}")
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

        # Pre-register loop index tensors to avoid torch.tensor() in forward
        self.register_buffer(
            "loop_indices", torch.arange(config.recursive_loops), persistent=False
        )

        # Store losses for training (load-balance and z-loss kept separate)
        self.load_balance_loss: Tensor | None = None
        self.z_loss: Tensor | None = None

        # Expert utilization stats (detached, for monitoring only)
        self.expert_counts: Tensor | None = None  # (num_routed,)

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
        self.expert_counts = None

        # Shared expert (unconditional)
        shared_out = self.shared_expert(x)

        # Loop-aware routing with cached index tensor
        loop_emb = self.loop_embeddings(
            self.loop_indices[loop_idx]
        ).unsqueeze(0).unsqueeze(0).expand(B, S, -1)  # (B, S, dim)

        router_input = torch.cat([x, loop_emb], dim=-1)  # (B, S, 2*dim)
        router_logits = self.router(router_input)  # (B, S, num_routed)

        # Sigmoid gating: each expert gate is independent (DeepSeek-V3 style)
        router_probs = torch.sigmoid(router_logits)

        # Top-k selection (no renormalization needed with sigmoid)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Compute load balancing + z-loss (kept separate for independent scaling)
        if self.training:
            self._compute_losses(router_logits, router_probs, top_k_indices, B, S)

        # Batched scatter-gather dispatch: each expert runs exactly once
        routed_out = self._dispatch_experts(
            x.reshape(-1, D), top_k_indices.reshape(-1, self.top_k),
            top_k_weights.reshape(-1, self.top_k), D,
        )

        return shared_out + routed_out.view(B, S, D)

    def _dispatch_experts(
        self, flat_x: Tensor, flat_indices: Tensor, flat_weights: Tensor, D: int,
    ) -> Tensor:
        """Batch tokens by expert, run each expert once, scatter back.

        Instead of looping over top_k × num_experts (22 iterations), this
        sorts tokens by expert assignment and runs each expert on its full
        batch in a single call.
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
        Also stores per-expert token counts for monitoring.
        """
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(top_k_indices, self.num_routed).float()  # (B, S, top_k, num_routed)
        tokens_per_expert = one_hot.sum(dim=(0, 1, 2))  # (num_routed,)
        fraction_routed = tokens_per_expert / (B * S * self.top_k)

        # Average router probability per expert (sigmoid — matches gating)
        avg_prob = router_probs.mean(dim=(0, 1))  # (num_routed,)

        # Load balancing loss (Switch Transformer style)
        self.load_balance_loss = self.num_routed * (fraction_routed * avg_prob).sum()

        # Router z-loss: penalise large logits to prevent overconfident routing
        # For sigmoid routing, penalise squared logits directly (large |logit|
        # pushes sigmoid toward 0 or 1, reducing routing flexibility).
        self.z_loss = router_logits.pow(2).mean()

        # Expert utilization counts (detached for monitoring; no gradient needed)
        self.expert_counts = tokens_per_expert.detach()


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

        # Learnable gates for parallel dense + MoE residual scaling
        # Initialised to 0.5 so combined contribution starts at 1.0×
        self.dense_gate = nn.Parameter(torch.tensor(0.5))
        self.moe_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        loop_idx: int,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
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

        # Apply RoPE to new positions only (cos/sin pre-sliced by caller)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Transpose to (B, heads, S, head_dim) for attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Concatenate with cached KV from previous generation steps
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Preserve autoregressive masking for cached multi-token decode.
        # - Prefill (Q_len == K_len): use the built-in causal path.
        # - Single-token decode (Q_len == 1): no mask is needed.
        # - Cached chunked decode (past_len > 0 and Q_len > 1): provide an
        #   explicit causal mask shifted by the cache length so earlier new
        #   tokens cannot attend to later new tokens in the same chunk.
        q_len = q.shape[2]
        k_len = k.shape[2]
        if past_len > 0 and q_len > 1:
            attn_mask = torch.full(
                (q_len, k_len),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )
            attn_mask = torch.triu(attn_mask, diagonal=1 + past_len)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        else:
            is_causal = (q_len == k_len)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        x = x_mod + self.out_proj(attn_out)

        # ── Parallel Dense + MoE FFN with learnable scaling ──
        h_dense = self.ffn_dense(self.norm_dense(x))
        h_moe = self.moe(self.norm_moe(x), loop_idx)
        x = x + self.dense_gate * h_dense + self.moe_gate * h_moe

        return x, present_kv


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

        # Gradient checkpointing (disabled by default, enable for long sequences)
        self.gradient_checkpointing = False

        self.post_init()

    @torch.no_grad()
    def get_moe_stats(self) -> dict[str, float]:
        """Collect routing stats from all MoE layers.

        Call after forward(). Returns dict with:
            moe/expert_balance_cv: coefficient of variation
            moe/max_expert_share: busiest expert fraction
            moe/dead_experts: experts that received 0 tokens
        """
        total_counts = None
        for block in self.blocks:
            if block.moe.expert_counts is not None:
                if total_counts is None:
                    total_counts = block.moe.expert_counts.clone()
                else:
                    total_counts = total_counts + block.moe.expert_counts
        if total_counts is None:
            return {}
        total = total_counts.sum().item()
        if total == 0:
            return {}
        fractions = total_counts / total
        mean_frac = fractions.mean().item()
        std_frac = fractions.std().item()
        return {
            "moe/expert_balance_cv": std_frac / mean_frac if mean_frac > 0 else 0.0,
            "moe/max_expert_share": fractions.max().item(),
            "moe/dead_experts": int((total_counts == 0).sum().item()),
        }

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[Tensor, list[Tensor], Tensor, Tensor, list[tuple[Tensor, Tensor]] | None]:
        """Forward pass.

        Args:
            input_ids: Token indices (B, S).
            past_key_values: KV cache — list of (K, V) tuples, one per
                effective layer (num_blocks × recursive_loops entries).
                Each tensor has shape (B, heads, cached_S, head_dim).
            use_cache: Whether to return updated KV cache for generation.

        Returns:
            (hidden_states, loop_rms_list, total_load_balance_loss,
             total_z_loss, present_key_values)
        """
        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        expected_past_layers = self.config.num_blocks * self.config.recursive_loops

        # Validate KV cache shape/length before indexing into it.
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != expected_past_layers:
                raise ValueError(
                    "Invalid past_key_values: expected "
                    f"{expected_past_layers} entries (num_blocks * recursive_loops), "
                    f"got {len(past_key_values)}."
                )
            for idx, layer_past in enumerate(past_key_values):
                if layer_past is not None and (
                    not isinstance(layer_past, tuple) or len(layer_past) != 2
                ):
                    raise ValueError(
                        "Invalid past_key_values: each entry must be None or a "
                        f"(key, value) tuple, but entry {idx} has type "
                        f"{type(layer_past).__name__}."
                    )
            if past_key_values[0] is not None:
                key, value = past_key_values[0]
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    raise ValueError(
                        "Invalid past_key_values: each non-None entry must contain "
                        "torch.Tensor key/value tensors."
                    )
                past_length = key.shape[2]

        required_seq_len = past_length + S
        if required_seq_len <= self.rope_cos.shape[1]:
            cos = self.rope_cos[:, past_length:required_seq_len, :, :]
            sin = self.rope_sin[:, past_length:required_seq_len, :, :]
        else:
            rope_cos, rope_sin = compute_rope_freqs(
                required_seq_len,
                self.rope_cos.shape[-1],
                theta=getattr(self.config, "rope_theta", 10000.0),
                device=x.device,
            )
            cos = rope_cos[:, past_length:required_seq_len, :, :].to(
                device=self.rope_cos.device, dtype=self.rope_cos.dtype
            )
            sin = rope_sin[:, past_length:required_seq_len, :, :].to(
                device=self.rope_sin.device, dtype=self.rope_sin.dtype
            )

        loop_rms: list[Tensor] = []
        total_lb_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)

        use_ckpt = self.gradient_checkpointing and self.training
        if use_ckpt and (use_cache or past_key_values is not None):
            raise ValueError(
                "KV caching (use_cache=True or past_key_values) is incompatible with "
                "gradient checkpointing. Disable gradient_checkpointing or set "
                "use_cache=False."
            )
        presents: list[tuple[Tensor, Tensor]] | None = [] if use_cache else None

        for loop in range(self.config.recursive_loops):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.config.num_blocks + block_idx
                adapter_a = self.adapters_a[idx]
                adapter_b = self.adapters_b[idx]

                layer_past = past_key_values[idx] if past_key_values is not None else None

                if use_ckpt:
                    # Gradient checkpointing (training only, never with cache).
                    # Wrap in closure to capture non-tensor args.
                    def _block_fn(
                        _x, _a, _b, _cos, _sin,
                        _block=block, _scale=self.adapter_scale, _loop=loop,
                    ):
                        return _block(_x, _a, _b, _scale, _cos, _sin, _loop)[0]

                    x = gradient_checkpoint(
                        _block_fn, x, adapter_a, adapter_b, cos, sin,
                        use_reentrant=False,
                    )
                else:
                    x, present_kv = block(
                        x, adapter_a, adapter_b,
                        self.adapter_scale, cos, sin,
                        loop_idx=loop,
                        past_key_value=layer_past,
                        use_cache=use_cache,
                    )
                    if presents is not None:
                        presents.append(present_kv)

                # Accumulate MoE losses (kept separate for independent scaling)
                if block.moe.load_balance_loss is not None:
                    total_lb_loss = total_lb_loss + block.moe.load_balance_loss
                if block.moe.z_loss is not None:
                    total_z_loss = total_z_loss + block.moe.z_loss

            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < self.config.recursive_loops - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        return x, loop_rms, total_lb_loss, total_z_loss, presents


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
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden, loop_rms, lb_loss, z_loss, presents = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

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
            # MoE losses scaled independently:
            #   load-balance at router_aux_loss_coeff  (default 0.01)
            #   z-loss at router_z_loss_coeff          (default 0.001)
            loss = (
                loss
                + self.config.router_aux_loss_coeff * lb_loss
                + self.config.router_z_loss_coeff * z_loss
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
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
        use_cache: bool = True,
        **kwargs,
    ) -> Tensor:
        """Generate tokens autoregressively with KV cache and top-p/top-k sampling.

        On the first step the full prompt is processed and KV states are
        cached.  Subsequent steps feed only the latest token, reusing the
        cache for O(1) attention per new token instead of O(n).
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                # Incremental decode: only the latest token
                model_input = generated[:, -1:]
            else:
                # Prefill: full context (capped to max position embeddings)
                model_input = generated[:, -self.config.max_position_embeddings:]

            outputs = self.forward(
                model_input,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            if use_cache:
                past_key_values = outputs.past_key_values
                # Trim KV cache to a sliding window so memory stays bounded and
                # RoPE positions never exceed max_position_embeddings.
                max_len = self.config.max_position_embeddings
                first = past_key_values[0] if past_key_values else None
                if (
                    first is not None
                    and isinstance(first, tuple)
                    and len(first) == 2
                    and isinstance(first[0], torch.Tensor)
                ):
                    cached_len = first[0].shape[2]
                    if cached_len > max_len:
                        past_key_values = [
                            (kv[0][:, :, -max_len:, :], kv[1][:, :, -max_len:, :])
                            if kv is not None else None
                            for kv in past_key_values
                        ]

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
