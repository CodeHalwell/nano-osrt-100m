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

import math

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
    """Mixture of Experts with shared expert + soft→hard top-k routing.

    Architecture:
        - 1 shared expert (always active, unconditional)
        - N routed experts, dispatched via a three-phase schedule:
            mode 0 (soft_all): every expert runs on every token, weighted
              by softmax(router_logits / tau). Used for soft_warmup_steps
              at the start of training so every expert gets gradient from
              step 0 and no expert can "die" before learning.
            mode 1 (blend): soft_out and hard top-k out are blended by a
              scalar alpha ∈ [0, 1]. alpha anneals 0 → 1 across
              blend_anneal_steps so the transition to hard routing is
              smooth rather than a cliff where collapse can resume.
            mode 2 (hard_topk): classic top-k sigmoid gating with
              scatter-gather dispatch. Active from the end of blend
              annealing onward.
        - Loop-aware router: adds a learned per-loop embedding to x.
        - Differentiable balance losses on the softmax distribution
          (importance + logit bias) + router z-loss.

    routing_mode is a Python int attribute — changing it triggers exactly
    two torch.compile recompilations per run (at the soft→blend and
    blend→hard transitions), which is cheap. routing_alpha is a scalar
    tensor buffer so the training loop can update it in-place every step
    without recompiling during the blend phase.
    """

    def __init__(self, config: NanoOSRTv4Config) -> None:
        super().__init__()
        self.num_routed = config.num_routed_experts
        self.top_k = config.top_k_experts
        self.num_loops = config.recursive_loops
        self.temperature = config.router_softmax_temperature
        self.capacity_capped = config.router_capacity_capped
        self.candidate_k = min(config.router_candidate_k, self.num_routed)
        self.capacity_factor = config.router_capacity_factor

        # Routing phase. 0 = soft_all, 1 = blend, 2 = hard_topk. Default
        # hard so eval / from-scratch inference acts like a normal MoE
        # without anyone having to set it. The training loop flips it
        # to 0 at step 0 and progresses through 1 → 2 on schedule.
        self.routing_mode: int = 2

        # Blend mixing coefficient. 0.0 = pure soft, 1.0 = pure hard.
        # Only consulted in mode 1. Scalar tensor buffer so in-place
        # updates don't recompile.
        self.register_buffer(
            "routing_alpha",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )

        # Router noise schedule (legacy, optional). Scalar tensor buffer.
        # Defaults to 0.0 now that soft warmup handles tie-locking, but
        # left in so it can be re-enabled from config if we ever want
        # extra jitter during hard-phase training.
        self.register_buffer(
            "noise_std",
            torch.tensor(config.router_noise_std_init, dtype=torch.float32),
            persistent=False,
        )
        # Gumbel-top-k temperature. Scalar tensor buffer so training loop
        # updates in-place without recompiles. Used inside _hard_gate to
        # add Gumbel(0, tau) noise to the selection logits, breaking
        # deterministic winner lock-in. Weights are still computed from
        # the clean sigmoid — only the *selection* is noised.
        self.register_buffer(
            "gumbel_tau",
            torch.tensor(config.router_gumbel_tau_init, dtype=torch.float32),
            persistent=False,
        )

        # ── Balance-bias controller ──
        # Per-expert additive bias applied to selection_logits as part
        # of the routing mechanism (both train and eval). Controls load
        # imbalance by pushing over-used experts down and under-used
        # experts up. Persistent so it survives checkpoint save/load —
        # the trained bias is part of the model, not a training-only
        # crutch.
        #
        # IMPORTANT: the controller runs at once-per-optimizer-step
        # cadence, NOT per MoE call. The 6 recursive-loop calls within
        # a forward pass are a sequence through evolving hidden states
        # and are not independent samples — updating the bias after
        # each call causes later calls to react to bias changes driven
        # by earlier calls, producing oscillation. We accumulate clean
        # biased assignment counts across the 6 loops in per-block
        # buffers, then the training loop calls `apply_balance_update`
        # after optimizer.step() to consume the accumulator and update
        # the bias exactly once.
        self.bias_enabled = config.router_balance_bias_enabled
        self.bias_update_rate = config.router_balance_bias_update_rate
        self.bias_ema_rate = config.router_balance_bias_ema_rate
        self.bias_max = config.router_balance_bias_max
        self.register_buffer(
            "router_balance_bias",
            torch.zeros(self.num_routed, dtype=torch.float32),
            persistent=True,
        )
        # Per-step accumulator. Reset by apply_balance_update().
        self.register_buffer(
            "balance_count_accum",
            torch.zeros(self.num_routed, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "balance_total_accum",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        # Non-persistent EMA of recent assignment fractions — used for
        # diagnostic logging only. Updated inside apply_balance_update.
        self.register_buffer(
            "expert_ema_fraction",
            torch.full((self.num_routed,), 1.0 / self.num_routed, dtype=torch.float32),
            persistent=False,
        )

        # Shared expert (always active)
        self.shared_expert = ExpertFFN(config.dim, config.expert_hidden)

        # Routed experts
        self.experts = nn.ModuleList(
            [ExpertFFN(config.dim, config.expert_hidden) for _ in range(self.num_routed)]
        )

        # Router: projects hidden state to expert scores. Loop-aware: we
        # add a learned per-loop embedding before the projection.
        # Additive (not concat) keeps router params at dim → num_routed.
        #
        # Loop embeddings get a larger init std (config.loop_embedding_init_std)
        # so routing actually differs across loops at init — default 0.02
        # is too small vs the x-signal magnitude. The _osrt_init_std tag
        # survives HF's post_init() which would otherwise stomp it.
        self.loop_embeddings = nn.Embedding(config.recursive_loops, config.dim)
        self.loop_embeddings._osrt_init_std = config.loop_embedding_init_std
        self.router = nn.Linear(config.dim, self.num_routed, bias=False)
        # Tag so _init_weights row-normalises this linear layer: every
        # expert row starts with equal norm, preventing any single row
        # from winning globally at init (independent of x).
        self.router._osrt_router_row_norm = True

        # Pre-register loop index tensors to avoid torch.tensor() in forward
        self.register_buffer(
            "loop_indices", torch.arange(config.recursive_loops), persistent=False
        )

        # Per-layer losses, set during training forward, accumulated by
        # the outer model. Kept separate so the wrapper can apply distinct
        # coefficients without re-deriving ratios.
        self.importance_loss: Tensor | None = None
        self.logit_bias_loss: Tensor | None = None
        self.z_loss: Tensor | None = None

        # Per-loop telemetry — indexed [0..num_loops-1], overwritten on
        # each forward. Plain Python so logging is free.
        #
        # last_router_entropy: entropy of the softmax distribution. Used
        #   during soft warmup as a secondary metric.
        # last_assignment_entropy: entropy of the NOISY top-k assignment
        #   counts (training-time: what actually happens during
        #   forward/backward). Uniform routing ~ ln(N); collapse ~ ln(k).
        # last_clean_assignment_entropy: entropy of the CLEAN top-k
        #   assignment counts (no Gumbel noise) — what deterministic
        #   inference would produce. Divergence between noisy and clean
        #   means training is propped up by sampling, not by learned
        #   preferences. Clean is the real health metric.
        # last_expert_fraction / last_clean_expert_fraction: per-expert
        #   fractions corresponding to the two histograms above.
        self.last_router_entropy: list[float] = [0.0] * config.recursive_loops
        # Noisy: training-time selection (bias + Gumbel noise)
        self.last_assignment_entropy: list[float] = [0.0] * config.recursive_loops
        # Clean biased: inference-time selection (bias, no noise) —
        # THIS is the decisive health metric because the bias is part
        # of the routing mechanism.
        self.last_clean_assignment_entropy: list[float] = [0.0] * config.recursive_loops
        # Raw: router_logits alone, no bias, no noise. Diagnostic only —
        # shows what the raw router has learned. May collapse even when
        # the biased/clean view stays healthy; that's acceptable as long
        # as the deployed model includes the bias.
        self.last_raw_assignment_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_expert_fraction: list[list[float]] = [
            [0.0] * config.num_routed_experts for _ in range(config.recursive_loops)
        ]
        self.last_clean_expert_fraction: list[list[float]] = [
            [0.0] * config.num_routed_experts for _ in range(config.recursive_loops)
        ]
        self.last_raw_expert_fraction: list[list[float]] = [
            [0.0] * config.num_routed_experts for _ in range(config.recursive_loops)
        ]
        # Capacity-capped telemetry
        self.last_overflow_rate: list[float] = [0.0] * config.recursive_loops
        self.last_assigned_per_token: list[float] = [float(self.top_k)] * config.recursive_loops
        self.last_candidate_rank_mean: list[float] = [0.0] * config.recursive_loops

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
        self.importance_loss = None
        self.logit_bias_loss = None
        self.z_loss = None

        # Shared expert (unconditional)
        shared_out = self.shared_expert(x)

        # Loop-aware routing: add the learned per-loop embedding to x.
        loop_emb = self.loop_embeddings(self.loop_indices[loop_idx])  # (dim,)
        router_input = x + loop_emb  # (B, S, dim)
        router_logits = self.router(router_input)  # (B, S, num_routed)

        # Legacy router noise (optional — defaults to 0). Left for hard
        # phase jitter only; applying it during soft dispatch would
        # perturb the balance loss targets without any routing benefit.
        if self.training and self.routing_mode == 2:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # Differentiable balance losses always use the softmax distribution,
        # regardless of routing mode. We compute soft_probs here so it's
        # available to both the soft dispatch path and the loss path.
        soft_probs = F.softmax(router_logits / self.temperature, dim=-1)

        if self.training:
            self._compute_losses(router_logits, soft_probs)

        mode = self.routing_mode
        if mode == 0:
            # Pure soft dispatch. Shadow telemetry shows what the
            # deployed path (capped or uncapped hard) would produce.
            routed_out = self._soft_dispatch(x, soft_probs)
            if self.training:
                with torch.no_grad():
                    if self.capacity_capped:
                        # Shadow capped dispatch for telemetry only
                        sig = torch.sigmoid(router_logits)
                        N = B * S
                        cap = math.ceil(
                            self.capacity_factor * N * self.top_k / self.num_routed
                        )
                        cand_s, cand_i = torch.topk(
                            sig.reshape(N, self.num_routed),
                            self.candidate_k, dim=-1,
                        )
                        shadow_assigned, shadow_raw = self._shadow_capped(
                            cand_i, N, cap, sig, B, S, loop_idx,
                        )
                        self._record_hard_telemetry(
                            soft_probs, shadow_assigned, shadow_assigned,
                            shadow_raw, loop_idx,
                        )
                    else:
                        noisy_idx, clean_biased_idx, clean_raw_idx = (
                            self._shadow_selections(router_logits)
                        )
                        self._record_hard_telemetry(
                            soft_probs, noisy_idx, clean_biased_idx,
                            clean_raw_idx, loop_idx,
                        )
                if self.bias_enabled:
                    self._accumulate_balance_counts(
                        shadow_assigned if self.capacity_capped
                        else clean_biased_idx
                    )
        elif mode == 2:
            # Pure hard dispatch
            if self.capacity_capped:
                routed_out = self._capped_dispatch(x, router_logits, loop_idx)
            else:
                top_k_weights, top_k_indices = self._hard_gate(router_logits)
                routed_flat = self._dispatch_experts(
                    x.reshape(-1, D),
                    top_k_indices.reshape(-1, self.top_k),
                    top_k_weights.reshape(-1, self.top_k),
                    D,
                )
                routed_out = routed_flat.view(B, S, D)
                if self.training:
                    with torch.no_grad():
                        raw_idx = torch.topk(
                            router_logits, self.top_k, dim=-1
                        ).indices
                    self._record_hard_telemetry(
                        soft_probs, top_k_indices, top_k_indices,
                        raw_idx, loop_idx,
                    )
        else:
            # Blend — soft + hard, mixed by alpha.
            soft_out = self._soft_dispatch(x, soft_probs)
            if self.capacity_capped:
                hard_out = self._capped_dispatch(x, router_logits, loop_idx)
            else:
                top_k_weights, top_k_indices = self._hard_gate(router_logits)
                hard_flat = self._dispatch_experts(
                    x.reshape(-1, D),
                    top_k_indices.reshape(-1, self.top_k),
                    top_k_weights.reshape(-1, self.top_k),
                    D,
                )
                hard_out = hard_flat.view(B, S, D)
            alpha = self.routing_alpha
            routed_out = alpha * hard_out + (1.0 - alpha) * soft_out

        return shared_out + routed_out

    def _shadow_capped(
        self,
        cand_indices: Tensor,
        N: int,
        capacity: int,
        sig: Tensor,
        B: int,
        S: int,
        loop_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Shadow capacity-capped assignment for telemetry during soft.

        Also records overflow/assigned/rank telemetry into the per-loop
        fields so the training loop can log them during soft phase
        (where the main capped dispatch doesn't run).

        Returns (capped_indices, raw_indices) both (B, S, top_k).
        """
        assigned, _, slots_filled, rank_sum = self._assign_with_caps(
            cand_indices, None, N, capacity, sig.device,
        )

        # Record capacity telemetry from the shadow computation
        if 0 <= loop_idx < len(self.last_overflow_rate):
            self.last_overflow_rate[loop_idx] = (
                (slots_filled < self.top_k).float().mean().item()
            )
            self.last_assigned_per_token[loop_idx] = (
                slots_filled.float().mean().item()
            )
            filled = slots_filled.clamp_min(1).float()
            self.last_candidate_rank_mean[loop_idx] = (
                (rank_sum / filled).mean().item()
            )

        raw_idx = torch.topk(
            sig.reshape(N, self.num_routed), self.top_k, dim=-1,
        ).indices.view(B, S, self.top_k)
        return assigned.view(B, S, self.top_k), raw_idx

    def _assign_with_caps(
        self,
        cand_indices: Tensor,
        cand_scores: Tensor | None,
        N: int,
        capacity: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Race-free capacity-capped assignment.

        Iterates over (candidate_rank, expert_id) and for each expert
        assigns at most `remaining_capacity` tokens. This avoids the
        vectorised race where multiple tokens pass the capacity check
        simultaneously and all get assigned, blowing past the cap.

        O(candidate_k × num_routed) = 66 iterations for default config.

        Returns (assigned_indices, assigned_weights, slots_filled, rank_sum)
        where assigned is (N, top_k), weights is (N, top_k),
        slots_filled is (N,) counting how many experts each token got,
        and rank_sum is (N,) total candidate rank across assigned slots
        (used to compute candidate_rank_mean = rank_sum / slots_filled).
        Unfilled slots have index -1 and weight 0. Callers must clamp
        index to 0 before passing to dispatch (so the unfilled slot
        routes through expert 0 with zero weight — harmless). Telemetry
        must filter out -1 entries to avoid inflating expert 0 counts.
        """
        assigned = torch.full((N, self.top_k), -1, dtype=torch.long, device=device)
        assigned_w = torch.zeros(N, self.top_k, device=device)
        expert_load = torch.zeros(self.num_routed, dtype=torch.long, device=device)
        slots_filled = torch.zeros(N, dtype=torch.long, device=device)
        rank_sum = torch.zeros(N, dtype=torch.float32, device=device)

        for rank in range(self.candidate_k):
            if (slots_filled >= self.top_k).all():
                break

            eid_col = cand_indices[:, rank]                       # (N,)
            eligible = slots_filled < self.top_k                  # (N,) bool

            eligible_idx = eligible.nonzero(as_tuple=True)[0]     # (M,)
            if eligible_idx.numel() == 0:
                break
            eligible_eid = eid_col[eligible_idx]                  # (M,)

            # Sort eligible tokens by expert ID for grouping
            sorted_order = eligible_eid.argsort(stable=True)      # (M,)
            sorted_eid = eligible_eid[sorted_order]               # (M,)
            M = sorted_eid.shape[0]

            # Compute within-group position via segment-cumsum trick:
            # group_change[i]=1 at group boundaries, 0 otherwise
            group_change = torch.ones(M, dtype=torch.long, device=device)
            if M > 1:
                group_change[1:] = (sorted_eid[1:] != sorted_eid[:-1]).long()
            # group_id per position (0-based)
            group_id = torch.cumsum(group_change, dim=0) - 1      # (M,)
            # Index of the first element of each group
            group_start_positions = group_change.nonzero(as_tuple=True)[0]  # (G,)
            # Map each position to its group's start index
            start_of_group = group_start_positions[group_id]      # (M,)
            # Within-group position = position - start of group
            arange_M = torch.arange(M, device=device)
            within_group_pos = arange_M - start_of_group          # (M,)

            # Remaining capacity per expert, indexed per sorted token
            remaining = (capacity - expert_load).clamp(min=0)     # (num_routed,)
            remaining_per_token = remaining[sorted_eid]           # (M,)

            # Token fits if its within-group position < remaining capacity
            fits = within_group_pos < remaining_per_token         # (M,) bool

            # Map back from sorted order to eligible-token order
            fits_orig = torch.zeros(M, dtype=torch.bool, device=device)
            fits_orig[sorted_order] = fits
            # Indices into the full N-sized arrays for tokens that get assigned
            take = eligible_idx[fits_orig]                        # (T,)

            if take.numel() == 0:
                continue

            take_eid = eid_col[take]                              # (T,)
            sidx = slots_filled[take]
            assigned[take, sidx] = take_eid
            if cand_scores is not None:
                assigned_w[take, sidx] = cand_scores[take, rank].float()
            slots_filled[take] += 1
            rank_sum[take] += rank

            # Update expert load counts
            assigned_counts = torch.zeros(
                self.num_routed, dtype=torch.long, device=device,
            )
            assigned_counts.scatter_add_(
                0, take_eid, torch.ones_like(take_eid),
            )
            expert_load += assigned_counts

        return assigned, assigned_w, slots_filled, rank_sum

    def _shadow_selections(
        self, router_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute three top-k index sets for telemetry + bias update.

        Returns (noisy_biased, clean_biased, clean_raw):
          - noisy_biased: router_logits + bias + Gumbel(τ)
              What the training-time hard path would select right now.
              Used as the source for the bias controller's update.
          - clean_biased: router_logits + bias, no noise
              What deterministic inference produces. The DECISIVE
              health metric — if this is balanced the deployed model
              is fine.
          - clean_raw: router_logits alone, no bias, no noise
              Diagnostic only. Shows what the raw router has learned.
              May look collapsed even when clean_biased is healthy;
              that's acceptable since the bias is part of the model.
        """
        if self.bias_enabled:
            biased = router_logits + self.router_balance_bias.view(1, 1, -1)
        else:
            biased = router_logits

        clean_biased_idx = torch.topk(biased, self.top_k, dim=-1).indices
        clean_raw_idx = torch.topk(router_logits, self.top_k, dim=-1).indices

        if self.gumbel_tau > 0:
            u = torch.rand_like(router_logits).clamp_min(1e-9)
            gumbel = -torch.log(-torch.log(u))
            noisy_logits = biased + self.gumbel_tau * gumbel
            noisy_biased_idx = torch.topk(noisy_logits, self.top_k, dim=-1).indices
        else:
            noisy_biased_idx = clean_biased_idx

        return noisy_biased_idx, clean_biased_idx, clean_raw_idx

    @torch._dynamo.disable
    @torch.no_grad()
    def _accumulate_balance_counts(self, top_k_indices: Tensor) -> None:
        """Accumulate per-expert assignment counts for the balance
        controller. Called from forward() once per MoE invocation
        during training; reset by `apply_balance_update()` once per
        optimizer step.
        """
        if not self.bias_enabled:
            return
        flat = top_k_indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_routed).float()
        self.balance_count_accum.add_(counts)
        self.balance_total_accum.add_(counts.sum())

    @torch.no_grad()
    def apply_balance_update(self) -> None:
        """Consume the accumulator and update the balance bias once.

        Called from the training loop after optimizer.step(), not from
        forward. This gives the controller integrated feedback over all
        6 recursive-loop calls of this block's MoELayer in the step
        (plus any gradient-accumulation micro-batches), so it sees the
        full step's routing distribution as one signal instead of
        reacting to intra-step bias drift from earlier loops.

        Resets the accumulator afterwards. If the accumulator is empty
        (e.g. an eval-only step, or apply called twice), this is a
        no-op — we must NOT update bias from an all-zero histogram,
        which would push every expert toward +1/N uniformly.
        """
        if not self.bias_enabled:
            return
        if self.balance_total_accum.item() == 0.0:
            return
        total = self.balance_total_accum
        current_frac = self.balance_count_accum / total

        # EMA for smoothed telemetry (not used by the controller)
        self.expert_ema_fraction.lerp_(current_frac, self.bias_ema_rate)

        target = 1.0 / self.num_routed
        delta = current_frac - target
        self.router_balance_bias.add_(delta, alpha=-self.bias_update_rate)
        self.router_balance_bias.clamp_(-self.bias_max, self.bias_max)

        # Reset for next step
        self.balance_count_accum.zero_()
        self.balance_total_accum.zero_()

    def _hard_gate(self, router_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Selection-logit gating + top-k + weight renormalization.

        Three sources are combined to form the selection logits:
          1. router_logits (what the router learned)
          2. router_balance_bias (per-expert additive — part of the
             routing mechanism, applied in both training and eval so
             the deployed path matches the training path)
          3. Gumbel(0, tau) noise during training only

        The *weights* returned come from the CLEAN sigmoid of
        router_logits alone (no bias, no noise), so gradient flows to
        the router through a meaningful path for the experts that
        actually were selected.

        Returns (top_k_weights, top_k_indices) both shaped (B, S, top_k).
        """
        # Bias is part of the routing mechanism — applied always
        if self.bias_enabled:
            selection_base = router_logits + self.router_balance_bias.view(1, 1, -1)
        else:
            selection_base = router_logits

        # Gumbel noise during training only
        if self.training and self.gumbel_tau > 0:
            u = torch.rand_like(router_logits).clamp_min(1e-9)
            gumbel = -torch.log(-torch.log(u))
            selection_logits = selection_base + self.gumbel_tau * gumbel
        else:
            selection_logits = selection_base

        _, top_k_indices = torch.topk(selection_logits, self.top_k, dim=-1)

        # Gather CLEAN sigmoid weights for the selected experts
        clean_sig = torch.sigmoid(router_logits)
        top_k_weights = clean_sig.gather(-1, top_k_indices)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return top_k_weights, top_k_indices

    @torch._dynamo.disable
    def _soft_dispatch(self, x: Tensor, soft_probs: Tensor) -> Tensor:
        """Soft all-expert dispatch: every expert runs on every token.

        This is ~num_routed× more FLOPs than top-k routing but guarantees
        that every expert receives gradient at every step — no dead
        experts, no self-reinforcing winners. Only used during the
        soft_warmup / blend phases (total ~1000 steps by default).

        Accumulates one expert at a time to avoid materialising the
        (B, S, num_routed, D) stacked tensor, which would be ~140MB at
        seq_len=2048, dim=1536, fp16.
        """
        out = torch.zeros_like(x)
        for eid in range(self.num_routed):
            w = soft_probs[..., eid].unsqueeze(-1)  # (B, S, 1)
            out = out + w * self.experts[eid](x)
        return out

    @torch._dynamo.disable
    def _capped_dispatch(
        self, x: Tensor, router_logits: Tensor, loop_idx: int,
    ) -> Tensor:
        """Capacity-capped expert dispatch.

        For each token, scans candidates in descending sigmoid score
        and assigns the first top_k experts that still have capacity.
        This structurally guarantees bounded max by construction —
        no expert can hoard more than `capacity` tokens regardless of
        what the router prefers.

        Same path in train and eval. No Gumbel, no bias — the cap IS
        the balance mechanism.

        Gradient flows through the sigmoid weights of the chosen
        experts exactly like uncapped top-k; only the selection is
        changed by the capacity constraint.

        Returns the routed output tensor (B, S, D).
        """
        B, S, D = x.shape
        N = B * S

        sig = torch.sigmoid(router_logits)
        cand_scores, cand_indices = torch.topk(
            sig.reshape(N, self.num_routed), self.candidate_k, dim=-1,
        )
        flat_x = x.reshape(N, D)

        capacity = math.ceil(
            self.capacity_factor * N * self.top_k / self.num_routed
        )

        assigned, assigned_w, slots_filled, rank_sum = self._assign_with_caps(
            cand_indices, cand_scores, N, capacity, x.device,
        )

        # Renormalize weights over actually-assigned experts (unfilled
        # slots keep weight 0). Clamp indices to 0 for dispatch only —
        # unfilled slots route through expert 0 with zero weight
        # (harmless, bounded by overflow rate).
        weight_sum = assigned_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        assigned_w = assigned_w / weight_sum

        dispatch_idx = assigned.clamp(min=0)
        routed_out = self._dispatch_experts(
            flat_x, dispatch_idx, assigned_w, D,
        )

        # Record capacity telemetry
        if 0 <= loop_idx < len(self.last_overflow_rate):
            self.last_overflow_rate[loop_idx] = (
                (slots_filled < self.top_k).float().mean().item()
            )
            self.last_assigned_per_token[loop_idx] = (
                slots_filled.float().mean().item()
            )
            filled = slots_filled.clamp_min(1).float()
            self.last_candidate_rank_mean[loop_idx] = (
                (rank_sum / filled).mean().item()
            )

        # Record routing histogram telemetry (reuse existing method).
        # Capped assignment is both the training and inference path, so
        # we pass it as both "noisy" and "clean_biased". Raw uncapped
        # top-k stays as diagnostic.
        if self.training:
            with torch.no_grad():
                raw_idx = torch.topk(
                    sig.reshape(N, self.num_routed), self.top_k, dim=-1,
                ).indices.view(B, S, self.top_k)
            soft_probs = F.softmax(router_logits / self.temperature, dim=-1)
            capped_idx = assigned.view(B, S, self.top_k)
            self._record_hard_telemetry(
                soft_probs, capped_idx, capped_idx, raw_idx, loop_idx,
            )

        return routed_out.view(B, S, D)

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

    def _compute_losses(self, router_logits: Tensor, soft_probs: Tensor) -> None:
        """Differentiable routing losses.

        All three losses are computed from the softmax router distribution
        and the raw logits — they have meaningful gradient regardless of
        routing_mode, so they work during soft warmup, blend, and hard
        phases alike.

        - importance_loss: ``N * sum(importance**2)`` where importance is
          the per-expert mean softmax probability. Minimum 1.0 when
          experts are loaded uniformly; grows quadratically with
          concentration. Unlike the Switch-Transformer load balance loss
          (which multiplies mean-prob by hard-assignment fractions), this
          does NOT saturate when an expert dies — dead experts still get
          gradient pushing their importance back up.

        - logit_bias_loss: mean squared deviation of each expert's mean
          logit from the grand mean. Attacks the global expert ordering
          that hard top-k exploits to pick consistent winners regardless
          of token content.

        - z_loss: mean squared logit, preventing the router from pushing
          logits to extreme magnitudes.
        """
        # Importance balance (differentiable surrogate for load balance)
        importance = soft_probs.mean(dim=(0, 1))  # (num_routed,)
        self.importance_loss = self.num_routed * (importance * importance).sum()

        # Logit centering: penalise per-expert mean logit deviation
        mean_logits = router_logits.mean(dim=(0, 1))  # (num_routed,)
        centered = mean_logits - mean_logits.mean()
        self.logit_bias_loss = centered.pow(2).mean()

        # Z-loss: squared logit magnitudes
        self.z_loss = router_logits.pow(2).mean()

    @torch._dynamo.disable
    @torch.no_grad()
    def _record_soft_telemetry(self, soft_probs: Tensor, loop_idx: int) -> None:
        """Telemetry for the soft-routing phase.

        No hard assignments exist yet, so last_assignment_entropy is set
        equal to the softmax entropy (best proxy we have) and
        last_expert_fraction holds the per-expert importance vector
        (softmax mean over tokens) so the logger always has something
        sensible to print.
        """
        prob_entropy = -(soft_probs * (soft_probs + 1e-8).log()).sum(dim=-1).mean()
        if 0 <= loop_idx < len(self.last_router_entropy):
            self.last_router_entropy[loop_idx] = prob_entropy.item()

        importance = soft_probs.mean(dim=(0, 1))  # (num_routed,)
        fractions = importance.tolist()
        if 0 <= loop_idx < len(self.last_expert_fraction):
            self.last_expert_fraction[loop_idx] = fractions

        # Entropy of the importance distribution — during healthy soft
        # routing this should also be near ln(num_routed).
        imp_entropy = -(importance * (importance + 1e-8).log()).sum()
        if 0 <= loop_idx < len(self.last_assignment_entropy):
            self.last_assignment_entropy[loop_idx] = imp_entropy.item()

    @torch._dynamo.disable
    @torch.no_grad()
    def _record_hard_telemetry(
        self,
        soft_probs: Tensor,
        top_k_indices: Tensor,
        clean_biased_indices: Tensor | None,
        clean_raw_indices: Tensor | None,
        loop_idx: int,
    ) -> None:
        """Record three histograms per loop:
          - noisy (actually used during training) → last_assignment_*
          - clean biased (inference path) → last_clean_assignment_*
          - clean raw (no bias, no noise, diagnostic) → last_raw_*

        The decisive health metric is the clean biased view because
        the bias is part of the routing mechanism.
        """
        def _hist(indices: Tensor) -> tuple[list[float], float]:
            flat = indices.reshape(-1)
            # Filter out -1 entries (unfilled capacity-capped slots)
            # to avoid inflating expert 0 counts.
            valid = flat[flat >= 0]
            if valid.numel() == 0:
                return [0.0] * self.num_routed, 0.0
            counts = torch.bincount(valid, minlength=self.num_routed).float()
            total = counts.sum().clamp_min(1)
            fractions_tensor = counts / total
            fractions = fractions_tensor.tolist()
            entropy = -(fractions_tensor * (fractions_tensor + 1e-8).log()).sum().item()
            return fractions, entropy

        prob_entropy = -(soft_probs * (soft_probs + 1e-8).log()).sum(dim=-1).mean()
        if 0 <= loop_idx < len(self.last_router_entropy):
            self.last_router_entropy[loop_idx] = prob_entropy.item()

        # Noisy (training) histogram
        fractions, entropy = _hist(top_k_indices)
        if 0 <= loop_idx < len(self.last_expert_fraction):
            self.last_expert_fraction[loop_idx] = fractions
            self.last_assignment_entropy[loop_idx] = entropy

        # Clean biased (inference path) histogram
        cb_src = clean_biased_indices if clean_biased_indices is not None else top_k_indices
        fractions, entropy = _hist(cb_src)
        if 0 <= loop_idx < len(self.last_clean_expert_fraction):
            self.last_clean_expert_fraction[loop_idx] = fractions
            self.last_clean_assignment_entropy[loop_idx] = entropy

        # Raw (diagnostic) histogram
        cr_src = clean_raw_indices if clean_raw_indices is not None else cb_src
        fractions, entropy = _hist(cr_src)
        if 0 <= loop_idx < len(self.last_raw_expert_fraction):
            self.last_raw_expert_fraction[loop_idx] = fractions
            self.last_raw_assignment_entropy[loop_idx] = entropy


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
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
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

        # Attention residual + adapter (parallel branches into x)
        x = x + self.out_proj(attn_out) + adapter_out

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
            # Router row-norm: if this linear layer is tagged as a router,
            # rescale its rows to have equal norm so no expert starts
            # with a larger weight row than any other. Without this, the
            # expert whose row norm is luckiest at init gets selected
            # disproportionately and hard top-k locks that advantage in
            # before the router has learned anything about tokens.
            if getattr(module, "_osrt_router_row_norm", False):
                with torch.no_grad():
                    w = module.weight.data
                    row_norms = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    mean_norm = row_norms.mean()
                    module.weight.data = (w / row_norms) * mean_norm
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

        # Note: _no_ post_init() here. The outer NanoOSRTv4ForCausalLM
        # wrapper calls post_init() exactly once on the full tree, which
        # also initialises this inner model. Calling it in both places
        # would re-init every parameter twice — not incorrect (the final
        # values are still the tagged-custom ones), just wasteful and
        # confusing when debugging init.

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[Tensor, list[Tensor], Tensor, Tensor, Tensor, list[tuple[Tensor, Tensor]] | None]:
        """Forward pass.

        Args:
            input_ids: Token indices (B, S).
            past_key_values: KV cache — list of (K, V) tuples, one per
                effective layer (num_blocks × recursive_loops entries).
                Each tensor has shape (B, heads, cached_S, head_dim).
            use_cache: Whether to return updated KV cache for generation.

        Returns:
            (hidden_states, loop_rms_list, importance_loss, logit_bias_loss,
             z_loss, present_key_values)
            where the three MoE losses are summed across every MoE
            application (num_blocks × recursive_loops).
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
            # Derive past_length from the first non-None layer and
            # verify every other layer's cache length matches. A
            # partially stale or malformed cache would produce
            # inconsistent RoPE positions across effective layers.
            past_length = 0
            for idx, layer_past in enumerate(past_key_values):
                if layer_past is None:
                    continue
                key, value = layer_past
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"past_key_values[{idx}] must contain Tensor "
                        f"key/value pairs, got {type(key).__name__}/{type(value).__name__}."
                    )
                layer_len = key.shape[2]
                if past_length == 0:
                    past_length = layer_len
                elif layer_len != past_length:
                    raise ValueError(
                        f"past_key_values cache length mismatch: layer 0 "
                        f"has {past_length} cached positions but layer "
                        f"{idx} has {layer_len}."
                    )

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
                scaling=getattr(self.config, "rope_scaling", None),
            )
            cos = rope_cos[:, past_length:required_seq_len, :, :].to(
                device=self.rope_cos.device, dtype=self.rope_cos.dtype
            )
            sin = rope_sin[:, past_length:required_seq_len, :, :].to(
                device=self.rope_sin.device, dtype=self.rope_sin.dtype
            )

        loop_rms: list[Tensor] = []
        total_imp_loss = torch.tensor(0.0, device=x.device)
        total_bias_loss = torch.tensor(0.0, device=x.device)
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
                if block.moe.importance_loss is not None:
                    total_imp_loss = total_imp_loss + block.moe.importance_loss
                if block.moe.logit_bias_loss is not None:
                    total_bias_loss = total_bias_loss + block.moe.logit_bias_loss
                if block.moe.z_loss is not None:
                    total_z_loss = total_z_loss + block.moe.z_loss

            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < self.config.recursive_loops - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        return x, loop_rms, total_imp_loss, total_bias_loss, total_z_loss, presents


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
        hidden, loop_rms, imp_loss, bias_loss, z_loss, presents = self.model(
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
            # MoE losses are accumulated across all (num_blocks × recursive_loops)
            # effective MoE applications (18 for default v4). Normalise by that
            # count so the configured coefficients match the per-layer
            # weights instead of being silently multiplied by 18.
            n_moe_layers = self.config.num_blocks * self.config.recursive_loops
            imp_norm = imp_loss / n_moe_layers
            bias_norm = bias_loss / n_moe_layers
            z_norm = z_loss / n_moe_layers
            loss = (
                loss
                + self.config.router_aux_loss_coeff * imp_norm
                + self.config.router_logit_bias_coeff * bias_norm
                + self.config.router_z_loss_coeff * z_norm
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

            if (next_token == eos_token_id).all():
                break

        return generated
