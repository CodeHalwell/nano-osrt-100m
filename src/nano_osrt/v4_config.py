"""Configuration for NanoOSRT v4 — Recursive MoE.

3 physical blocks × 6 loops = 18 effective layers.
Dense FFN + MoE (1 shared + 11 routed experts, top-2) in parallel residual.
Loop-aware routing with learned loop embeddings.
"""

from transformers import PretrainedConfig


class NanoOSRTv4Config(PretrainedConfig):
    """HuggingFace-compatible config for NanoOSRT v4.

    Architecture: ~306M physical params, ~130M active per token,
    ~1.8B effective via recursive weight sharing.

    32K vocab with dim=1536 uses ~50M params for embeddings (16%
    of model). The remaining 256M powers the transformer body
    including attention, dense FFN, and MoE experts — more of the
    parameter budget goes to reasoning capacity vs embedding lookup.
    """

    model_type = "nano-osrt-v4"

    def __init__(
        self,
        # Core dimensions
        dim: int = 1536,
        heads: int = 24,
        head_dim: int = 64,
        vocab_size: int = 32768,   # 32K vocab — more of the budget in the transformer body
        real_vocab_size: int = 32768,

        # Recursive structure
        num_blocks: int = 3,
        recursive_loops: int = 6,
        adapter_rank: int = 16,
        adapter_alpha: float = 16.0,

        # Dense FFN
        dense_hidden: int = 4096,

        # MoE
        num_experts: int = 12,
        num_shared_experts: int = 1,
        num_routed_experts: int = 11,
        top_k_experts: int = 2,
        expert_hidden: int = 1024,
        # --- Routing regularisation (differentiable) ---
        # Hard top-k is a non-differentiable gate; any balance loss computed
        # from hard-assignment fractions saturates once experts die. We use
        # a differentiable surrogate from the softmax router distribution
        # instead: the "importance" loss
        #     N * sum(importance^2) with importance = softmax_probs.mean(seq).
        # Minimum 1.0 when uniform, grows quadratically with concentration.
        #
        # NOTE: We originally also used logit_bias (mean-centered logit
        # squared deviation) and z_loss (squared logits) as extra guards.
        # Sanity run showed these are actively harmful: they have a global
        # minimum at logit=0, so during soft warmup they drive every router
        # output to zero magnitude. A zero-logit router has no token-
        # dependent preference, so the blend→hard transition finds nothing
        # for top-k to grab onto and collapses immediately.
        #
        # Row-normed router init already addresses the global ordering
        # concern at step 0, and the task loss punishes misrouting. The
        # importance loss alone is enough to prevent marginal collapse
        # while leaving the router free to develop opinions.
        router_aux_loss_coeff: float = 0.05,
        router_logit_bias_coeff: float = 0.0,
        router_z_loss_coeff: float = 0.0,
        # Softmax temperature for the differentiable balance losses and
        # for soft-routing dispatch. Separate from sigmoid-gating scale.
        router_softmax_temperature: float = 1.0,
        # --- Soft→hard routing schedule ---
        # The hard top-k winner-takes-all pattern self-reinforces from
        # the very first step; even tiny initial logit differences become
        # permanent winners. Bootstrap by running every routed expert
        # every step (soft dispatch, softmax-weighted sum) for the first
        # soft_warmup_steps, then blend into hard top-k linearly over
        # blend_anneal_steps so dead experts never get a chance to form.
        soft_warmup_steps: int = 500,
        blend_anneal_steps: int = 500,
        # --- Gumbel-top-k selection ---
        # Deterministic top-k lets tiny logit ordering differences lock in
        # permanent winners. First sanity run confirmed this: even after
        # we removed the zero-logit pressure (z_loss, logit_bias), the
        # shadow hard-top-k collapsed within 40 steps because task loss
        # kept rewarding whichever experts happened to score highest on
        # the first few batches.
        #
        # Fix: during training, add Gumbel(0, tau) noise to router logits
        # BEFORE the top-k operation. The selected *weights* still come
        # from the clean sigmoid of the original logits, so gradient flow
        # to the chosen experts is undisturbed. Only the *selection*
        # becomes stochastic — which breaks deterministic winner lock-in
        # because no expert is guaranteed to be picked by tie-breaking.
        #
        # tau anneals linearly from router_gumbel_tau_init down to
        # router_gumbel_tau_final over router_gumbel_anneal_steps. At
        # step 0 of hard phase the noise magnitude should still be
        # meaningful (tau=1.0 matches raw logit scale once logits are
        # ~order 1). By the end of the anneal window the router is on
        # its own and should have learned robust token-dependent
        # preferences that top-k picks consistently even without noise.
        # Long anneal: Gumbel noise is NOT a short warmup jitter; it's
        # anti-lock-in pressure the router needs for as long as it takes
        # to learn robust preferences. 10K steps holds the noise through
        # most of the Foundation phase so the router has time to
        # accumulate token-dependent gradient without getting locked
        # into early winners. Final value is 0 so the production
        # inference path is deterministic.
        router_gumbel_tau_init: float = 0.0,
        router_gumbel_tau_final: float = 0.0,
        router_gumbel_anneal_steps: int = 10000,
        # --- Balance-bias controller (DeepSeek-V3 style) ---
        # Per-expert additive bias applied to router logits *as part of
        # the routing mechanism* (both training and inference). The bias
        # is updated per step by an additive controller:
        #     delta = current_assignment_fraction - 1/N
        #     bias -= update_rate * delta
        #     bias = clamp(bias, -max, +max)
        # Over-used experts get their logits pushed down; under-used
        # experts get theirs pushed up. This directly attacks the
        # failure mode we saw in the Gumbel run: soft-dispatch + Gumbel
        # could keep noisy training healthy but the clean logit ordering
        # still collapsed because task loss kept rewarding whichever
        # experts scored highest. The bias counter-rotates the ordering
        # whenever the assignment distribution deviates from uniform.
        #
        # Because the bias is part of the routing mechanism (persistent
        # buffer, applied in eval too), "clean biased top-k" is the new
        # inference health metric. Raw un-biased top-k is kept only as
        # a diagnostic — its collapse is acceptable as long as biased
        # routing remains balanced.
        router_balance_bias_enabled: bool = False,
        router_balance_bias_update_rate: float = 0.25,
        router_balance_bias_ema_rate: float = 0.05,
        router_balance_bias_max: float = 2.0,
        # --- Capacity-capped routing ---
        # Structural load balancing: each expert has a hard token cap
        # per batch. For each token, scan candidates in descending
        # router score and assign the first top_k experts that still
        # have capacity. Guarantees bounded max by construction.
        # candidate_k = num_routed is cheap (11 experts, 11 scans) and
        # ensures tokens can always fall through to underused experts
        # outside their top-6 preference list when the popular ones
        # are full. Overflow should be near zero with candidate_k=11.
        router_capacity_capped: bool = True,
        router_candidate_k: int = 11,
        router_capacity_factor: float = 1.25,
        # --- Router noise (legacy, optional) ---
        router_noise_std_init: float = 0.0,
        router_noise_std_final: float = 0.0,
        router_noise_anneal_steps: int = 5000,
        # Loop embedding init std — raised from the default initializer_range
        # (0.02) so that x + loop_emb actually produces per-loop routing
        # differentiation at init. At 0.02, loop_emb contribution to router
        # logits is ~0.016 vs ~0.78 from x → effectively invisible.
        loop_embedding_init_std: float = 0.1,

        # Sequence length
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,

        # Training defaults
        initializer_range: float = 0.02,

        # Token IDs (must match tokenizer special tokens)
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        unk_token_id: int = 3,
        fim_prefix_id: int = 4,
        fim_middle_id: int = 5,
        fim_suffix_id: int = 6,
        think_open_id: int = 7,
        think_close_id: int = 8,
        answer_open_id: int = 9,
        answer_close_id: int = 10,
        user_token_id: int = 11,
        assistant_token_id: int = 12,
        system_token_id: int = 13,

        **kwargs,
    ):
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.real_vocab_size = real_vocab_size

        self.num_blocks = num_blocks
        self.recursive_loops = recursive_loops
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha

        self.dense_hidden = dense_hidden

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k_experts = top_k_experts
        self.expert_hidden = expert_hidden
        self.router_aux_loss_coeff = router_aux_loss_coeff
        self.router_logit_bias_coeff = router_logit_bias_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.router_softmax_temperature = router_softmax_temperature
        self.soft_warmup_steps = soft_warmup_steps
        self.blend_anneal_steps = blend_anneal_steps
        self.router_gumbel_tau_init = router_gumbel_tau_init
        self.router_gumbel_tau_final = router_gumbel_tau_final
        self.router_gumbel_anneal_steps = router_gumbel_anneal_steps
        self.router_capacity_capped = router_capacity_capped
        self.router_candidate_k = router_candidate_k
        self.router_capacity_factor = router_capacity_factor
        self.router_balance_bias_enabled = router_balance_bias_enabled
        self.router_balance_bias_update_rate = router_balance_bias_update_rate
        self.router_balance_bias_ema_rate = router_balance_bias_ema_rate
        self.router_balance_bias_max = router_balance_bias_max
        self.router_noise_std_init = router_noise_std_init
        self.router_noise_std_final = router_noise_std_final
        self.router_noise_anneal_steps = router_noise_anneal_steps
        self.loop_embedding_init_std = loop_embedding_init_std

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.initializer_range = initializer_range

        # Special token IDs
        self.unk_token_id = unk_token_id
        self.fim_prefix_id = fim_prefix_id
        self.fim_middle_id = fim_middle_id
        self.fim_suffix_id = fim_suffix_id
        self.think_open_id = think_open_id
        self.think_close_id = think_close_id
        self.answer_open_id = answer_open_id
        self.answer_close_id = answer_close_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id
        self.system_token_id = system_token_id

        super().__init__(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self._validate()

    def _validate(self) -> None:
        if self.dim % self.heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by heads ({self.heads})"
            )
        if self.top_k_experts > self.num_routed_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) must be <= "
                f"num_routed_experts ({self.num_routed_experts})"
            )
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.recursive_loops < 1:
            raise ValueError(
                f"recursive_loops must be >= 1, got {self.recursive_loops}"
            )
