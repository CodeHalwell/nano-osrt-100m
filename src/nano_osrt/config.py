"""Configuration for NanoOSRT — Mixtral-style MoE without dense FFN.

3 physical blocks × 6 loops = 18 effective layers.
MoE only (1 shared + 8 routed experts, top-2 routing).
No dense FFN — shared expert replaces it.
Switch-style balance loss (minimises at uniform without enforcing it).
DeepSeek-style per-expert balance bias controller.
No importance loss, no soft warmup.
Training can use annealed Gumbel top-k noise to prevent early dead experts.
Orthogonal per-expert initialisation breaks symmetry at step 0.

Architecture: ~363M physical params, ~192M active per token (52.9%),
~1.15B effective via recursive weight sharing.
"""

from transformers import PretrainedConfig


class NanoOSRTConfig(PretrainedConfig):
    """HuggingFace-compatible config for NanoOSRT.

    Design lessons from v4:
      - Dense FFN made MoE optional → removed
      - Importance loss minimises at uniform → replaced with Switch balance
      - Tight capacity cap hid router decisions → factor 2.0, near-zero overflow
      - Soft warmup + blend didn't prevent collapse → no all-expert soft path
      - Small experts (1024 hidden) × many (11) couldn't specialise → 2048 × 8
      - Top-2 is fine — v4's real issues were the above, not top-k itself
        (Mixtral uses top-2 of 8 successfully with Switch balance + no dense FFN)
    """

    model_type = "nano-osrt"

    def __init__(
        self,
        # Core dimensions (unchanged from v4)
        dim: int = 1536,
        heads: int = 24,
        head_dim: int = 64,
        vocab_size: int = 32768,
        real_vocab_size: int = 32768,

        # Recursive structure (unchanged from v4)
        num_blocks: int = 3,
        recursive_loops: int = 6,
        adapter_rank: int = 16,
        adapter_alpha: float = 16.0,

        # --- MoE (v5 architecture) ---
        # No dense FFN. Shared expert replaces it at larger hidden size.
        # 8 routed experts × hidden 2048, top-2 (Mixtral-style).
        num_routed_experts: int = 8,
        top_k_experts: int = 2,
        expert_hidden: int = 2048,          # routed experts
        shared_expert_hidden: int = 4096,   # shared expert (replaces dense FFN)

        # --- Routing (Switch-style + explicit load controller) ---
        # Balance loss: num_experts * sum(f_i * p_i) where
        #   f_i = fraction of tokens routed to expert i (hard assignment)
        #   p_i = mean softmax prob for expert i
        # Minimised at uniform for BOTH f and p. Unlike importance loss,
        # this penalises imbalance without forcing the router to produce
        # uniform probabilities — the router can develop sharp preferences
        # as long as tokens are roughly evenly distributed across experts.
        #
        # v5 sanity runs showed the original Switch default (0.01) let the
        # 8-expert router collapse to roughly two active experts. 0.03 moved
        # it to roughly three active experts without hurting task loss, but
        # the 1200-step sanity (W&B run 08o88sq0) showed partial collapse
        # resurfacing once LR reached ~25% of peak: drop rate 0.6%→12%,
        # clean marginal entropy 1.97→1.46, one expert to ~1.5% of tokens.
        # Bumped to 0.10 after sanity showed 0.03 was too weak. The
        # clean-path variant still collapsed, so this aux now acts on raw
        # pre-bias router logits while the explicit bias controller below
        # handles deployed clean-routing load.
        router_aux_loss_coeff: float = 0.10,

        # --- Router Z-loss (numerical safety + early router health) ---
        # Penalty on the router's pre-softmax log-partition:
        #   L_z = mean_token (logsumexp(router_logits))^2
        # Two effects: (1) keeps logit magnitudes bounded so bf16/fp8
        # softmax exponentials don't overflow, and (2) keeps early
        # softmax distributions flatter, so cold experts retain a
        # non-zero gradient through the LR warmup. The standard ST-MoE
        # value is ~1e-3; small enough not to compete with task loss but
        # large enough to keep raw logit magnitudes O(1).
        router_z_loss_coeff: float = 1e-3,

        # --- Sequence-wise balance loss (DeepSeek-V3 §5.2) ---
        # Per-sequence Switch-style loss penalising imbalance INSIDE a
        # single sequence. Useful at long context where one document can
        # dominate one micro-batch even when the global batch is
        # balanced. Default 0.0 — opt in once the global aux is stable.
        # When non-zero, a small value (1e-3 to 1e-2) is enough; the
        # global aux loss does the heavy lifting.
        router_seq_balance_loss_coeff: float = 0.0,

        # --- Per-loop auxiliary LM-head loss (architecture fix) ---
        # When > 0, an aux CE loss is computed on the hidden state at
        # the END of each recursive loop except the last one, by
        # applying norm_out + (weight-tied) LM head and comparing to
        # the same next-token labels. Each per-loop loss is added to
        # the main loss with this weight. Forces gradient signal into
        # intermediate loops that would otherwise collapse to "minor
        # refinement" mode (see probe_recursion findings 2026-06-05).
        # Total aux contribution scales with (n_loops - 1) × weight;
        # at 6 loops and weight 0.1 the aux block is +0.5× main loss
        # which is large enough to drive learning without overpowering
        # the primary objective. Set to 0 to disable.
        aux_loop_loss_weight: float = 0.0,

        # --- Balance-bias controller (DeepSeek-style) ---
        # Per-expert additive bias applied to router logits as part of the
        # routing mechanism in both train and eval. Bias is per recursive loop
        # because capacity/drop is enforced per MoE call; aggregating all loops
        # into one block-level bias lets loop-specific imbalances cancel. The
        # bias is updated once per optimizer step from observed clean top-k load:
        #   bias -= update_rate * (current_fraction - uniform_fraction)
        # Over-used experts get pushed down; under-used experts get lifted.
        #
        # This is intentionally non-gradient control: the three 1200-step v5
        # sanity runs showed Switch aux loss, even on the clean path, can
        # delay but not prevent task-gradient concentration onto a favorite
        # subset once LR ramps. The bias directly counter-rotates that load
        # imbalance. Since it is persistent and applied at eval, clean router
        # health means "router logits + loop-specific balance bias, no Gumbel".
        router_balance_bias_enabled: bool = True,
        router_balance_bias_update_rate: float = 0.10,
        router_balance_bias_ema_rate: float = 0.05,
        router_balance_bias_max: float = 1.5,

        # Training-time noisy top-k. The training loop anneals this buffer;
        # default stays 0.0 so unit tests and standalone/eval forwards are
        # deterministic unless the trainer explicitly enables exploration.
        router_gumbel_tau_init: float = 0.0,

        # Capacity factor: expert_capacity = capacity_factor * N / num_experts
        # 2.0 is loose enough that router preferences actually drive routing.
        # Dropped tokens (exceeded capacity) skip the MoE path and get
        # only the shared expert + residual. Switch uses 1.25; we loosen
        # to 2.0 because v4 showed tight caps hide router decisions.
        router_capacity_factor: float = 2.0,

        # Orthogonal expert initialisation:
        # Initialise each routed expert's projection weights with mutually
        # orthogonal feature subspaces so experts start in different
        # directions of the hidden space. Uses QR decomposition of random
        # matrices with a per-expert seed offset.
        expert_orthogonal_init: bool = True,

        # Loop embedding init std (kept from v4 — was correct)
        loop_embedding_init_std: float = 0.1,

        # --- Sequence length (unchanged from v4) ---
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,

        # --- Training defaults ---
        initializer_range: float = 0.02,

        # --- Token IDs (must match tokenizer special tokens) ---
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

        self.num_routed_experts = num_routed_experts
        self.top_k_experts = top_k_experts
        self.expert_hidden = expert_hidden
        self.shared_expert_hidden = shared_expert_hidden

        self.router_aux_loss_coeff = router_aux_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.router_seq_balance_loss_coeff = router_seq_balance_loss_coeff
        self.aux_loop_loss_weight = aux_loop_loss_weight
        self.router_balance_bias_enabled = router_balance_bias_enabled
        self.router_balance_bias_update_rate = router_balance_bias_update_rate
        self.router_balance_bias_ema_rate = router_balance_bias_ema_rate
        self.router_balance_bias_max = router_balance_bias_max
        self.router_gumbel_tau_init = router_gumbel_tau_init
        self.router_capacity_factor = router_capacity_factor
        self.expert_orthogonal_init = expert_orthogonal_init
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
        # head_dim is an explicit config (not dim // heads), so verify
        # the three are mutually consistent. Without this, shape errors
        # only surface at the first forward pass inside attention.
        if self.heads * self.head_dim != self.dim:
            raise ValueError(
                f"heads ({self.heads}) * head_dim ({self.head_dim}) = "
                f"{self.heads * self.head_dim} must equal dim ({self.dim})"
            )
        # RoPE rotates pairs of dimensions, so head_dim must be even.
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for RoPE "
                f"(apply_rope splits the last dim into two halves)"
            )
        if self.top_k_experts > self.num_routed_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) must be <= "
                f"num_routed_experts ({self.num_routed_experts})"
            )
        if self.top_k_experts < 1:
            raise ValueError(
                f"top_k_experts must be >= 1, got {self.top_k_experts}"
            )
        if self.top_k_experts > self.num_routed_experts // 2:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) > "
                f"num_routed_experts/2 ({self.num_routed_experts // 2}) "
                f"defeats the sparsity benefit of MoE. "
                f"Reduce top-k or increase expert count."
            )
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.recursive_loops < 1:
            raise ValueError(
                f"recursive_loops must be >= 1, got {self.recursive_loops}"
            )
        if self.router_capacity_factor <= 1.0:
            raise ValueError(
                f"router_capacity_factor must be > 1.0, got "
                f"{self.router_capacity_factor}. Below 1.0 guarantees "
                f"dropped tokens even with perfect balance."
            )
        if self.router_aux_loss_coeff < 0:
            raise ValueError(
                f"router_aux_loss_coeff must be >= 0, got "
                f"{self.router_aux_loss_coeff}"
            )
        if self.router_z_loss_coeff < 0:
            raise ValueError(
                f"router_z_loss_coeff must be >= 0, got "
                f"{self.router_z_loss_coeff}"
            )
        if self.router_seq_balance_loss_coeff < 0:
            raise ValueError(
                f"router_seq_balance_loss_coeff must be >= 0, got "
                f"{self.router_seq_balance_loss_coeff}"
            )
        if self.router_balance_bias_update_rate < 0:
            raise ValueError(
                f"router_balance_bias_update_rate must be >= 0, got "
                f"{self.router_balance_bias_update_rate}"
            )
        if not 0 <= self.router_balance_bias_ema_rate <= 1:
            raise ValueError(
                f"router_balance_bias_ema_rate must be in [0, 1], got "
                f"{self.router_balance_bias_ema_rate}"
            )
        if self.router_balance_bias_max < 0:
            raise ValueError(
                f"router_balance_bias_max must be >= 0, got "
                f"{self.router_balance_bias_max}"
            )
        if self.router_gumbel_tau_init < 0:
            raise ValueError(
                f"router_gumbel_tau_init must be >= 0, got "
                f"{self.router_gumbel_tau_init}"
            )
