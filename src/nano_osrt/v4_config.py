"""Configuration for NanoOSRT v4 — Recursive MoE.

3 physical blocks × 6 loops = 18 effective layers.
Dense FFN + MoE (1 shared + 11 routed experts, top-2) in parallel residual.
Loop-aware routing with learned loop embeddings.
"""

from transformers import PretrainedConfig


class NanoOSRTv4Config(PretrainedConfig):
    """HuggingFace-compatible config for NanoOSRT v4.

    Architecture: ~208M physical params, ~131M active per token,
    ~790M effective via recursive weight sharing.

    Vocab reduced from 128K to 64K to rebalance parameter budget:
    128K vocab with dim=1536 consumed 197M params (64% of model) in
    embeddings alone. 64K vocab uses ~101M, freeing ~96M params
    worth of capacity for transformer blocks and MoE experts.
    Trade-off: slightly more tokens per sequence vs 128K, but far
    better allocation of the overall parameter budget.
    """

    model_type = "nano-osrt-v4"

    def __init__(
        self,
        # Core dimensions
        dim: int = 1536,
        heads: int = 24,
        head_dim: int = 64,
        vocab_size: int = 65536,   # 64K vocab — balances tokenisation quality vs param budget
        real_vocab_size: int = 65536,

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
        router_aux_loss_coeff: float = 0.01,
        router_z_loss_coeff: float = 0.001,

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
        self.router_z_loss_coeff = router_z_loss_coeff

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
