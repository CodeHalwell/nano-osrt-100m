"""Configuration for GRPO (Group Relative Policy Optimization) training.

GRPO uses verifiable math rewards instead of a learned reward model.
Pipeline: pretrain → SFT (math) → GRPO (this) → General SFT
"""


class GRPOConfig:
    """Hyperparameters for GRPO reinforcement learning.

    Group Relative Policy Optimization generates multiple completions
    per prompt, scores them with verifiable rewards, and trains the
    policy to increase probability of high-reward completions relative
    to low-reward ones within each group.
    """

    # Architecture (must match pretrained/SFT model)
    dim: int = 1280
    heads: int = 20
    head_dim: int = 64
    seq_len: int = 4096
    vocab_size: int = -1
    real_vocab_size: int = -1
    num_blocks: int = 2
    recursive_loops: int = 6
    adapter_rank: int = 16
    adapter_alpha: float = 16.0

    # High Rank Adaptation (must match SFT)
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_lr: float = 2.5e-5  # 5x base GRPO LR
    hra_freeze_pretrained: bool = False

    # GRPO-specific hyperparameters
    group_size: int = 8  # completions per prompt
    max_gen_len: int = 512  # max tokens to generate per completion
    temperature: float = 0.8  # sampling temperature for generation
    top_p: float = 0.95  # nucleus sampling threshold

    # KL penalty (keeps policy close to reference)
    kl_coeff: float = 0.05  # β in GRPO objective
    clip_range: float = 0.2  # PPO-style ratio clipping ε

    # Training
    batch_size: int = 4  # prompts per micro-batch
    grad_accum_steps: int = 4  # effective = 16 prompts per step
    total_steps: int = 1_500
    warmup_steps: int = 75
    peak_lr: float = 5e-6
    min_lr: float = 5e-7
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    ckpt_interval: int = 250
    optimizer_name: str = "adamw"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-100m"
    wandb_run_name: str = "osrt-grpo-v1"
    wandb_run_id: str = ""

    # SFT checkpoint to start from
    pretrained_checkpoint: str = "/vol/checkpoints/osrt100m_sft_final.pt"

    # Chat format (must match SFT)
    user_prefix: str = "user: "
    assistant_prefix: str = "assistant: "
    think_open: str = "<think>"
    think_close: str = "</think>"

    # Reward weights
    correctness_reward: float = 1.0  # correct final answer
    format_reward: float = 0.2  # used <think>...</think> correctly
    length_penalty: float = -0.001  # per-token penalty to encourage conciseness

    # Prompt source
    prompt_dataset: str = "openai/gsm8k"
    prompt_config: str = "main"
    prompt_split: str = "train"
