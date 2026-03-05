"""Configuration dataclasses for nano-osrt-100m."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Hyperparameters for the NanoOSRT transformer model.

    Target ~100M parameters with these defaults:
        vocab_size=50257, n_layer=12, n_head=12, n_embd=768
        => ~117M parameters (GPT-2 small scale).
    """

    # Vocabulary
    vocab_size: int = 50257  # GPT-2 / tiktoken cl100k_base compatible

    # Architecture
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    ffn_hidden_mult: int = 4  # FFN hidden dim = ffn_hidden_mult * n_embd
    dropout: float = 0.1
    bias: bool = True

    # Sequence length
    block_size: int = 1024  # maximum context length

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    dataset: str = "openwebtext"
    data_dir: str = "data"

    # Batching
    batch_size: int = 12
    block_size: int = 1024
    grad_accumulation_steps: int = 40  # effective batch ~480 sequences

    # Optimiser
    learning_rate: float = 6e-4
    max_iters: int = 600_000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule (cosine decay)
    warmup_iters: int = 2_000
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5

    # Evaluation
    eval_interval: int = 2_000
    eval_iters: int = 200
    log_interval: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5_000
    resume: bool = False

    # Device / precision
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True

    # Logging
    wandb_log: bool = False
    wandb_project: str = "nano-osrt-100m"
    wandb_run_name: str = "run"

    # Seed
    seed: int = 1337

    # Model config embedded here for convenience
    model: ModelConfig = field(default_factory=ModelConfig)
