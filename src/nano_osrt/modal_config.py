"""Configuration for the recursive-block NanoOSRT Modal deployment."""


class ModalConfig:
    """Hyperparameters for the recursive-block NanoOSRT variant.

    104.5M physical parameters simulating 302M equivalent dense via
    recursive weight sharing.  2 physical blocks × 6 recursive loops = 12
    effective layers, each with unique per-pass residual adapters.

    Attributes:
        dim: Hidden dimension.
        heads: Number of attention heads.
        head_dim: Dimension per attention head.
        seq_len: Maximum sequence length.
        vocab_size: Padded vocabulary size (set at runtime from tokenizer).
        real_vocab_size: Actual tokenizer vocabulary size (set at runtime).
        num_blocks: Number of physical transformer blocks.
        recursive_loops: Number of times each block is re-used.
        adapter_rank: Rank of the per-pass residual adapters.
        adapter_alpha: Scaling numerator for adapter outputs.
        batch_size: Micro-batch size.
        grad_accum_steps: Gradient accumulation steps.
        total_steps: Total training steps.
        warmup_steps: Linear warmup steps (longer for sign-based optimizers).
        peak_lr: Peak learning rate.
        min_lr: Minimum learning rate after cosine decay.
        weight_decay: Weight decay (Lion: 3-10× higher than AdamW).
        grad_clip: Gradient clipping norm.
        log_interval: Steps between log messages.
        ckpt_interval: Steps between checkpoints.
        optimizer_name: Optimizer to use ("lion" or "adamw").
        phases: Ordered training-data curriculum phases.
    """

    dim: int = 1280
    heads: int = 20
    head_dim: int = 64
    seq_len: int = 2048

    # Overwritten dynamically at runtime from tokenizer
    vocab_size: int = -1
    real_vocab_size: int = -1

    num_blocks: int = 2
    recursive_loops: int = 6

    adapter_rank: int = 16
    adapter_alpha: float = 16.0

    batch_size: int = 16
    grad_accum_steps: int = 4
    total_steps: int = 150_000
    warmup_steps: int = 2000
    peak_lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.3
    grad_clip: float = 1.0
    log_interval: int = 50
    ckpt_interval: int = 2000
    optimizer_name: str = "lion"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-100m"
    wandb_run_name: str = "osrt-v3.2"

    # Tokens/step = batch_size * seq_len * grad_accum = 16 * 2048 * 4 = 131,072
    phases: dict = {  # noqa: RUF012
        "tinystories": {
            "start": 0,
            "end": 8_000,
            "dataset": "roneneldan/TinyStories",
        },
        "fineweb": {
            "start": 8_000,
            "end": 140_000,
            "dataset": "HuggingFaceFW/fineweb-edu",
        },
        "smoltalk": {
            "start": 140_000,
            "end": 150_000,
            "dataset": "HuggingFaceTB/smoltalk",
        },
    }
