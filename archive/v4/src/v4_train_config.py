"""Training configurations for NanoOSRT v4.

Progressive curriculum:
  Phase 1 (Foundation):  seq_len 2048, FineWeb-Edu + CodeParrot
  Phase 2 (Knowledge):   seq_len 4096, FineWeb-Edu + CodeParrot + Wikipedia
  Phase 3 (Instruction): seq_len 8192, SmolTalk + Evol-Code + OpenHermes

Post-training:
  SFT:  Balanced math + code + STEM + general (native tag format)
  GRPO: Verifiable math rewards (retry with larger model)
"""


class V4PretrainConfig:
    """Pre-training hyperparameters for v4."""

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    total_steps: int = 300_000
    warmup_steps: int = 3_000
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.3
    grad_clip: float = 1.0
    log_interval: int = 50
    eval_interval: int = 1_000
    eval_steps: int = 20  # number of batches per eval
    ckpt_interval: int = 1_000
    optimizer_name: str = "lion"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-v4"
    wandb_run_name: str = "osrt-v4-pretrain"
    wandb_run_id: str = ""

    # Progressive seq_len curriculum
    # Tokens per step at each phase:
    #   Phase 1 (foundation, 10K steps):  8 × 8  × 2048 = 131K tok/step → ~1.3B tokens
    #   Phase 2 (knowledge, 240K steps):  4 × 16 × 4096 = 262K tok/step → ~63B tokens
    #   Phase 3 (instruction, 50K steps): 2 × 32 × 8192 = 524K tok/step → ~26B tokens
    # Total budget: ~90B tokens if the full 300K schedule completes.
    # Batch sizes reduced at longer seq_len to fit H100 80GB VRAM.
    phases: dict = {  # noqa: RUF012
        "foundation": {
            "start": 0,
            "end": 10_000,
            "seq_len": 2048,
            "grad_accum_steps": 8,
            "datasets": [
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.6,
                },
                {
                    "name": "codeparrot-clean",
                    "hf_id": "codeparrot/codeparrot-clean",
                    "weight": 0.4,
                },
            ],
        },
        "knowledge": {
            "start": 10_000,
            "end": 250_000,
            "seq_len": 4096,
            "batch_size": 4,
            "grad_accum_steps": 16,
            "datasets": [
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.50,
                },
                {
                    "name": "codeparrot-clean",
                    "hf_id": "codeparrot/codeparrot-clean",
                    "weight": 0.30,
                },
                {
                    "name": "wikipedia",
                    "hf_id": "wikimedia/wikipedia",
                    "hf_config": "20231101.en",
                    "weight": 0.20,
                },
            ],
        },
        "instruction": {
            "start": 250_000,
            "end": 300_000,
            "seq_len": 8192,
            "batch_size": 2,
            "grad_accum_steps": 32,  # increased to compensate for smaller batch_size
            "datasets": [
                {
                    "name": "smoltalk",
                    "hf_id": "HuggingFaceTB/smoltalk",
                    "hf_config": "all",
                    "weight": 0.50,
                },
                {
                    "name": "evol-instruct-code",
                    "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
                    "weight": 0.30,
                },
                {
                    "name": "openhermes",
                    "hf_id": "teknium/OpenHermes-2.5",
                    "weight": 0.20,
                },
            ],
        },
    }

    # Budget note: the schedule is aspirational — the user runs in chunks as
    # Modal credits allow. Checkpoints every 5K steps make stop/resume cheap.
    # Any early stopping still leaves a usable model for SFT.


class V4SFTConfig:
    """Balanced SFT config for v4 — math + code + STEM + general."""

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    total_steps: int = 5_000
    warmup_steps: int = 250
    peak_lr: float = 1.5e-5
    min_lr: float = 1.5e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 25
    ckpt_interval: int = 500
    optimizer_name: str = "adamw"
    seq_len: int = 2048

    # HRA
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_lr: float = 7.5e-5
    hra_freeze_pretrained: bool = False
    hra_before_load: bool = False
    stage_prefix: str = "sft"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-v4"
    wandb_run_name: str = "osrt-v4-sft"
    wandb_run_id: str = ""

    # Pretrained checkpoint
    pretrained_checkpoint: str = "/vol/checkpoints/v4/osrt_v4_step_9000.pt"

    # Chat format (uses native single-token tags)
    # Format: <|user|>{prompt}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"

    # Balanced dataset mixture
    datasets: list = [  # noqa: RUF012
        # Math (25%)
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.10,
            "format": "gsm8k",
        },
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.15,
            "format": "numina_math",
        },
        # Code (25%)
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.15,
            "format": "evol_code",
        },
        {
            "name": "code-instructions-122k",
            "hf_id": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca_code",
        },
        # STEM (20%)
        {
            "name": "orca-math",
            "hf_id": "microsoft/orca-math-word-problems-200k",
            "split": "train",
            "weight": 0.10,
            "format": "orca_math",
        },
        {
            "name": "math-instruct",
            "hf_id": "TIGER-Lab/MathInstruct",
            "split": "train",
            "weight": 0.10,
            "format": "math_instruct",
        },
        # General (20%)
        {
            "name": "alpaca-cleaned",
            "hf_id": "yahma/alpaca-cleaned",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca",
        },
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.10,
            "format": "openhermes",
        },
        # Instruction following (10%)
        {
            "name": "ifeval-like",
            "hf_id": "argilla/ifeval-like-data",
            "hf_config": "filtered",
            "split": "train",
            "weight": 0.05,
            "format": "ifeval",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.05,
            "format": "longform",
        },
    ]


class V4GRPOConfig:
    """GRPO config for v4 — retry with larger model."""

    batch_size: int = 4
    grad_accum_steps: int = 4
    total_steps: int = 2_000
    warmup_steps: int = 100
    peak_lr: float = 3e-6
    min_lr: float = 3e-7
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    ckpt_interval: int = 250
    seq_len: int = 8192

    # GRPO-specific
    group_size: int = 16  # more completions (model should be more capable)
    max_gen_len: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coeff: float = 0.05
    clip_range: float = 0.2

    # HRA
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_lr: float = 1.5e-5
    hra_before_load: bool = True

    # Rewards
    correctness_reward: float = 1.0
    format_reward: float = 0.2
    reasoning_bonus: float = 0.3
    truncation_penalty: float = -0.5
    empty_think_penalty: float = -0.1
    length_penalty: float = 0.0

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-v4"
    wandb_run_name: str = "osrt-v4-grpo"
    wandb_run_id: str = ""

    # Checkpoint
    pretrained_checkpoint: str = "/vol/checkpoints/v4/osrt_v4_sft_final.pt"
    stage_prefix: str = "grpo"

    # Prompt source
    prompt_dataset: str = "openai/gsm8k"
    prompt_config: str = "main"
    prompt_split: str = "train"

    # Chat format (native single-token tags)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"


class V4MoERecoverConfig:
    """MoE recovery config — wake up the dead router.

    Strategy:
      1. Re-init all routed experts from shared expert + small noise
      2. Freeze everything except router, routed experts, and gates
      3. Remove importance loss (it enforces uniformity)
      4. Loosen capacity cap so router preferences matter
      5. Add diversity loss pushing experts to produce different outputs
    """

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    total_steps: int = 2_000
    warmup_steps: int = 100
    peak_lr: float = 5e-5          # higher than SFT since only MoE bits train
    min_lr: float = 5e-6
    grad_clip: float = 1.0
    log_interval: int = 25
    ckpt_interval: int = 500
    seq_len: int = 2048

    # Routing overrides (applied to model_config at runtime)
    capacity_factor: float = 2.5   # loosened from 1.25 — router actually drives routing
    gumbel_tau: float = 0.3        # keep some noise so router can explore
    diversity_coeff: float = 0.1   # push experts apart
    reinit_noise_std: float = 0.01
    reinit_seed: int = 1337

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-v4"
    wandb_run_name: str = "osrt-v4-moe-recover"

    # Checkpoint to start from (latest SFT checkpoint)
    pretrained_checkpoint: str = "/vol/checkpoints/v4/osrt_v4_sft_step_3500.pt"

    # Datasets (Foundation phase: FineWeb-Edu + CodeParrot)
    datasets: list = [  # noqa: RUF012
        {
            "name": "fineweb-edu",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "weight": 0.6,
        },
        {
            "name": "codeparrot-clean",
            "hf_id": "codeparrot/codeparrot-clean",
            "weight": 0.4,
        },
    ]
