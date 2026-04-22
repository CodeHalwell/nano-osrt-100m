"""Training configurations for NanoOSRT.

v5 architecture: Mixtral-style MoE (8 routed × top-2, 1 shared, no dense FFN),
Switch balance loss, orthogonal expert init, eval-time drop-free capacity.

Progressive curriculum:
  Phase 1 (Foundation):  seq_len 2048, FineWeb-Edu + CodeParrot
  Phase 2 (Knowledge):   seq_len 4096, FineWeb-Edu + CodeParrot + Wikipedia
  Phase 3 (Instruction): seq_len 8192, SmolTalk + Evol-Code + OpenHermes

Post-training:
  SFT:  Balanced math + code + STEM + general (native tag format)
  GRPO: Verifiable math rewards
"""


class PretrainConfig:
    """Pre-training hyperparameters for v5."""

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
    eval_steps: int = 20           # number of batches per eval
    ckpt_interval: int = 1_000
    optimizer_name: str = "lion"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt"
    wandb_run_name: str = "osrt-pretrain"
    wandb_run_id: str = ""

    # --- Success criteria for Phase 1 (Foundation) ---
    # v4's router was never alive. v5 uses a four-metric check that would
    # have correctly diagnosed v4's failure (batch-marginal entropy stayed
    # high while per-token entropy also stayed high). All four must hold by
    # step `early_stop_check_step` or training is considered failed.
    early_stop_check_step: int = 5_000
    min_per_token_entropy_drop: float = 0.55   # init 2.08 → 1.53 (ln 8 = 2.08)
    min_raw_max_prob: float = 0.30             # well above uniform 1/8 = 0.125
    min_top_margin: float = 0.10               # clear gap between rank 0 and 1
    min_marginal_entropy: float = 1.80         # balanced across experts

    # --- Router exploration ---
    # Sanity runs showed experts can die during the first 20 optimizer steps:
    # once an expert falls out of top-k it receives no task gradient. Add
    # noisy top-k exploration early. Anneal after LR warmup peaks so the
    # router does not face a rising task-loss gradient with no exploration,
    # but finish 1k steps before the 5k clean-router health gate.
    router_gumbel_tau_init: float = 0.5
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 4_000

    # Progressive seq_len curriculum
    # Tokens per step per phase:
    #   Phase 1 (foundation, 10K steps):  8 × 8  × 2048 = 131K tok/step → ~1.3B
    #   Phase 2 (knowledge, 240K steps):  4 × 16 × 4096 = 262K tok/step → ~63B
    #   Phase 3 (instruction, 50K steps): 2 × 32 × 8192 = 524K tok/step → ~26B
    # Total budget: ~90B tokens if the full 300K schedule completes.
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
            "grad_accum_steps": 32,
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
    # Modal credits allow. Checkpoints every 1K steps keep stop/resume cheap.
    # Any early stopping still leaves a usable model for SFT.


class SFTConfig:
    """Balanced SFT config for v5 — math + code + STEM + general."""

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
    seq_len: int = 2048              # short seq_len + packing (see v4_sft_data)

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
    wandb_project: str = "nano-osrt"
    wandb_run_name: str = "osrt-sft"
    wandb_run_id: str = ""

    # Pretrained checkpoint
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_final.pt"

    # Chat format (native single-token tags)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"

    # Balanced dataset mixture (same as v4)
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


class GRPOConfig:
    """GRPO config for v5 — verifiable math rewards on top of SFT."""

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
    group_size: int = 16
    max_gen_len: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coeff: float = 0.05
    clip_range: float = 0.2

    # HRA (inherited on top of SFT-injected HRA weights)
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
    wandb_project: str = "nano-osrt"
    wandb_run_name: str = "osrt-grpo"
    wandb_run_id: str = ""

    # Checkpoint
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_final.pt"
    stage_prefix: str = "grpo"

    # Prompt source
    prompt_dataset: str = "openai/gsm8k"
    prompt_config: str = "main"
    prompt_split: str = "train"

    # Chat format (native single-token tags, same as v4/SFT)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"
