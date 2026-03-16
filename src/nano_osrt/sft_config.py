"""Configuration for supervised fine-tuning of nano-osrt-100m.

Two SFT stages:
1. SFTConfig — math/reasoning with <think>...</think> (post-pretrain)
2. GeneralSFTConfig — broad instruction following (post-GRPO)
"""


class SFTConfig:
    """Hyperparameters for chain-of-thought SFT.

    Full-parameter fine-tuning on the pretrained 104.5M recursive
    transformer. Trains on reasoning data with loss masking: only
    assistant/thinking tokens contribute to the loss.
    """

    # Architecture (must match pretrained model)
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

    # High Rank Adaptation (HRA) — adds capacity for reasoning
    hra_enabled: bool = True
    hra_rank: int = 256  # ~11M extra params at rank 256
    hra_scale: float = 1.0
    hra_lr: float = 1e-4  # 5x base LR for fresh adapter params
    hra_freeze_pretrained: bool = False  # train everything, differential LR
    hra_before_load: bool = False  # inject HRA before loading checkpoint
    stage_prefix: str = "sft"  # checkpoint naming prefix

    # SFT training hyperparameters
    batch_size: int = 8
    grad_accum_steps: int = 8
    total_steps: int = 3_000
    warmup_steps: int = 150
    peak_lr: float = 2e-5
    min_lr: float = 2e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 25
    ckpt_interval: int = 500
    optimizer_name: str = "adamw"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt-100m"
    wandb_run_name: str = "osrt-sft-v1"
    wandb_run_id: str = ""

    # Pretrained checkpoint
    pretrained_checkpoint: str = "/vol/checkpoints/osrt100m_final.pt"

    # Chat format
    user_prefix: str = "user: "
    assistant_prefix: str = "assistant: "
    think_open: str = "<think>"
    think_close: str = "</think>"

    # Dataset mixture
    # GSM8K (grade school math) + Orca-Math (word problems) + NuminaMath-CoT
    # (competition math, AI Math Olympiad winner) + MathInstruct + LongForm
    datasets: list = [  # noqa: RUF012
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.25,
            "format": "gsm8k",
        },
        {
            "name": "orca-math",
            "hf_id": "microsoft/orca-math-word-problems-200k",
            "split": "train",
            "weight": 0.25,
            "format": "orca_math",
        },
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.20,
            "format": "numina_math",
        },
        {
            "name": "math-instruct",
            "hf_id": "TIGER-Lab/MathInstruct",
            "split": "train",
            "weight": 0.15,
            "format": "math_instruct",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.15,
            "format": "longform",
        },
    ]


class GeneralSFTConfig(SFTConfig):
    """General instruction following after GRPO.

    Pipeline: pretrain -> SFT (math) -> GRPO -> GeneralSFT (this)
    """

    total_steps: int = 2_000
    warmup_steps: int = 100
    peak_lr: float = 1e-5
    min_lr: float = 1e-6
    batch_size: int = 8
    grad_accum_steps: int = 4

    wandb_run_name: str = "osrt-general-sft-v1"
    wandb_run_id: str = ""

    pretrained_checkpoint: str = "/vol/checkpoints/osrt100m_grpo_final.pt"
    hra_before_load: bool = True
    stage_prefix: str = "general"

    datasets: list = [  # noqa: RUF012
        {
            "name": "alpaca-cleaned",
            "hf_id": "yahma/alpaca-cleaned",
            "split": "train",
            "weight": 0.25,
            "format": "alpaca",
        },
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.20,
            "format": "openhermes",
        },
        {
            "name": "slimorca",
            "hf_id": "Open-Orca/SlimOrca-Dedup",
            "split": "train",
            "weight": 0.20,
            "format": "slimorca",
        },
        {
            "name": "ifeval-like",
            "hf_id": "argilla/ifeval-like-data",
            "hf_config": "filtered",
            "split": "train",
            "weight": 0.10,
            "format": "ifeval",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.15,
            "format": "longform",
        },
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.10,
            "format": "gsm8k",
        },
    ]


class CodeSFTConfig(SFTConfig):
    """Stage 4: Code fine-tuning after GRPO.

    Teaches the model to reason about code using <think>...</think> format.
    Focused on Python and general programming with math retention.

    Pipeline: pretrain -> SFT (math) -> GRPO -> CodeSFT (this)
    """

    total_steps: int = 7_000  # ~2 epochs over 220K examples
    warmup_steps: int = 200
    peak_lr: float = 1.5e-5
    min_lr: float = 1.5e-6
    batch_size: int = 8
    grad_accum_steps: int = 8
    hra_lr: float = 7.5e-5  # 5x base LR

    # Weights & Biases
    wandb_run_name: str = "osrt-code-sft-v1"
    wandb_run_id: str = ""

    # Load from GRPO checkpoint (has HRA structure)
    pretrained_checkpoint: str = "/vol/checkpoints/osrt100m_grpo_final.pt"
    hra_before_load: bool = True
    stage_prefix: str = "code"

    # Code-focused datasets with math retention
    datasets: list = [  # noqa: RUF012
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.30,
            "format": "evol_code",
        },
        {
            "name": "code-instructions-122k",
            "hf_id": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "weight": 0.30,
            "format": "alpaca_code",
        },
        {
            "name": "python-code-instructions",
            "hf_id": "iamtarun/python_code_instructions_18k_alpaca",
            "split": "train",
            "weight": 0.20,
            "format": "alpaca_code",
        },
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.10,
            "format": "gsm8k",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.10,
            "format": "longform",
        },
    ]
