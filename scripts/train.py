"""CLI entry-point for training nano-osrt-100m.

Usage::

    uv run python scripts/train.py

Or, if the package is installed::

    uv run train
"""

import argparse

from nano_osrt.config import ModelConfig, TrainConfig
from nano_osrt.train import train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train nano-osrt-100m")

    # Data
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--data-dir", default="data")

    # Training
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-iters", type=int, default=600_000)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--grad-accumulation-steps", type=int, default=40)

    # Model
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Device
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"]
    )
    parser.add_argument("--no-compile", action="store_true")

    # Logging / checkpointing
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--wandb-log", action="store_true")
    parser.add_argument("--wandb-project", default="nano-osrt-100m")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    model_cfg = ModelConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        block_size=args.block_size,
    )

    return TrainConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
        device=args.device,
        dtype=args.dtype,
        compile=not args.no_compile,
        checkpoint_dir=args.checkpoint_dir,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        resume=args.resume,
        seed=args.seed,
        model=model_cfg,
    )


def main() -> None:
    cfg = parse_args()
    print("Starting training with config:")
    print(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
