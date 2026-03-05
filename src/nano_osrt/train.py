"""Training loop for nano-osrt-100m."""

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from nano_osrt.config import TrainConfig
from nano_osrt.data import get_batch, load_data_split
from nano_osrt.model import NanoOSRT


def cosine_lr(
    it: int,
    *,
    warmup_iters: int,
    lr_decay_iters: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    """Cosine learning-rate schedule with linear warm-up."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train(cfg: TrainConfig) -> NanoOSRT:
    """Run the training loop.

    Args:
        cfg: A fully-populated :class:`TrainConfig` instance.

    Returns:
        The trained :class:`NanoOSRT` model.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Mixed precision context
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.cuda.GradScaler(enabled=(cfg.dtype == "float16"))

    # Data
    train_data = load_data_split(cfg.data_dir, "train")
    val_data = load_data_split(cfg.data_dir, "val")

    # Model
    model = NanoOSRT(cfg.model).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Optional: resume from checkpoint
    start_iter = 0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume:
        ckpt_path = ckpt_dir / "latest.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_iter = checkpoint["iter"] + 1
            print(f"Resumed from checkpoint at iter {start_iter}")

    # Logging
    if cfg.wandb_log:
        import wandb

        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    # Training loop
    best_val_loss = float("inf")
    t0 = time.time()

    for iter_num in range(start_iter, cfg.max_iters):
        # Update LR
        lr = cosine_lr(
            iter_num,
            warmup_iters=cfg.warmup_iters,
            lr_decay_iters=cfg.lr_decay_iters,
            learning_rate=cfg.learning_rate,
            min_lr=cfg.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(cfg.grad_accumulation_steps):
            x, y = get_batch(
                train_data, cfg.block_size, cfg.batch_size, device
            )
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accumulation_steps
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            lossf = loss.item() * cfg.grad_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, dt {dt*1000:.0f}ms")
            if cfg.wandb_log:
                wandb.log({"train/loss": lossf, "lr": lr}, step=iter_num)

        # Evaluation
        if iter_num % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_data, cfg, device, ctx)
            print(f"val loss: {val_loss:.4f}")
            if cfg.wandb_log:
                wandb.log({"val/loss": val_loss}, step=iter_num)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, optimizer, iter_num, ckpt_dir / "best.pt")

        # Periodic checkpoint
        if iter_num % cfg.checkpoint_interval == 0 and iter_num > 0:
            _save_checkpoint(model, optimizer, iter_num, ckpt_dir / "latest.pt")

    return model


@torch.no_grad()
def evaluate(
    model: NanoOSRT,
    val_data,
    cfg: TrainConfig,
    device: torch.device,
    ctx,
) -> float:
    """Estimate validation loss over *cfg.eval_iters* batches."""
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for k in range(cfg.eval_iters):
        x, y = get_batch(val_data, cfg.block_size, cfg.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def _save_checkpoint(
    model: NanoOSRT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    path: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter_num,
            "config": model.config,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")
