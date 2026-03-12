"""Training utilities for the Modal deployment of nano-osrt-100m.

This module contains the core training loop and helpers used by the Modal
``app.py`` entrypoint.  All functions are pure-Python / PyTorch so they can
be unit-tested independently of Modal infrastructure.
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.modal_config import ModalConfig
from nano_osrt.modal_data import make_loader
from nano_osrt.recursive_model import RecursiveNanoOSRT

# ------------------------------------------------------------------
# LR schedule & phase helpers
# ------------------------------------------------------------------


def get_lr(step: int, cfg: ModalConfig) -> float:
    """Cosine learning-rate with linear warmup.

    Args:
        step: Current training step.
        cfg: Training configuration.

    Returns:
        Learning rate for *step*.
    """
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def get_phase(step: int, cfg: ModalConfig) -> tuple[str, str]:
    """Determine the curriculum phase for a given *step*.

    Args:
        step: Current training step.
        cfg: Training configuration.

    Returns:
        ``(phase_name, dataset_name)`` tuple.
    """
    for name, p in cfg.phases.items():
        if p["start"] <= step < p["end"]:
            return name, p["dataset"]
    return "fineweb", cfg.phases["fineweb"]["dataset"]


# ------------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------------


def log_step(
    step: int,
    accum_loss: torch.Tensor,
    lr: float,
    start_step: int,
    start_time: float,
    tok_per_step: int,
    current_phase: str,
    last_loop_rms: list[torch.Tensor] | None,
    model: nn.Module,
    cfg: ModalConfig,
) -> dict:
    """Print a formatted log line and return metrics dict for W&B."""
    elapsed = time.time() - start_time
    tokens_so_far = (step - start_step) * tok_per_step
    tok_per_sec = tokens_so_far / elapsed if elapsed > 0 else 0
    total_tokens = step * tok_per_step
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    intra_sims = []
    inter_sims = []
    with torch.no_grad():
        inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        b_mats = [b.flatten().float() for b in inner_model.adapters_b]
        if b_mats[0].norm() > 1e-6:
            b0_vecs = [b_mats[i] for i in range(0, len(b_mats), cfg.num_blocks)]
            intra_sims = [
                F.cosine_similarity(b0_vecs[0], v, dim=0).item()
                for v in b0_vecs[1:]
            ]
            intra_str = "[" + ", ".join(f"{s:.2f}" for s in intra_sims) + "]"
            inter_sims = [
                F.cosine_similarity(
                    b_mats[loop * cfg.num_blocks],
                    b_mats[loop * cfg.num_blocks + 1],
                    dim=0,
                ).item()
                for loop in range(cfg.recursive_loops)
            ]
            inter_str = "[" + ", ".join(f"{s:.2f}" for s in inter_sims) + "]"
        else:
            intra_str = "[waking up...]"
            inter_str = "[waking up...]"

    rms_str = (
        "[" + ", ".join(f"{r.item():.3f}" for r in last_loop_rms) + "]"
        if last_loop_rms
        else "[n/a]"
    )

    print(
        f"step {step:>7d} | loss {accum_loss.item():.4f} | "
        f"lr {lr:.2e} | vram {vram_gb:.1f}GB | "
        f"tok/s {tok_per_sec:,.0f} | phase {current_phase}\n"
        f"           intra-block (b0: L0 vs L1..5): {intra_str}\n"
        f"           inter-block (b0 vs b1 per loop): {inter_str}\n"
        f"           loop RMS: {rms_str}"
    )

    # Build metrics dict for W&B
    metrics: dict = {
        "train/loss": accum_loss.item(),
        "train/lr": lr,
        "train/vram_gb": vram_gb,
        "train/tok_per_sec": tok_per_sec,
        "train/total_tokens": total_tokens,
        "train/phase": current_phase,
    }
    if last_loop_rms:
        for i, r in enumerate(last_loop_rms):
            metrics[f"loop_rms/loop_{i}"] = r.item()
    for i, s in enumerate(intra_sims):
        metrics[f"adapter/intra_block_sim_{i}"] = s
    for i, s in enumerate(inter_sims):
        metrics[f"adapter/inter_block_sim_loop_{i}"] = s

    return metrics


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Save a training checkpoint to *path*.

    Handles ``torch.compile``-wrapped models by extracting ``_orig_mod``.
    """
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"  -> checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    """Resume from a checkpoint at *path*.

    Returns:
        The training step to resume from.
    """
    if not os.path.exists(path):
        return 0
    print(f"Resuming from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"Resumed at step {start_step}")
    return start_step


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------


def run_training(cfg: ModalConfig, vol, tokenizer_name: str) -> None:
    """Execute the full Modal training loop.

    Args:
        cfg: Fully-populated :class:`ModalConfig` with vocab sizes set.
        vol: Modal :class:`Volume` for checkpoint persistence.
        tokenizer_name: HuggingFace tokenizer identifier.
    """
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Nano-OSRT 100M Trainer (v3.2)")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = RecursiveNanoOSRT(cfg).to(device=device)

    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in model.adapters_a) + sum(
        p.numel() for p in model.adapters_b
    )
    eff_batch = cfg.batch_size * cfg.grad_accum_steps
    tok_per_step = eff_batch * cfg.seq_len

    print(
        f"Vocab size          : {cfg.real_vocab_size} (padded to {cfg.vocab_size})"
    )
    print(f"Physical parameters : {total_params:>12,}")
    print(
        f"  of which adapters : {adapter_params:>12,} "
        f"({adapter_params / total_params * 100:.2f}%)"
    )
    adapter_scale_val = cfg.adapter_alpha / cfg.adapter_rank
    print(
        f"Adapter scale       : {cfg.adapter_alpha}/{cfg.adapter_rank} "
        f"= {adapter_scale_val:.1f}"
    )
    print(
        f"Effective depth     : {cfg.recursive_loops} loops x "
        f"{cfg.num_blocks} blocks = {cfg.recursive_loops * cfg.num_blocks} layers"
    )
    print(f"Micro-batch         : {cfg.batch_size}")
    print(f"Grad accum steps    : {cfg.grad_accum_steps}")
    print(f"Effective batch     : {eff_batch}")
    print(f"Tokens per step     : {tok_per_step:,}")
    print(f"Total token budget  : ~{cfg.total_steps * tok_per_step / 1e9:.1f}B")
    print(f"Optimizer           : {cfg.optimizer_name}")
    print(
        "Precision           : FP32 master weights, BF16 compute (autocast)"
    )
    print()

    model = torch.compile(model, mode="max-autotune")

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                "dim": cfg.dim,
                "heads": cfg.heads,
                "head_dim": cfg.head_dim,
                "seq_len": cfg.seq_len,
                "vocab_size": cfg.vocab_size,
                "real_vocab_size": cfg.real_vocab_size,
                "num_blocks": cfg.num_blocks,
                "recursive_loops": cfg.recursive_loops,
                "adapter_rank": cfg.adapter_rank,
                "adapter_alpha": cfg.adapter_alpha,
                "batch_size": cfg.batch_size,
                "grad_accum_steps": cfg.grad_accum_steps,
                "total_steps": cfg.total_steps,
                "warmup_steps": cfg.warmup_steps,
                "peak_lr": cfg.peak_lr,
                "min_lr": cfg.min_lr,
                "weight_decay": cfg.weight_decay,
                "grad_clip": cfg.grad_clip,
                "optimizer": cfg.optimizer_name,
                "total_params": total_params,
                "adapter_params": adapter_params,
                "tok_per_step": tok_per_step,
            },
            resume="allow",
        )
        print("W&B logging enabled.")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if cfg.optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.peak_lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        print("Using AdamW (fallback, wd=0.1)")
    else:
        from lion_pytorch import Lion

        optimizer = Lion(
            model.parameters(),
            lr=cfg.peak_lr,
            weight_decay=cfg.weight_decay,
        )
        print(f"Using Lion (sign-based, wd={cfg.weight_decay})")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    rescue_path = "/vol/checkpoints/osrt100m_rescue.pt"
    start_step = load_checkpoint(model, optimizer, rescue_path, device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    current_phase: str | None = None
    loader_iter = None
    current_loader = None
    start_time = time.time()
    step = start_step

    while step < cfg.total_steps:
        phase_name, dataset_name = get_phase(step, cfg)

        if phase_name != current_phase:
            current_phase = phase_name
            print(
                f"\n>>> Phase: {current_phase} | "
                f"Dataset: {dataset_name} | Step: {step}"
            )
            if current_loader is not None:
                del current_loader
            current_loader = make_loader(
                dataset_name, cfg.seq_len, tokenizer_name, cfg.batch_size, step
            )
            loader_iter = iter(current_loader)

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.tensor(0.0, device=device)
        last_loop_rms = None

        for micro in range(cfg.grad_accum_steps):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                _, ds_name = get_phase(step, cfg)
                if current_loader is not None:
                    del current_loader
                current_loader = make_loader(
                    ds_name,
                    cfg.seq_len,
                    tokenizer_name,
                    cfg.batch_size,
                    step,
                )
                loader_iter = iter(current_loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loop_rms_tensors = model(input_ids)
                active_logits = (
                    logits[..., : cfg.real_vocab_size].contiguous().float()
                )
                loss = F.cross_entropy(
                    active_logits.view(-1, cfg.real_vocab_size),
                    labels.view(-1),
                )
                scaled_loss = loss / cfg.grad_accum_steps

            scaled_loss.backward()
            accum_loss += loss.detach() / cfg.grad_accum_steps

            if micro == cfg.grad_accum_steps - 1:
                last_loop_rms = [r.detach() for r in loop_rms_tensors]

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # --- Logging ---
        if step % cfg.log_interval == 0:
            metrics = log_step(
                step,
                accum_loss,
                lr,
                start_step,
                start_time,
                tok_per_step,
                current_phase,
                last_loop_rms,
                model,
                cfg,
            )
            if use_wandb:
                wandb.log(metrics, step=step)

        # --- Checkpoints ---
        if step > 0 and step % cfg.ckpt_interval == 0:
            path = f"/vol/checkpoints/osrt100m_step_{step}.pt"
            save_checkpoint(model, optimizer, step, path)
            vol.commit()

        # --- 23h Modal safety ---
        if time.time() - start_time > 82_800:
            save_checkpoint(model, optimizer, step, rescue_path)
            vol.commit()
            print(
                f"\n23h boundary. Rescue checkpoint at step {step}. "
                "Re-run to resume."
            )
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # --- Final ---
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = "/vol/checkpoints/osrt100m_final.pt"
    torch.save(inner.state_dict(), final_path)
    vol.commit()
    elapsed_total = time.time() - start_time
    print(f"\nTraining complete. {step:,} steps in {elapsed_total / 3600:.1f}h")
    print(f"Final model: {final_path}")
    if use_wandb:
        wandb.finish()
