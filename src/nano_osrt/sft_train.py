"""SFT training loop for nano-osrt-100m chain-of-thought fine-tuning."""

import glob
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.hra import get_param_groups, inject_hra
from nano_osrt.recursive_model import RecursiveNanoOSRT
from nano_osrt.sft_config import SFTConfig
from nano_osrt.sft_data import IGNORE_INDEX, make_sft_loader


def get_sft_lr(step: int, cfg: SFTConfig) -> float:
    """Cosine learning-rate with linear warmup for SFT."""
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def load_pretrained(model: nn.Module, path: str, device: torch.device) -> None:
    """Load pretrained weights into model (weights only, fresh optimizer)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}. "
            "Run pretraining first or provide a valid checkpoint path."
        )

    print(f"Loading pretrained weights from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        pretrain_step = ckpt.get("step", "unknown")
        print(f"  Checkpoint from pretraining step {pretrain_step}")
    else:
        state_dict = ckpt

    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    missing, unexpected = inner.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (initialized to defaults): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    print("  Pretrained weights loaded successfully.")


def save_sft_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Save an SFT checkpoint."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_stage": "sft",
        },
        path,
    )
    print(f"  -> SFT checkpoint saved: {path}")


def load_sft_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    """Resume SFT from a checkpoint. Returns step to resume from."""
    if not os.path.exists(path):
        return 0
    print(f"Resuming SFT from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"Resumed SFT at step {start_step}")
    return start_step


def run_sft(cfg: SFTConfig, vol, tokenizer_name: str) -> None:
    """Execute the SFT training loop on Modal."""
    device = torch.device("cuda")

    print("=" * 60)
    print("Nano-OSRT 100M — SFT Training (Chain-of-Thought)")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = RecursiveNanoOSRT(cfg).to(device=device)

    base_params = sum(p.numel() for p in model.parameters())
    eff_batch = cfg.batch_size * cfg.grad_accum_steps
    tok_per_step = eff_batch * cfg.seq_len

    print(f"Base parameters     : {base_params:>12,}")
    print(f"SFT learning rate   : {cfg.peak_lr}")
    print(f"Micro-batch         : {cfg.batch_size}")
    print(f"Grad accum steps    : {cfg.grad_accum_steps}")
    print(f"Effective batch     : {eff_batch}")
    print(f"Max tokens per step : {tok_per_step:,}")
    print(f"Total SFT steps     : {cfg.total_steps}")
    print(f"Optimizer           : {cfg.optimizer_name}")
    print(f"Chat format         : user/assistant with <think>...</think>")
    print()

    # ------------------------------------------------------------------
    # High Rank Adaptation (HRA) — expand model capacity
    # ------------------------------------------------------------------
    hra_params = []
    if cfg.hra_enabled and getattr(cfg, "hra_before_load", False):
        # Inject HRA BEFORE loading — needed when checkpoint already has HRA keys
        print(f"\nInjecting HRA adapters before load (rank={cfg.hra_rank})...")
        hra_params = inject_hra(
            model,
            rank=cfg.hra_rank,
            scale=cfg.hra_scale,
            freeze_pretrained=cfg.hra_freeze_pretrained,
        )

    load_pretrained(model, cfg.pretrained_checkpoint, device)

    if cfg.hra_enabled and not getattr(cfg, "hra_before_load", False):
        # Inject HRA AFTER loading — for fresh pretrained checkpoints
        print(f"\nInjecting HRA adapters (rank={cfg.hra_rank})...")
        hra_params = inject_hra(
            model,
            rank=cfg.hra_rank,
            scale=cfg.hra_scale,
            freeze_pretrained=cfg.hra_freeze_pretrained,
        )

    if cfg.hra_enabled:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters  : {total_params:>12,} (+{total_params - base_params:,} HRA)")
        print(f"  Trainable         : {trainable:>12,}")
    else:
        total_params = base_params

    print("\nCompiling model with torch.compile...")
    compile_start = time.time()
    model = torch.compile(model)
    print(f"Model compile done in {time.time() - compile_start:.1f}s")

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": cfg.wandb_project,
            "name": cfg.wandb_run_name,
            "config": {
                "stage": "sft",
                "base_params": base_params,
                "total_params": total_params,
                "hra_enabled": cfg.hra_enabled,
                "hra_rank": cfg.hra_rank if cfg.hra_enabled else 0,
                "hra_extra_params": total_params - base_params,
                "peak_lr": cfg.peak_lr,
                "hra_lr": cfg.hra_lr if cfg.hra_enabled else 0,
                "batch_size": cfg.batch_size,
                "grad_accum_steps": cfg.grad_accum_steps,
                "total_steps": cfg.total_steps,
                "optimizer": cfg.optimizer_name,
                "datasets": [d["name"] for d in cfg.datasets],
            },
        }
        if cfg.wandb_run_id:
            wandb_kwargs["id"] = cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # ------------------------------------------------------------------
    # Optimizer (fresh — no pretrain optimizer state)
    # ------------------------------------------------------------------
    if cfg.hra_enabled and hra_params:
        # Differential LR: pretrained weights at base LR, HRA adapters at higher LR
        param_groups = get_param_groups(
            model, hra_params,
            base_lr=cfg.peak_lr,
            hra_lr=cfg.hra_lr,
            weight_decay=cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            param_groups, betas=(0.9, 0.95), eps=1e-8
        )
        print(f"Using AdamW with differential LR (base={cfg.peak_lr}, hra={cfg.hra_lr})")
    elif cfg.optimizer_name.lower() == "lion":
        from lion_pytorch import Lion

        optimizer = Lion(
            model.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay
        )
        print(f"Using Lion (wd={cfg.weight_decay})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.peak_lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        print(f"Using AdamW (wd={cfg.weight_decay})")

    # ------------------------------------------------------------------
    # Resume from SFT checkpoint if available
    # ------------------------------------------------------------------
    prefix = getattr(cfg, "stage_prefix", "sft")
    sft_rescue_path = f"/vol/checkpoints/osrt100m_{prefix}_rescue.pt"
    ckpt_dir = "/vol/checkpoints"
    best_ckpt = sft_rescue_path
    best_step = -1

    if os.path.isdir(ckpt_dir):
        for f in glob.glob(os.path.join(ckpt_dir, f"osrt100m_{prefix}_step_*.pt")):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
                if s > best_step:
                    best_step = s
                    best_ckpt = f
            except (ValueError, IndexError):
                continue

    start_step = 0
    if best_step > 0:
        print(f"Found SFT checkpoint at step {best_step}: {best_ckpt}")
        start_step = load_sft_checkpoint(model, optimizer, best_ckpt, device)

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    print("\nLoading SFT datasets...")
    load_t = time.time()
    loader = make_sft_loader(
        dataset_configs=cfg.datasets,
        seq_len=cfg.seq_len,
        tokenizer_name=tokenizer_name,
        batch_size=cfg.batch_size,
        seed=42 + start_step,
        think_open=cfg.think_open,
        think_close=cfg.think_close,
        user_prefix=cfg.user_prefix,
        assistant_prefix=cfg.assistant_prefix,
    )
    loader_iter = iter(loader)
    print(f"SFT DataLoader ready in {time.time() - load_t:.1f}s")

    # ------------------------------------------------------------------
    # SFT Training loop
    # ------------------------------------------------------------------
    start_time = time.time()
    step = start_step

    while step < cfg.total_steps:
        lr = get_sft_lr(step, cfg)
        for pg in optimizer.param_groups:
            if cfg.hra_enabled and pg.get("group_name") == "hra":
                # HRA LR follows same cosine shape but scaled to hra_lr
                hra_ratio = cfg.hra_lr / cfg.peak_lr
                pg["lr"] = lr * hra_ratio
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.tensor(0.0, device=device)
        accum_trained_tokens = 0

        for _micro in range(cfg.grad_accum_steps):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                loader = make_sft_loader(
                    dataset_configs=cfg.datasets,
                    seq_len=cfg.seq_len,
                    tokenizer_name=tokenizer_name,
                    batch_size=cfg.batch_size,
                    seed=42 + step,
                    think_open=cfg.think_open,
                    think_close=cfg.think_close,
                    user_prefix=cfg.user_prefix,
                    assistant_prefix=cfg.assistant_prefix,
                )
                loader_iter = iter(loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            accum_trained_tokens += (labels != IGNORE_INDEX).sum().item()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(input_ids)
                active_logits = (
                    logits[..., : cfg.real_vocab_size].contiguous().float()
                )
                loss = F.cross_entropy(
                    active_logits.view(-1, cfg.real_vocab_size),
                    labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                )
                scaled_loss = loss / cfg.grad_accum_steps

            scaled_loss.backward()
            accum_loss += loss.detach() / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # --- Logging ---
        should_log = (
            step % cfg.log_interval == 0
            or step == 0
            or (step < 50 and step % 5 == 0)
        )
        if should_log:
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

            total_positions = eff_batch * cfg.seq_len
            token_util = (
                accum_trained_tokens / total_positions if total_positions > 0 else 0
            )

            print(
                f"step {step:>6d}/{cfg.total_steps} | "
                f"loss {accum_loss.item():.4f} | lr {lr:.2e} | "
                f"vram {vram_gb:.1f}GB | "
                f"tok_util {token_util:.1%} | "
                f"elapsed {elapsed:.0f}s"
            )

            if use_wandb:
                wandb.log(
                    {
                        "sft/loss": accum_loss.item(),
                        "sft/lr": lr,
                        "sft/vram_gb": vram_gb,
                        "sft/token_utilization": token_util,
                        "sft/trained_tokens": accum_trained_tokens,
                    },
                    step=step,
                )
        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()

        # --- Checkpoints ---
        if step > 0 and step % cfg.ckpt_interval == 0:
            path = f"/vol/checkpoints/osrt100m_{prefix}_step_{step}.pt"
            save_sft_checkpoint(model, optimizer, step, path)
            vol.commit()

        # --- 23h Modal safety ---
        if time.time() - start_time > 82_800:
            save_sft_checkpoint(model, optimizer, step, sft_rescue_path)
            vol.commit()
            print(f"\n23h boundary. SFT rescue checkpoint at step {step}.")
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # --- Final ---
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = f"/vol/checkpoints/osrt100m_{prefix}_final.pt"
    torch.save(
        {
            "model_state_dict": inner.state_dict(),
            "training_stage": "sft",
            "total_steps": cfg.total_steps,
        },
        final_path,
    )
    vol.commit()
    elapsed_total = time.time() - start_time
    print(f"\nSFT complete. {step:,} steps in {elapsed_total / 3600:.1f}h")
    print(f"Final SFT model: {final_path}")
    if use_wandb:
        wandb.finish()
