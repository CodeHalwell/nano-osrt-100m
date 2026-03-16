"""SFT training loop for NanoOSRT v4.

Balanced fine-tuning: math + code + STEM + general.
Uses native single-token tags for chat format.
Supports HRA injection for expanded capacity.
"""

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

from nano_osrt.v4_config import NanoOSRTv4Config
from nano_osrt.v4_model import NanoOSRTv4ForCausalLM
from nano_osrt.v4_sft_data import IGNORE_INDEX, make_v4_sft_loader


def get_sft_lr(step: int, total_steps: int, warmup: int, peak: float, minimum: float) -> float:
    """Cosine LR with linear warmup."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def run_v4_sft(model_config: NanoOSRTv4Config, sft_cfg, vol, tokenizer) -> None:
    """Execute the v4 SFT training loop."""
    device = torch.device("cuda")

    print("=" * 60)
    print("NanoOSRT v4 — SFT Training (Balanced)")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = NanoOSRTv4ForCausalLM(model_config).to(device=device)
    base_params = sum(p.numel() for p in model.parameters())

    eff_batch = sft_cfg.batch_size * sft_cfg.grad_accum_steps
    print(f"Base parameters     : {base_params:>12,}")
    print(f"SFT learning rate   : {sft_cfg.peak_lr}")
    print(f"Effective batch     : {eff_batch}")
    print(f"Seq len             : {sft_cfg.seq_len}")
    print(f"Total SFT steps     : {sft_cfg.total_steps}")
    print(f"HRA enabled         : {sft_cfg.hra_enabled}")
    print()

    # HRA injection (before or after load depending on config)
    hra_params = []
    if sft_cfg.hra_enabled and getattr(sft_cfg, "hra_before_load", False):
        from nano_osrt.hra import inject_hra
        print(f"Injecting HRA before load (rank={sft_cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=sft_cfg.hra_rank, scale=sft_cfg.hra_scale,
                                freeze_pretrained=sft_cfg.hra_freeze_pretrained)

    # Load pretrained weights
    ckpt_path = sft_cfg.pretrained_checkpoint
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Weights loaded.")
    else:
        print(f"WARNING: Checkpoint not found: {ckpt_path}")

    if sft_cfg.hra_enabled and not getattr(sft_cfg, "hra_before_load", False):
        from nano_osrt.hra import inject_hra
        print(f"Injecting HRA after load (rank={sft_cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=sft_cfg.hra_rank, scale=sft_cfg.hra_scale,
                                freeze_pretrained=sft_cfg.hra_freeze_pretrained)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters    : {total_params:>12,}")

    print("\nCompiling model...")
    compile_t = time.time()
    model = torch.compile(model)
    print(f"Compile done in {time.time() - compile_t:.1f}s")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = sft_cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": sft_cfg.wandb_project,
            "name": sft_cfg.wandb_run_name,
            "config": {
                "stage": "sft",
                "total_params": total_params,
                "hra_enabled": sft_cfg.hra_enabled,
                "datasets": [d["name"] for d in sft_cfg.datasets],
            },
        }
        if sft_cfg.wandb_run_id:
            wandb_kwargs["id"] = sft_cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if sft_cfg.hra_enabled and hra_params:
        from nano_osrt.hra import get_param_groups
        param_groups = get_param_groups(model, hra_params,
                                        base_lr=sft_cfg.peak_lr,
                                        hra_lr=sft_cfg.hra_lr,
                                        weight_decay=sft_cfg.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
        print(f"AdamW with differential LR (base={sft_cfg.peak_lr}, hra={sft_cfg.hra_lr})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg.peak_lr,
                                       weight_decay=sft_cfg.weight_decay,
                                       betas=(0.9, 0.95), eps=1e-8)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    prefix = getattr(sft_cfg, "stage_prefix", "sft")
    ckpt_dir = "/vol/checkpoints/v4"
    os.makedirs(ckpt_dir, exist_ok=True)
    rescue_path = f"{ckpt_dir}/osrt_v4_{prefix}_rescue.pt"
    best_step = -1
    best_ckpt = rescue_path

    for f in glob.glob(f"{ckpt_dir}/osrt_v4_{prefix}_step_*.pt"):
        try:
            s = int(f.rsplit("_", 1)[1].split(".")[0])
            if s > best_step:
                best_step = s
                best_ckpt = f
        except (ValueError, IndexError):
            continue

    start_step = 0
    if best_step > 0:
        print(f"Found {prefix} checkpoint at step {best_step}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        inner.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
        start_step = ckpt["step"] + 1
        print(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nLoading SFT datasets...")
    load_t = time.time()
    loader = make_v4_sft_loader(
        dataset_configs=sft_cfg.datasets,
        seq_len=sft_cfg.seq_len,
        tokenizer=tokenizer,
        batch_size=sft_cfg.batch_size,
        seed=42 + start_step,
        user_tag=sft_cfg.user_tag,
        assistant_tag=sft_cfg.assistant_tag,
        think_open=sft_cfg.think_open,
        think_close=sft_cfg.think_close,
        answer_open=sft_cfg.answer_open,
        answer_close=sft_cfg.answer_close,
    )
    loader_iter = iter(loader)
    print(f"DataLoader ready in {time.time() - load_t:.1f}s")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_time = time.time()
    step = start_step

    while step < sft_cfg.total_steps:
        lr = get_sft_lr(step, sft_cfg.total_steps, sft_cfg.warmup_steps,
                         sft_cfg.peak_lr, sft_cfg.min_lr)
        for pg in optimizer.param_groups:
            if sft_cfg.hra_enabled and pg.get("group_name") == "hra":
                pg["lr"] = lr * (sft_cfg.hra_lr / sft_cfg.peak_lr)
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.tensor(0.0, device=device)
        accum_trained_tokens = 0

        for _micro in range(sft_cfg.grad_accum_steps):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                loader = make_v4_sft_loader(
                    dataset_configs=sft_cfg.datasets,
                    seq_len=sft_cfg.seq_len,
                    tokenizer=tokenizer,
                    batch_size=sft_cfg.batch_size,
                    seed=42 + step,
                    user_tag=sft_cfg.user_tag,
                    assistant_tag=sft_cfg.assistant_tag,
                    think_open=sft_cfg.think_open,
                    think_close=sft_cfg.think_close,
                    answer_open=sft_cfg.answer_open,
                    answer_close=sft_cfg.answer_close,
                )
                loader_iter = iter(loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            accum_trained_tokens += (labels != IGNORE_INDEX).sum().item()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / sft_cfg.grad_accum_steps

            loss.backward()
            accum_loss += outputs.loss.detach() / sft_cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
        optimizer.step()

        # --- Logging ---
        should_log = (
            step % sft_cfg.log_interval == 0
            or step == 0
            or (step < 50 and step % 5 == 0)
        )
        if should_log:
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            total_positions = eff_batch * sft_cfg.seq_len
            tok_util = accum_trained_tokens / total_positions if total_positions > 0 else 0

            print(
                f"step {step:>6d}/{sft_cfg.total_steps} | "
                f"loss {accum_loss.item():.4f} | lr {lr:.2e} | "
                f"vram {vram_gb:.1f}GB | tok_util {tok_util:.1%} | "
                f"elapsed {elapsed:.0f}s"
            )

            if use_wandb:
                wandb.log({
                    f"{prefix}/loss": accum_loss.item(),
                    f"{prefix}/lr": lr,
                    f"{prefix}/vram_gb": vram_gb,
                    f"{prefix}/token_utilization": tok_util,
                    f"{prefix}/trained_tokens": accum_trained_tokens,
                }, step=step)

        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()

        # --- Checkpoints ---
        if step > 0 and step % sft_cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v4_{prefix}_step_{step}.pt"
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_stage": prefix,
            }, path)
            vol.commit()
            print(f"  -> Checkpoint saved: {path}")

        # --- 23h safety ---
        if time.time() - start_time > 82_800:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_stage": prefix,
            }, rescue_path)
            vol.commit()
            print(f"\n23h boundary. Rescue checkpoint at step {step}.")
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # --- Final ---
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = f"{ckpt_dir}/osrt_v4_{prefix}_final.pt"
    torch.save({
        "model_state_dict": inner.state_dict(),
        "training_stage": prefix,
        "total_steps": sft_cfg.total_steps,
    }, final_path)
    vol.commit()
    elapsed_total = time.time() - start_time
    print(f"\n{prefix.upper()} complete. {step:,} steps in {elapsed_total / 3600:.1f}h")
    print(f"Final model: {final_path}")
    if use_wandb:
        wandb.finish()
