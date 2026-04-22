"""SFT training loop for NanoOSRT.

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

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.config import NanoOSRTConfig
from nano_osrt.model import NanoOSRTForCausalLM
from nano_osrt.sft_data import IGNORE_INDEX, make_sft_loader


def get_sft_lr(
    step: int, total_steps: int, warmup: int, peak: float, minimum: float,
) -> float:
    """Cosine LR with linear warmup."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def run_sft(model_config: NanoOSRTConfig, sft_cfg, vol, tokenizer) -> None:
    """Execute the v5 SFT training loop."""
    device = torch.device("cuda")

    print("=" * 60)
    print("NanoOSRT — SFT Training (Balanced)")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = NanoOSRTForCausalLM(model_config).to(device=device)
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

    # Load pretrained weights — SFT MUST start from a real pretrained
    # checkpoint. Running SFT on a randomly-initialised model would waste
    # compute and silently produce a garbage model.
    ckpt_path = sft_cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"SFT refuses to start: pretrained checkpoint not found at {ckpt_path}. "
            "Run pretrain first (modal run app_v4.py --stage pretrain)."
        )

    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  MISSING keys ({len(missing)}): sample {missing[:3]}")
    if unexpected:
        print(f"  UNEXPECTED keys ({len(unexpected)}): sample {unexpected[:3]}")
    if not missing and not unexpected:
        print("  Clean load: all keys matched.")
    else:
        print("  Partial load: review the keys above if this is unexpected.")

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
        print(
            f"AdamW with differential LR (base={sft_cfg.peak_lr}, "
            f"hra={sft_cfg.hra_lr})"
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg.peak_lr,
                                       weight_decay=sft_cfg.weight_decay,
                                       betas=(0.9, 0.95), eps=1e-8)

    # ------------------------------------------------------------------
    # Resume.
    # Two resumable patterns (matches pretrain):
    #   osrt_v5_{prefix}_step_N.pt          — interval save
    #   osrt_v5_{prefix}_rescue_step_N.pt   — 23h boundary save
    # When steps tie, rescue wins (same "latest optimizer state" argument
    # as pretrain).
    # ------------------------------------------------------------------
    prefix = getattr(sft_cfg, "stage_prefix", "sft")
    ckpt_dir = "/vol/checkpoints/v5"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_step = -1
    best_ckpt: str | None = None
    for pattern in (
        f"{ckpt_dir}/osrt_v5_{prefix}_step_*.pt",
        f"{ckpt_dir}/osrt_v5_{prefix}_rescue_step_*.pt",
    ):
        for f in glob.glob(pattern):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if s > best_step or (s == best_step and "rescue" in f):
                best_step = s
                best_ckpt = f

    start_step = 0
    if best_step > 0 and best_ckpt is not None:
        print(f"Found {prefix} checkpoint at step {best_step}: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
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
    loader = make_sft_loader(
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
                loader = make_sft_loader(
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
            tok_util = (
                accum_trained_tokens / total_positions
                if total_positions > 0 else 0
            )

            print(
                f"step {step:>6d}/{sft_cfg.total_steps} | "
                f"loss {accum_loss.item():.4f} | lr {lr:.2e} | "
                f"vram {vram_gb:.1f}GB | tok_util {tok_util:.1%} | "
                f"elapsed {elapsed:.0f}s"
            )

            # --- MoE telemetry (condensed, v5 names) ---
            # v5 has no dense_gate (no dense FFN) — only moe_gate on the
            # routed branch. The MoE layer exposes clean_* metrics with
            # Gumbel noise stripped, which is what matters at SFT time
            # (noise schedule is inherited from pretrain but should have
            # annealed to ~0 by then).
            moe_metrics: dict = {}
            try:
                inner = model._orig_mod if hasattr(model, "_orig_mod") else model
                base = inner.model if hasattr(inner, "model") else inner

                def _mean(xs: list[float]) -> float:
                    return sum(xs) / len(xs) if xs else 0.0

                moe_gates: list[float] = []
                prob_ents: list[float] = []
                assign_ents: list[float] = []
                drop_rates: list[float] = []
                for bi, blk in enumerate(base.blocks):
                    mg = blk.moe_gate.item()
                    moe_gates.append(mg)
                    moe_metrics[f"moe/moe_gate_b{bi}"] = mg
                    prob_ents.extend(blk.moe.last_clean_per_token_entropy)
                    assign_ents.extend(blk.moe.last_clean_assignment_entropy)
                    drop_rates.extend(blk.moe.last_drop_rate)
                moe_metrics["moe/clean_per_token_entropy_mean"] = _mean(prob_ents)
                moe_metrics["moe/clean_assignment_entropy_mean"] = _mean(assign_ents)
                moe_metrics["moe/drop_rate_mean"] = _mean(drop_rates)
                moe_metrics["moe/moe_gate_mean"] = _mean(moe_gates)
                print(
                    f"           moe: pte={_mean(prob_ents):.3f} "
                    f"assn={_mean(assign_ents):.3f} "
                    f"drop={_mean(drop_rates):.4f} "
                    f"gate={_mean(moe_gates):.3f}"
                )
            except AttributeError:
                pass

            if use_wandb:
                log_dict = {
                    f"{prefix}/loss": accum_loss.item(),
                    f"{prefix}/lr": lr,
                    f"{prefix}/vram_gb": vram_gb,
                    f"{prefix}/token_utilization": tok_util,
                    f"{prefix}/trained_tokens": accum_trained_tokens,
                }
                log_dict.update(moe_metrics)
                wandb.log(log_dict, step=step)

        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()

        # --- Checkpoints ---
        if step > 0 and step % sft_cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v5_{prefix}_step_{step}.pt"
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
        # Include step number in rescue filename so the resume scanner
        # can rank it alongside numbered checkpoints. Without this the
        # scanner either misses rescue files entirely or can't break
        # ties against same-step interval saves.
        if time.time() - start_time > 82_800:
            rescue_path = (
                f"{ckpt_dir}/osrt_v5_{prefix}_rescue_step_{step}.pt"
            )
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_stage": prefix,
            }, rescue_path)
            vol.commit()
            print(
                f"\n23h boundary. Rescue checkpoint at step {step}: "
                f"{rescue_path}",
            )
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # --- Final ---
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = f"{ckpt_dir}/osrt_v5_{prefix}_final.pt"
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
