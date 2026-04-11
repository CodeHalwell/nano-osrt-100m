"""Pre-training loop for NanoOSRT v4.

Progressive curriculum with increasing seq_len:
  Phase 1 (Foundation):  2048, text + code syntax
  Phase 2 (Knowledge):   4096, dense knowledge + code patterns
  Phase 3 (Instruction): 8192, chat + code instruction format

Handles:
- Phase transitions with automatic seq_len changes
- MoE load balancing auxiliary loss
- Loop RMS and adapter monitoring
- torch.compile for performance
- Checkpoint resume with 23h Modal safety
"""

import glob
import math
import os
import sys
import time

import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.v4_config import NanoOSRTv4Config
from nano_osrt.v4_data import make_v4_loader
from nano_osrt.v4_model import NanoOSRTv4ForCausalLM
from nano_osrt.v4_train_config import V4PretrainConfig


def get_lr(step: int, cfg: V4PretrainConfig) -> float:
    """Cosine LR with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def get_phase(step: int, cfg: V4PretrainConfig) -> tuple[str, dict]:
    """Get current phase config for a given step.

    Returns:
        (phase_name, phase_config_dict)
    """
    for name, p in cfg.phases.items():
        if p["start"] <= step < p["end"]:
            return name, p
    # Default to last phase
    last_name = list(cfg.phases.keys())[-1]
    return last_name, cfg.phases[last_name]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Save a training checkpoint."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    # For HF models, get the underlying model
    if hasattr(inner, "model"):
        state_dict = inner.state_dict()
    else:
        state_dict = inner.state_dict()
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "training_stage": "pretrain",
        },
        path,
    )
    print(f"  -> Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    """Load from checkpoint. Returns step to resume from."""
    if not os.path.exists(path):
        return 0
    print(f"Resuming from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    missing, unexpected = inner.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        print("  Skipping optimizer state (parameter count changed)")
    else:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, RuntimeError) as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    start_step = ckpt["step"] + 1
    print(f"  Resumed at step {start_step}")
    return start_step


@torch.no_grad()
def run_eval(
    model: nn.Module,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
    real_vocab_size: int,
) -> dict:
    """Run evaluation on held-out data. Returns dict of metrics.

    Uses a fixed-seed FineWeb-Edu stream so the same data is evaluated
    every time, never overlapping with training data (different seed).
    """
    model.eval()

    eval_loader = make_v4_loader(
        dataset_configs=[
            {"name": "fineweb-edu-eval", "hf_id": "HuggingFaceFW/fineweb-edu", "weight": 1.0},
        ],
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        step_num=999999,  # fixed seed, never matches training seeds
    )
    eval_iter = iter(eval_loader)

    total_loss = 0.0
    total_tokens = 0

    for _ in range(eval_steps):
        try:
            input_ids, labels = next(eval_iter)
        except StopIteration:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)

        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    model.train()

    eval_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(eval_loss, 20.0))  # cap to avoid overflow

    return {"eval/loss": eval_loss, "eval/perplexity": perplexity, "eval/tokens": total_tokens}


def run_v4_training(model_config: NanoOSRTv4Config, train_cfg: V4PretrainConfig, vol, tokenizer_name: str) -> None:
    """Execute the v4 pre-training loop."""
    device = torch.device("cuda")

    print("=" * 60)
    print("NanoOSRT v4 — Recursive MoE Pre-training")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = NanoOSRTv4ForCausalLM(model_config).to(device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Physical parameters : {total_params:>12,}")
    print(f"Blocks              : {model_config.num_blocks}")
    print(f"Recursive loops     : {model_config.recursive_loops}")
    print(f"Effective layers    : {model_config.num_blocks * model_config.recursive_loops}")
    print(f"Experts             : {model_config.num_experts} ({model_config.num_shared_experts} shared + {model_config.num_routed_experts} routed, top-{model_config.top_k_experts})")
    print(f"Hidden dim          : {model_config.dim}")
    print(f"Peak LR             : {train_cfg.peak_lr}")
    print(f"Optimizer           : {train_cfg.optimizer_name}")
    print(f"Total steps         : {train_cfg.total_steps}")
    print()

    print("Compiling model with torch.compile...")
    compile_start = time.time()
    model = torch.compile(model)
    print(f"Model compile done in {time.time() - compile_start:.1f}s")

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    use_wandb = train_cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": train_cfg.wandb_project,
            "name": train_cfg.wandb_run_name,
            "config": {
                "stage": "pretrain",
                "total_params": total_params,
                "architecture": "recursive_moe",
                "num_blocks": model_config.num_blocks,
                "recursive_loops": model_config.recursive_loops,
                "num_experts": model_config.num_experts,
                "top_k": model_config.top_k_experts,
                "peak_lr": train_cfg.peak_lr,
                "optimizer": train_cfg.optimizer_name,
                "total_steps": train_cfg.total_steps,
            },
        }
        if train_cfg.wandb_run_id:
            wandb_kwargs["id"] = train_cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # ------------------------------------------------------------------
    # Optimizer — differential weight decay (router params get wd=0)
    # ------------------------------------------------------------------
    # Router weights are sensitive to weight decay — too much decay pushes
    # them toward zero, causing uniform routing (defeating MoE purpose).
    # Separate into router params (wd=0) and everything else (full wd).
    inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    router_params = []
    other_params = []
    for name, param in inner_model.named_parameters():
        if not param.requires_grad:
            continue
        if "router" in name or "loop_embeddings" in name:
            router_params.append(param)
        else:
            other_params.append(param)

    print(f"Param groups: {len(other_params)} standard, {len(router_params)} router (wd=0)")

    if train_cfg.optimizer_name.lower() == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(
            [
                {"params": other_params, "weight_decay": train_cfg.weight_decay},
                {"params": router_params, "weight_decay": 0.0},
            ],
            lr=train_cfg.peak_lr,
        )
        print(f"Using Lion (wd={train_cfg.weight_decay}, router_wd=0.0)")
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": train_cfg.weight_decay},
                {"params": router_params, "weight_decay": 0.0},
            ],
            lr=train_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        print(f"Using AdamW (wd={train_cfg.weight_decay}, router_wd=0.0)")

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    ckpt_dir = "/vol/checkpoints/v4"
    os.makedirs(ckpt_dir, exist_ok=True)
    rescue_path = f"{ckpt_dir}/osrt_v4_rescue.pt"
    best_ckpt = rescue_path
    best_step = -1

    for f in glob.glob(f"{ckpt_dir}/osrt_v4_step_*.pt"):
        try:
            s = int(f.rsplit("_", 1)[1].split(".")[0])
            if s > best_step:
                best_step = s
                best_ckpt = f
        except (ValueError, IndexError):
            continue

    start_step = 0
    if best_step > 0:
        print(f"Found checkpoint at step {best_step}: {best_ckpt}")
        start_step = load_checkpoint(model, optimizer, best_ckpt, device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_time = time.time()
    step = start_step
    current_phase = None
    current_loader = None
    loader_iter = None
    current_seq_len = 2048
    current_batch_size = train_cfg.batch_size

    while step < train_cfg.total_steps:
        phase_name, phase_cfg = get_phase(step, train_cfg)

        # Phase transition — reload data with new seq_len / datasets
        if phase_name != current_phase:
            current_phase = phase_name
            current_seq_len = phase_cfg["seq_len"]
            grad_accum = phase_cfg.get("grad_accum_steps", train_cfg.grad_accum_steps)
            current_batch_size = phase_cfg.get("batch_size", train_cfg.batch_size)

            print(f"\n>>> Phase: {current_phase} | seq_len: {current_seq_len} | "
                  f"batch: {current_batch_size} | accum: {grad_accum} | Step: {step}")
            print(f"    Datasets: {[d['name'] for d in phase_cfg['datasets']]}")

            # Enable gradient checkpointing for long sequences
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            if hasattr(inner, "model") and hasattr(inner.model, "gradient_checkpointing"):
                if current_seq_len >= 4096:
                    inner.model.gradient_checkpointing = True
                    print("    Gradient checkpointing: ENABLED")
                else:
                    inner.model.gradient_checkpointing = False

            if current_loader is not None:
                del current_loader
            load_t = time.time()
            current_loader = make_v4_loader(
                phase_cfg["datasets"],
                current_seq_len,
                tokenizer_name,
                current_batch_size,
                step,
            )
            loader_iter = iter(current_loader)
            print(f"    DataLoader ready in {time.time() - load_t:.1f}s")
        else:
            grad_accum = phase_cfg.get("grad_accum_steps", train_cfg.grad_accum_steps)
            current_batch_size = phase_cfg.get("batch_size", train_cfg.batch_size)

        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Anneal router noise linearly from init to final over
        #    router_noise_anneal_steps (default 5000). Updates the
        #    per-MoE noise_std buffer in-place so torch.compile sees a
        #    fresh tensor each step without recompiling.
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        base = inner.model if hasattr(inner, "model") else inner
        ns_init = model_config.router_noise_std_init
        ns_final = model_config.router_noise_std_final
        ns_anneal = max(model_config.router_noise_anneal_steps, 1)
        noise_progress = min(step / ns_anneal, 1.0)
        current_noise = ns_init + (ns_final - ns_init) * noise_progress
        for blk in base.blocks:
            blk.moe.noise_std.fill_(current_noise)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.tensor(0.0, device=device)

        if step == start_step:
            print("Fetching first batch...")
            batch_t = time.time()

        for micro in range(grad_accum):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                _, p_cfg = get_phase(step, train_cfg)
                if current_loader is not None:
                    del current_loader
                current_loader = make_v4_loader(
                    p_cfg["datasets"],
                    p_cfg["seq_len"],
                    tokenizer_name,
                    p_cfg.get("batch_size", train_cfg.batch_size),
                    step,
                )
                loader_iter = iter(current_loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if step == start_step and micro == 0:
                print(f"First batch fetched in {time.time() - batch_t:.1f}s")
                print("Running first forward pass (torch.compile tracing)...")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum

            loss.backward()
            accum_loss += outputs.loss.detach() / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        # --- Logging ---
        should_log = (
            step % train_cfg.log_interval == 0
            or step == 0
            or (step < 100 and step % 10 == 0)
        )
        if should_log:
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

            eff_batch = current_batch_size * grad_accum
            tok_per_sec = eff_batch * current_seq_len / max(elapsed / max(step - start_step, 1), 1e-8)

            # ── MoE diagnostics (cheap, high-value for debugging) ──
            # Pull gate values, aux losses, router entropy, expert usage
            # fractions, and (most importantly) assignment entropy from the
            # compiled model via the unwrapped inner module. Values are per
            # physical block and per loop. Key summary numbers are also
            # printed to stdout so we catch routing collapse in real time
            # without needing the W&B dashboard.
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            moe_metrics = {}
            # Collected for stdout summary line
            moe_summary = None
            try:
                base = inner.model if hasattr(inner, "model") else inner
                block_lb_losses = []
                block_z_losses = []
                prob_entropies = []
                assign_entropies = []
                max_fracs = []
                min_fracs = []
                dense_gates = []
                moe_gates = []
                for bi, blk in enumerate(base.blocks):
                    dg = blk.dense_gate.item()
                    mg = blk.moe_gate.item()
                    dense_gates.append(dg)
                    moe_gates.append(mg)
                    moe_metrics[f"moe/dense_gate_b{bi}"] = dg
                    moe_metrics[f"moe/moe_gate_b{bi}"] = mg
                    if blk.moe.load_balance_loss is not None:
                        block_lb_losses.append(blk.moe.load_balance_loss.item())
                    if blk.moe.z_loss is not None:
                        block_z_losses.append(blk.moe.z_loss.item())

                    # Per-loop router-prob entropy (can be misleading) +
                    # per-loop assignment entropy (the real metric).
                    for li, ent in enumerate(blk.moe.last_router_entropy):
                        moe_metrics[f"moe/prob_entropy_b{bi}_l{li}"] = ent
                        prob_entropies.append(ent)
                    for li, ent in enumerate(blk.moe.last_assignment_entropy):
                        moe_metrics[f"moe/assign_entropy_b{bi}_l{li}"] = ent
                        assign_entropies.append(ent)

                    # Per-loop expert usage: track max and min fraction
                    # (tight gap = collapsed routing, wide gap = specialisation)
                    for li, fracs in enumerate(blk.moe.last_expert_fraction):
                        if fracs:
                            mx = max(fracs)
                            mn = min(fracs)
                            moe_metrics[f"moe/expert_max_b{bi}_l{li}"] = mx
                            moe_metrics[f"moe/expert_min_b{bi}_l{li}"] = mn
                            max_fracs.append(mx)
                            min_fracs.append(mn)

                def _mean(xs):
                    return sum(xs) / len(xs) if xs else 0.0

                if block_lb_losses:
                    moe_metrics["moe/load_balance_loss_mean"] = _mean(block_lb_losses)
                if block_z_losses:
                    moe_metrics["moe/z_loss_mean"] = _mean(block_z_losses)
                if prob_entropies:
                    moe_metrics["moe/prob_entropy_mean"] = _mean(prob_entropies)
                if assign_entropies:
                    moe_metrics["moe/assign_entropy_mean"] = _mean(assign_entropies)
                if max_fracs:
                    moe_metrics["moe/expert_max_mean"] = _mean(max_fracs)
                if min_fracs:
                    moe_metrics["moe/expert_min_mean"] = _mean(min_fracs)

                moe_summary = {
                    "assign_entropy": _mean(assign_entropies),
                    "expert_max": _mean(max_fracs),
                    "expert_min": _mean(min_fracs),
                    "dense_gate": _mean(dense_gates),
                    "moe_gate": _mean(moe_gates),
                    "lb_loss": _mean(block_lb_losses),
                    "z_loss": _mean(block_z_losses),
                }

                # ── Clean-routing diagnostic ──
                # The noisy numbers above include router_noise_std
                # perturbation, so they optimistically bias assign_H
                # upward. Zero the noise buffers, run ONE extra forward
                # pass on the same data (no gradients), read the
                # telemetry, then restore the noise schedule. This
                # gives us the "true" learned routing every log step.
                clean_assign = None
                clean_max = None
                try:
                    saved_noise = [blk.moe.noise_std.clone() for blk in base.blocks]
                    for blk in base.blocks:
                        blk.moe.noise_std.fill_(0.0)
                    with torch.no_grad():
                        _ = model(input_ids)
                    # telemetry was refreshed by the clean forward
                    clean_assign_entropies = []
                    clean_max_fracs = []
                    clean_min_fracs = []
                    for blk in base.blocks:
                        clean_assign_entropies.extend(blk.moe.last_assignment_entropy)
                        for fracs in blk.moe.last_expert_fraction:
                            if fracs:
                                clean_max_fracs.append(max(fracs))
                                clean_min_fracs.append(min(fracs))
                    clean_assign = _mean(clean_assign_entropies)
                    clean_max = _mean(clean_max_fracs)
                    clean_min = _mean(clean_min_fracs)
                    moe_metrics["moe/clean_assign_entropy_mean"] = clean_assign
                    moe_metrics["moe/clean_expert_max_mean"] = clean_max
                    moe_metrics["moe/clean_expert_min_mean"] = clean_min
                    moe_summary["clean_assign_entropy"] = clean_assign
                    moe_summary["clean_expert_max"] = clean_max
                    moe_summary["clean_expert_min"] = clean_min
                finally:
                    # ALWAYS restore noise schedule, even if the
                    # diagnostic forward raised.
                    for blk, ns in zip(base.blocks, saved_noise):
                        blk.moe.noise_std.copy_(ns)
                    # Re-run one noisy forward to repopulate telemetry
                    # for the next log cycle — otherwise the stored
                    # last_* lists would reflect the clean forward and
                    # skew the noisy training metrics next time.
                    with torch.no_grad():
                        _ = model(input_ids)
            except AttributeError:
                pass  # model not fully set up yet (first step)

            print(
                f"step {step:>7d}/{train_cfg.total_steps} | "
                f"loss {accum_loss.item():.4f} | lr {lr:.2e} | "
                f"vram {vram_gb:.1f}GB | tok/s {tok_per_sec:,.0f} | "
                f"phase {current_phase} | seq_len {current_seq_len}",
                flush=True,
            )

            # Critical MoE health line — prints to stdout so we see
            # routing collapse immediately in the live Modal log instead
            # of having to check W&B. Key number is the CLEAN assign_H
            # (no router noise), which reflects what the router has
            # actually learned:
            #   ~2.40 (ln 11) = uniform, healthy
            #   ~2.00        = mildly concentrated
            #   ~1.00        = collapsing
            #   ~0.69 (ln 2) = full top-2 collapse
            if moe_summary is not None:
                clean_H = moe_summary.get("clean_assign_entropy", float("nan"))
                clean_max = moe_summary.get("clean_expert_max", float("nan"))
                clean_min = moe_summary.get("clean_expert_min", float("nan"))
                noise_std_now = current_noise
                print(
                    f"           moe: clean assign_H={clean_H:.3f} "
                    f"max={clean_max:.3f} "
                    f"min={clean_min:.3f} | "
                    f"noisy assign_H={moe_summary['assign_entropy']:.3f} "
                    f"(noise_std={noise_std_now:.3f}) | "
                    f"gates d={moe_summary['dense_gate']:.4f} "
                    f"m={moe_summary['moe_gate']:.4f} | "
                    f"lb={moe_summary['lb_loss']:.3f} "
                    f"z={moe_summary['z_loss']:.3f}",
                    flush=True,
                )

                # ── Early-stop guard ──
                # Criteria from the review: stop immediately if routing
                # has collapsed in a way we can't recover from. We apply
                # these to the CLEAN metrics (not the noisy training
                # ones) and only after step 50 so random init noise
                # doesn't trigger false alarms.
                if step >= 50 and clean_H == clean_H:  # not NaN
                    collapse_reasons = []
                    if clean_H < 1.5:
                        collapse_reasons.append(
                            f"clean assign_H {clean_H:.3f} < 1.5 (severe collapse)"
                        )
                    if clean_max > 0.35:
                        collapse_reasons.append(
                            f"clean expert_max {clean_max:.3f} > 0.35 (concentrated)"
                        )
                    if clean_min < 1e-5:
                        collapse_reasons.append(
                            f"clean expert_min {clean_min:.5f} ~ 0 (dead experts)"
                        )
                    if collapse_reasons:
                        print(
                            "\n!!! MoE ROUTING COLLAPSE DETECTED — aborting run:",
                            flush=True,
                        )
                        for r in collapse_reasons:
                            print(f"    - {r}", flush=True)
                        print(
                            "Saving rescue checkpoint for post-mortem and exiting.",
                            flush=True,
                        )
                        save_checkpoint(model, optimizer, step, rescue_path)
                        vol.commit()
                        if use_wandb:
                            wandb.log({"train/aborted": 1.0}, step=step)
                            wandb.finish()
                        return

            if use_wandb:
                log_dict = {
                    "train/loss": accum_loss.item(),
                    "train/lr": lr,
                    "train/vram_gb": vram_gb,
                    "train/tok_per_sec": tok_per_sec,
                    "train/phase": current_phase,
                    "train/seq_len": current_seq_len,
                }
                log_dict.update(moe_metrics)
                wandb.log(log_dict, step=step)

        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()

        # --- Eval on held-out data ---
        # Use phase-specific batch_size to avoid OOM at seq_len 4096 / 8192.
        if step > 0 and step % train_cfg.eval_interval == 0:
            eval_metrics = run_eval(
                model, tokenizer_name, current_seq_len,
                current_batch_size, train_cfg.eval_steps,
                device, model_config.real_vocab_size,
            )
            print(
                f"  EVAL step {step} | "
                f"loss {eval_metrics['eval/loss']:.4f} | "
                f"ppl {eval_metrics['eval/perplexity']:.1f}"
            )
            if use_wandb:
                wandb.log(eval_metrics, step=step)

        # --- Checkpoints ---
        if step > 0 and step % train_cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v4_step_{step}.pt"
            save_checkpoint(model, optimizer, step, path)
            vol.commit()

        # --- 23h Modal safety ---
        if time.time() - start_time > 82_800:
            save_checkpoint(model, optimizer, step, rescue_path)
            vol.commit()
            print(f"\n23h boundary. Rescue checkpoint at step {step}.")
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # --- Final ---
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = f"{ckpt_dir}/osrt_v4_final.pt"
    torch.save(
        {
            "model_state_dict": inner.state_dict(),
            "training_stage": "pretrain",
            "total_steps": train_cfg.total_steps,
        },
        final_path,
    )
    vol.commit()
    elapsed_total = time.time() - start_time
    print(f"\nPre-training complete. {step:,} steps in {elapsed_total / 3600:.1f}h")
    print(f"Final model: {final_path}")
    if use_wandb:
        wandb.finish()
