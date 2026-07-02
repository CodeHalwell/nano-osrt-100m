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
            {
                "name": "fineweb-edu-eval",
                "hf_id": "HuggingFaceFW/fineweb-edu",
                "weight": 1.0,
            },
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

    return {
        "eval/loss": eval_loss,
        "eval/perplexity": perplexity,
        "eval/tokens": total_tokens,
    }


def run_v4_training(
    model_config: NanoOSRTv4Config,
    train_cfg: V4PretrainConfig,
    vol,
    tokenizer_name: str,
) -> None:
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
    print(
        f"Effective layers    : {model_config.num_blocks * model_config.recursive_loops}"
    )
    print(
        f"Experts             : {model_config.num_experts} ({model_config.num_shared_experts} shared + {model_config.num_routed_experts} routed, top-{model_config.top_k_experts})"
    )
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

    print(
        f"Param groups: {len(other_params)} standard, {len(router_params)} router (wd=0)"
    )

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

            print(
                f"\n>>> Phase: {current_phase} | seq_len: {current_seq_len} | "
                f"batch: {current_batch_size} | accum: {grad_accum} | Step: {step}"
            )
            print(f"    Datasets: {[d['name'] for d in phase_cfg['datasets']]}")

            # (Per-step logic below owns gradient_checkpointing toggling —
            # it accounts for both seq_len and routing phase. No need to
            # set it here.)
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

        # ── Routing schedule ──
        # Three phases:
        #   [0, soft_warmup_steps):     mode 0 (soft_all),  alpha=0
        #   [soft_warmup, soft+blend):  mode 1 (blend),     alpha=linear 0→1
        #   [soft+blend, ∞):            mode 2 (hard_topk), alpha=1
        # Mode is a Python int → changing it triggers exactly two
        # torch.compile recompilations per run, which is fine. Alpha is
        # a scalar buffer → in-place update, no recompile.
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        base = inner.model if hasattr(inner, "model") else inner
        soft_steps = max(model_config.soft_warmup_steps, 0)
        blend_steps = max(model_config.blend_anneal_steps, 0)
        if step < soft_steps:
            target_mode = 0
            current_alpha = 0.0
        elif step < soft_steps + blend_steps:
            target_mode = 1
            current_alpha = (
                (step - soft_steps) / blend_steps if blend_steps > 0 else 1.0
            )
        else:
            target_mode = 2
            current_alpha = 1.0
        routing_phase = {0: "soft", 1: "blend", 2: "hard"}[target_mode]
        for blk in base.blocks:
            blk.moe.routing_mode = target_mode
            blk.moe.routing_alpha.fill_(current_alpha)

        # Soft / blend phases run every routed expert on every token, which
        # is ~num_routed× the activation memory of hard top-k. On an 80GB
        # A100 the uncheckpointed soft pass OOMs (first run crashed at step
        # 0 with 74GB allocated). Force gradient checkpointing on whenever
        # routing is not fully hard; also keep it on for long seq_len phases
        # as before. Toggling a bool triggers at most 2 torch.compile
        # recompiles per run (soft→blend→hard boundaries).
        need_ckpt = (target_mode != 2) or (current_seq_len >= 4096)
        if (
            hasattr(base, "gradient_checkpointing")
            and base.gradient_checkpointing != need_ckpt
        ):
            base.gradient_checkpointing = need_ckpt

        # Legacy router noise anneal (defaults to 0.0 → 0.0). Keep the
        # buffer update here so setting non-zero values in config still
        # works if someone wants extra jitter during hard phase.
        ns_init = model_config.router_noise_std_init
        ns_final = model_config.router_noise_std_final
        ns_anneal = max(model_config.router_noise_anneal_steps, 1)
        noise_progress = min(step / ns_anneal, 1.0)
        current_noise = ns_init + (ns_final - ns_init) * noise_progress
        for blk in base.blocks:
            blk.moe.noise_std.fill_(current_noise)

        # Gumbel-top-k anneal. Linearly decay gumbel_tau from init to
        # final over router_gumbel_anneal_steps. This is the anti
        # deterministic-winner-lock-in pressure: while tau > 0, top-k
        # selection is stochastic and no expert can win by tiny
        # logit-ordering tie-breaking alone. The anneal window (default
        # 10K) is intentionally long so the router has time to learn
        # robust preferences before determinism returns.
        gt_init = model_config.router_gumbel_tau_init
        gt_final = model_config.router_gumbel_tau_final
        gt_anneal = max(model_config.router_gumbel_anneal_steps, 1)
        gumbel_progress = min(step / gt_anneal, 1.0)
        current_gumbel_tau = gt_init + (gt_final - gt_init) * gumbel_progress
        for blk in base.blocks:
            blk.moe.gumbel_tau.fill_(current_gumbel_tau)

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

        # ── Balance-bias controller: once-per-step update ──
        # Each MoELayer has accumulated clean-biased assignment counts
        # across all 6 recursive-loop calls (and any grad-accum
        # micro-batches) during the just-completed forward passes. Now
        # that the optimizer has stepped, compute one bias update per
        # block from the aggregated counts and reset the accumulator.
        for blk in base.blocks:
            blk.moe.apply_balance_update()

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
            tok_per_sec = (
                eff_batch
                * current_seq_len
                / max(elapsed / max(step - start_step, 1), 1e-8)
            )

            # ── MoE diagnostics (cheap, high-value for debugging) ──
            # Pull gate values, aux losses, router entropy, expert usage
            # fractions, and (most importantly) assignment entropy from the
            # compiled model via the unwrapped inner module. Values are per
            # physical block and per loop. Key summary numbers are also
            # printed to stdout so we catch routing collapse in real time
            # without needing the W&B dashboard.
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            moe_metrics = {
                "moe/routing_mode": target_mode,
                "moe/routing_alpha": current_alpha,
                "moe/gumbel_tau": current_gumbel_tau,
            }
            # Collected for stdout summary line
            moe_summary = None
            try:
                base = inner.model if hasattr(inner, "model") else inner
                block_imp_losses = []
                block_bias_losses = []
                block_z_losses = []
                prob_entropies = []
                assign_entropies = []
                clean_assign_entropies = []
                raw_assign_entropies = []
                max_fracs = []
                min_fracs = []
                clean_max_fracs = []
                clean_min_fracs = []
                raw_max_fracs = []
                raw_min_fracs = []
                bias_abs_maxes = []
                dense_gates = []
                moe_gates = []
                for bi, blk in enumerate(base.blocks):
                    dg = blk.dense_gate.item()
                    mg = blk.moe_gate.item()
                    dense_gates.append(dg)
                    moe_gates.append(mg)
                    moe_metrics[f"moe/dense_gate_b{bi}"] = dg
                    moe_metrics[f"moe/moe_gate_b{bi}"] = mg
                    if blk.moe.importance_loss is not None:
                        block_imp_losses.append(blk.moe.importance_loss.item())
                    if blk.moe.logit_bias_loss is not None:
                        block_bias_losses.append(blk.moe.logit_bias_loss.item())
                    if blk.moe.z_loss is not None:
                        block_z_losses.append(blk.moe.z_loss.item())

                    # Per-loop softmax entropy (soft-phase primary
                    # signal) + per-loop assignment entropy (hard-phase
                    # primary signal; during soft phase this holds
                    # importance entropy as a proxy).
                    for li, ent in enumerate(blk.moe.last_router_entropy):
                        moe_metrics[f"moe/prob_entropy_b{bi}_l{li}"] = ent
                        prob_entropies.append(ent)
                    for li, ent in enumerate(blk.moe.last_assignment_entropy):
                        moe_metrics[f"moe/assign_entropy_b{bi}_l{li}"] = ent
                        assign_entropies.append(ent)

                    # Clean BIASED assignment entropy (inference path:
                    # router_logits + router_balance_bias, no noise) —
                    # the decisive health signal.
                    for li, ent in enumerate(blk.moe.last_clean_assignment_entropy):
                        moe_metrics[f"moe/clean_assign_entropy_b{bi}_l{li}"] = ent
                        clean_assign_entropies.append(ent)

                    # Raw assignment entropy (router_logits only, no bias,
                    # no noise) — diagnostic. May be collapsed even
                    # when clean_biased is healthy; that's fine if
                    # the deployed routing path includes the bias.
                    for li, ent in enumerate(blk.moe.last_raw_assignment_entropy):
                        moe_metrics[f"moe/raw_assign_entropy_b{bi}_l{li}"] = ent
                        raw_assign_entropies.append(ent)

                    # Per-block bias magnitude
                    if hasattr(blk.moe, "router_balance_bias"):
                        bam = blk.moe.router_balance_bias.abs().max().item()
                        bias_abs_maxes.append(bam)
                        moe_metrics[f"moe/bias_abs_max_b{bi}"] = bam

                    # Capacity-capped telemetry
                    for li, ofr in enumerate(blk.moe.last_overflow_rate):
                        moe_metrics[f"moe/overflow_rate_b{bi}_l{li}"] = ofr
                    for li, apt in enumerate(blk.moe.last_assigned_per_token):
                        moe_metrics[f"moe/assigned_per_token_b{bi}_l{li}"] = apt

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

                    # Clean BIASED per-loop expert usage
                    for li, fracs in enumerate(blk.moe.last_clean_expert_fraction):
                        if fracs:
                            mx = max(fracs)
                            mn = min(fracs)
                            moe_metrics[f"moe/clean_expert_max_b{bi}_l{li}"] = mx
                            moe_metrics[f"moe/clean_expert_min_b{bi}_l{li}"] = mn
                            clean_max_fracs.append(mx)
                            clean_min_fracs.append(mn)

                    # RAW (unbiased) per-loop expert usage — diagnostic
                    for li, fracs in enumerate(blk.moe.last_raw_expert_fraction):
                        if fracs:
                            mx = max(fracs)
                            mn = min(fracs)
                            moe_metrics[f"moe/raw_expert_max_b{bi}_l{li}"] = mx
                            moe_metrics[f"moe/raw_expert_min_b{bi}_l{li}"] = mn
                            raw_max_fracs.append(mx)
                            raw_min_fracs.append(mn)

                def _mean(xs):
                    return sum(xs) / len(xs) if xs else 0.0

                if block_imp_losses:
                    moe_metrics["moe/importance_loss_mean"] = _mean(block_imp_losses)
                if block_bias_losses:
                    moe_metrics["moe/logit_bias_loss_mean"] = _mean(block_bias_losses)
                if block_z_losses:
                    moe_metrics["moe/z_loss_mean"] = _mean(block_z_losses)
                if prob_entropies:
                    moe_metrics["moe/prob_entropy_mean"] = _mean(prob_entropies)
                if assign_entropies:
                    moe_metrics["moe/assign_entropy_mean"] = _mean(assign_entropies)
                if clean_assign_entropies:
                    moe_metrics["moe/clean_assign_entropy_mean"] = _mean(
                        clean_assign_entropies
                    )
                if raw_assign_entropies:
                    moe_metrics["moe/raw_assign_entropy_mean"] = _mean(
                        raw_assign_entropies
                    )
                if max_fracs:
                    moe_metrics["moe/expert_max_mean"] = _mean(max_fracs)
                if min_fracs:
                    moe_metrics["moe/expert_min_mean"] = _mean(min_fracs)
                if clean_max_fracs:
                    moe_metrics["moe/clean_expert_max_mean"] = _mean(clean_max_fracs)
                if clean_min_fracs:
                    moe_metrics["moe/clean_expert_min_mean"] = _mean(clean_min_fracs)
                if raw_max_fracs:
                    moe_metrics["moe/raw_expert_max_mean"] = _mean(raw_max_fracs)
                if raw_min_fracs:
                    moe_metrics["moe/raw_expert_min_mean"] = _mean(raw_min_fracs)
                if bias_abs_maxes:
                    moe_metrics["moe/bias_abs_max_mean"] = _mean(bias_abs_maxes)

                moe_summary = {
                    "assign_entropy": _mean(assign_entropies),
                    "clean_assign_entropy": _mean(clean_assign_entropies),
                    "raw_assign_entropy": _mean(raw_assign_entropies),
                    "prob_entropy": _mean(prob_entropies),
                    "expert_max": _mean(max_fracs),
                    "expert_min": _mean(min_fracs),
                    "clean_expert_max": _mean(clean_max_fracs),
                    "clean_expert_min": _mean(clean_min_fracs),
                    "raw_expert_max": _mean(raw_max_fracs),
                    "raw_expert_min": _mean(raw_min_fracs),
                    "bias_abs_max": _mean(bias_abs_maxes),
                    "dense_gate": _mean(dense_gates),
                    "moe_gate": _mean(moe_gates),
                    "imp_loss": _mean(block_imp_losses),
                    "bias_loss": _mean(block_bias_losses),
                    "z_loss": _mean(block_z_losses),
                }

                # Aggregate capacity-capped telemetry
                all_overflow = []
                all_assigned = []
                all_rank = []
                for blk in base.blocks:
                    all_overflow.extend(blk.moe.last_overflow_rate)
                    all_assigned.extend(blk.moe.last_assigned_per_token)
                    all_rank.extend(blk.moe.last_candidate_rank_mean)
                if all_overflow:
                    moe_metrics["moe/overflow_rate_mean"] = _mean(all_overflow)
                    moe_metrics["moe/assigned_per_token_mean"] = _mean(all_assigned)
                    moe_metrics["moe/candidate_rank_mean"] = _mean(all_rank)
                    moe_summary["overflow_rate"] = _mean(all_overflow)
                    moe_summary["assigned_per_token"] = _mean(all_assigned)
                    moe_summary["candidate_rank"] = _mean(all_rank)
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
            # routing collapse immediately in the live Modal log. Which
            # metric matters depends on the routing phase:
            #   soft phase  → prob_H (softmax entropy) is primary;
            #                 assign_H holds importance entropy (proxy)
            #   blend phase → both matter; assign_H is from top-k
            #   hard phase  → assign_H is primary
            if moe_summary is not None:
                assign_H = moe_summary["assign_entropy"]
                clean_H = moe_summary["clean_assign_entropy"]
                raw_H = moe_summary["raw_assign_entropy"]
                prob_H = moe_summary["prob_entropy"]
                clean_max = moe_summary["clean_expert_max"]
                clean_min = moe_summary["clean_expert_min"]
                raw_max = moe_summary["raw_expert_max"]
                raw_min = moe_summary["raw_expert_min"]
                bias_mag = moe_summary["bias_abs_max"]
                overflow = moe_summary.get("overflow_rate", 0.0)
                assigned_tok = moe_summary.get("assigned_per_token", 0.0)
                cand_rank = moe_summary.get("candidate_rank", 0.0)
                print(
                    f"           moe[{routing_phase}]: "
                    f"capped H={clean_H:.3f} max={clean_max:.3f} min={clean_min:.3f} | "
                    f"raw H={raw_H:.3f} max={raw_max:.3f} | "
                    f"overflow={overflow:.4f} assigned/tok={assigned_tok:.2f} rank={cand_rank:.2f} | "
                    f"gates d={moe_summary['dense_gate']:.4f} "
                    f"m={moe_summary['moe_gate']:.4f} | "
                    f"imp={moe_summary['imp_loss']:.3f}",
                    flush=True,
                )

                # ── Phase-aware early-stop guard ──
                # Judge on CLEAN metrics (deterministic top-k), not the
                # noisy training-time ones. Gumbel noise can keep the
                # noisy histogram healthy-looking while the clean one
                # collapses; the clean is the real inference signal.
                #
                # Only enforce once we've reached hard phase AND at
                # least 50 steps past the blend→hard boundary, so the
                # router has some post-transition adjustment window
                # before we start auto-aborting.
                hard_start = soft_steps + blend_steps
                if (
                    target_mode == 2
                    and step >= hard_start + 50
                    and clean_H == clean_H  # not NaN
                ):
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
                model,
                tokenizer_name,
                current_seq_len,
                current_batch_size,
                train_cfg.eval_steps,
                device,
                model_config.real_vocab_size,
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
