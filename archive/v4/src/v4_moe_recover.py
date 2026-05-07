"""MoE Recovery training for NanoOSRT v4.

Experiment: wake up the dead router by:
  1. Re-initialising all 11 routed experts from the shared expert + small noise
     (breaks symmetry so experts can diverge with different init)
  2. Freezing everything except router + routed experts + gates
  3. Removing importance/balance losses (they enforce uniformity)
  4. Loosening the capacity cap so router preferences drive routing
  5. Adding a diversity loss that pushes experts to produce different outputs
     on the same input (gives experts a reason to specialise)

Philosophy: the current model has a dead router (prob_entropy=ln(11)) because
the importance loss pushes to uniformity, the capacity cap hides routing
decisions, and the experts are interchangeable. This run gives the MoE
machinery a fresh training signal that rewards differentiation.
"""

import glob
import math
import os
import time

import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.v4_config import NanoOSRTv4Config
from nano_osrt.v4_data import make_v4_loader
from nano_osrt.v4_model import NanoOSRTv4ForCausalLM


def get_lr(step: int, total_steps: int, warmup: int, peak: float, minimum: float) -> float:
    """Cosine LR with linear warmup."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def reinit_experts_from_shared(model: NanoOSRTv4ForCausalLM, noise_std: float = 0.01) -> None:
    """Copy shared expert weights to all routed experts + add per-expert noise.

    This breaks symmetry: experts start from a known-good distribution
    (the shared expert has been training and carries useful features) but
    each gets a small unique perturbation so gradients can push them apart.
    """
    base = model.model
    for bi, block in enumerate(base.blocks):
        moe = block.moe
        shared = moe.shared_expert
        with torch.no_grad():
            for ei, expert in enumerate(moe.experts):
                for src, dst in zip(
                    (shared.w_gate, shared.w_up, shared.w_down),
                    (expert.w_gate, expert.w_up, expert.w_down),
                    strict=True,
                ):
                    dst.weight.copy_(src.weight)
                    dst.weight.add_(torch.randn_like(dst.weight) * noise_std)
        print(f"  Block {bi}: re-initialised {len(moe.experts)} routed experts from shared expert (+ noise σ={noise_std})")


def freeze_all_except_moe(model: NanoOSRTv4ForCausalLM) -> tuple[int, int]:
    """Freeze everything except router, routed experts, and dense/moe gates.

    Returns (trainable_count, frozen_count) as param element counts.
    """
    base = model.model
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = []
    for block in base.blocks:
        # Router
        for p in block.moe.router.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        # Loop embeddings (for per-loop router input)
        for p in block.moe.loop_embeddings.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        # Routed experts
        for expert in block.moe.experts:
            for p in expert.parameters():
                p.requires_grad = True
                trainable_params.append(p)
        # Gates
        block.dense_gate.requires_grad = True
        block.moe_gate.requires_grad = True
        trainable_params.append(block.dense_gate)
        trainable_params.append(block.moe_gate)

    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_count, frozen_count


def compute_diversity_loss(model: NanoOSRTv4ForCausalLM, x_sample: torch.Tensor) -> torch.Tensor:
    """Push routed experts toward producing different outputs on the same input.

    Loss = -mean_i mean_{j!=i} ||expert_i(x) - expert_j(x)||^2 / ||x||^2

    Negative because we want to MAXIMISE pairwise distance (minimise the
    negative). Normalised so it's scale-invariant to hidden-state magnitudes.
    """
    base = model.model
    total = x_sample.new_zeros(())
    count = 0
    # Take a small sample per block for efficiency
    for block in base.blocks:
        experts = block.moe.experts
        # Compute each expert's output
        outs = torch.stack([e(x_sample) for e in experts], dim=0)  # (E, B, S, D)
        E = outs.shape[0]
        # Pairwise squared distances
        # outs flattened per expert: (E, B*S*D)
        flat = outs.flatten(1)
        flat_norm = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine similarity matrix
        sim = flat_norm @ flat_norm.T  # (E, E)
        # Zero diagonal, average off-diagonal similarity
        mask = 1.0 - torch.eye(E, device=sim.device, dtype=sim.dtype)
        mean_sim = (sim * mask).sum() / mask.sum()
        # Minimising similarity pushes experts apart
        total = total + mean_sim
        count += 1
    return total / max(count, 1)


def run_v4_moe_recover(model_config: NanoOSRTv4Config, cfg, vol, tokenizer_name: str) -> None:
    """Execute the MoE recovery training loop."""
    device = torch.device("cuda")

    print("=" * 60)
    print("NanoOSRT v4 — MoE Recovery")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Override routing config for recovery
    # ------------------------------------------------------------------
    # Kill importance loss — this is what made the router uniform.
    model_config.router_aux_loss_coeff = 0.0
    model_config.router_logit_bias_coeff = 0.0
    model_config.router_z_loss_coeff = 0.0
    # Loosen capacity cap so router preferences actually matter.
    model_config.router_capacity_factor = cfg.capacity_factor
    # Skip soft warmup — go straight to hard routing so the router's
    # choices are visible and get gradient.
    model_config.router_soft_warmup_steps = 0
    model_config.router_blend_steps = 0
    # No bias controller during recovery
    model_config.router_balance_bias_enabled = False
    model_config.router_gumbel_tau_init = cfg.gumbel_tau
    model_config.router_gumbel_tau_final = cfg.gumbel_tau * 0.1

    print(f"  importance_coeff      : 0.0 (was 0.05) — removed")
    print(f"  capacity_factor       : {cfg.capacity_factor} (was 1.25) — loosened")
    print(f"  soft_warmup / blend   : 0 / 0 — straight to hard")
    print(f"  balance_bias          : disabled")
    print(f"  gumbel_tau            : {cfg.gumbel_tau} → {cfg.gumbel_tau * 0.1}")
    print(f"  diversity_loss_coeff  : {cfg.diversity_coeff}")
    print()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = NanoOSRTv4ForCausalLM(model_config).to(device=device)

    # Load checkpoint
    ckpt_path = cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MoE recovery needs a checkpoint at {ckpt_path}")
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  MISSING keys ({len(missing)}): sample {missing[:3]}")
    if unexpected:
        print(f"  UNEXPECTED keys ({len(unexpected)}): sample {unexpected[:3]}")

    # Re-init experts from shared expert + noise
    print("\nRe-initialising routed experts from shared expert...")
    torch.manual_seed(cfg.reinit_seed)
    reinit_experts_from_shared(model, noise_std=cfg.reinit_noise_std)

    # Freeze everything except MoE bits
    print("\nFreezing all non-MoE parameters...")
    trainable, frozen = freeze_all_except_moe(model)
    print(f"  Trainable: {trainable:>12,} params ({trainable / 1e6:.1f}M)")
    print(f"  Frozen   : {frozen:>12,} params ({frozen / 1e6:.1f}M)")
    print()

    # No torch.compile for recovery — we want clean debugging visibility on
    # the MoE telemetry and the diversity loss hook.
    # (compile also interferes with dynamic gradient flow through frozen params)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                "total_steps": cfg.total_steps,
                "peak_lr": cfg.peak_lr,
                "capacity_factor": cfg.capacity_factor,
                "diversity_coeff": cfg.diversity_coeff,
                "trainable_params": trainable,
            },
        )
        print("W&B logging enabled.")

    # ------------------------------------------------------------------
    # Optimizer — only trainable params
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.peak_lr, weight_decay=0.0, betas=(0.9, 0.95)
    )

    # ------------------------------------------------------------------
    # Data — use Foundation phase datasets (FineWeb-Edu + CodeParrot)
    # ------------------------------------------------------------------
    print("Loading pretraining data (FineWeb-Edu + CodeParrot)...")
    loader = make_v4_loader(
        cfg.datasets,
        cfg.seq_len,
        tokenizer_name,
        batch_size=cfg.batch_size,
        step_num=0,
    )
    loader_iter = iter(loader)
    print("DataLoader ready.\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ckpt_dir = os.path.dirname(cfg.pretrained_checkpoint)
    start_time = time.time()
    eff_batch = cfg.batch_size * cfg.grad_accum_steps

    for step in range(cfg.total_steps):
        lr = get_lr(step, cfg.total_steps, cfg.warmup_steps, cfg.peak_lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_task = torch.zeros((), device=device)
        accum_div = torch.zeros((), device=device)

        for _ in range(cfg.grad_accum_steps):
            batch = next(loader_iter)
            input_ids, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                task_loss = outputs.loss

                # Sample a small chunk of hidden state to compute diversity loss
                # (computing on full sequence is expensive — use first 128 tokens)
                if cfg.diversity_coeff > 0:
                    with torch.no_grad():
                        # Get a hidden state sample by running embedding on a small slice
                        emb = model.model.embedding(input_ids[:, :64])
                    div_loss = compute_diversity_loss(model, emb)
                else:
                    div_loss = torch.zeros((), device=device)

                total = task_loss + cfg.diversity_coeff * div_loss

            (total / cfg.grad_accum_steps).backward()
            accum_task += task_loss.detach() / cfg.grad_accum_steps
            accum_div += div_loss.detach() / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        optimizer.step()

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

            # Collect MoE telemetry
            base = model.model
            _mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
            dense_gates, moe_gates, prob_ents, assign_ents, raw_ents = [], [], [], [], []
            all_overflow = []
            raw_maxes = []
            for block in base.blocks:
                dense_gates.append(block.dense_gate.item())
                moe_gates.append(block.moe_gate.item())
                prob_ents.extend(block.moe.last_router_entropy)
                assign_ents.extend(block.moe.last_assignment_entropy)
                raw_ents.extend(block.moe.last_raw_assignment_entropy)
                all_overflow.extend(block.moe.last_overflow_rate)
                for fracs in block.moe.last_raw_expert_fraction:
                    if fracs:
                        raw_maxes.append(max(fracs))

            print(
                f"step {step:>5d}/{cfg.total_steps} | "
                f"task {accum_task.item():.4f} | div {accum_div.item():+.4f} | "
                f"lr {lr:.2e} | vram {vram_gb:.1f}GB | "
                f"elapsed {elapsed:.0f}s"
            )
            print(
                f"         moe: prob_H={_mean(prob_ents):.3f} "
                f"assign_H={_mean(assign_ents):.3f} "
                f"raw_H={_mean(raw_ents):.3f} "
                f"raw_max={_mean(raw_maxes):.3f} "
                f"overflow={_mean(all_overflow):.4f} | "
                f"gates d={_mean(dense_gates):.3f} m={_mean(moe_gates):.3f}"
            )

            if use_wandb:
                wandb.log({
                    "recover/task_loss": accum_task.item(),
                    "recover/diversity_loss": accum_div.item(),
                    "recover/lr": lr,
                    "recover/vram_gb": vram_gb,
                    "moe/prob_entropy_mean": _mean(prob_ents),
                    "moe/assign_entropy_mean": _mean(assign_ents),
                    "moe/raw_assign_entropy_mean": _mean(raw_ents),
                    "moe/raw_expert_max_mean": _mean(raw_maxes),
                    "moe/overflow_rate_mean": _mean(all_overflow),
                    "moe/dense_gate_mean": _mean(dense_gates),
                    "moe/moe_gate_mean": _mean(moe_gates),
                }, step=step)

        # Checkpoints
        if step > 0 and step % cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v4_moe_recover_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_stage": "moe_recover",
            }, path)
            vol.commit()
            print(f"  -> Checkpoint saved: {path}")

    # Final
    final_path = f"{ckpt_dir}/osrt_v4_moe_recover_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "training_stage": "moe_recover",
    }, final_path)
    vol.commit()
    elapsed_total = time.time() - start_time
    print(f"\nMoE recovery complete. {cfg.total_steps:,} steps in {elapsed_total / 3600:.1f}h")
    print(f"Final model: {final_path}")
    if use_wandb:
        wandb.finish()
