"""Pre-training loop for NanoOSRT.

Simpler than v4_train.py because v5 removes:
  - Soft warmup / blend / routing_mode switching
  - router_noise anneal

What remains: cosine LR, phase transitions (seq_len + dataset swap),
eval, checkpointing, W&B, compile, resume, 23h rescue, and annealed
Gumbel top-k exploration plus a DeepSeek-style per-expert bias controller
to prevent early dead experts.

v5-specific telemetry (new metrics, all logged to W&B and stdout):
  - per_token_entropy — the real router-sharpness signal
  - marginal_entropy — balance proxy (stays high if globally balanced)
  - assignment_entropy — hard f entropy
  - raw_max_prob — pre-renormalisation top-1 confidence
  - top_margin — gap between rank 0 and rank 1 probs
  - drop_rate — fraction of token-expert pairs dropped by capacity cap
  - dense_gate removed (no dense FFN in v5); log moe_gate per block

Early-stop check: at `early_stop_check_step`, verifies the router has
made the four v5 success criteria. If not, prints a clear diagnosis and
stops so further compute isn't wasted on a known-bad run.
"""

import glob
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch import Tensor

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.config import NanoOSRTConfig
from nano_osrt.data import make_loader  # reused unchanged
from nano_osrt.model import NanoOSRTForCausalLM
from nano_osrt.train_config import PretrainConfig


def get_lr(step: int, cfg: PretrainConfig) -> float:
    """Cosine LR with linear warmup, returning the AdamW/Lion peak_lr."""
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def _set_param_group_lrs(
    optimizer, step: int, cfg: PretrainConfig,
) -> float:
    """Apply the cosine schedule to every param group, respecting the
    per-group `_peak_lr` / `_min_lr` tags written at construction.

    Returns the AdamW/Lion-scale LR for logging. Muon groups (when
    present) have their own peak_lr/min_lr stored on the group dict and
    are scaled along the same cosine ratio so all optimizers reach
    their min at the end of training.

    `lr_anchor_step` (optional cfg attr, default 0) lets a resumed
    extension run re-warm without an LR jump-up: the warmup/cosine
    treats `step - anchor` as the effective step, so a continuation
    starts cleanly from warmup again instead of recomputing cosine
    progress against the new total_steps (which would jump LR back to
    near-peak mid-cool). Same mechanism as GRPOConfig.lr_anchor_step.
    """
    anchor = getattr(cfg, "lr_anchor_step", 0)
    eff_step = max(step - anchor, 0)
    eff_total = max(cfg.total_steps - anchor, 1)
    if eff_step < cfg.warmup_steps:
        ratio = eff_step / cfg.warmup_steps
        for pg in optimizer.param_groups:
            peak = pg.get("_peak_lr", cfg.peak_lr)
            pg["lr"] = peak * ratio
        return cfg.peak_lr * ratio

    progress = (eff_step - cfg.warmup_steps) / max(
        eff_total - cfg.warmup_steps, 1,
    )
    cosine_half = 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        peak = pg.get("_peak_lr", cfg.peak_lr)
        floor = pg.get("_min_lr", cfg.min_lr)
        pg["lr"] = floor + (peak - floor) * cosine_half
    return cfg.min_lr + (cfg.peak_lr - cfg.min_lr) * cosine_half


def get_router_gumbel_tau(step: int, cfg: PretrainConfig) -> float:
    """Linear Gumbel top-k noise schedule for early router exploration."""
    init = cfg.router_gumbel_tau_init
    final = cfg.router_gumbel_tau_final
    anneal = max(cfg.router_gumbel_anneal_steps, 1)
    progress = min(step / anneal, 1.0)
    return init + (final - init) * progress


def set_router_gumbel_tau(model: nn.Module, tau: float) -> None:
    """Set the per-MoE Gumbel tau buffer on compiled or eager v5 models."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    for block in base.blocks:
        block.moe.gumbel_tau.fill_(tau)


def apply_router_balance_updates(model: nn.Module) -> None:
    """Apply once-per-step balance-bias updates on compiled or eager models."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    for block in base.blocks:
        block.moe.apply_balance_update()


def get_phase(step: int, cfg: PretrainConfig) -> tuple[str, dict]:
    """Get current phase config for a given step."""
    for name, p in cfg.phases.items():
        if p["start"] <= step < p["end"]:
            return name, p
    last_name = list(cfg.phases.keys())[-1]
    return last_name, cfg.phases[last_name]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Save a training checkpoint (v5 format)."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_stage": "pretrain_v5",
        },
        path,
    )
    print(f"  -> Checkpoint saved: {path}")


def load_model_state_or_raise(
    model: nn.Module,
    state_dict: dict,
    context: str,
) -> None:
    """Load model weights and fail on any key drift."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    missing, unexpected = inner.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        missing_sample = ", ".join(missing[:8])
        unexpected_sample = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"{context}: checkpoint/model key mismatch. "
            f"missing={len(missing)} [{missing_sample}], "
            f"unexpected={len(unexpected)} [{unexpected_sample}]. "
            "Use an explicit migration or start a fresh checkpoint directory."
        )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    """Load from checkpoint. Returns step to resume from (0 if path missing)."""
    if not os.path.exists(path):
        return 0
    print(f"Resuming from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    load_model_state_or_raise(
        model,
        ckpt["model_state_dict"],
        context=f"pretrain resume from {path}",
    )
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except (ValueError, RuntimeError) as e:
        print(f"  Optimizer state mismatch, starting fresh: {e}")
    start_step = ckpt["step"] + 1
    print(f"  Resumed at step {start_step}")
    return start_step


# Eval batches are materialised once per process and replayed on every
# subsequent call. `ds.skip(N)` is O(N) iteration through FineWeb-Edu's
# streaming shards, so paying the skip cost every eval interval was
# wasting hours across a long run. The skip offset (see run_eval
# docstring) is also deliberately large — it has to sit past the full
# training-budget consumption so cached eval samples never leak back
# into the training set late in the run. Caching amortises that
# one-time cost and guarantees identical samples across steps, making
# loss curves strictly comparable.
_EVAL_BATCH_CACHE: dict[tuple, list[tuple[Tensor, Tensor]]] = {}


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
    """Run held-out evaluation on a cached FineWeb-Edu slice.

    FineWeb-Edu has no upstream validation split, so we carve out a
    held-out subset by starting the stream at a 100M-example skip
    offset. Budget math: the schedule consumes ~32B FineWeb tokens
    (0.8B in phase 1 + 31.5B in phase 2, see PretrainConfig:63). At
    ~1000 tokens per FineWeb-Edu record that's ~32M training records,
    so a 100M skip leaves a ~3x safety margin over the full planned
    budget. The previous 500k skip was blown past within hours of
    phase-2 training, meaning cached eval samples could re-appear as
    training data later in the run.

    `ds.skip(N)` is O(N) iteration, so this first-call cost is real
    (~10-20 min on Modal, depending on HF streaming throughput). The
    first call materialises `eval_steps` batches and caches them on
    CPU; every subsequent call replays the cache, so we pay the skip
    and tokenisation cost exactly once per process. This also
    guarantees identical samples across steps, making loss curves
    strictly comparable.

    Switches model to inference mode (drops off, aux loss excluded).
    """
    was_training = model.training
    model.train(False)  # disable capacity drops + dropout-like behaviour

    cache_key = (tokenizer_name, seq_len, batch_size, eval_steps)
    cached = _EVAL_BATCH_CACHE.get(cache_key)
    if cached is None:
        loader = make_loader(
            dataset_configs=[
                {
                    "name": "fineweb-edu-eval",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 1.0,
                    # Held-out offset past the full training budget.
                    # See docstring for the ~32M-record budget math.
                    "skip": 100_000_000,
                },
            ],
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            step_num=999999,  # fixed seed, never matches training seeds
        )
        data_iter = iter(loader)
        cached = []
        for _ in range(eval_steps):
            try:
                cached.append(next(data_iter))
            except StopIteration:
                break
        _EVAL_BATCH_CACHE[cache_key] = cached
        # Drop the iterator before the loader so worker processes are
        # reaped cleanly. `del loader, data_iter` evaluates left-to-right
        # and dropping `loader` first does nothing observable because
        # `data_iter` still holds the loader (and therefore the spawn
        # workers); the actual teardown only fires when `data_iter` is
        # released, racing finalize and producing
        # "Fatal Python error: PyGILState_Release" in one of the workers
        # — a leak per eval call. Force order + GC so workers exit cleanly.
        import gc
        del data_iter
        del loader
        gc.collect()

    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in cached:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)
        # In inference mode, outputs.loss is pure task CE (no aux pollution).
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    if was_training:
        model.train(True)

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))
    return {
        "eval/loss": mean_loss,
        "eval/perplexity": perplexity,
        "eval/tokens": total_tokens,
    }


def _collect_moe_metrics(model: nn.Module) -> tuple[dict, dict]:
    """Pull v5 MoE telemetry from each block. Returns (wandb_dict, stdout_summary)."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    per_token_ents: list[float] = []
    marginal_ents: list[float] = []
    assign_ents: list[float] = []
    clean_per_token_ents: list[float] = []
    clean_marginal_ents: list[float] = []
    clean_assign_ents: list[float] = []
    raw_max_probs: list[float] = []
    top_margins: list[float] = []
    clean_raw_max_probs: list[float] = []
    clean_top_margins: list[float] = []
    drop_rates: list[float] = []
    moe_gates: list[float] = []
    expert_maxes: list[float] = []
    expert_mins: list[float] = []
    clean_expert_maxes: list[float] = []
    clean_expert_mins: list[float] = []
    prebias_per_token_ents: list[float] = []
    prebias_marginal_ents: list[float] = []
    prebias_assign_ents: list[float] = []
    prebias_raw_max_probs: list[float] = []
    prebias_top_margins: list[float] = []
    prebias_expert_maxes: list[float] = []
    prebias_expert_mins: list[float] = []
    balance_losses: list[float] = []
    bias_abs_maxes: list[float] = []
    bias_ema_maxes: list[float] = []
    bias_ema_mins: list[float] = []

    metrics: dict = {}
    for bi, blk in enumerate(base.blocks):
        # Log the EFFECTIVE gate (post-softplus) — the raw parameter is
        # an unconstrained pre-image and doesn't reflect the actual
        # routed-branch scaling. softplus(raw) is what multiplies h_routed
        # in RecursiveBlock.forward, so that's what matters for
        # interpreting "is the routed branch contributing".
        mg = blk.effective_moe_gate().item()
        moe_gates.append(mg)
        metrics[f"moe/moe_gate_b{bi}"] = mg

        if blk.moe.balance_loss is not None:
            balance_losses.append(blk.moe.balance_loss.item())
        if hasattr(blk.moe, "router_balance_bias"):
            bias_abs_max = blk.moe.router_balance_bias.abs().max().item()
            bias_abs_maxes.append(bias_abs_max)
            metrics[f"moe/bias_abs_max_b{bi}"] = bias_abs_max
        if hasattr(blk.moe, "expert_ema_fraction"):
            ema = blk.moe.expert_ema_fraction
            ema_max = ema.max().item()
            ema_min = ema.min().item()
            bias_ema_maxes.append(ema_max)
            bias_ema_mins.append(ema_min)
            metrics[f"moe/bias_ema_max_b{bi}"] = ema_max
            metrics[f"moe/bias_ema_min_b{bi}"] = ema_min

        for li, v in enumerate(blk.moe.last_per_token_entropy):
            metrics[f"moe/per_token_entropy_b{bi}_l{li}"] = v
            per_token_ents.append(v)
        for li, v in enumerate(blk.moe.last_marginal_entropy):
            metrics[f"moe/marginal_entropy_b{bi}_l{li}"] = v
            marginal_ents.append(v)
        for li, v in enumerate(blk.moe.last_assignment_entropy):
            metrics[f"moe/assignment_entropy_b{bi}_l{li}"] = v
            assign_ents.append(v)
        for li, v in enumerate(blk.moe.last_clean_per_token_entropy):
            metrics[f"moe/clean_per_token_entropy_b{bi}_l{li}"] = v
            clean_per_token_ents.append(v)
        for li, v in enumerate(blk.moe.last_clean_marginal_entropy):
            metrics[f"moe/clean_marginal_entropy_b{bi}_l{li}"] = v
            clean_marginal_ents.append(v)
        for li, v in enumerate(blk.moe.last_clean_assignment_entropy):
            metrics[f"moe/clean_assignment_entropy_b{bi}_l{li}"] = v
            clean_assign_ents.append(v)
        for li, v in enumerate(blk.moe.last_raw_max_prob):
            metrics[f"moe/raw_max_prob_b{bi}_l{li}"] = v
            raw_max_probs.append(v)
        for li, v in enumerate(blk.moe.last_top_margin):
            metrics[f"moe/top_margin_b{bi}_l{li}"] = v
            top_margins.append(v)
        for li, v in enumerate(blk.moe.last_clean_raw_max_prob):
            metrics[f"moe/clean_raw_max_prob_b{bi}_l{li}"] = v
            clean_raw_max_probs.append(v)
        for li, v in enumerate(blk.moe.last_clean_top_margin):
            metrics[f"moe/clean_top_margin_b{bi}_l{li}"] = v
            clean_top_margins.append(v)
        for li, v in enumerate(blk.moe.last_drop_rate):
            metrics[f"moe/drop_rate_b{bi}_l{li}"] = v
            drop_rates.append(v)
        for li, fracs in enumerate(blk.moe.last_expert_fraction):
            if fracs:
                mx = max(fracs)
                mn = min(fracs)
                metrics[f"moe/expert_max_b{bi}_l{li}"] = mx
                metrics[f"moe/expert_min_b{bi}_l{li}"] = mn
                expert_maxes.append(mx)
                expert_mins.append(mn)
        for li, fracs in enumerate(blk.moe.last_clean_expert_fraction):
            if fracs:
                mx = max(fracs)
                mn = min(fracs)
                metrics[f"moe/clean_expert_max_b{bi}_l{li}"] = mx
                metrics[f"moe/clean_expert_min_b{bi}_l{li}"] = mn
                clean_expert_maxes.append(mx)
                clean_expert_mins.append(mn)

        # Prebias raw-router telemetry — exposed by the model when the
        # bias controller is enabled. Guarded with hasattr so older
        # checkpoints / tests without the attribute don't crash.
        if hasattr(blk.moe, "last_prebias_per_token_entropy"):
            for li, v in enumerate(blk.moe.last_prebias_per_token_entropy):
                metrics[f"moe/prebias_per_token_entropy_b{bi}_l{li}"] = v
                prebias_per_token_ents.append(v)
            for li, v in enumerate(blk.moe.last_prebias_marginal_entropy):
                metrics[f"moe/prebias_marginal_entropy_b{bi}_l{li}"] = v
                prebias_marginal_ents.append(v)
            for li, v in enumerate(blk.moe.last_prebias_assignment_entropy):
                metrics[f"moe/prebias_assignment_entropy_b{bi}_l{li}"] = v
                prebias_assign_ents.append(v)
            for li, v in enumerate(blk.moe.last_prebias_raw_max_prob):
                metrics[f"moe/prebias_raw_max_prob_b{bi}_l{li}"] = v
                prebias_raw_max_probs.append(v)
            for li, v in enumerate(blk.moe.last_prebias_top_margin):
                metrics[f"moe/prebias_top_margin_b{bi}_l{li}"] = v
                prebias_top_margins.append(v)
            for li, fracs in enumerate(blk.moe.last_prebias_expert_fraction):
                if fracs:
                    mx = max(fracs)
                    mn = min(fracs)
                    metrics[f"moe/prebias_expert_max_b{bi}_l{li}"] = mx
                    metrics[f"moe/prebias_expert_min_b{bi}_l{li}"] = mn
                    prebias_expert_maxes.append(mx)
                    prebias_expert_mins.append(mn)

    metrics["moe/per_token_entropy_mean"] = _mean(per_token_ents)
    metrics["moe/marginal_entropy_mean"] = _mean(marginal_ents)
    metrics["moe/assignment_entropy_mean"] = _mean(assign_ents)
    metrics["moe/clean_per_token_entropy_mean"] = _mean(clean_per_token_ents)
    metrics["moe/clean_marginal_entropy_mean"] = _mean(clean_marginal_ents)
    metrics["moe/clean_assignment_entropy_mean"] = _mean(clean_assign_ents)
    metrics["moe/raw_max_prob_mean"] = _mean(raw_max_probs)
    metrics["moe/top_margin_mean"] = _mean(top_margins)
    metrics["moe/clean_raw_max_prob_mean"] = _mean(clean_raw_max_probs)
    metrics["moe/clean_top_margin_mean"] = _mean(clean_top_margins)
    metrics["moe/drop_rate_mean"] = _mean(drop_rates)
    metrics["moe/moe_gate_mean"] = _mean(moe_gates)
    metrics["moe/expert_max_mean"] = _mean(expert_maxes)
    metrics["moe/expert_min_mean"] = _mean(expert_mins)
    metrics["moe/clean_expert_max_mean"] = _mean(clean_expert_maxes)
    metrics["moe/clean_expert_min_mean"] = _mean(clean_expert_mins)
    metrics["moe/balance_loss_mean"] = _mean(balance_losses)
    metrics["moe/bias_abs_max_mean"] = _mean(bias_abs_maxes)
    metrics["moe/bias_ema_max_mean"] = _mean(bias_ema_maxes)
    metrics["moe/bias_ema_min_mean"] = _mean(bias_ema_mins)
    metrics["moe/prebias_per_token_entropy_mean"] = _mean(prebias_per_token_ents)
    metrics["moe/prebias_marginal_entropy_mean"] = _mean(prebias_marginal_ents)
    metrics["moe/prebias_assignment_entropy_mean"] = _mean(prebias_assign_ents)
    metrics["moe/prebias_raw_max_prob_mean"] = _mean(prebias_raw_max_probs)
    metrics["moe/prebias_top_margin_mean"] = _mean(prebias_top_margins)
    metrics["moe/prebias_expert_max_mean"] = _mean(prebias_expert_maxes)
    metrics["moe/prebias_expert_min_mean"] = _mean(prebias_expert_mins)

    summary = {
        "per_token_H": _mean(per_token_ents),
        "marginal_H": _mean(marginal_ents),
        "assign_H": _mean(assign_ents),
        "clean_per_token_H": _mean(clean_per_token_ents),
        "clean_marginal_H": _mean(clean_marginal_ents),
        "clean_assign_H": _mean(clean_assign_ents),
        "raw_max": _mean(raw_max_probs),
        "top_margin": _mean(top_margins),
        "clean_raw_max": _mean(clean_raw_max_probs),
        "clean_top_margin": _mean(clean_top_margins),
        "drop_rate": _mean(drop_rates),
        "moe_gate": _mean(moe_gates),
        "expert_max": _mean(expert_maxes),
        "expert_min": _mean(expert_mins),
        "clean_expert_max": _mean(clean_expert_maxes),
        "clean_expert_min": _mean(clean_expert_mins),
        "balance_loss": _mean(balance_losses),
        "bias_abs_max": _mean(bias_abs_maxes),
        "bias_ema_max": _mean(bias_ema_maxes),
        "bias_ema_min": _mean(bias_ema_mins),
        "prebias_per_token_H": _mean(prebias_per_token_ents),
        "prebias_marginal_H": _mean(prebias_marginal_ents),
        "prebias_assign_H": _mean(prebias_assign_ents),
        "prebias_raw_max": _mean(prebias_raw_max_probs),
        "prebias_top_margin": _mean(prebias_top_margins),
        "prebias_expert_max": _mean(prebias_expert_maxes),
        "prebias_expert_min": _mean(prebias_expert_mins),
    }
    return metrics, summary


def _average_moe_snapshots(
    snapshots: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Average a list of per-micro-batch MoE snapshots into one dict.

    Each snapshot is a wandb-keyed metrics dict from _collect_moe_metrics.
    Averaging element-wise gives us the per-step MoE state averaged over
    all grad_accum micro-batches — much less noisy than taking only the
    last micro-batch, which matters for the 5k early-stop gate where a
    single-batch outlier on clean_raw_max_prob could false-trip.

    Returns (metrics_avg, summary). summary is rebuilt from the averaged
    *_mean keys so stdout formatting is unchanged.
    """
    if not snapshots:
        return {}, {}
    n = len(snapshots)
    keys = snapshots[0].keys()
    avg: dict[str, float] = {}
    for k in keys:
        total = 0.0
        for snap in snapshots:
            total += snap.get(k, 0.0)
        avg[k] = total / n

    summary = {
        "per_token_H": avg.get("moe/per_token_entropy_mean", 0.0),
        "marginal_H": avg.get("moe/marginal_entropy_mean", 0.0),
        "assign_H": avg.get("moe/assignment_entropy_mean", 0.0),
        "clean_per_token_H": avg.get("moe/clean_per_token_entropy_mean", 0.0),
        "clean_marginal_H": avg.get("moe/clean_marginal_entropy_mean", 0.0),
        "clean_assign_H": avg.get("moe/clean_assignment_entropy_mean", 0.0),
        "raw_max": avg.get("moe/raw_max_prob_mean", 0.0),
        "top_margin": avg.get("moe/top_margin_mean", 0.0),
        "clean_raw_max": avg.get("moe/clean_raw_max_prob_mean", 0.0),
        "clean_top_margin": avg.get("moe/clean_top_margin_mean", 0.0),
        "drop_rate": avg.get("moe/drop_rate_mean", 0.0),
        "moe_gate": avg.get("moe/moe_gate_mean", 0.0),
        "expert_max": avg.get("moe/expert_max_mean", 0.0),
        "expert_min": avg.get("moe/expert_min_mean", 0.0),
        "clean_expert_max": avg.get("moe/clean_expert_max_mean", 0.0),
        "clean_expert_min": avg.get("moe/clean_expert_min_mean", 0.0),
        "balance_loss": avg.get("moe/balance_loss_mean", 0.0),
        "bias_abs_max": avg.get("moe/bias_abs_max_mean", 0.0),
        "bias_ema_max": avg.get("moe/bias_ema_max_mean", 0.0),
        "bias_ema_min": avg.get("moe/bias_ema_min_mean", 0.0),
        "prebias_per_token_H": avg.get(
            "moe/prebias_per_token_entropy_mean", 0.0,
        ),
        "prebias_marginal_H": avg.get(
            "moe/prebias_marginal_entropy_mean", 0.0,
        ),
        "prebias_assign_H": avg.get(
            "moe/prebias_assignment_entropy_mean", 0.0,
        ),
        "prebias_raw_max": avg.get("moe/prebias_raw_max_prob_mean", 0.0),
        "prebias_top_margin": avg.get("moe/prebias_top_margin_mean", 0.0),
        "prebias_expert_max": avg.get("moe/prebias_expert_max_mean", 0.0),
        "prebias_expert_min": avg.get("moe/prebias_expert_min_mean", 0.0),
    }
    return avg, summary


def _check_early_stop_criteria(
    step: int, summary: dict, cfg: PretrainConfig, model_cfg: NanoOSRTConfig,
) -> list[str]:
    """Return list of failing criteria (empty means all pass)."""
    failures: list[str] = []
    per_token_h = summary.get("clean_per_token_H", summary["per_token_H"])
    raw_max = summary.get("clean_raw_max", summary["raw_max"])
    top_margin = summary.get("clean_top_margin", summary["top_margin"])
    marginal_h = summary.get("clean_marginal_H", summary["marginal_H"])
    # per_token_entropy target: init near ln(num_routed) = ln(8) ≈ 2.079.
    # Require it to drop by at least min_per_token_entropy_drop.
    target_pte = 2.079 - cfg.min_per_token_entropy_drop
    if per_token_h > target_pte:
        failures.append(
            f"clean_per_token_entropy {per_token_h:.3f} > "
            f"{target_pte:.3f} (router hasn't sharpened)"
        )
    if raw_max < cfg.min_raw_max_prob:
        failures.append(
            f"clean_raw_max_prob {raw_max:.3f} < "
            f"{cfg.min_raw_max_prob:.3f} (no strong primary pick)"
        )
    if top_margin < cfg.min_top_margin:
        failures.append(
            f"clean_top_margin {top_margin:.3f} < "
            f"{cfg.min_top_margin:.3f} (no gap between rank 0 and rank 1)"
        )
    if marginal_h < cfg.min_marginal_entropy:
        failures.append(
            f"clean_marginal_entropy {marginal_h:.3f} < "
            f"{cfg.min_marginal_entropy:.3f} (dead/overloaded experts)"
        )
    prebias_marginal_h = summary.get("prebias_marginal_H", marginal_h)
    min_prebias_marginal = getattr(
        cfg, "min_prebias_marginal_entropy", cfg.min_marginal_entropy,
    )
    if prebias_marginal_h < min_prebias_marginal:
        failures.append(
            f"prebias_marginal_entropy {prebias_marginal_h:.3f} < "
            f"{min_prebias_marginal:.3f} "
            "(learned router collapsed before balance bias)"
        )
    prebias_expert_min = summary.get("prebias_expert_min", 1.0)
    min_prebias_expert = getattr(cfg, "min_prebias_expert_fraction", 0.0)
    if prebias_expert_min < min_prebias_expert:
        failures.append(
            f"prebias_expert_min {prebias_expert_min:.4f} < "
            f"{min_prebias_expert:.4f} "
            "(one or more experts are dead before balance bias)"
        )
    bias_limit = (
        model_cfg.router_balance_bias_max
        * getattr(cfg, "max_bias_saturation_fraction", 1.0)
    )
    if model_cfg.router_balance_bias_enabled and bias_limit > 0:
        bias_abs_max = summary.get("bias_abs_max", 0.0)
        if bias_abs_max > bias_limit:
            failures.append(
                f"router_balance_bias_abs_max {bias_abs_max:.3f} > "
                f"{bias_limit:.3f} "
                "(bias controller is near saturation and may be masking collapse)"
            )
    return failures


def run_training(
    model_config: NanoOSRTConfig,
    train_cfg: PretrainConfig,
    vol,
    tokenizer_name: str,
    ckpt_dir: str = "/vol/checkpoints/v5",
) -> None:
    """Execute the v5 pre-training loop.

    Args:
        model_config: v5 model configuration.
        train_cfg: training hyperparameters + phase schedule.
        vol: Modal Volume for checkpoints.
        tokenizer_name: path or HF id of the tokenizer.
        ckpt_dir: directory for checkpoints. Defaults to v5 production dir;
            sanity/test runs should pass a distinct dir to avoid colliding
            with real checkpoints.
    """
    device = torch.device("cuda")

    print("=" * 60)
    print("NanoOSRT — Mixtral MoE Pre-training")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Model setup
    model = NanoOSRTForCausalLM(model_config).to(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    n_experts = 1 + model_config.num_routed_experts  # +1 shared
    print(f"Physical parameters : {total_params:>12,}")
    print(f"Blocks              : {model_config.num_blocks}")
    print(f"Recursive loops     : {model_config.recursive_loops}")
    print(
        f"Effective layers    : "
        f"{model_config.num_blocks * model_config.recursive_loops}"
    )
    print(
        f"Experts             : {n_experts} "
        f"(1 shared + {model_config.num_routed_experts} routed, "
        f"top-{model_config.top_k_experts})"
    )
    print(f"Hidden dim          : {model_config.dim}")
    print(f"Peak LR             : {train_cfg.peak_lr}")
    print(f"Optimizer           : {train_cfg.optimizer_name}")
    print(f"Total steps         : {train_cfg.total_steps}")
    print(f"Aux loss coeff      : {model_config.router_aux_loss_coeff}")
    print(
        f"Balance bias        : {model_config.router_balance_bias_enabled} "
        f"(rate={model_config.router_balance_bias_update_rate}, "
        f"max={model_config.router_balance_bias_max})"
    )
    print(
        f"Router Gumbel tau   : {train_cfg.router_gumbel_tau_init} -> "
        f"{train_cfg.router_gumbel_tau_final} over "
        f"{train_cfg.router_gumbel_anneal_steps} steps"
    )
    print()

    print("Compiling model with torch.compile...")
    compile_start = time.time()
    model = torch.compile(model)
    print(f"Model compile done in {time.time() - compile_start:.1f}s")

    # W&B
    use_wandb = train_cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": train_cfg.wandb_project,
            "name": train_cfg.wandb_run_name,
            "config": {
                "stage": "pretrain",
                "total_params": total_params,
                "architecture": "mixtral_moe",
                "num_blocks": model_config.num_blocks,
                "recursive_loops": model_config.recursive_loops,
                "num_routed_experts": model_config.num_routed_experts,
                "top_k": model_config.top_k_experts,
                "expert_hidden": model_config.expert_hidden,
                "shared_expert_hidden": model_config.shared_expert_hidden,
                "capacity_factor": model_config.router_capacity_factor,
                "aux_loss_coeff": model_config.router_aux_loss_coeff,
                "balance_bias_enabled": (
                    model_config.router_balance_bias_enabled
                ),
                "balance_bias_update_rate": (
                    model_config.router_balance_bias_update_rate
                ),
                "balance_bias_max": model_config.router_balance_bias_max,
                "router_gumbel_tau_init": train_cfg.router_gumbel_tau_init,
                "router_gumbel_tau_final": train_cfg.router_gumbel_tau_final,
                "router_gumbel_anneal_steps": (
                    train_cfg.router_gumbel_anneal_steps
                ),
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

    # Optimizer — router/loop_embeddings get wd=0 (they're routing-sensitive).
    # Three branches today:
    #   "lion"  — single Lion over all params (router gets a wd=0 group)
    #   "muon"  — hybrid Muon (2D matrix weights) + AdamW (embeddings,
    #             norms, scalars, router/loop_embeddings)
    #   else    — AdamW fallback
    inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    if train_cfg.optimizer_name.lower() == "muon":
        from nano_osrt.muon import (
            HybridMuonAdamW,
            Muon,
            build_param_groups,
        )

        muon_params, adamw_groups = build_param_groups(
            inner_model.named_parameters(),
            weight_decay=train_cfg.weight_decay,
        )
        # Muon LR is much smaller-magnitude than Lion/AdamW because the
        # Newton-Schulz update is normalised. Keep it as a separate
        # config knob so users can A/B without nuking the Lion peak_lr.
        muon_lr = getattr(train_cfg, "muon_lr", train_cfg.peak_lr)
        muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=train_cfg.weight_decay,
        )
        adamw = torch.optim.AdamW(
            adamw_groups,
            lr=train_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        # Tag each group with its peak/min for the cosine schedule.
        muon_min = getattr(train_cfg, "muon_min_lr", muon_lr * 0.1)
        for pg in muon.param_groups:
            pg["_peak_lr"] = muon_lr
            pg["_min_lr"] = muon_min
        for pg in adamw.param_groups:
            pg["_peak_lr"] = train_cfg.peak_lr
            pg["_min_lr"] = train_cfg.min_lr
        optimizer = HybridMuonAdamW(muon, adamw)
        n_muon = len(muon_params)
        n_adamw = sum(len(g["params"]) for g in adamw_groups)
        print(
            f"Using Muon+AdamW hybrid: {n_muon} matrix tensors → Muon "
            f"(lr={muon_lr}), {n_adamw} other tensors → AdamW "
            f"(lr={train_cfg.peak_lr}, wd={train_cfg.weight_decay} "
            f"on non-norm/non-embed only)"
        )
    else:
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
            f"Param groups: {len(other_params)} standard, "
            f"{len(router_params)} router (wd=0)"
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

    # Checkpoint resume.
    # Three kinds of checkpoints with different naming:
    #   osrt_v5_step_N.pt          — normal interval save (resumable)
    #   osrt_v5_rescue_step_N.pt   — 23h timeout rescue    (resumable)
    #   osrt_v5_failed_step_N.pt   — failed early-stop     (NOT resumed; the
    #     run declared itself bad. If you want to investigate or force-resume
    #     anyway, rename the file to osrt_v5_step_N.pt explicitly.)
    # Resume scans the first two patterns and picks the highest step.
    os.makedirs(ckpt_dir, exist_ok=True)

    def _extract_step(path: str) -> int | None:
        """Extract the step number from an osrt_v5_..._step_N.pt path."""
        try:
            return int(path.rsplit("_", 1)[1].split(".")[0])
        except (ValueError, IndexError):
            return None

    best_step = -1
    best_ckpt: str | None = None
    # Scan order matters for tie-breaking: if the 23h rescue fires on a
    # ckpt_interval step, both osrt_v5_step_N.pt and
    # osrt_v5_rescue_step_N.pt exist at step N. The files contain
    # identical optimizer state (same end-of-step save) but we prefer
    # the rescue variant because it's the intentional "resume here"
    # marker — scanning rescue AFTER normal with `>=` makes rescue win
    # on ties. When steps differ, higher step always wins regardless
    # of pattern.
    for pattern in (
        f"{ckpt_dir}/osrt_v5_step_*.pt",
        f"{ckpt_dir}/osrt_v5_rescue_step_*.pt",
    ):
        for f in glob.glob(pattern):
            s = _extract_step(f)
            if s is None:
                continue
            if s > best_step or (s == best_step and "rescue" in f):
                best_step = s
                best_ckpt = f

    # Explicit notice if there's a FAILED checkpoint — user should know.
    failed_paths = sorted(glob.glob(f"{ckpt_dir}/osrt_v5_failed_step_*.pt"))
    if failed_paths:
        print(
            f"WARNING: Found {len(failed_paths)} failed-early-stop "
            f"checkpoint(s): {[os.path.basename(p) for p in failed_paths]}. "
            f"These are NOT resumed automatically. Rename to "
            f"osrt_v5_step_N.pt if you want to force-resume.",
        )

    start_step = 0
    if best_step > 0 and best_ckpt is not None:
        print(f"Found checkpoint at step {best_step}: {best_ckpt}")
        start_step = load_checkpoint(model, optimizer, best_ckpt, device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_time = time.time()
    step = start_step
    current_phase: str | None = None
    current_loader = None
    loader_iter = None
    current_seq_len = 2048
    current_batch_size = train_cfg.batch_size
    grad_accum = train_cfg.grad_accum_steps
    early_stop_triggered = False

    while step < train_cfg.total_steps and not early_stop_triggered:
        phase_name, phase_cfg = get_phase(step, train_cfg)

        if phase_name != current_phase:
            current_phase = phase_name
            current_seq_len = phase_cfg["seq_len"]
            grad_accum = phase_cfg.get(
                "grad_accum_steps", train_cfg.grad_accum_steps,
            )
            current_batch_size = phase_cfg.get(
                "batch_size", train_cfg.batch_size,
            )

            print(
                f"\n>>> Phase: {current_phase} | seq_len: {current_seq_len} | "
                f"batch: {current_batch_size} | accum: {grad_accum} | "
                f"Step: {step}"
            )
            print(
                f"    Datasets: {[d['name'] for d in phase_cfg['datasets']]}"
            )

            if current_loader is not None:
                # Tear down the previous phase's loader BEFORE building the
                # new one. The iterator owns the loader, the loader owns
                # the worker processes; if we just rebind both vars in
                # sequence (assigning current_loader first, then
                # loader_iter), the old workers only get reaped when
                # loader_iter is overwritten — which happens *after* the
                # new workers have already spawned, racing teardown
                # against startup. With persistent_workers=True +
                # multiprocessing_context="spawn" + live HF streaming
                # connections (aiohttp/fsspec), that race manifested at
                # the foundation→knowledge transition (step 10000) as
                # "Fatal Python error: PyGILState_Release". Drop both
                # refs and force a GC pass so old workers fully exit
                # before any new ones come up.
                import gc
                loader_iter = None
                current_loader = None
                gc.collect()
            load_t = time.time()
            current_loader = make_loader(
                phase_cfg["datasets"],
                current_seq_len,
                tokenizer_name,
                current_batch_size,
                step,
            )
            loader_iter = iter(current_loader)
            print(f"    DataLoader ready in {time.time() - load_t:.1f}s")
        else:
            grad_accum = phase_cfg.get(
                "grad_accum_steps", train_cfg.grad_accum_steps,
            )
            current_batch_size = phase_cfg.get(
                "batch_size", train_cfg.batch_size,
            )

        # Schedule writes per-group LRs honouring _peak_lr / _min_lr
        # tags, returning the AdamW/Lion-scale LR for stdout/W&B logs.
        lr = _set_param_group_lrs(optimizer, step, train_cfg)
        router_gumbel_tau = get_router_gumbel_tau(step, train_cfg)
        set_router_gumbel_tau(model, router_gumbel_tau)

        # Gradient checkpointing: on only for very long seq_len. The
        # original threshold was 4096 (Phase 2's seq_len) but H100 80GB
        # only fills ~14 GB at Phase 2 sizes (batch 4 × accum 16 ×
        # seq 4096), so checkpointing was throwing away ~50% of
        # throughput for memory headroom we don't need. Bumped to 8192
        # so Phase 3 still gets the memory relief while Phase 2 runs at
        # full speed. If activation memory ever crowds the H100 budget
        # at seq_len 4096 (e.g. larger batch on H200 141GB), raise
        # batch_size in train_config first; only revert this threshold
        # if that doesn't fit either.
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        base = inner.model if hasattr(inner, "model") else inner
        need_ckpt = current_seq_len >= 8192
        if (hasattr(base, "gradient_checkpointing")
                and base.gradient_checkpointing != need_ckpt):
            base.gradient_checkpointing = need_ckpt

        optimizer.zero_grad(set_to_none=True)
        accum_task_loss = torch.tensor(0.0, device=device)
        accum_balance_norm = torch.tensor(0.0, device=device)
        # Accumulate MoE telemetry across all grad_accum micro-batches so
        # the per-step metrics (and the 5k gate) average over the full
        # effective batch instead of reading only the last micro-batch.
        moe_snapshots: list[dict[str, float]] = []

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
                current_loader = make_loader(
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
            # Pull separated components from the unwrapped model for clean
            # logging (total loss includes aux; these are the components).
            if inner.last_task_loss is not None:
                accum_task_loss += (
                    inner.last_task_loss.detach() / grad_accum
                )
            if inner.last_balance_loss_normalised is not None:
                accum_balance_norm += (
                    inner.last_balance_loss_normalised.detach() / grad_accum
                )

            # Snapshot per-micro-batch MoE telemetry. Cheap — the last_*
            # lists are already Python floats, and reading moe_gate/
            # balance_loss involves a handful of .item() CPU syncs.
            micro_metrics, _ = _collect_moe_metrics(model)
            moe_snapshots.append(micro_metrics)

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        apply_router_balance_updates(model)

        # Average snapshots once per step. Used for both logging and the
        # early-stop gate so both see the same grad-accum-averaged values.
        moe_metrics, moe_summary = _average_moe_snapshots(moe_snapshots)

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
            steps_done = max(step - start_step, 1)
            tok_per_sec = eff_batch * current_seq_len / max(
                elapsed / steps_done, 1e-8,
            )

            # moe_metrics / moe_summary are already computed above as the
            # average over grad_accum micro-batches. No need to re-collect.

            print(
                f"step {step:>7d}/{train_cfg.total_steps} | "
                f"task {accum_task_loss.item():.4f} | "
                f"bal {accum_balance_norm.item():.4f} | "
                f"lr {lr:.2e} | gumbel {router_gumbel_tau:.3f} | "
                f"vram {vram_gb:.1f}GB | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"phase {current_phase} | seq_len {current_seq_len}",
                flush=True,
            )
            print(
                f"           moe: "
                f"pte={moe_summary['per_token_H']:.3f} "
                f"marg={moe_summary['marginal_H']:.3f} "
                f"assn={moe_summary['assign_H']:.3f} "
                f"raw_max={moe_summary['raw_max']:.3f} "
                f"margin={moe_summary['top_margin']:.3f} "
                f"drop={moe_summary['drop_rate']:.4f} "
                f"gate={moe_summary['moe_gate']:.3f} "
                f"bias={moe_summary['bias_abs_max']:.3f} "
                f"emax={moe_summary['expert_max']:.3f} "
                f"emin={moe_summary['expert_min']:.3f} "
                f"bal={moe_summary['balance_loss']:.3f}",
                flush=True,
            )
            print(
                f"           clean: "
                f"pte={moe_summary['clean_per_token_H']:.3f} "
                f"marg={moe_summary['clean_marginal_H']:.3f} "
                f"raw_max={moe_summary['clean_raw_max']:.3f} "
                f"margin={moe_summary['clean_top_margin']:.3f} "
                f"emax={moe_summary['clean_expert_max']:.3f} "
                f"emin={moe_summary['clean_expert_min']:.3f}",
                flush=True,
            )
            print(
                f"           prebias: "
                f"pte={moe_summary['prebias_per_token_H']:.3f} "
                f"marg={moe_summary['prebias_marginal_H']:.3f} "
                f"raw_max={moe_summary['prebias_raw_max']:.3f} "
                f"margin={moe_summary['prebias_top_margin']:.3f} "
                f"emax={moe_summary['prebias_expert_max']:.3f} "
                f"emin={moe_summary['prebias_expert_min']:.3f}",
                flush=True,
            )

            if use_wandb:
                log_dict = {
                    "train/task_loss": accum_task_loss.item(),
                    "train/balance_loss_normalised": accum_balance_norm.item(),
                    "train/lr": lr,
                    "moe/gumbel_tau": router_gumbel_tau,
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

        # --- Eval on held-out FineWeb-Edu ---
        if step > 0 and step % train_cfg.eval_interval == 0:
            eval_metrics = run_eval(
                model, tokenizer_name, current_seq_len,
                current_batch_size, train_cfg.eval_steps,
                device, model_config.real_vocab_size,
            )
            print(
                f"  EVAL step {step} | "
                f"loss {eval_metrics['eval/loss']:.4f} | "
                f"ppl {eval_metrics['eval/perplexity']:.1f}",
                flush=True,
            )
            if use_wandb:
                wandb.log(eval_metrics, step=step)

        # --- Phase-1 early-stop check (router health) ---
        # MUST run BEFORE the numbered checkpoint save on the same step.
        # Otherwise a failed run writes osrt_v5_step_N.pt and a later launch
        # would resume past the gate and ignore the failure diagnosis.
        # On failure we save a DIFFERENT filename (osrt_v5_failed_step_N.pt)
        # that the resume scanner explicitly ignores.
        if step == train_cfg.early_stop_check_step:
            # Use the grad-accum-averaged summary from this step — not a
            # single-batch snapshot — so the gate isn't tripped by sample
            # noise on a 16k-token micro-batch.
            failures = _check_early_stop_criteria(
                step, moe_summary, train_cfg, model_config,
            )
            if failures:
                print(
                    f"\n>>> EARLY STOP at step {step}: "
                    f"router-health criteria failed:",
                    flush=True,
                )
                for f in failures:
                    print(f"      - {f}")
                print(
                    "\n  The v5 architecture bets are not paying off on "
                    "this run. Saving failed-state checkpoint (not "
                    "auto-resumable) and exiting so compute isn't wasted. "
                    "Review telemetry and consider architecture changes "
                    "before retrying.",
                    flush=True,
                )
                failed_path = f"{ckpt_dir}/osrt_v5_failed_step_{step}.pt"
                save_checkpoint(model, optimizer, step, failed_path)
                vol.commit()
                early_stop_triggered = True
                break
            else:
                print(
                    f"\n>>> Router health check at step {step}: "
                    f"all criteria PASS. Continuing training.",
                    flush=True,
                )

        # --- Checkpoints (numbered, resumable) ---
        # Runs AFTER early-stop check so failed runs never produce a
        # step_N.pt that would bypass the gate on resume.
        if step > 0 and step % train_cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v5_step_{step}.pt"
            save_checkpoint(model, optimizer, step, path)
            vol.commit()

        # --- 23h Modal safety (rescue checkpoint + clean exit) ---
        # Rescue filename includes the step so resume scanner can rank it
        # against numbered checkpoints.
        if time.time() - start_time > 82_800:
            rescue_path = f"{ckpt_dir}/osrt_v5_rescue_step_{step}.pt"
            save_checkpoint(model, optimizer, step, rescue_path)
            vol.commit()
            print(
                f"\n23h boundary reached at step {step}. "
                f"Rescue checkpoint saved; exiting cleanly for resume.",
                flush=True,
            )
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # Final checkpoint (full run completed or early stopped)
    if not early_stop_triggered:
        final_path = f"{ckpt_dir}/osrt_v5_final.pt"
        save_checkpoint(model, optimizer, step, final_path)
        vol.commit()
        elapsed_total = time.time() - start_time
        print(
            f"\nPretrain complete. {step:,} steps in "
            f"{elapsed_total / 3600:.1f}h",
            flush=True,
        )
        print(f"Final checkpoint: {final_path}", flush=True)
    if use_wandb:
        wandb.finish()


# ============================================================================
# CONTINUED PRE-TRAINING ("EXTEND" / MID-TRAINING)
# ============================================================================


def _freeze_hra_params(model: nn.Module) -> tuple[int, int]:
    """Freeze every HRA adapter parameter on a (possibly compiled) model.

    HRA wraps Linear layers with a parallel `adapter_a @ adapter_b`
    pair (singular, see hra.py::HRALinear). State-dict names end in
    ".adapter_a" / ".adapter_b" — e.g. "model.blocks.0.qkv.adapter_a".

    CRITICAL: do NOT use substring `"adapter_a" in name` — that also
    matches the recursive-architecture loop adapters at "model.adapters_a"
    (plural, ModuleList from config.adapter_rank=16). Endswith with the
    leading "." disambiguates: HRA uses `.adapter_a` singular, loop
    adapters are `model.adapters_a.<i>` plural. A previous version of
    this function used the substring check and only froze 884k params
    (the loop adapters!), leaving the 86.1M HRA tensors fully trainable
    despite "Frozen HRA" log messages — the bug was caught during
    validation by counting trainable params before/after.

    Setting requires_grad=False excludes them from both Muon and
    AdamW optimizer groups (build_param_groups in muon.py and the
    AdamW path in run_training both check requires_grad).

    Returns (n_frozen_params, n_frozen_tensors) for logging.
    """
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_params = 0
    n_tensors = 0
    for name, p in inner.named_parameters():
        if name.endswith(".adapter_a") or name.endswith(".adapter_b"):
            p.requires_grad = False
            n_params += p.numel()
            n_tensors += 1
    return n_params, n_tensors


def run_pretrain_extend(
    model_config: NanoOSRTConfig,
    extend_cfg,
    vol,
    tokenizer_name: str,
    ckpt_dir: str = "/vol/checkpoints/v5",
) -> None:
    """Continued pre-training on top of an SFT checkpoint.

    Differs from `run_training` in four bounded ways:
      1. Initial weights come from `extend_cfg.pretrained_checkpoint`
         (an SFT ckpt with HRA params), not from a glob-scan.
      2. HRA structure is injected before load so SFT-trained HRA
         tensors land in their slots correctly.
      3. HRA params are frozen post-load if `hra_frozen=True` (the
         default for this stage).
      4. Checkpoints are written under `osrt_v5_{stage_prefix}_step_*.pt`
         so they don't collide with base pretrain checkpoints under
         the resume scanner.

    Held-out validation pass is intentionally skipped for this stage —
    the run is short (1,800 steps), training task_loss + MoE telemetry
    give continuous monitoring, and the held-out cache builder eats
    10-20 min of first-call time we don't want to burn here. If a
    longer extend run is ever needed, lift the validation block from
    run_training.
    """
    device = torch.device("cuda")

    print("=" * 60)
    print("NanoOSRT — Continued Pre-training (Extend / Mid-training)")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Build model ─────────────────────────────────────────────────
    model = NanoOSRTForCausalLM(model_config).to(device=device)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"Base parameters     : {base_params:>12,}")
    print(f"Resume from         : {extend_cfg.pretrained_checkpoint}")
    print(f"Stage prefix        : {extend_cfg.stage_prefix}")
    print(f"Total steps         : {extend_cfg.total_steps}")
    print(f"Peak LR             : {extend_cfg.peak_lr}")
    print(f"HRA enabled         : {extend_cfg.hra_enabled}")
    print(f"HRA frozen          : {getattr(extend_cfg, 'hra_frozen', False)}")
    print()

    # ── HRA injection (BEFORE state_dict load) ─────────────────────
    if extend_cfg.hra_enabled:
        from nano_osrt.hra import inject_hra
        print(f"Injecting HRA before load (rank={extend_cfg.hra_rank})...")
        inject_hra(
            model,
            rank=extend_cfg.hra_rank,
            scale=getattr(extend_cfg, "hra_scale", 1.0),
            freeze_pretrained=False,
        )
        with_hra_params = sum(p.numel() for p in model.parameters())
        added = with_hra_params - base_params
        print(f"  HRA injected: +{added:,} params ({added / 1e6:.1f}M)")

    # ── Load SFT checkpoint ────────────────────────────────────────
    ckpt_path = extend_cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"pretrain_extend refuses to start: pretrained checkpoint "
            f"not found at {ckpt_path}. Run SFT first or correct the "
            f"`pretrained_checkpoint` path in the extend config.",
        )
    print(f"Loading initial weights from {ckpt_path}...")
    init_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    init_state = init_ckpt.get("model_state_dict", init_ckpt)
    load_model_state_or_raise(
        model,
        init_state,
        context=f"pretrain_extend initial load from {ckpt_path}",
    )
    print("  Clean load: all keys matched.")

    # ── Freeze HRA after load ──────────────────────────────────────
    if extend_cfg.hra_enabled and getattr(extend_cfg, "hra_frozen", False):
        n_frozen_params, n_frozen_tensors = _freeze_hra_params(model)
        trainable_after = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f"Froze {n_frozen_tensors} HRA tensors "
            f"({n_frozen_params:,} params, "
            f"{n_frozen_params / 1e6:.1f}M). "
            f"Trainable: {trainable_after:,} "
            f"({trainable_after / 1e6:.1f}M)",
        )

    # `compile_enabled` opt-out lets a stage skip torch.compile entirely.
    # Eager is ~2-3× slower per step but starts emitting step events
    # immediately — useful for sanity stages and for workspaces where
    # silent compile time triggers Modal's idle-cancellation heuristic.
    if getattr(extend_cfg, "compile_enabled", True):
        print("\nCompiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model)
        print(f"Model compile done in {time.time() - compile_start:.1f}s")
    else:
        print("\nSkipping torch.compile (compile_enabled=False, eager mode).")

    # ── W&B ────────────────────────────────────────────────────────
    use_wandb = extend_cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": extend_cfg.wandb_project,
            "name": extend_cfg.wandb_run_name,
            "config": {
                "stage": "pretrain_extend",
                "base_params": base_params,
                "hra_enabled": extend_cfg.hra_enabled,
                "hra_frozen": getattr(extend_cfg, "hra_frozen", False),
                "pretrained_checkpoint": extend_cfg.pretrained_checkpoint,
                "peak_lr": extend_cfg.peak_lr,
                "min_lr": extend_cfg.min_lr,
                "warmup_steps": extend_cfg.warmup_steps,
                "total_steps": extend_cfg.total_steps,
                "datasets": [
                    d["name"]
                    for d in extend_cfg.phases["extend"]["datasets"]
                ],
            },
        }
        if extend_cfg.wandb_run_id:
            wandb_kwargs["id"] = extend_cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # ── Optimizer ──────────────────────────────────────────────────
    inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if extend_cfg.optimizer_name.lower() == "muon":
        from nano_osrt.muon import (
            HybridMuonAdamW,
            Muon,
            build_param_groups,
        )
        muon_params, adamw_groups = build_param_groups(
            inner_model.named_parameters(),
            weight_decay=extend_cfg.weight_decay,
        )
        muon_lr = getattr(extend_cfg, "muon_lr", extend_cfg.peak_lr)
        muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=extend_cfg.weight_decay,
        )
        adamw = torch.optim.AdamW(
            adamw_groups,
            lr=extend_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        muon_min = getattr(extend_cfg, "muon_min_lr", muon_lr * 0.1)
        for pg in muon.param_groups:
            pg["_peak_lr"] = muon_lr
            pg["_min_lr"] = muon_min
        for pg in adamw.param_groups:
            pg["_peak_lr"] = extend_cfg.peak_lr
            pg["_min_lr"] = extend_cfg.min_lr
        optimizer = HybridMuonAdamW(muon, adamw)
        n_muon = len(muon_params)
        n_adamw = sum(len(g["params"]) for g in adamw_groups)
        print(
            f"Muon+AdamW hybrid: {n_muon} matrix tensors → Muon "
            f"(lr={muon_lr}), {n_adamw} other tensors → AdamW "
            f"(lr={extend_cfg.peak_lr})",
        )
    else:
        router_params = []
        other_params = []
        for name, param in inner_model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name or "loop_embeddings" in name:
                router_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": extend_cfg.weight_decay},
                {"params": router_params, "weight_decay": 0.0},
            ],
            lr=extend_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        for pg in optimizer.param_groups:
            pg["_peak_lr"] = extend_cfg.peak_lr
            pg["_min_lr"] = extend_cfg.min_lr
        print(f"AdamW (lr={extend_cfg.peak_lr}, wd={extend_cfg.weight_decay})")

    # ── Resume from prior extend checkpoints ────────────────────────
    prefix = extend_cfg.stage_prefix
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
        load_model_state_or_raise(
            model,
            ckpt["model_state_dict"],
            context=f"{prefix} resume from {best_ckpt}",
        )
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, RuntimeError) as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
        start_step = ckpt["step"] + 1
        print(f"Resumed at step {start_step}")

    # ── Training loop (single phase, no curriculum, no held-out val) ──
    start_time = time.time()
    step = start_step
    extend_phase = extend_cfg.phases["extend"]
    seq_len = extend_phase["seq_len"]
    batch_size = extend_phase["batch_size"]
    grad_accum = extend_phase["grad_accum_steps"]

    print(
        f"\n>>> Extend phase | seq_len: {seq_len} | "
        f"batch: {batch_size} | accum: {grad_accum} | "
        f"effective batch: {batch_size * grad_accum}",
    )
    print(f"    Datasets: {[d['name'] for d in extend_phase['datasets']]}")

    load_t = time.time()
    loader = make_loader(
        extend_phase["datasets"],
        seq_len,
        tokenizer_name,
        batch_size,
        step,
        num_workers=getattr(extend_cfg, "dataloader_num_workers", 4),
        prefetch_factor=getattr(extend_cfg, "dataloader_prefetch_factor", 4),
    )
    loader_iter = iter(loader)
    print(f"    DataLoader ready in {time.time() - load_t:.1f}s")

    # Activation checkpointing only for very long seq.
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    need_ckpt = seq_len >= 8192
    if (hasattr(base, "gradient_checkpointing")
            and base.gradient_checkpointing != need_ckpt):
        base.gradient_checkpointing = need_ckpt

    # Gumbel buffer fill (no-op since extend_cfg sets tau init = 0).
    set_router_gumbel_tau(model, extend_cfg.router_gumbel_tau_init)

    while step < extend_cfg.total_steps:
        lr = _set_param_group_lrs(optimizer, step, extend_cfg)
        optimizer.zero_grad(set_to_none=True)

        accum_task_loss = torch.tensor(0.0, device=device)
        accum_balance_norm = torch.tensor(0.0, device=device)
        moe_snapshots: list[dict[str, float]] = []

        if step == start_step:
            print("Fetching first batch...")
            batch_t = time.time()

        for micro in range(grad_accum):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                # Loader exhausted — rebuild with a different seed.
                import gc
                loader_iter = None
                del loader
                gc.collect()
                loader = make_loader(
                    extend_phase["datasets"],
                    seq_len,
                    tokenizer_name,
                    batch_size,
                    step,
                )
                loader_iter = iter(loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if step == start_step and micro == 0:
                print(f"First batch fetched in {time.time() - batch_t:.1f}s")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum
            loss.backward()

            if inner.last_task_loss is not None:
                accum_task_loss += (
                    inner.last_task_loss.detach() / grad_accum
                )
            if inner.last_balance_loss_normalised is not None:
                accum_balance_norm += (
                    inner.last_balance_loss_normalised.detach() / grad_accum
                )
            micro_metrics, _ = _collect_moe_metrics(model)
            moe_snapshots.append(micro_metrics)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), extend_cfg.grad_clip,
        )
        optimizer.step()
        apply_router_balance_updates(model)

        moe_metrics, moe_summary = _average_moe_snapshots(moe_snapshots)

        # ── Logging ────────────────────────────────────────────────
        should_log = (
            step % extend_cfg.log_interval == 0
            or step == 0
            or (step < 100 and step % 10 == 0)
        )
        if should_log:
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            eff_batch = batch_size * grad_accum
            steps_done = max(step - start_step, 1)
            tok_per_sec = eff_batch * seq_len / max(
                elapsed / steps_done, 1e-8,
            )
            print(
                f"step {step:>5d}/{extend_cfg.total_steps} | "
                f"task {accum_task_loss.item():.4f} | "
                f"bal {accum_balance_norm.item():.4f} | "
                f"lr {lr:.2e} | vram {vram_gb:.1f}GB | "
                f"tok/s {tok_per_sec:,.0f}",
                flush=True,
            )
            print(
                f"           moe: pte={moe_summary['per_token_H']:.3f} "
                f"assn={moe_summary['assign_H']:.3f} "
                f"drop={moe_summary['drop_rate']:.4f} "
                f"gate={moe_summary['moe_gate']:.3f}",
                flush=True,
            )
            if use_wandb:
                log_dict = {
                    "extend/task_loss": accum_task_loss.item(),
                    "extend/balance_loss_normalised": accum_balance_norm.item(),
                    "extend/lr": lr,
                    "extend/vram_gb": vram_gb,
                    "extend/tok_per_sec": tok_per_sec,
                }
                log_dict.update(moe_metrics)
                wandb.log(log_dict, step=step)
        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()

        # ── Numbered checkpoints ───────────────────────────────────
        if step > 0 and step % extend_cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v5_{prefix}_step_{step}.pt"
            save_checkpoint(model, optimizer, step, path)
            vol.commit()

        # ── 23h Modal safety rescue ────────────────────────────────
        if time.time() - start_time > 82_800:
            rescue_path = (
                f"{ckpt_dir}/osrt_v5_{prefix}_rescue_step_{step}.pt"
            )
            save_checkpoint(model, optimizer, step, rescue_path)
            vol.commit()
            print(
                f"\n23h boundary reached at step {step}. "
                f"Rescue checkpoint saved; exiting cleanly for resume.",
                flush=True,
            )
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # Final checkpoint
    final_path = f"{ckpt_dir}/osrt_v5_{prefix}_final.pt"
    save_checkpoint(model, optimizer, step, final_path)
    vol.commit()
    elapsed_total = time.time() - start_time
    print(
        f"\n{prefix} complete. {step:,} steps in "
        f"{elapsed_total / 3600:.1f}h",
        flush=True,
    )
    print(f"Final checkpoint: {final_path}", flush=True)
    if use_wandb:
        wandb.finish()
