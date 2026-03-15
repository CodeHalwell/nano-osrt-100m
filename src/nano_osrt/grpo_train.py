"""GRPO training loop for nano-osrt-100m.

Group Relative Policy Optimization: generate multiple completions per prompt,
score with verifiable rewards, train policy to favor high-reward completions.
No reward model needed — uses rule-based math verification.
"""

import copy
import glob
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from nano_osrt.grpo_config import GRPOConfig
from nano_osrt.hra import get_param_groups, inject_hra
from nano_osrt.recursive_model import RecursiveNanoOSRT
from nano_osrt.rewards import compute_group_advantages, compute_reward


def get_grpo_lr(step: int, cfg: GRPOConfig) -> float:
    """Cosine LR with linear warmup for GRPO."""
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def generate_completions(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    cfg: GRPOConfig,
    tokenizer,
) -> list[torch.Tensor]:
    """Generate group_size completions for a single prompt.

    Uses top-p sampling at the configured temperature.
    Returns a list of token tensors (prompt + completion).
    """
    device = prompt_ids.device
    completions = []

    for _ in range(cfg.group_size):
        generated = prompt_ids.clone()  # (1, prompt_len)

        for _t in range(cfg.max_gen_len):
            # Truncate to seq_len if needed
            input_seq = generated[:, -cfg.seq_len :]
            logits, _ = model(input_seq)
            next_logits = logits[:, -1, : cfg.real_vocab_size].float()

            # Temperature + top-p sampling
            next_logits = next_logits / cfg.temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob above top_p
            sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= cfg.top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back and sample
            next_logits.scatter_(1, sorted_indices, sorted_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

        completions.append(generated.squeeze(0))  # (seq_len,)

    return completions


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    real_vocab_size: int,
) -> torch.Tensor:
    """Compute per-token log probs for the completion portion only.

    Returns log_probs tensor of shape (completion_len,).
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids.unsqueeze(0))
        logits = logits[0, :, :real_vocab_size].float()  # (S, V)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[prompt_len - 1 : -1]  # (completion_len, V)
    shift_labels = input_ids[prompt_len:]  # (completion_len,)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    return token_log_probs


def save_grpo_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Save a GRPO checkpoint."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_stage": "grpo",
        },
        path,
    )
    print(f"  -> GRPO checkpoint saved: {path}")


def load_grpo_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    """Resume GRPO from checkpoint. Returns step to resume from."""
    if not os.path.exists(path):
        return 0
    print(f"Resuming GRPO from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"Resumed GRPO at step {start_step}")
    return start_step


def run_grpo(cfg: GRPOConfig, vol, tokenizer_name: str) -> None:
    """Execute the GRPO training loop on Modal."""
    device = torch.device("cuda")

    print("=" * 60)
    print("Nano-OSRT 100M — GRPO Training")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = RecursiveNanoOSRT(cfg).to(device=device)
    base_params = sum(p.numel() for p in model.parameters())

    print(f"Base parameters     : {base_params:>12,}")
    print(f"GRPO learning rate  : {cfg.peak_lr}")
    print(f"Group size          : {cfg.group_size}")
    print(f"Max gen length      : {cfg.max_gen_len}")
    print(f"KL coefficient      : {cfg.kl_coeff}")
    print(f"Clip range          : {cfg.clip_range}")
    print(f"Total GRPO steps    : {cfg.total_steps}")
    print()

    # Inject HRA adapters (must happen before loading SFT weights)
    hra_params = []
    if cfg.hra_enabled:
        print(f"Injecting HRA adapters (rank={cfg.hra_rank})...")
        hra_params = inject_hra(
            model,
            rank=cfg.hra_rank,
            scale=cfg.hra_scale,
            freeze_pretrained=cfg.hra_freeze_pretrained,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters  : {total_params:>12,} (+{total_params - base_params:,} HRA)")
    else:
        total_params = base_params

    # Load SFT weights (includes HRA adapter weights from SFT)
    sft_path = cfg.pretrained_checkpoint
    if not os.path.exists(sft_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {sft_path}. Run SFT first."
        )
    print(f"Loading SFT weights from {sft_path}...")
    ckpt = torch.load(sft_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    print("  SFT weights loaded.")

    # Reference model (frozen copy — includes HRA weights)
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("  Reference model ready.")

    # Compile policy model (not ref — ref is inference only)
    print("Compiling policy model...")
    compile_start = time.time()
    model = torch.compile(model)
    print(f"  Compile done in {time.time() - compile_start:.1f}s")

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": cfg.wandb_project,
            "name": cfg.wandb_run_name,
            "config": {
                "stage": "grpo",
                "total_params": total_params,
                "peak_lr": cfg.peak_lr,
                "group_size": cfg.group_size,
                "kl_coeff": cfg.kl_coeff,
                "clip_range": cfg.clip_range,
                "total_steps": cfg.total_steps,
            },
        }
        if cfg.wandb_run_id:
            wandb_kwargs["id"] = cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if cfg.hra_enabled and hra_params:
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
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.peak_lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    # ------------------------------------------------------------------
    # Resume from GRPO checkpoint if available
    # ------------------------------------------------------------------
    grpo_rescue_path = "/vol/checkpoints/osrt100m_grpo_rescue.pt"
    ckpt_dir = "/vol/checkpoints"
    best_ckpt = grpo_rescue_path
    best_step = -1

    if os.path.isdir(ckpt_dir):
        for f in glob.glob(os.path.join(ckpt_dir, "osrt100m_grpo_step_*.pt")):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
                if s > best_step:
                    best_step = s
                    best_ckpt = f
            except (ValueError, IndexError):
                continue

    start_step = 0
    if best_step > 0:
        print(f"Found GRPO checkpoint at step {best_step}: {best_ckpt}")
        start_step = load_grpo_checkpoint(model, optimizer, best_ckpt, device)

    # ------------------------------------------------------------------
    # Prompt dataset (GSM8K questions)
    # ------------------------------------------------------------------
    print("\nLoading prompt dataset...")
    load_kwargs = {"split": cfg.prompt_split, "streaming": True}
    if cfg.prompt_config:
        load_kwargs["name"] = cfg.prompt_config
    prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
    prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42 + start_step)
    prompt_iter = iter(prompt_ds)
    print("  Prompt stream ready.")

    # ------------------------------------------------------------------
    # GRPO Training loop
    # ------------------------------------------------------------------
    start_time = time.time()
    step = start_step

    while step < cfg.total_steps:
        lr = get_grpo_lr(step, cfg)
        for pg in optimizer.param_groups:
            if cfg.hra_enabled and pg.get("group_name") == "hra":
                hra_ratio = cfg.hra_lr / cfg.peak_lr
                pg["lr"] = lr * hra_ratio
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_rewards = []
        step_correct = 0
        step_total = 0
        step_kl = 0.0

        for _accum in range(cfg.grad_accum_steps):
            # Get a prompt
            try:
                example = next(prompt_iter)
            except StopIteration:
                prompt_ds = load_dataset(
                    cfg.prompt_dataset, split=cfg.prompt_split, streaming=True
                )
                prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42 + step)
                prompt_iter = iter(prompt_ds)
                example = next(prompt_iter)

            question = example["question"]
            ground_truth = example["answer"]

            # Format prompt
            prompt_text = f"{cfg.user_prefix}{question}\n{cfg.assistant_prefix}"
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor(
                [prompt_ids], dtype=torch.long, device=device
            )
            prompt_len = len(prompt_ids)

            # Generate completions
            model.eval()
            completions = generate_completions(model, prompt_tensor, cfg, tokenizer)
            model.train()

            # Score completions
            rewards = []
            for comp_ids in completions:
                comp_text = tokenizer.decode(
                    comp_ids[prompt_len:].tolist(), skip_special_tokens=True
                )
                reward, breakdown = compute_reward(
                    comp_text,
                    ground_truth,
                    correctness_weight=cfg.correctness_reward,
                    format_weight=cfg.format_reward,
                    length_penalty=cfg.length_penalty,
                    think_open=cfg.think_open,
                    think_close=cfg.think_close,
                )
                rewards.append(reward)
                if breakdown["correct"]:
                    step_correct += 1
                step_total += 1

            step_rewards.extend(rewards)

            # Compute group advantages
            advantages = compute_group_advantages(rewards)

            # Policy gradient with KL penalty
            for comp_ids, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue  # skip zero-advantage completions

                # Truncate to seq_len
                comp_ids = comp_ids[: cfg.seq_len].to(device)
                comp_len = len(comp_ids) - prompt_len

                if comp_len <= 0:
                    continue

                # Get policy log probs
                policy_log_probs = compute_log_probs(
                    model, comp_ids, prompt_len, cfg.real_vocab_size
                )

                # Get reference log probs
                with torch.no_grad():
                    ref_log_probs = compute_log_probs(
                        ref_model, comp_ids, prompt_len, cfg.real_vocab_size
                    )

                # KL divergence (per-token)
                kl = (policy_log_probs - ref_log_probs).mean()
                step_kl += kl.item()

                # GRPO objective: maximize advantage-weighted log prob - KL penalty
                # With PPO-style clipping
                ratio = torch.exp(policy_log_probs - ref_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
                )
                adv_tensor = torch.tensor(adv, device=device, dtype=torch.float32)

                surrogate1 = ratio * adv_tensor
                surrogate2 = clipped_ratio * adv_tensor
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                kl_loss = cfg.kl_coeff * kl

                loss = (policy_loss + kl_loss) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()

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

            mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
            accuracy = step_correct / step_total if step_total > 0 else 0
            mean_kl = step_kl / max(step_total, 1)

            print(
                f"step {step:>6d}/{cfg.total_steps} | "
                f"loss {step_loss:.4f} | reward {mean_reward:.3f} | "
                f"acc {accuracy:.1%} | kl {mean_kl:.4f} | "
                f"lr {lr:.2e} | vram {vram_gb:.1f}GB | "
                f"elapsed {elapsed:.0f}s"
            )

            if use_wandb:
                wandb.log(
                    {
                        "grpo/loss": step_loss,
                        "grpo/mean_reward": mean_reward,
                        "grpo/accuracy": accuracy,
                        "grpo/kl_divergence": mean_kl,
                        "grpo/lr": lr,
                        "grpo/vram_gb": vram_gb,
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
            path = f"/vol/checkpoints/osrt100m_grpo_step_{step}.pt"
            save_grpo_checkpoint(model, optimizer, step, path)
            vol.commit()

        # --- 23h Modal safety ---
        if time.time() - start_time > 82_800:
            save_grpo_checkpoint(model, optimizer, step, grpo_rescue_path)
            vol.commit()
            print(f"\n23h boundary. GRPO rescue checkpoint at step {step}.")
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # --- Final ---
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = "/vol/checkpoints/osrt100m_grpo_final.pt"
    torch.save(
        {
            "model_state_dict": inner.state_dict(),
            "training_stage": "grpo",
            "total_steps": cfg.total_steps,
        },
        final_path,
    )
    vol.commit()
    elapsed_total = time.time() - start_time
    print(f"\nGRPO complete. {step:,} steps in {elapsed_total / 3600:.1f}h")
    print(f"Final GRPO model: {final_path}")
    if use_wandb:
        wandb.finish()
