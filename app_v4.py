"""NanoOSRT v4 — Modal deployment entrypoint.

~306M physical params (32K vocab + 1536 dim), ~130M active/token,
~1.8B effective via recursive weight sharing.
3 physical blocks × 6 loops = 18 effective layers.
Dense FFN + MoE (1 shared + 11 routed experts, top-2) in parallel residual.

Stages:
    modal run app_v4.py --stage tokenizer    Train custom 32K BPE tokenizer
    modal run app_v4.py --stage pretrain     Pre-training (progressive seq_len)
    modal run app_v4.py --stage sft          Balanced SFT (math + code + STEM + general)
    modal run app_v4.py --stage grpo         GRPO reinforcement learning
    modal run app_v4.py --stage eval         Benchmark evaluation
"""

import modal

# =============================================================================
# MODAL INFRASTRUCTURE
# =============================================================================

app = modal.App("nano-osrt-v4")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"TORCH_LOGS": "perf_hints"})
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers", "datasets", "lion-pytorch", "triton", "wandb",
        "tokenizers", "sentencepiece", "safetensors",
    )
    .pip_install("lm-eval", "langdetect", "immutabledict")
    .add_local_dir("src/nano_osrt", remote_path="/root/nano_osrt")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

vol = modal.Volume.from_name("osrt-v4-checkpoints", create_if_missing=True)
tokenizer_vol = modal.Volume.from_name("osrt-v4-tokenizer", create_if_missing=True)


# =============================================================================
# TOKENIZER TRAINING
# =============================================================================


@app.function(
    gpu="A100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=14400,  # 4 hours
)
def train_tokenizer():
    """Train custom 32K BPE tokenizer on pre-training data mix."""
    import sys
    sys.path.insert(0, "/root")

    from scripts.train_tokenizer import sample_training_data, train_with_hf_tokenizers

    print("=" * 60)
    print("NanoOSRT v4 — Custom 32K Tokenizer Training")
    print("=" * 60)

    # Sample 10GB of training data (proportional to pre-training mix)
    print("\nSampling training data...")
    data_path = sample_training_data(sample_size=10_000_000_000)

    # Train tokenizer
    output_dir = "/vol/tokenizer"
    train_with_hf_tokenizers(data_path, vocab_size=32768, output_dir=output_dir)

    # Cleanup temp file
    import os
    os.remove(data_path)

    print("\nTokenizer saved to Modal volume 'osrt-v4-tokenizer'")


# =============================================================================
# PRE-TRAINING
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def pretrain():
    """Run v4 pre-training with progressive seq_len curriculum."""
    # Force reload volume to avoid stale cache
    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.v4_config import NanoOSRTv4Config
    from nano_osrt.v4_train import run_v4_training
    from nano_osrt.v4_train_config import V4PretrainConfig
    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    # Load custom tokenizer
    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

    # Debug: list what's on the volume
    import os
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    # Sanity check — if tokenizer is wrong size, warn loudly
    expected_vocab = 32768
    if len(tok) != expected_vocab:
        print(f"WARNING: Expected {expected_vocab} vocab but got {len(tok)}!")
        print("  Retrain tokenizer: modal run app_v4.py --stage tokenizer")

    model_config = NanoOSRTv4Config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    train_cfg = V4PretrainConfig()

    total_params = (
        model_config.vocab_size * model_config.dim  # embedding
        + model_config.num_blocks * (
            model_config.dim * model_config.dim * 4  # attention (approx)
            + model_config.dim * model_config.dense_hidden * 3  # dense FFN
            + (model_config.num_shared_experts + model_config.num_routed_experts)
            * model_config.dim * model_config.expert_hidden * 3  # MoE
        )
    )
    print(f"Estimated parameters: ~{total_params / 1e6:.0f}M")

    run_v4_training(model_config, train_cfg, vol, tokenizer_name)


# =============================================================================
# SFT (BALANCED)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def sft():
    """Run balanced SFT: math + code + STEM + general."""
    from transformers import AutoTokenizer

    from nano_osrt.v4_config import NanoOSRTv4Config
    from nano_osrt.v4_sft_train import run_v4_sft
    from nano_osrt.v4_train_config import V4SFTConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = NanoOSRTv4Config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_cfg = V4SFTConfig()
    run_v4_sft(model_config, sft_cfg, vol, tok)


# =============================================================================
# GRPO (REINFORCEMENT LEARNING)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def grpo():
    """Run GRPO with verifiable math rewards."""
    import copy
    import math
    import os
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        import wandb
    except ImportError:
        wandb = None

    from nano_osrt.hra import get_param_groups, inject_hra
    from nano_osrt.rewards import compute_group_advantages, compute_reward
    from nano_osrt.v4_config import NanoOSRTv4Config
    from nano_osrt.v4_model import NanoOSRTv4ForCausalLM
    from nano_osrt.v4_train_config import V4GRPOConfig

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = V4GRPOConfig()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    print("=" * 60)
    print("NanoOSRT v4 — GRPO Training")
    print("=" * 60)

    model_config = NanoOSRTv4Config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    model = NanoOSRTv4ForCausalLM(model_config).to(device)

    # Inject HRA before loading SFT checkpoint
    hra_params = []
    if cfg.hra_enabled:
        print(f"Injecting HRA (rank={cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=cfg.hra_rank)

    # Load SFT weights
    ckpt_path = cfg.pretrained_checkpoint
    if os.path.exists(ckpt_path):
        print(f"Loading SFT weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("  Loaded.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Group size: {cfg.group_size}")
    print(f"Total steps: {cfg.total_steps}")

    # Reference model
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Compiling policy model...")
    model = torch.compile(model)

    # W&B
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config={"stage": "grpo"})

    # Optimizer
    if hra_params:
        param_groups = get_param_groups(model, hra_params, cfg.peak_lr, cfg.hra_lr, cfg.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr,
                                       weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # Prompt dataset
    print("Loading prompt dataset...")
    load_kwargs = {"split": cfg.prompt_split, "streaming": True}
    if cfg.prompt_config:
        load_kwargs["name"] = cfg.prompt_config
    prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
    prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42)
    prompt_iter = iter(prompt_ds)

    # Training loop
    start_time = time.time()
    ckpt_dir = "/vol/checkpoints/v4"
    os.makedirs(ckpt_dir, exist_ok=True)

    for step in range(cfg.total_steps):
        # LR schedule
        if step < cfg.warmup_steps:
            lr = cfg.peak_lr * step / cfg.warmup_steps
        else:
            progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
            lr = cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            if pg.get("group_name") == "hra":
                pg["lr"] = lr * (cfg.hra_lr / cfg.peak_lr)
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_kl = 0.0
        step_rewards = []
        step_correct = 0
        step_total = 0

        for _accum in range(cfg.grad_accum_steps):
            try:
                example = next(prompt_iter)
            except StopIteration:
                prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
                prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42 + step)
                prompt_iter = iter(prompt_ds)
                example = next(prompt_iter)

            question = example["question"]
            ground_truth = example["answer"]

            prompt_text = f"{cfg.user_tag}{question}{cfg.assistant_tag}"
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)

            # Generate completions
            model.eval()
            completions = []
            for _ in range(cfg.group_size):
                generated = prompt_tensor.clone()
                for _t in range(cfg.max_gen_len):
                    input_seq = generated[:, -cfg.seq_len:]
                    with torch.no_grad():
                        out = model(input_seq)
                        next_logits = out.logits[:, -1, :model_config.real_vocab_size].float()
                    next_logits = next_logits / cfg.temperature
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= cfg.top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    next_logits.scatter_(1, sorted_indices, sorted_logits)
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
                    if next_token.item() == tok.eos_token_id:
                        break
                completions.append(generated.squeeze(0))
            model.train()

            # Score — IMPORTANT: skip_special_tokens=False so native tags
            # like <|think|>, <|answer|> survive decoding for the reward
            # scorer. And explicitly pass the v4 native tag strings so
            # the reward function doesn't fall back to v3 defaults.
            rewards = []
            for comp_ids in completions:
                comp_text = tok.decode(
                    comp_ids[prompt_len:].tolist(),
                    skip_special_tokens=False,
                )
                comp_tokens = len(comp_ids) - prompt_len
                reward, breakdown = compute_reward(
                    comp_text, ground_truth,
                    correctness_weight=cfg.correctness_reward,
                    format_weight=cfg.format_reward,
                    length_penalty=cfg.length_penalty,
                    think_open=cfg.think_open,
                    think_close=cfg.think_close,
                    answer_open=cfg.answer_open,
                    answer_close=cfg.answer_close,
                    max_tokens=cfg.max_gen_len,
                    completion_tokens=comp_tokens,
                    reasoning_bonus=cfg.reasoning_bonus,
                    truncation_penalty=cfg.truncation_penalty,
                    empty_think_penalty=cfg.empty_think_penalty,
                )
                rewards.append(reward)
                if breakdown["correct"]:
                    step_correct += 1
                step_total += 1
            step_rewards.extend(rewards)

            advantages = compute_group_advantages(rewards)

            for comp_ids, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue
                comp_ids = comp_ids[:cfg.seq_len].to(device)
                comp_len = len(comp_ids) - prompt_len
                if comp_len <= 0:
                    continue

                # Policy log probs on the sampled completion
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(comp_ids.unsqueeze(0))
                    logits = out.logits[0, :, :model_config.real_vocab_size].float()
                shift_logits = logits[prompt_len - 1:-1]
                shift_labels = comp_ids[prompt_len:]
                policy_lp = F.log_softmax(shift_logits, dim=-1).gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                # Reference log probs (frozen, no grad)
                with torch.no_grad():
                    ref_out = ref_model(comp_ids.unsqueeze(0))
                    ref_logits = ref_out.logits[
                        0, :, :model_config.real_vocab_size
                    ].float()
                ref_shift = ref_logits[prompt_len - 1:-1]
                ref_lp = F.log_softmax(ref_shift, dim=-1).gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                # Direct policy gradient weighted by group-normalised advantage.
                # Since we perform only one gradient step per sampled batch,
                # importance-sampling ratio ~= 1, so PPO clipping is a no-op
                # here. We keep the formulation simple and correct.
                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -(policy_lp * adv_t).mean()

                # Schulman's unbiased non-negative KL approximation:
                #   approx_kl = exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
                # Always >= 0 (unlike the simple mean(policy_lp - ref_lp) which
                # can go negative and give a bogus "negative KL" penalty).
                log_ratio = ref_lp - policy_lp
                approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()

                loss = (policy_loss + cfg.kl_coeff * approx_kl) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()
                step_kl += approx_kl.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
            accuracy = step_correct / step_total if step_total > 0 else 0
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            mean_kl = step_kl / max(step_total, 1)
            print(f"step {step:>6d}/{cfg.total_steps} | loss {step_loss:.4f} | "
                  f"reward {mean_reward:.3f} | acc {accuracy:.1%} | "
                  f"kl {mean_kl:.4f} | lr {lr:.2e} | "
                  f"vram {vram:.1f}GB | elapsed {elapsed:.0f}s")
            if use_wandb:
                wandb.log({
                    "grpo/loss": step_loss,
                    "grpo/mean_reward": mean_reward,
                    "grpo/accuracy": accuracy,
                    "grpo/approx_kl": mean_kl,
                    "grpo/lr": lr,
                }, step=step)

        # Checkpoints
        if step > 0 and step % cfg.ckpt_interval == 0:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model_state_dict": inner.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       f"{ckpt_dir}/osrt_v4_grpo_step_{step}.pt")
            vol.commit()

        # 23h safety
        if time.time() - start_time > 82_800:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model_state_dict": inner.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       f"{ckpt_dir}/osrt_v4_grpo_rescue.pt")
            vol.commit()
            if use_wandb:
                wandb.finish()
            return

    # Final
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model_state_dict": inner.state_dict(), "training_stage": "grpo"},
               f"{ckpt_dir}/osrt_v4_grpo_final.pt")
    vol.commit()
    print(f"\nGRPO complete. {cfg.total_steps} steps in {(time.time() - start_time) / 3600:.1f}h")
    if use_wandb:
        wandb.finish()


# =============================================================================
# EVALUATION
# =============================================================================


@app.function(
    gpu="A100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=43200,
)
def evaluate(tasks: str = "ifeval", limit: int = 0):
    """Run lm-evaluation-harness benchmarks."""
    import torch
    import torch.nn.functional as F
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from transformers import AutoTokenizer

    from nano_osrt.v4_config import NanoOSRTv4Config
    from nano_osrt.v4_model import NanoOSRTv4ForCausalLM

    class V4EvalModel(LM):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cuda")
            self._batch_size = 1

            tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
            model_config = NanoOSRTv4Config(
                vocab_size=len(tok),
                real_vocab_size=len(tok),
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

            self.model = NanoOSRTv4ForCausalLM(model_config).to("cuda")

            # Resolve latest checkpoint: priority grpo_final > sft_final > final.
            import os

            from nano_osrt.hra import inject_hra

            ckpt_dir = "/vol/checkpoints/v4"
            resolved_path = None
            resolved_stage = None
            for stage, name in [
                ("grpo", "osrt_v4_grpo_final.pt"),
                ("sft", "osrt_v4_sft_final.pt"),
                ("pretrain", "osrt_v4_final.pt"),
            ]:
                path = os.path.join(ckpt_dir, name)
                if os.path.exists(path):
                    resolved_path = path
                    resolved_stage = stage
                    break

            if resolved_path is None:
                raise FileNotFoundError(
                    f"No v4 checkpoint found in {ckpt_dir}. "
                    "Run pretrain/sft/grpo first."
                )

            print(f"Loading {resolved_path} (stage={resolved_stage})...")
            ckpt = torch.load(resolved_path, map_location="cuda", weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # SFT/GRPO checkpoints have HRA-wrapped linear layers (keys like
            # 'blocks.0.qkv.adapter_a', 'blocks.0.qkv.original.weight').
            # Inject HRA into a plain base model so the key names match,
            # otherwise load_state_dict silently drops everything HRA-related.
            if resolved_stage in ("sft", "grpo"):
                has_hra_keys = any(
                    "adapter_a" in k or "adapter_b" in k or "original.weight" in k
                    for k in state_dict
                )
                if has_hra_keys:
                    print("  Detected HRA keys in checkpoint, injecting adapters...")
                    inject_hra(self.model, rank=256)

            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=False
            )
            if missing:
                print(f"  MISSING keys ({len(missing)}): sample {missing[:3]}")
            if unexpected:
                print(f"  UNEXPECTED keys ({len(unexpected)}): sample {unexpected[:3]}")
            if not missing and not unexpected:
                print("  Clean load: all keys matched.")

            self.model.eval()
            self.tokenizer = tok
            self.vocab_size = model_config.real_vocab_size
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        @property
        def eot_token_id(self): return self.tokenizer.eos_token_id
        @property
        def max_length(self): return self.model.config.max_position_embeddings
        @property
        def max_gen_toks(self): return 256
        @property
        def batch_size(self): return self._batch_size
        @property
        def device(self): return self._device

        def tok_encode(self, s, **kw): return self.tokenizer.encode(s, add_special_tokens=False)
        def tok_decode(self, t, **kw): return self.tokenizer.decode(t, skip_special_tokens=True)

        def _model_call(self, inps):
            with torch.no_grad():
                return self.model(inps.to(self._device)).logits[:, :, :self.vocab_size]

        def loglikelihood(self, requests, **kw):
            results = []
            for req in requests:
                ctx, cont = req.args
                ctx_ids = self.tok_encode(ctx)
                cont_ids = self.tok_encode(cont)
                full = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
                if full.shape[1] > self.max_length:
                    full = full[:, -self.max_length:]
                    ctx_len = max(0, len(ctx_ids) - (len(ctx_ids) + len(cont_ids) - self.max_length))
                else:
                    ctx_len = len(ctx_ids)
                logits = self._model_call(full)
                sl = logits[0, ctx_len-1:-1, :]
                labels = full[0, ctx_len:].to(self._device)
                lp = F.log_softmax(sl.float(), dim=-1)
                tlp = lp.gather(1, labels.unsqueeze(1)).squeeze(1)
                results.append((tlp.sum().item(), (sl.argmax(-1) == labels).all().item()))
            return results

        def loglikelihood_rolling(self, requests, **kw):
            results = []
            for req in requests:
                (s,) = req.args
                ids = self.tok_encode(s)
                full = torch.tensor([ids], dtype=torch.long)
                if full.shape[1] > self.max_length:
                    full = full[:, -self.max_length:]
                logits = self._model_call(full)
                sl = logits[0, :-1, :]
                labels = full[0, 1:].to(self._device)
                lp = F.log_softmax(sl.float(), dim=-1)
                tlp = lp.gather(1, labels.unsqueeze(1)).squeeze(1)
                results.append((tlp.sum().item(),))
            return results

        def generate_until(self, requests, **kw):
            results = []
            for i, req in enumerate(requests):
                ctx = req.args[0]
                gen_kw = req.args[1] if len(req.args) > 1 else {}
                until = gen_kw.get("until", [self.tokenizer.eos_token])
                max_gen = gen_kw.get("max_gen_toks", self.max_gen_toks)
                ctx_ids = self.tok_encode(ctx)
                ctx_t = torch.tensor([ctx_ids], dtype=torch.long)
                if ctx_t.shape[1] > self.max_length - max_gen:
                    ctx_t = ctx_t[:, -(self.max_length - max_gen):]
                out = self.model.generate(ctx_t.to(self._device), max_new_tokens=max_gen,
                                          temperature=0.0, eos_token_id=self.tokenizer.eos_token_id)
                resp = self.tok_decode(out[0, ctx_t.shape[1]:].tolist())
                for stop in until:
                    if stop in resp:
                        resp = resp[:resp.index(stop)]
                results.append(resp)
                if (i+1) % 50 == 0:
                    print(f"  Generated {i+1}/{len(requests)}")
            return results

    lm = V4EvalModel()
    task_list = [t.strip() for t in tasks.split(",")]
    print(f"\nRunning: {task_list}")

    eval_kw = {"model": lm, "tasks": task_list}
    if limit > 0:
        eval_kw["limit"] = limit
    results = evaluator.simple_evaluate(**eval_kw)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS — NanoOSRT v4")
    print("=" * 60)
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
            else:
                print(f"  {k}: {v}")
    return results["results"]


# =============================================================================
# ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main(stage: str = "pretrain"):
    """Run v4 training stages.

    --stage tokenizer  Train custom 64K BPE tokenizer
    --stage pretrain   Pre-training (progressive seq_len)
    --stage sft        Balanced SFT
    --stage grpo       GRPO reinforcement learning
    --stage eval       Benchmark evaluation
    """
    if stage == "tokenizer":
        train_tokenizer.remote()
    elif stage == "sft":
        sft.remote()
    elif stage == "grpo":
        grpo.remote()
    elif stage == "eval":
        results = evaluate.remote("ifeval,gsm8k,hellaswag")
        print("\nResults:")
        for task, metrics in results.items():
            print(f"\n{task}:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
                else:
                    print(f"  {k}: {v}")
    else:
        pretrain.remote()
