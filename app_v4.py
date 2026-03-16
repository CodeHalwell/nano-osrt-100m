"""NanoOSRT v4 — Modal deployment entrypoint.

453M physical params, ~275M active/token, ~2.7B effective via recursive MoE.
3 physical blocks × 6 loops = 18 effective layers.
Dense FFN + MoE (1 shared + 11 routed experts, top-2) in parallel residual.

Stages:
    modal run app_v4.py --stage tokenizer    Train custom 128K tokenizer
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
    """Train custom 128K SuperBPE/BPE tokenizer on pre-training data mix."""
    import sys
    sys.path.insert(0, "/root")

    from scripts.train_tokenizer import sample_training_data, train_with_hf_tokenizers, _verify_tokenizer

    print("=" * 60)
    print("NanoOSRT v4 — Custom 128K Tokenizer Training")
    print("=" * 60)

    # Sample 100MB of training data (proportional to pre-training mix)
    print("\nSampling training data...")
    data_path = sample_training_data(sample_size=100_000_000)

    # Train tokenizer
    output_dir = "/vol/tokenizer"
    train_with_hf_tokenizers(data_path, vocab_size=128256, output_dir=output_dir)

    # Cleanup temp file
    import os
    os.remove(data_path)

    print(f"\nTokenizer saved to Modal volume 'osrt-v4-tokenizer'")


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
    from transformers import AutoTokenizer

    from nano_osrt.v4_config import NanoOSRTv4Config
    from nano_osrt.v4_train import run_v4_training
    from nano_osrt.v4_train_config import V4PretrainConfig

    # Load custom tokenizer
    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

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
    import glob
    import math
    import os
    import sys
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        import wandb
    except ImportError:
        wandb = None

    from nano_osrt.v4_config import NanoOSRTv4Config
    from nano_osrt.v4_model import NanoOSRTv4ForCausalLM
    from nano_osrt.v4_train_config import V4GRPOConfig
    from nano_osrt.rewards import compute_group_advantages, compute_reward
    from nano_osrt.hra import inject_hra, get_param_groups

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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

            # Score
            rewards = []
            for comp_ids in completions:
                comp_text = tok.decode(comp_ids[prompt_len:].tolist(), skip_special_tokens=True)
                comp_tokens = len(comp_ids) - prompt_len
                reward, breakdown = compute_reward(
                    comp_text, ground_truth,
                    correctness_weight=cfg.correctness_reward,
                    format_weight=cfg.format_reward,
                    length_penalty=cfg.length_penalty,
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

                # Policy log probs
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(comp_ids.unsqueeze(0))
                    logits = out.logits[0, :, :model_config.real_vocab_size].float()
                shift_logits = logits[prompt_len - 1:-1]
                shift_labels = comp_ids[prompt_len:]
                policy_lp = F.log_softmax(shift_logits, dim=-1).gather(1, shift_labels.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    ref_out = ref_model(comp_ids.unsqueeze(0))
                    ref_logits = ref_out.logits[0, :, :model_config.real_vocab_size].float()
                ref_shift = ref_logits[prompt_len - 1:-1]
                ref_lp = F.log_softmax(ref_shift, dim=-1).gather(1, shift_labels.unsqueeze(1)).squeeze(1)

                ratio = torch.exp(policy_lp - ref_lp)
                clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
                kl_loss = cfg.kl_coeff * (policy_lp - ref_lp).mean()
                loss = (policy_loss + kl_loss) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
            accuracy = step_correct / step_total if step_total > 0 else 0
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            print(f"step {step:>6d}/{cfg.total_steps} | loss {step_loss:.4f} | "
                  f"reward {mean_reward:.3f} | acc {accuracy:.1%} | "
                  f"lr {lr:.2e} | vram {vram:.1f}GB | elapsed {elapsed:.0f}s")
            if use_wandb:
                wandb.log({"grpo/loss": step_loss, "grpo/mean_reward": mean_reward,
                           "grpo/accuracy": accuracy, "grpo/lr": lr}, step=step)

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
    from transformers import AutoTokenizer
    from lm_eval.api.model import LM
    from lm_eval import evaluator

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

            # Load latest checkpoint
            import glob, os
            ckpt_dir = "/vol/checkpoints/v4"
            # Priority: grpo_final > sft_final > final
            for name in ["osrt_v4_grpo_final.pt", "osrt_v4_sft_final.pt", "osrt_v4_final.pt"]:
                path = os.path.join(ckpt_dir, name)
                if os.path.exists(path):
                    print(f"Loading {path}...")
                    ckpt = torch.load(path, map_location="cuda", weights_only=False)
                    self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                    break

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

    --stage tokenizer  Train custom 128K tokenizer
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
