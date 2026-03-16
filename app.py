"""Nano-OSRT 100M — Modal deployment entrypoint (v3.2).

104.5M physical parameters simulating 302M equivalent dense via
recursive weight sharing. 2 physical blocks x 6 recursive loops = 12
effective layers, each with unique per-pass residual adapters.

Deploy::

    modal run app.py

Resume after 24h timeout (automatically resumes from latest rescue
checkpoint)::

    modal run app.py
"""

import modal

from nano_osrt.modal_config import ModalConfig

# NOTE: torch imports are deliberately kept INSIDE train() because Modal
# parses this file locally before sending to the container. If torch isn't
# installed on the local machine, top-level imports would crash.

# =============================================================================
# MODAL INFRASTRUCTURE
# =============================================================================

app = modal.App("nano-osrt-100m")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"TORCH_LOGS": "perf_hints"})
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "lion-pytorch", "triton", "wandb")
    .pip_install("lm-eval", "langdetect", "immutabledict")
    .add_local_dir("src/nano_osrt", remote_path="/root/nano_osrt")
)

vol = modal.Volume.from_name("osrt-checkpoints", create_if_missing=True)


# =============================================================================
# REMOTE TRAINING FUNCTION
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def train():
    """Run the full training loop inside a Modal H100 container."""
    from transformers import AutoTokenizer

    from nano_osrt.modal_train import run_training

    cfg = ModalConfig()

    # Tokenizer + dynamic vocab alignment
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg.real_vocab_size = len(tokenizer)
    cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

    run_training(cfg, vol, tokenizer_name)


# =============================================================================
# SFT (SUPERVISED FINE-TUNING)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def sft():
    """Run SFT with chain-of-thought reasoning data."""
    from transformers import AutoTokenizer

    from nano_osrt.sft_config import SFTConfig
    from nano_osrt.sft_train import run_sft

    cfg = SFTConfig()

    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg.real_vocab_size = len(tokenizer)
    cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

    run_sft(cfg, vol, tokenizer_name)


# =============================================================================
# GRPO (REINFORCEMENT LEARNING)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def grpo():
    """Run GRPO reinforcement learning with verifiable math rewards."""
    from transformers import AutoTokenizer

    from nano_osrt.grpo_config import GRPOConfig
    from nano_osrt.grpo_train import run_grpo

    cfg = GRPOConfig()

    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg.real_vocab_size = len(tokenizer)
    cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

    run_grpo(cfg, vol, tokenizer_name)


# =============================================================================
# GENERAL SFT (POST-GRPO)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def general_sft():
    """Run general instruction SFT after GRPO."""
    from transformers import AutoTokenizer

    from nano_osrt.sft_config import GeneralSFTConfig
    from nano_osrt.sft_train import run_sft

    cfg = GeneralSFTConfig()

    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg.real_vocab_size = len(tokenizer)
    cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

    run_sft(cfg, vol, tokenizer_name)


# =============================================================================
# CODE SFT (POST-GRPO)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def code_sft():
    """Run code-focused SFT after GRPO."""
    from transformers import AutoTokenizer

    from nano_osrt.sft_config import CodeSFTConfig
    from nano_osrt.sft_train import run_sft

    cfg = CodeSFTConfig()

    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg.real_vocab_size = len(tokenizer)
    cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

    run_sft(cfg, vol, tokenizer_name)


# =============================================================================
# EVALUATION
# =============================================================================


@app.function(
    gpu="A100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=43200,
)
def evaluate(tasks: str = "ifeval", limit: int = 0):
    """Run lm-evaluation-harness benchmarks on A100."""
    import json
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from lm_eval.api.model import LM
    from lm_eval import evaluator

    from nano_osrt.hf_model import NanoOSRTConfig, NanoOSRTForCausalLM

    class NanoOSRTEval(LM):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cuda")
            self._batch_size = 1

            print("Loading model from /vol/checkpoints/osrt100m_code_final.pt...")
            config = NanoOSRTConfig()
            self.model = NanoOSRTForCausalLM(config)

            import copy
            ckpt = torch.load("/vol/checkpoints/osrt100m_code_final.pt",
                              map_location="cuda", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to("cuda")
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = config.real_vocab_size

            total = sum(p.numel() for p in self.model.parameters())
            print(f"Parameters: {total:,}")

        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            return self.model.config.seq_len

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string, **kwargs):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens, **kwargs):
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_call(self, inps):
            with torch.no_grad():
                out = self.model(inps.to(self._device))
                return out["logits"][:, :, :self.vocab_size]

        def loglikelihood(self, requests, **kwargs):
            results = []
            for req in requests:
                context, continuation = req.args
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation)
                full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)

                if full_ids.shape[1] > self.max_length:
                    full_ids = full_ids[:, -self.max_length:]
                    ctx_len = max(0, len(ctx_ids) - (len(ctx_ids) + len(cont_ids) - self.max_length))
                else:
                    ctx_len = len(ctx_ids)

                logits = self._model_call(full_ids)
                shift_logits = logits[0, ctx_len - 1:-1, :]
                shift_labels = full_ids[0, ctx_len:].to(self._device)

                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

                total_ll = token_log_probs.sum().item()
                is_greedy = (shift_logits.argmax(dim=-1) == shift_labels).all().item()
                results.append((total_ll, is_greedy))
            return results

        def loglikelihood_rolling(self, requests, **kwargs):
            results = []
            for req in requests:
                (string,) = req.args
                ids = self.tok_encode(string)
                full_ids = torch.tensor([ids], dtype=torch.long)
                if full_ids.shape[1] > self.max_length:
                    full_ids = full_ids[:, -self.max_length:]

                logits = self._model_call(full_ids)
                shift_logits = logits[0, :-1, :]
                shift_labels = full_ids[0, 1:].to(self._device)

                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                results.append((token_log_probs.sum().item(),))
            return results

        def generate_until(self, requests, **kwargs):
            results = []
            for i, req in enumerate(requests):
                context = req.args[0]
                gen_kwargs = req.args[1] if len(req.args) > 1 else {}
                until = gen_kwargs.get("until", [self.tokenizer.eos_token])
                max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

                ctx_ids = self.tok_encode(context)
                ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long)
                if ctx_tensor.shape[1] > self.max_length - max_gen:
                    ctx_tensor = ctx_tensor[:, -(self.max_length - max_gen):]

                output = self.model.generate(
                    ctx_tensor.to(self._device),
                    max_new_tokens=max_gen,
                    temperature=0.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                new_tokens = output[0, ctx_tensor.shape[1]:]
                response = self.tok_decode(new_tokens.tolist())

                for stop_seq in until:
                    if stop_seq in response:
                        response = response[:response.index(stop_seq)]

                results.append(response)

                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  Generated {i + 1}/{len(requests)}")

            return results

    lm = NanoOSRTEval()

    task_list = [t.strip() for t in tasks.split(",")]
    print(f"\nRunning benchmarks: {task_list}")
    if limit > 0:
        print(f"Limiting to {limit} examples per task")

    eval_kwargs = {"model": lm, "tasks": task_list}
    if limit > 0:
        eval_kwargs["limit"] = limit

    results = evaluator.simple_evaluate(**eval_kwargs)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS — NanoOSRT 100M (115.7M with HRA)")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in sorted(task_results.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")
            else:
                print(f"  {metric}: {value}")

    return results["results"]


# =============================================================================
# ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main(stage: str = "pretrain"):
    """Run training stages.

    --stage pretrain  Pre-training (default)
    --stage sft       Math/reasoning SFT
    --stage grpo      GRPO reinforcement learning
    --stage general   General instruction SFT (post-GRPO)
    --stage code      Code SFT (post-GRPO)
    --stage eval      Run benchmarks (IFEval, GSM8K, HellaSwag)
    """
    if stage == "sft":
        sft.remote()
    elif stage == "grpo":
        grpo.remote()
    elif stage == "general":
        general_sft.remote()
    elif stage == "code":
        code_sft.remote()
    elif stage == "eval":
        results = evaluate.remote("ifeval")
        print("\nResults returned:")
        for task, metrics in results.items():
            print(f"\n{task}:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
                else:
                    print(f"  {k}: {v}")
    else:
        train.remote()
