"""NanoOSRT v3 — Re-benchmark with corrected generation params.

The original v3 benchmark (26.7% IFEval) silently inherited
`repetition_penalty=1.2` from hf_model.py's generate signature. This hurts
format-following tasks (IFEval) where the model needs to repeat bullet
markers, required keywords, or structural tokens.

Stages:
    greedy      T=0.0, rep_penalty=1.0                        (Config 1)
    original    T=0.0, rep_penalty=1.2 (reproduce 26.7% bug)   (Config 2)
    sampled     T=0.3, top_p=0.9, top_k=50, rep_penalty=1.0    (Config 3)
"""

import modal

app = modal.App("nano-osrt-v3-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"PYTHONUNBUFFERED": "1"})
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers", "datasets", "tokenizers", "safetensors",
        "lm-eval", "langdetect", "immutabledict",
    )
    .add_local_dir("src/nano_osrt", remote_path="/root/nano_osrt")
)

vol = modal.Volume.from_name("osrt-v3-checkpoints", create_if_missing=True)


def _run_benchmark(
    config_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    tasks: str = "ifeval,gsm8k,hellaswag",
    limit: int = 0,
) -> None:
    import time

    import torch
    import torch.nn.functional as F
    from lm_eval import evaluator as lmeval
    from lm_eval.api.model import LM
    from transformers import AutoTokenizer

    from nano_osrt.hf_model import NanoOSRTConfig, NanoOSRTForCausalLM

    print("=" * 60, flush=True)
    print(f"v3 BENCHMARK -- Config {config_name}", flush=True)
    print("=" * 60, flush=True)
    print(f"  temperature        : {temperature}", flush=True)
    print(f"  top_p              : {top_p}", flush=True)
    print(f"  top_k              : {top_k}", flush=True)
    print(f"  repetition_penalty : {repetition_penalty}", flush=True)
    print(f"  tasks              : {tasks}", flush=True)
    if limit > 0:
        print(f"  limit              : {limit}", flush=True)
    print(flush=True)

    class NanoOSRTEval(LM):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cuda")
            self._batch_size = 1

            ckpt_path = "/vol/checkpoints/osrt100m_grpo_final.pt"
            print(f"Loading model from {ckpt_path}...", flush=True)
            config = NanoOSRTConfig()
            self.model = NanoOSRTForCausalLM(config)
            ckpt = torch.load(
                ckpt_path, map_location="cuda", weights_only=True,
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to("cuda")
            self.model.train(False)

            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = config.real_vocab_size

            total = sum(p.numel() for p in self.model.parameters())
            print(f"Parameters: {total:,}", flush=True)

            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty

            self._ll_done = 0
            self._gen_done = 0
            self._ll_start = None
            self._gen_start = None

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

        def _log_ll_progress(self, total):
            self._ll_done += 1
            if self._ll_start is None:
                self._ll_start = time.time()
            should = (
                self._ll_done == 1
                or self._ll_done % 200 == 0
                or self._ll_done == total
            )
            if should:
                elapsed = time.time() - self._ll_start
                rate = self._ll_done / max(elapsed, 1e-6)
                eta = (total - self._ll_done) / max(rate, 1e-6)
                print(
                    f"  [loglikelihood] {self._ll_done}/{total} "
                    f"| {rate:.1f} items/s | ETA {eta / 60:.1f} min",
                    flush=True,
                )

        def _log_gen_progress(self, total):
            self._gen_done += 1
            if self._gen_start is None:
                self._gen_start = time.time()
            should = (
                self._gen_done == 1
                or self._gen_done % 25 == 0
                or self._gen_done == total
            )
            if should:
                elapsed = time.time() - self._gen_start
                rate = self._gen_done / max(elapsed, 1e-6)
                eta = (total - self._gen_done) / max(rate, 1e-6)
                print(
                    f"  [generate_until] {self._gen_done}/{total} "
                    f"| {rate:.2f} items/s | ETA {eta / 60:.1f} min",
                    flush=True,
                )

        def loglikelihood(self, requests, **kwargs):
            self._ll_done = 0
            self._ll_start = None
            total = len(requests)
            results = []
            for req in requests:
                context, continuation = req.args
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation)
                full_ids = torch.tensor(
                    [ctx_ids + cont_ids], dtype=torch.long, device=self._device,
                )
                if full_ids.shape[1] > self.max_length:
                    full_ids = full_ids[:, -self.max_length:]
                    ctx_len = max(
                        0,
                        len(ctx_ids) - (len(ctx_ids) + len(cont_ids) - self.max_length),
                    )
                else:
                    ctx_len = len(ctx_ids)

                logits = self._model_call(full_ids)
                shift_logits = logits[0, ctx_len - 1:-1, :]
                shift_labels = full_ids[0, ctx_len:]

                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs.gather(
                    1, shift_labels.unsqueeze(1),
                ).squeeze(1)

                total_ll = token_log_probs.sum().item()
                is_greedy = (
                    shift_logits.argmax(dim=-1) == shift_labels
                ).all().item()
                results.append((total_ll, is_greedy))
                self._log_ll_progress(total)
            return results

        def loglikelihood_rolling(self, requests, **kwargs):
            self._ll_done = 0
            self._ll_start = None
            total = len(requests)
            results = []
            for req in requests:
                (string,) = req.args
                ids = self.tok_encode(string)
                full_ids = torch.tensor(
                    [ids], dtype=torch.long, device=self._device,
                )
                if full_ids.shape[1] > self.max_length:
                    full_ids = full_ids[:, -self.max_length:]

                logits = self._model_call(full_ids)
                shift_logits = logits[0, :-1, :]
                shift_labels = full_ids[0, 1:]

                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs.gather(
                    1, shift_labels.unsqueeze(1),
                ).squeeze(1)
                results.append((token_log_probs.sum().item(),))
                self._log_ll_progress(total)
            return results

        def generate_until(self, requests, **kwargs):
            self._gen_done = 0
            self._gen_start = None
            total = len(requests)
            results = []
            for req in requests:
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
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                new_tokens = output[0, ctx_tensor.shape[1]:]
                response = self.tok_decode(new_tokens.tolist())
                for stop_seq in until:
                    if stop_seq in response:
                        response = response[:response.index(stop_seq)]
                results.append(response)
                self._log_gen_progress(total)
            return results

    lm = NanoOSRTEval()
    task_list = [t.strip() for t in tasks.split(",")]
    print(f"\nRunning benchmarks: {task_list}", flush=True)

    run_kwargs = {"model": lm, "tasks": task_list}
    if limit > 0:
        run_kwargs["limit"] = limit

    results = lmeval.simple_evaluate(**run_kwargs)

    print("\n" + "=" * 60, flush=True)
    print(f"RESULTS -- Config {config_name}", flush=True)
    print("=" * 60, flush=True)
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:", flush=True)
        for metric, value in sorted(task_results.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)", flush=True)
            else:
                print(f"  {metric}: {value}", flush=True)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=14400,
)
def greedy(tasks: str = "ifeval,gsm8k,hellaswag"):
    _run_benchmark(
        config_name="GREEDY (T=0.0, rep_penalty=1.0)",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        tasks=tasks,
    )


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=14400,
)
def original():
    _run_benchmark(
        config_name="ORIGINAL (T=0.0, rep_penalty=1.2 -- the bug)",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.2,
    )


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=14400,
)
def sampled():
    _run_benchmark(
        config_name="SAMPLED (T=0.3, top_p=0.9, top_k=50, rep_penalty=1.0)",
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
    )


@app.local_entrypoint()
def main(stage: str = "greedy", tasks: str = "ifeval,gsm8k,hellaswag"):
    """Run v3 benchmark with corrected generation params.

    --stage greedy     Config 1: T=0.0, rep_penalty=1.0 (best for IFEval)
    --stage original   Config 2: T=0.0, rep_penalty=1.2 (reproduce 26.7%)
    --stage sampled    Config 3: T=0.3, top_p=0.9, top_k=50, rep_penalty=1.0
    --tasks <list>     Comma-separated tasks, e.g. "ifeval" or "ifeval,gsm8k"
    """
    if stage == "greedy":
        greedy.remote(tasks=tasks)
    elif stage == "original":
        original.remote()
    elif stage == "sampled":
        sampled.remote()
    else:
        print(f"Unknown stage: {stage}. Use greedy/original/sampled.")
