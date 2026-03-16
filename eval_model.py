#!/usr/bin/env python3
"""Evaluate NanoOSRT on benchmarks using lm-evaluation-harness.

Usage:
    # IFEval benchmark
    uv run python eval_model.py --tasks ifeval --model ./nano-osrt-model

    # Multiple benchmarks
    uv run python eval_model.py --tasks ifeval,gsm8k,hellaswag --model ./nano-osrt-model

    # With specific generation params
    uv run python eval_model.py --tasks ifeval --model ./nano-osrt-model --batch-size 4
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import evaluator

from src.nano_osrt.hf_model import NanoOSRTConfig, NanoOSRTForCausalLM


@register_model("nano-osrt")
class NanoOSRTHarnessModel(LM):
    """Wrapper for lm-evaluation-harness."""

    def __init__(self, model_path: str = "./nano-osrt-model", device: str = "auto", batch_size: int = 1, **kwargs):
        super().__init__()

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._device = torch.device(device)
        self._batch_size = batch_size

        print(f"Loading NanoOSRT from {model_path} on {device}...")
        self.model = NanoOSRTForCausalLM.from_pretrained(model_path, device=device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.model.config.real_vocab_size

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
        return 512

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(inps.to(self._device))
            return out["logits"][:, :, :self.vocab_size]

    def _model_generate(self, context: torch.Tensor, max_length: int, stop=None, **kwargs) -> torch.Tensor:
        max_new = max_length - context.shape[1]
        with torch.no_grad():
            return self.model.generate(
                context.to(self._device),
                max_new_tokens=max_new,
                temperature=0.0,  # greedy for eval
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def loglikelihood(self, requests, **kwargs):
        results = []
        for context, continuation in [req.args for req in requests]:
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)

            # Truncate to max_length
            if full_ids.shape[1] > self.max_length:
                full_ids = full_ids[:, -self.max_length:]
                ctx_len = max(0, len(ctx_ids) - (len(ctx_ids) + len(cont_ids) - self.max_length))
            else:
                ctx_len = len(ctx_ids)

            logits = self._model_call(full_ids)
            # Shift: logits[t] predicts token[t+1]
            shift_logits = logits[0, ctx_len - 1:-1, :]
            shift_labels = full_ids[0, ctx_len:]

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

            total_ll = token_log_probs.sum().item()
            is_greedy = (shift_logits.argmax(dim=-1) == shift_labels).all().item()

            results.append((total_ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests, **kwargs):
        results = []
        for (string,) in [req.args for req in requests]:
            ids = self.tok_encode(string)
            full_ids = torch.tensor([ids], dtype=torch.long)

            if full_ids.shape[1] > self.max_length:
                full_ids = full_ids[:, -self.max_length:]

            logits = self._model_call(full_ids)
            shift_logits = logits[0, :-1, :]
            shift_labels = full_ids[0, 1:]

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

            results.append((token_log_probs.sum().item(),))
        return results

    def generate_until(self, requests, **kwargs):
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
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            new_tokens = output[0, ctx_tensor.shape[1]:]
            response = self.tok_decode(new_tokens.tolist())

            # Apply stop sequences
            for stop_seq in until:
                if stop_seq in response:
                    response = response[:response.index(stop_seq)]

            results.append(response)
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate NanoOSRT on benchmarks")
    parser.add_argument("--model", type=str, default="./nano-osrt-model")
    parser.add_argument("--tasks", type=str, default="ifeval", help="Comma-separated task list")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="./eval_results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for quick testing)")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",")]

    lm = NanoOSRTHarnessModel(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\nRunning benchmarks: {task_list}")
    if args.limit:
        print(f"Limiting to {args.limit} examples per task")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        limit=args.limit,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in sorted(task_results.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")
            else:
                print(f"  {metric}: {value}")

    print(f"\nFull results saved to {args.output}/")


if __name__ == "__main__":
    main()
