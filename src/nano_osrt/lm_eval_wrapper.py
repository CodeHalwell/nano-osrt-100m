"""lm-evaluation-harness wrapper for NanoOSRT.

Implements the `lm_eval.api.model.LM` interface so the standard
benchmark harness (gsm8k, IFEval, MMLU, etc.) can score our custom
NanoOSRTForCausalLM. Two methods cover the bulk of HF-style benchmarks:

  loglikelihood(requests)  — multiple-choice scoring
                              (MMLU, ARC, hellaswag, BoolQ, ...)
  generate_until(requests) — open-ended generation with stop strings
                              (gsm8k, IFEval, HumanEval, ...)

`loglikelihood_rolling` (rolling perplexity) is not implemented — none
of our target benchmarks use it. If that changes, plumb it like
`loglikelihood` but with sliding-window context.

The wrapper assumes the checkpoint may include HRA adapter parameters
(injected during SFT). If `hra_enabled=True`, HRA is injected BEFORE
the state_dict load so adapter rows in the saved file land in the
right place.

Usage from app.py::evaluate:

    wrapper = NanoOSRTLMEval(
        ckpt_path="/vol/checkpoints/v5/osrt_v5_sft_ultralong_final.pt",
        tokenizer_path="/vol/tokenizer",
        hra_enabled=True,
        hra_rank=256,
        batch_size=8,
    )
    results = lm_eval.simple_evaluate(
        model=wrapper, tasks=["gsm8k", "ifeval", "mmlu_stem"],
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


class NanoOSRTLMEval(LM):
    """LM-eval-harness adapter over NanoOSRTForCausalLM.

    Loads the model + tokenizer once at construction. Subsequent
    .loglikelihood / .generate_until calls run against the same
    in-memory model — important for benchmark suites that issue
    thousands of requests per task.

    Batching strategy:
      - loglikelihood: right-pad each batch of requests to the max
        sequence length, run a single forward, gather per-token
        log-probs and ignore padded positions. Default batch=8 fits
        comfortably on a single H100.
      - generate_until: batched at request level — each request gets
        its own .generate call. Generation is autoregressive and
        sequence lengths diverge fast on stop-string termination, so
        true batching gains very little here. KV cache inside
        .generate keeps per-step cost O(1).

    Numerical correctness:
      - `loglikelihood` returns log-probs in fp32 (after up-casting
        from bf16 logits). The bf16 → fp32 cast happens inside
        F.log_softmax to preserve precision when computing sums over
        long continuations.
      - Logits are clipped to `real_vocab_size` (32768) so probability
        mass on padding/unused vocab tail (when `vocab_size` was
        rounded up for tensor-core alignment) doesn't leak in. v5 has
        vocab_size == real_vocab_size already so the clip is a no-op,
        but the guard stays for portability.
    """

    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        hra_enabled: bool = True,
        hra_rank: int = 256,
        batch_size: int = 8,
        max_length: int = 8192,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length

        # Imports here to avoid loading torch dependencies at module
        # import time when this file might be parsed but not used.
        from transformers import AutoTokenizer

        from nano_osrt.config import NanoOSRTConfig
        from nano_osrt.model import NanoOSRTForCausalLM

        print(f"[lm_eval] Loading tokenizer from {tokenizer_path}", flush=True)
        self._tok = AutoTokenizer.from_pretrained(tokenizer_path)

        print("[lm_eval] Constructing model...", flush=True)
        cfg = NanoOSRTConfig(
            vocab_size=len(self._tok),
            real_vocab_size=len(self._tok),
            bos_token_id=self._tok.bos_token_id,
            eos_token_id=self._tok.eos_token_id,
            pad_token_id=self._tok.pad_token_id,
        )
        self._cfg = cfg

        model = NanoOSRTForCausalLM(cfg).to(self._device)

        # Inject HRA adapters BEFORE loading the state_dict. SFT
        # checkpoints have HRA params already in their state_dict; the
        # injection adds the matching nn.Parameter slots so they fill
        # in correctly during load_state_dict.
        if hra_enabled:
            from nano_osrt.hra import inject_hra
            print(f"[lm_eval] Injecting HRA (rank={hra_rank})", flush=True)
            inject_hra(model, rank=hra_rank)

        print(f"[lm_eval] Loading weights from {ckpt_path}", flush=True)
        ckpt = torch.load(
            ckpt_path, map_location=self._device, weights_only=True,
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"[lm_eval] WARNING: missing keys ({len(missing)}): "
                f"{missing[:3]}", flush=True,
            )
        if unexpected:
            print(
                f"[lm_eval] WARNING: unexpected keys ({len(unexpected)}): "
                f"{unexpected[:3]}", flush=True,
            )

        model.train(False)  # disables MoE capacity drops, enables KV-cache path
        self._model = model
        print("[lm_eval] Model ready.", flush=True)

    # ── Required LM interface ──────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return int(self._tok.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 512

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    def tok_encode(self, string: str, **kwargs) -> list[int]:  # noqa: ANN003
        return self._tok.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs) -> str:  # noqa: ANN001, ANN003
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tok.decode(tokens, skip_special_tokens=False)

    # ── loglikelihood (multiple-choice scoring) ────────────────────

    @torch.no_grad()
    def loglikelihood(
        self, requests: list[Instance],
    ) -> list[tuple[float, bool]]:
        """Score continuations given context.

        For each (context, continuation) pair, returns (sum_log_p,
        is_greedy) where:
          sum_log_p = sum over continuation tokens of log P(token | prev tokens)
          is_greedy = True iff every continuation token equals argmax of
                      its position's logits (no sampling needed)

        Used by MMLU / ARC / hellaswag / etc. — score the multiple
        choice options against each other and pick the one with
        highest sum_log_p.
        """
        results: list[tuple[float, bool]] = []
        real_vocab = self._cfg.real_vocab_size
        pad_id = int(self._tok.pad_token_id)

        # Pre-tokenize all requests so we can sort by length for
        # better batch padding efficiency. Track original index so we
        # can return results in the original order.
        items = []
        for orig_idx, req in enumerate(requests):
            context, continuation = req.args
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            # Truncate from the left if context+continuation exceeds
            # max_length. We keep the continuation intact and lop off
            # the front of the context.
            full = ctx_ids + cont_ids
            if len(full) > self._max_length:
                drop = len(full) - self._max_length
                ctx_ids = ctx_ids[drop:]
                full = ctx_ids + cont_ids
            items.append((orig_idx, ctx_ids, cont_ids, full))

        # Sort by total length for tighter batches.
        items.sort(key=lambda x: len(x[3]))

        # Process in batches.
        out: dict[int, tuple[float, bool]] = {}
        for batch_start in range(0, len(items), self._batch_size):
            batch = items[batch_start : batch_start + self._batch_size]
            max_len = max(len(it[3]) for it in batch)
            input_ids = torch.full(
                (len(batch), max_len), pad_id,
                dtype=torch.long, device=self._device,
            )
            for i, (_, _, _, full) in enumerate(batch):
                input_ids[i, : len(full)] = torch.tensor(
                    full, dtype=torch.long, device=self._device,
                )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self._model(input_ids).logits  # (B, T, V)

            # Slice to real_vocab and cast to fp32 for stable sums.
            logits = logits[..., :real_vocab].float()
            log_probs = F.log_softmax(logits, dim=-1)

            # For each sample: continuation lives at positions
            # [len(ctx), len(ctx)+len(cont)). The token at position p
            # is predicted by the logits at position p-1.
            for i, (orig_idx, ctx_ids, cont_ids, full) in enumerate(batch):
                cont_len = len(cont_ids)
                ctx_len = len(ctx_ids)
                if cont_len == 0:
                    out[orig_idx] = (0.0, True)
                    continue
                # Logits predicting continuation tokens.
                cont_logits = logits[
                    i, ctx_len - 1 : ctx_len - 1 + cont_len, :
                ]
                cont_log_probs = log_probs[
                    i, ctx_len - 1 : ctx_len - 1 + cont_len, :
                ]
                cont_target = torch.tensor(
                    cont_ids, dtype=torch.long, device=self._device,
                )
                token_log_probs = cont_log_probs.gather(
                    1, cont_target.unsqueeze(1),
                ).squeeze(1)
                argmax = cont_logits.argmax(dim=-1)
                is_greedy = bool((argmax == cont_target).all().item())
                out[orig_idx] = (
                    float(token_log_probs.sum().item()),
                    is_greedy,
                )

        # Reassemble in original order.
        for orig_idx in range(len(requests)):
            results.append(out[orig_idx])
        return results

    def loglikelihood_rolling(
        self, requests: list[Instance],
    ) -> list[float]:
        """Not used by gsm8k / IFEval / MMLU. Implement if a benchmark
        we care about (e.g. WikiText perplexity) needs it."""
        raise NotImplementedError(
            "loglikelihood_rolling not implemented — none of the v5 "
            "target benchmarks need it.",
        )

    # ── generate_until (open-ended generation) ─────────────────────

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate continuations up to a stop string or max_gen_toks.

        gsm8k passes `until=["</s>", "Question:", "\n\n"]` (or similar);
        IFEval passes `until=["\n\n"]`. We use the model's KV-cached
        .generate, then post-process to truncate at the first stop
        string encountered.

        Sampling is deterministic by default (temperature=0). Override
        per-request via `gen_kwargs["temperature"]` and
        `gen_kwargs["top_p"]` if a benchmark configures otherwise.
        """
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            top_p = float(gen_kwargs.get("top_p", 1.0))
            top_k = int(gen_kwargs.get("top_k", 0))

            ctx_ids = self.tok_encode(context)
            # Leave room for max_new tokens within the model's
            # max_position_embeddings. Drop oldest context if needed.
            keep = self._max_length - max_new
            if len(ctx_ids) > keep:
                ctx_ids = ctx_ids[-keep:]
            ctx_tensor = torch.tensor(
                [ctx_ids], dtype=torch.long, device=self._device,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_ids = self._model.generate(
                    ctx_tensor,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_id=self.eot_token_id,
                )

            gen_ids = out_ids[0, len(ctx_ids):].tolist()
            text = self.tok_decode(gen_ids)

            # Truncate at first occurrence of any stop string.
            min_idx = math.inf
            for stop in until:
                idx = text.find(stop)
                if idx != -1 and idx < min_idx:
                    min_idx = idx
            if min_idx != math.inf:
                text = text[: int(min_idx)]
            results.append(text)
        return results
