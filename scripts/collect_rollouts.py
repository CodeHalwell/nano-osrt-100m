"""MOPD teacher-rollout collector for nano-osrt distillation.

Calls Gemini 3.5 Flash with thinking enabled (search OFF) on a mixed
prompt distribution matched to nano-osrt's target capabilities:

    Math word problems        : gsm8k train
    Multi-step reasoning      : open-thoughts-114k prompts
    Code (basic Python)       : mbpp train
    Chat / instruction        : ultrachat-200k user turns
    Knowledge / STEM          : sciq train

Each rollout is one Gemini call producing (thinking_text, response_text).
Results are streamed to a JSONL file with one record per line:

    {"id": "<source>:<idx>", "source": str, "prompt": str,
     "thinking": str, "response": str, "ts": float,
     "input_tokens": int, "output_tokens": int}

The script is RESUMABLE — on restart it scans the JSONL for already-done
prompt IDs and skips them. Run with whatever concurrency your Gemini quota
supports; default 5 in-flight calls is conservative.

Usage:
    python scripts/collect_rollouts.py \
        --output rollouts/mopd_v1.jsonl \
        --sources math reasoning chat \
        --max-per-source 1000 \
        --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator

# ── env loading (no external dep) ──────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# Soft dependency check
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. Run: uv add google-genai",
          file=sys.stderr)
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets not installed. Run: uv add datasets",
          file=sys.stderr)
    sys.exit(1)

GEMINI_MODEL = "gemini-3.5-flash"
THINKING_LEVEL = "MEDIUM"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "rollouts" / "mopd_v1.jsonl"

# Pricing (per 1M tokens) — Gemini 3.5 Flash with thinking. THINKING
# tokens are billed as output, which dominates cost: a math problem
# with MEDIUM thinking generates 3K-5K thinking + 500-1K response
# tokens, so output token cost is the binding constraint. £300 budget
# ≈ $370 → ~10-15K rollouts (not 100K).
PRICE_INPUT_PER_M = 1.50
PRICE_OUTPUT_PER_M = 9.00


# ─────────────────────────────────────────────────────────────────────
# Prompt source registry
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Source:
    name: str
    hf_id: str
    split: str
    prompt_field: str | None     # if None, use a custom extractor
    extractor: callable | None
    hf_config: str | None = None


def _extract_gsm8k(row: dict) -> str:
    return row["question"]


def _extract_ultrachat(row: dict) -> str:
    # First user message only — we want a single-turn distillation rollout,
    # not multi-turn conversation continuation.
    msgs = row.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def _extract_mbpp(row: dict) -> str:
    # MBPP prompts are coding instructions like "Write a function to ..."
    # Test cases live in row["test_list"] — we exclude them so the
    # student learns to write code from the natural-language spec, not
    # to satisfy tests it's been shown.
    return row.get("text", "")


def _extract_open_thoughts(row: dict) -> str:
    # open-thoughts-114k has a 'conversations' field with user → assistant.
    convos = row.get("conversations", [])
    for c in convos:
        if c.get("from") == "user" or c.get("role") == "user":
            return c.get("value") or c.get("content", "")
    # Fallback: 'problem' field on some configurations
    return row.get("problem", "")


def _extract_sciq(row: dict) -> str:
    q = row.get("question", "")
    correct = row.get("correct_answer", "")
    # SciQ has 4 answer choices in fields correct_answer + distractor1/2/3.
    # We pose it as a free-form question (no MC choices) so Gemini reasons
    # rather than just picking from a given set.
    if q:
        return q
    return ""


SOURCES: dict[str, Source] = {
    "math": Source(
        name="gsm8k",
        hf_id="openai/gsm8k",
        hf_config="main",
        split="train",
        prompt_field="question",
        extractor=_extract_gsm8k,
    ),
    "reasoning": Source(
        name="open-thoughts-114k",
        hf_id="open-thoughts/OpenThoughts-114k",
        split="train",
        prompt_field=None,
        extractor=_extract_open_thoughts,
    ),
    "code": Source(
        name="mbpp",
        hf_id="google-research-datasets/mbpp",
        hf_config="full",
        split="train",
        prompt_field="text",
        extractor=_extract_mbpp,
    ),
    "chat": Source(
        name="ultrachat-200k",
        hf_id="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        prompt_field=None,
        extractor=_extract_ultrachat,
    ),
    "science": Source(
        name="sciq",
        hf_id="allenai/sciq",
        split="train",
        prompt_field="question",
        extractor=_extract_sciq,
    ),
}


def iter_prompts(source_key: str, max_n: int) -> Iterator[tuple[str, str]]:
    """Yield (rollout_id, prompt) for a given source, up to max_n."""
    src = SOURCES[source_key]
    print(f"[{source_key}] Loading {src.hf_id} ({src.split})...", flush=True)
    if src.hf_config is not None:
        ds = load_dataset(src.hf_id, src.hf_config, split=src.split,
                          streaming=True)
    else:
        ds = load_dataset(src.hf_id, split=src.split, streaming=True)
    n = 0
    for idx, row in enumerate(ds):
        if n >= max_n:
            break
        prompt = src.extractor(row) if src.extractor else row.get(src.prompt_field, "")
        prompt = (prompt or "").strip()
        if not prompt or len(prompt) > 4000:
            # Skip empty prompts and overlong ones (token budget control).
            continue
        rid = f"{source_key}:{idx}"
        yield rid, prompt
        n += 1


# ─────────────────────────────────────────────────────────────────────
# Resume support
# ─────────────────────────────────────────────────────────────────────
def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("id")
                if rid:
                    done.add(rid)
            except json.JSONDecodeError:
                continue
    return done


# ─────────────────────────────────────────────────────────────────────
# Gemini call (sync — wrap in to_thread for asyncio concurrency)
# ─────────────────────────────────────────────────────────────────────
def call_gemini(client: "genai.Client", prompt: str) -> dict:
    """Single Gemini call. Returns dict with thinking/response/tokens."""
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=THINKING_LEVEL,
            include_thoughts=True,
        ),
        # NOTE: Google Search is intentionally DISABLED. Our student model
        # has no web-retrieval capability — if Gemini answers via search,
        # the student will hallucinate facts at inference time.
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    thinking_parts: list[str] = []
    response_parts: list[str] = []
    last_usage = None

    for chunk in client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=cfg,
    ):
        if chunk.usage_metadata is not None:
            last_usage = chunk.usage_metadata
        if not chunk.candidates:
            continue
        candidate = chunk.candidates[0]
        if candidate.content is None or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if part.text is None:
                continue
            if getattr(part, "thought", False):
                thinking_parts.append(part.text)
            else:
                response_parts.append(part.text)

    input_tokens = getattr(last_usage, "prompt_token_count", 0) if last_usage else 0
    output_tokens = (
        getattr(last_usage, "candidates_token_count", 0)
        + getattr(last_usage, "thoughts_token_count", 0)
    ) if last_usage else 0

    return {
        "thinking": "".join(thinking_parts).strip(),
        "response": "".join(response_parts).strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def call_gemini_async(client: "genai.Client", prompt: str,
                            max_retries: int = 5) -> dict:
    """Async wrapper with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(call_gemini, client, prompt)
        except Exception as e:
            msg = str(e).lower()
            # Retry on rate limits and 5xx; fail fast on auth / 4xx that
            # are not 429.
            if "429" in msg or "rate" in msg or "503" in msg or "500" in msg:
                wait = 2 ** attempt + random.random()
                print(f"  [retry {attempt+1}/{max_retries} in {wait:.1f}s]: {e}",
                      flush=True)
                await asyncio.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Exhausted {max_retries} retries")


# ─────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────
async def collect(args: argparse.Namespace) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set (check .env or env)",
              file=sys.stderr)
        sys.exit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output)
    print(f"Output: {output}")
    print(f"Resume: {len(done_ids)} rollouts already on disk\n", flush=True)

    client = genai.Client(api_key=api_key)

    # Build the (rid, prompt) work queue, filtering already-done.
    queue: list[tuple[str, str, str]] = []  # (rid, source_key, prompt)
    for src_key in args.sources:
        if src_key not in SOURCES:
            print(f"WARN: unknown source '{src_key}', skipping", file=sys.stderr)
            continue
        count_added = 0
        for rid, prompt in iter_prompts(src_key, args.max_per_source):
            if rid in done_ids:
                continue
            queue.append((rid, src_key, prompt))
            count_added += 1
        print(f"[{src_key}] queued {count_added} prompts", flush=True)

    print(f"\nTotal queued: {len(queue)} prompts, concurrency={args.concurrency}",
          flush=True)
    if not queue:
        print("Nothing to do.", flush=True)
        return

    # Atomic-append: one writer task drains a queue that workers fill.
    write_q: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=64)

    async def writer():
        with output.open("a", buffering=1) as f:
            while True:
                rec = await write_q.get()
                if rec is None:
                    return
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                write_q.task_done()

    writer_task = asyncio.create_task(writer())

    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.time()
    stats = {"done": 0, "errors": 0, "input_toks": 0, "output_toks": 0,
             "cost_usd": 0.0}
    # Use an event to signal "budget exceeded — stop spawning new work and
    # stop in-flight workers from sending more requests"
    budget_stop = asyncio.Event()

    async def worker(rid: str, src_key: str, prompt: str) -> None:
        # Check budget BEFORE acquiring semaphore — cheap fast-exit
        if budget_stop.is_set():
            return
        async with semaphore:
            if budget_stop.is_set():
                return
            try:
                t0 = time.time()
                result = await call_gemini_async(client, prompt)
                rec = {
                    "id": rid,
                    "source": src_key,
                    "prompt": prompt,
                    "thinking": result["thinking"],
                    "response": result["response"],
                    "ts": time.time(),
                    "elapsed_s": round(time.time() - t0, 2),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                await write_q.put(rec)
                stats["done"] += 1
                stats["input_toks"] += result["input_tokens"]
                stats["output_toks"] += result["output_tokens"]
                stats["cost_usd"] += (
                    result["input_tokens"] * PRICE_INPUT_PER_M / 1e6
                    + result["output_tokens"] * PRICE_OUTPUT_PER_M / 1e6
                )
                # Hard stop on budget breach
                if stats["cost_usd"] >= args.budget_usd:
                    if not budget_stop.is_set():
                        print(f"\n💸 BUDGET CAP HIT (${stats['cost_usd']:.2f} "
                              f">= ${args.budget_usd:.2f}). Stopping new work.",
                              flush=True)
                        budget_stop.set()
            except Exception as e:
                stats["errors"] += 1
                print(f"  FAILED {rid}: {e}", flush=True)

        # Periodic progress (every 10 completions)
        if stats["done"] % 10 == 0 and stats["done"] > 0:
            elapsed = time.time() - start
            rate = stats["done"] / elapsed
            remaining = (len(queue) - stats["done"]) / max(rate, 1e-6)
            avg_cost = stats["cost_usd"] / stats["done"]
            print(f"  [{stats['done']}/{len(queue)}] "
                  f"{rate:.2f} rps  err={stats['errors']}  "
                  f"in={stats['input_toks']:,} out={stats['output_toks']:,}  "
                  f"$={stats['cost_usd']:.2f} (~${avg_cost:.3f}/roll)  "
                  f"ETA={remaining/60:.1f}m", flush=True)

    workers = [asyncio.create_task(worker(rid, src_key, prompt))
               for rid, src_key, prompt in queue]
    await asyncio.gather(*workers)
    await write_q.put(None)
    await writer_task

    elapsed = time.time() - start
    avg_cost = stats["cost_usd"] / max(stats["done"], 1)
    print(f"\n✅ Done in {elapsed/60:.1f}m. "
          f"completed={stats['done']}, errors={stats['errors']}, "
          f"input_toks={stats['input_toks']:,}, "
          f"output_toks={stats['output_toks']:,}, "
          f"cost=${stats['cost_usd']:.2f} (avg ${avg_cost:.3f}/rollout)",
          flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--sources", nargs="+",
                   default=["math", "reasoning", "code", "chat", "science"],
                   help="Subset of: math, reasoning, code, chat, science")
    p.add_argument("--max-per-source", type=int, default=2_000,
                   help="Per-source rollout cap (default 2K = ~10K total)")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Max in-flight Gemini calls")
    p.add_argument("--budget-usd", type=float, default=300.0,
                   help="Hard stop when accumulated cost reaches this many USD")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(collect(parse_args()))
