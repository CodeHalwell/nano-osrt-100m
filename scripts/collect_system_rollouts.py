"""Collect system-prompt-bearing rows from UltraChat + OpenHermes-2.5.

Strict filter: rows MUST have a non-empty system prompt. UltraChat
is mostly user-first (so most rows drop out); OpenHermes has system
roles for a much higher fraction.

Output JSONL one row per turn (system, user, assistant, source).
Schema matches what RolloutDataset expects:
    {
      "system": "...",
      "prompt": "...",          # user question
      "response": "...",        # assistant answer (no <|think|> wrap;
                                #  we add format at training time)
      "source": "ultrachat" | "openhermes",
      "id": "<dataset>:<row_id>:<turn_idx>"
    }

Format applied during SFT training:
    <|system|>{system}<|user|>{prompt}<|assistant|>{response}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset


def iter_ultrachat(split: str = "train_sft"):
    """Yield (system, user, assistant, row_id, turn_idx) from UltraChat.

    Filters to rows where messages[0].role == 'system'. Yields only
    the FIRST user/assistant pair after the system message — we want
    single-turn (system, user, asst) triples, not multi-turn dialogs.
    """
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
    for row in ds:
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "system":
            continue
        sys = (msgs[0].get("content") or "").strip()
        if not sys:
            continue
        # Find first user→assistant pair after system
        u = a = None
        for m in msgs[1:]:
            r = m.get("role")
            if u is None and r == "user":
                u = (m.get("content") or "").strip()
            elif u is not None and r == "assistant":
                a = (m.get("content") or "").strip()
                break
        if u and a:
            yield sys, u, a, row.get("prompt_id", ""), 0


def iter_openhermes():
    """Yield (system, user, assistant, row_id, turn_idx) from OpenHermes.

    Uses the explicit `system_prompt` field if non-empty. Falls back
    to checking conversations[0].from == 'system' for older rows.
    """
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    for row in ds:
        sys = (row.get("system_prompt") or "").strip()
        if not sys:
            # Fall back: check conversations[0]
            convs = row.get("conversations") or []
            if convs and convs[0].get("from") == "system":
                sys = (convs[0].get("value") or "").strip()
        if not sys:
            continue

        convs = row.get("conversations") or []
        # Find first human→gpt pair (skip system if present at index 0)
        u = a = None
        for c in convs:
            f = c.get("from")
            if u is None and f == "human":
                u = (c.get("value") or "").strip()
            elif u is not None and f == "gpt":
                a = (c.get("value") or "").strip()
                break
        if u and a:
            yield sys, u, a, str(row.get("id", "")), 0


SOURCES = {
    "ultrachat": iter_ultrachat,
    "openhermes": iter_openhermes,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="rollouts/system_prompt_sft.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=5000,
        help="Cap per source (UltraChat will likely hit the limit much later)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCES.keys()),
        choices=list(SOURCES.keys()),
    )
    parser.add_argument(
        "--min-user-len",
        type=int,
        default=10,
        help="Skip rows with user text shorter than N chars",
    )
    parser.add_argument(
        "--min-assistant-len",
        type=int,
        default=20,
        help="Skip rows with assistant text shorter than N chars",
    )
    parser.add_argument(
        "--max-assistant-len",
        type=int,
        default=4000,
        help="Skip rows with assistant text longer than N chars",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: track existing IDs so a re-run doesn't duplicate
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resume: {len(done_ids)} rows already in {out_path}")

    counts = {s: 0 for s in args.sources}
    skipped = {s: 0 for s in args.sources}

    t0 = time.time()
    with out_path.open("a") as f:
        for source in args.sources:
            print(f"\n== streaming {source} (max {args.max_per_source}) ==")
            iterator = SOURCES[source]()
            for sys, u, a, row_id, turn_idx in iterator:
                if counts[source] >= args.max_per_source:
                    break
                rid = f"{source}:{row_id}:{turn_idx}"
                if rid in done_ids:
                    skipped[source] += 1
                    continue
                if len(u) < args.min_user_len:
                    skipped[source] += 1
                    continue
                if len(a) < args.min_assistant_len or len(a) > args.max_assistant_len:
                    skipped[source] += 1
                    continue

                record = {
                    "id": rid,
                    "source": source,
                    "system": sys,
                    "prompt": u,
                    "response": a,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[source] += 1
                if counts[source] % 250 == 0:
                    elapsed = time.time() - t0
                    rate = sum(counts.values()) / max(elapsed, 1)
                    print(
                        f"  [{source}] {counts[source]:>5} kept, "
                        f"{skipped[source]:>4} skipped "
                        f"(total: {sum(counts.values())}, {rate:.1f} rows/sec)",
                        flush=True,
                    )

    elapsed = time.time() - t0
    print(f"\n== done in {elapsed:.0f}s ==")
    for s in args.sources:
        print(f"  {s:>12}: {counts[s]:>5} kept, {skipped[s]:>5} skipped")
    print(f"  TOTAL kept: {sum(counts.values())}")
    print(f"  Output:    {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
