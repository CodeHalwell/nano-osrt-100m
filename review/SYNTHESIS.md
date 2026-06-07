# Plan Review Synthesis

**Date:** 2026-06-07
**Source reviews:** `agy-plan-reviewed.md` (Antigravity), `codex-plan-review.md` (Codex)
**Reviewed docs:** `README.md`, `ARCHITECTURE.md`, `LEARNINGS.md`, `RESEARCH.md`

## Verdict

**Codex is the high-signal review.** Almost every Codex finding is
evidence-based (file:line citations) and points at hard contract
failures that would silently break training. Antigravity reads as a
research-side smoke check — useful for architectural reasoning, but
its claims overlap with v5 lessons we already documented, and its
"recommendations" don't catch the actual contract bugs.

**Net:** 21 Codex findings, 4 Antigravity recommendations.
~15 unique-to-Codex findings vs ~1 unique-to-Antigravity finding
(V-from-K expressivity argument). The two are not interchangeable.

## Severity tiers

### Tier 0 — Must fix before any code is written (5)

These are hard contract failures. Implementing v6 against the current
spec would silently produce a broken model.

| # | Issue | Source | Fix complexity |
|---|---|---|---|
| 1 | Tokenizer IDs in spec ≠ actual `tokenizer/tokenizer.json` (PAD/BOS/EOS swapped; ID 14 = `!` not `<\|end_turn\|>`) | Codex §2 | Mechanical (clarify spec is for v6 tokenizer to be retrained) |
| 2 | Repo doesn't import — `pytest` collection fails (`src/nano_osrt` missing, moved to `archive/v5/src/`) | Codex §1 | Mechanical (clarify root is design-only OR restore package) |
| 3 | mHC pseudocode dimensionally broken (3 separate bugs: shape after transpose, `expand()` aliasing, undefined final collapse) | Codex §6, §7, §8 | Spec rewrite + design decision on collapse head |
| 4 | Parameter accounting disagrees across files (32,768 typo, 206M vs 232.5M vs 210M active) | Codex §3 | Mechanical (pick one + propagate) |
| 5 | KV cache savings double-counted (claims 2× reduction that doesn't exist given K-only baseline already) | Codex §9 | Mechanical |

### Tier 1 — Important architectural decisions (5)

These need explicit design decisions before training starts.

| # | Issue | Source | Decision |
|---|---|---|---|
| 6 | Hash routing under recurrence — does it apply every loop, or only loop 0? Top-1 or top-2? | Codex §11 + Antigravity §1.3 | Pick semantic (Antigravity recommends `hash(token + loop_idx)`) |
| 7 | V-from-K rank/expressivity bottleneck | Antigravity §1.1 | Accept constraint, or widen latent K to 768/1024 |
| 8 | Final mHC collapse head undefined (uses stale `A_l`) | Codex §8 | Design a dedicated collapse head |
| 9 | Tier 1 cost: 50K H100-hr × $4/hr = $200K, not $15K | Codex §4 | Either reduce hour estimate, or change price assumption ($0.30/hr spot?) |
| 10 | Research provenance: LFM2 paper (arXiv 2511.23404) miscited as DeepSeek-V4 | Codex §5 | Mechanical (fix citation) |

### Tier 2 — Specification clarity (11)

These are cleanup work — spec is ambiguous, will cause implementer
churn but not silent breakage if caught at integration time.

| # | Issue | Source |
|---|---|---|
| 11 | Deployment memory budget: 318.5M @ 4-bit ≈ 159 MB, not claimed 80 MB | Codex §10 |
| 12 | HRA injection count 87 vs naive 132 — need enumeration table | Codex §12 |
| 13 | Three aux losses conflated (loop LM aux, router balance, router z) | Codex §13 |
| 14 | Speculative decoding isn't distribution-preserving (greedy only) | Codex §14 |
| 15 | `forward()` / `generate()` signatures don't match in pseudocode | Codex §15 |
| 16 | "Gated short convolutions" claimed in overview, never specified | Codex §16 |
| 17 | HCA claimed required for deployment, never specified | Codex §17 |
| 18 | Eval cadence (500B tokens) doesn't fit Tier 2 (12B tokens total) | Codex §18 |
| 19 | OOD probe size inconsistent (20-50, 50, 12) | Codex §19 |
| 20 | Tool use both "day-1 commitment" and "deferred to higher tier" | Codex §20 |
| 21 | mHC Sinkhorn runtime claim (6.7%) needs kernel assumption | Codex §21 + Antigravity §1.2 |

## What this synthesis applied immediately (mechanical fixes)

See `MECHANICAL_FIXES.md` for the diff applied. Specifically:

- ARCHITECTURE.md §1: typo fix (32,768 → 65,536)
- ARCHITECTURE.md §3: clarify tokenizer/ on disk is v5; v6 needs regen with spec IDs
- ARCHITECTURE.md §13/§14: KV cache double-count removed, one baseline only
- ARCHITECTURE.md §2: pick canonical active-param number
- README.md cost section: reconcile 50K H100-hr line with stated $/hr
- RESEARCH.md: fix DeepSeek-V4 citation (point at real DeepSeek-V4 paper, mark LFM2 separately)
- Repo state: README header note that `pyproject.toml`/`tests/` still
  point at archived v5 package; v6 package layout TBD
- ARCHITECTURE.md §17 / new §18: explicit `DECISION REQUIRED` callouts
  for Tier-1 items the user needs to resolve

## What this synthesis did NOT touch (needs your decision)

These are deferred to you, not silently changed:

1. **Hash routing semantic** — three plausible choices (Tier 1 #6)
2. **V-from-K vs widen-latent** — affects 1M params and expressivity (Tier 1 #7)
3. **Final mHC collapse head** — affects param count and architecture (Tier 1 #8)
4. **Tier 1 cost** — depends on real pricing assumption (Tier 1 #9)
5. **HCA / gated short conv scope** — Tier 2 #16, #17, defer or specify
6. **Speculative decoding mode** — greedy or sampling (Tier 2 #14)

## Recommended next moves

1. **Read SYNTHESIS.md (this doc) + the inline `DECISION REQUIRED`
   callouts now in ARCHITECTURE.md**
2. Resolve the 6 design decisions above
3. Then I write a `compute_budget.py` that generates all the
   parameter/FLOP/memory numbers from canonical config, replace
   hand-written totals in README/ARCHITECTURE with its output
4. Then I write the contract tests Codex recommends in Phase 4:
   tokenizer encode/decode test, mHC shape test, expand-aliasing
   test, hash routing top-k test, tied embedding identity test,
   loop aux output count test, cache prefill/decode shape test,
   active param accounting test
5. Then v6 implementation can begin

## Bottom line

Codex is right: the plans are a strong research notebook, not a
training launch plan. The Tier 0 fixes are mechanical and applied
in this pass. The Tier 1 decisions are yours.

Antigravity's V-from-K expressivity argument is worth reading and
deciding on — it's the one Tier 1 design point Codex didn't surface.
