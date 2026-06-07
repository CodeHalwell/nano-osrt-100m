# OSRT-100M (budget edition) — $280 from scratch

**Parent doc:** [`OSRT_600M.md`](OSRT_600M.md) — the no-budget-constraint version
**Date:** 2026-06-07
**Constraint:** $280 Modal budget (build-small workspace), ~70 H100-hours total

This is the budget-engineered scale-down of OSRT_600M.md. Same lessons,
same architecture family, same training pipeline philosophy — every
parameter, token count, and stage cost slashed to fit $280.

The headline trade: ship a **smaller model** (100M total, ~50M active)
trained on **fewer tokens** (~100B vs 3T), keep **all the architectural
and training discipline** wins. We retain the recursive MoE
differentiator, the Muon optimizer, the system-prompt pipeline, the
HRA-only GRPO, strict rewards, and OOD probes. We cut the size, the
data scale, and the optional stages (vision, multi-stage curation,
tool use).

---

## 0. Budget breakdown

| stage | H100-hr | $ | what's in vs cut |
|---|---|---|---|
| Pretraining | 45 | **$180** | 100B tokens, single-stage WSD, FineWeb-Edu + chat mix |
| SFT (system-prompt) | 12 | **$48** | 5K OpenHermes rollouts, 300 steps |
| HRA-only GRPO | 7 | **$28** | Math-only env, 100 steps, OOD probe + strict |
| Eval pipeline | 4 | **$16** | gsm8k full + IFEval + MMLU-Pro 200-subset |
| Buffer / reruns | 2 | **$8** | One small ablation re-run if needed |
| **Total** | **70** | **$280** | |

**Cut entirely:**
- Vision retrofit (~$200 in 600M doc) — can be Stage 2 in a future
  budget allocation
- Tool-use GRPO (~$15) — model is too small for tools to be the main
  unlock; revisit if math performance is poor enough that it matters
- Multi-teacher rollout collection (~$10-15 API) — use OpenHermes
  free + maybe 500 supplemental DeepSeek calls (~$2)
- Multi-stage Nemotron-CC ensemble curation — too complex; one
  high-quality web source (FineWeb-Edu) is enough at 100B tokens
- 50-prompt OOD probe — shrink to 12 prompts (we already validated
  this size in v5)
- μP / μTransfer HP search — use known-good Muon hparams from
  modded-nanoGPT speedrun directly

---

## 1. Architecture — OSRT-100M

Same recursive MoE family as v5/v6, scaled down. The 6-loop recurrence
gives ~300M-FLOPs-equivalent compute per token from a ~100M parameter
budget.

### 1.1 Parameter budget

```
Embedding (32K × 768, tied with LM head)        : 24,576,000   (always active)
Attention × 3 blocks (qkv + out_proj)           :  7,077,888   (always active)
Shared experts × 3 (SwiGLU h=2048)              : 14,155,776   (always active)
Routed experts: 3 × 4 × (SwiGLU h=1024)         : 14,155,776 total
  → with top-2 of 4, active per token           :  7,077,888
HRA adapters (rank 128, injected day 1)         : 14,155,776   (trainable)
Router + loop_emb + adapters + norms            :  ~0.5 M

Total physical params                           : ~74 M (call it "100M class")
Active per token                                : ~52 M (70 %)
Effective compute per token (× 6 loops)         : ~625 M FLOPs-equivalent
```

The active-fraction climbs to ~70% (vs v5's 53% and 600M's 34%) because
with only 4 routed experts and top-2 selection, most of the routed pool
is hit on average. That's the **right trade for a small model** — there
isn't enough capacity to specialise heavily.

### 1.2 Key dimensional changes from OSRT-600M

| param | 600M | **100M** | rationale |
|---|---|---|---|
| Hidden dim | 1792 | **768** | 4× smaller embedding, attention |
| Routed experts per block | 12 | **4** | Fine-grained specialisation needs scale we don't have |
| Routed expert hidden | 2304 | **1024** | Match dim |
| Shared expert hidden | 4608 | **2048** | Match dim |
| HRA adapter rank | 256 | **128** | Smaller adapters for smaller base |
| Vocab | 48K | **32K** | Save 8M params on embedding; English-only is fine |
| Loops | 6 | **6** (kept) | The architecture is the architecture |
| Physical blocks | 3 | **3** (kept) | Same |

### 1.3 Architecture stability — keep ALL of OSRT-600M's fixes

These are not optional. The recursive-MoE-under-Muon failure modes
don't get easier at smaller scale:

- **QK-norm** on every attention block ✓
- **Sandwich RMSNorm** (pre + post) in recurrent block ✓
- **Per-loop expert-load logging** ✓ (catches per-loop routing
  divergence)
- **Loss-free balancing** (DeepSeek-V3 bias-update method, γ=0.001)
- **Loop embeddings** capped at `min(r, 7)` — kept from v5
- **Tied embedding + LM head** ✓ (saves 24M params)

### 1.4 What we lose

- **Knowledge capacity** — 100M parameters genuinely can't store as
  many facts as 600M. Multi-digit arithmetic, world facts, niche
  knowledge will be worse. That's why tools are doubly important
  long-term — but they're out of scope for the $280 build.
- **Specialisation potential** — 4 routed experts can't carve niches
  as cleanly as 12 or 64. Routing decisions will look more uniform.
- **Multilingual / code performance** — 32K vocab + English-centric
  data; not aiming for it.

---

## 2. Optimizer — Muon (kept verbatim from OSRT-600M)

No scale-down here. Muon is FREE — no extra memory vs AdamW, no extra
compute, just better convergence. We get the 2× compute efficiency vs
AdamW which means **our $180 pretraining budget effectively buys $360
of training quality**.

- Muon for all 2D hidden matrices
- AdamW for embedding (tied), LM head, RMSNorm gains, biases, router
  bias accumulator
- Weight decay 0.01-0.10 on Muon params
- Update-RMS alignment across Muon/AdamW
- Gram Newton-Schulz (5 iterations, reset at step 2-3) if implementing
  fresh; standard NS5 acceptable for this scale
- WSD schedule: 1000-step warmup → stable at peak LR 3e-4 → 10% decay
  to high-quality math/code

**The Muon adoption is the single highest-ROI lesson we apply.** Lion
would have cost us ~2× more for the same final loss at this scale.

---

## 3. Tokenizer — chat template from day 1 (kept)

This is where we **don't compromise**. The v5 lesson — system prompts
must be in pretraining, not retrofitted — applies regardless of
budget.

Required single-token reservations:

| token | id (reserved) | purpose |
|---|---|---|
| `<|system|>` | 13 | system prompt opener |
| `<|user|>` | 11 | user turn |
| `<|assistant|>` | 12 | assistant turn |
| `<|end_turn|>` | NEW | turn separator (fixes v5's ambiguity) |
| `<|think|>` / `<|/think|>` | 7 / 8 | reasoning block |
| `<|answer|>` / `<|/answer|>` | 9 / 10 | answer block |

Tool tokens (`<|tool_call|>`) are **reserved but not actively trained**
on this budget. The token slots exist in the tokenizer so we can add
tool support later without retraining; the pretraining data doesn't
include tool-using examples.

---

## 4. Pretraining — 100B tokens, single-stage WSD

### 4.1 Token budget

100B tokens at 74M total params = **~1,350 tokens/param**. Above
Chinchilla (20×) but well below the SOTA SLM ratio (SmolLM2: 6,500×;
Gemma 3 270M: 22,000×). Under-trained relative to frontier — but
that's the budget.

This means OSRT-100M will plateau earlier on benchmark gains than a
600M model trained longer. Acceptable for a research artifact; not
acceptable if we're competing for production deployment.

### 4.2 Single-stage data mixture

Skip the multi-stage Nemotron-CC ensemble curation — too complex at
this scale. Use a single high-quality mix:

- **75% FineWeb-Edu** — quality-classified web text
- **10% chat-formatted** — OpenHermes-2.5 system-prompt-bearing rows
  (we built the filter for v5; reuse)
- **5% reasoning** — OpenThoughts subset + R1-traces
- **5% math** — Nemotron-CC-Math-v1 subset (we have access)
- **5% code** — Stack v2 filtered subset

Stream from HF directly, no preprocessing pipeline. Single epoch.

### 4.3 WSD schedule

- Warmup: 1000 steps to peak LR 3e-4
- Stable: bulk of training at peak LR
- Decay: final 10% linear-decay to 3e-5, with mix shifted to
  higher-quality math/code

### 4.4 Validation cadence

Every 10B tokens (~10 checkpoints over the run):

- **gsm8k 200-subset** (full eval too expensive per ckpt)
- **MMLU 1K-subset** (cheap version)
- **Per-loop CE loss** (Test 3 — the v5 depth-utilization probe)
- **Per-loop expert balance**
- **12-prompt OOD probe** (built in v5)

This is what v5 didn't have. Auto-pipelined via Modal volume-commit
hook.

---

## 5. SFT — 5K rollouts, 300 steps

### 5.1 Data — OpenHermes filtered (free)

Use the system-prompt-bearing rows we just collected for v5
(`rollouts/system_prompt_sft.jsonl`). 10K rows already in flight.
Cost: $0 API.

If quality is insufficient post-SFT, supplement with ~500 DeepSeek
v4-flash rollouts at varied system prompts (~$2). Stay under budget.

### 5.2 Training

- 300 steps from pretrained ckpt
- Peak LR 1e-6 → cosine 1e-7
- Format: `<|system|>{sys}<|user|>{q}<|assistant|>{response}<|end_turn|>`
- Loss masked on prefix
- aux_loop_loss_weight 0.05, loop_dropout 0.10 (preserve depth fix)
- Batch 4 × grad_accum 8 = effective batch 32

12 H100-hours total.

---

## 6. GRPO — HRA-only, math-only, 100 steps

### 6.1 Single-env focus

OSRT-600M plans multi-env (math + IFEval + MBPP). At $28 / 100 steps,
spreading across 3 envs gives us ~33 steps per env — not enough for
signal. **Pick one: math (gsm8k).**

Math is where:
- The model needs the most help
- Verifiable rewards work cleanly
- Real benchmark transfer happens
- The strict-extraction reward hack-prevention matters most

### 6.2 Configuration

- 100 steps from SFT checkpoint
- HRA-only (freeze base, train only 14M HRA params)
- Peak LR 5e-6 → cosine 5e-7
- kl_coeff 0.15 (math-only GRPO proven setting)
- Group size 6 (vs 8 — saves rollout compute)
- max_gen_len 384
- Strict extraction ON (the v5 anti-hacking gate)
- OOD probe every 20 steps (5 probes over the run)
- Stop early if OOD drops 2× in a row

7 H100-hours.

---

## 7. Eval — single comprehensive pass

### 7.1 Benchmarks (one full run)

- **gsm8k full (1319)** — math reasoning
- **IFEval full (541)** — instruction following
- **MMLU-Pro 200-subset** — knowledge sampling
- **HumanEval (164)** — code generation (basic check)
- **MT-Bench short** — chat quality
- **12-prompt OOD** — our generalisation probe

### 7.2 Output

- One JSON per benchmark with per-prompt scores
- Aggregate dashboard via W&B
- Comparison table: OSRT-100M vs Qwen3-0.6B / SmolLM2 / Gemma 3 270M
  on the same prompts at T=0.3

4 H100-hours total.

---

## 8. Stages explicitly CUT

For honesty about what we're giving up:

| cut stage | what it would have been | why we cut it |
|---|---|---|
| Vision retrofit | LLaVA-style projector, MMBench/ScienceQA | ~$200 — outweighs entire SFT+GRPO+eval combined |
| Tool-use GRPO | Calculator + python_exec native tools | $15-20 + needs token reservations trained; better as future Stage 2 |
| Multi-teacher rollouts | Gemini + DeepSeek + Gemma 3 self-hosted mix | OpenHermes free + 500 DeepSeek (~$2) is good enough at this scale |
| 3-stage WSD with anneal | SmolLM3-style mid-training stages | Complex; single-stage WSD captures 80% of the benefit at 1/3 the engineering |
| μP HP search | Tune at width 256, transfer to 1792 | We're at width 768 — proxy distance is small; use known-good Muon defaults |
| Speculative decoding via MTP heads | 2-3× inference speedup | Engineering cost too high at this budget; aux_loop_loss training stays so we COULD add it later |
| Per-env rich logging | math.exact, ifeval.constraints, code.tests_pass | Single-env GRPO; just need math.exact and OOD score |
| 50-prompt OOD probe | Diverse capability check | 12 prompts validated in v5; smaller is fine |

---

## 9. Decision tree if budget changes

**+$100 (total $380):**
- Add tool-use GRPO ($20) + double pretrain tokens ($80 → 180B tokens)

**+$300 (total $580):**
- Triple pretrain tokens (300B) + multi-env GRPO + brief vision SFT

**+$1000 (total $1280):**
- 600B-token pretrain + full vision retrofit + multi-teacher rollouts +
  full multi-stage WSD anneal

**-$100 (total $180):**
- Drop OSRT-100M from scratch entirely; use Gemma 3 270M as base,
  apply ONLY the SFT+GRPO+eval pipeline on top. Loses our recursive
  MoE thesis but gets a deployable model for $180. Track B fallback.

---

## 10. Expected outcomes

Honest estimates, calibrated against v5 results + research literature
scaling laws:

| benchmark | OSRT-100M target | Qwen3-0.6B baseline | Gemma 3 270M |
|---|---|---|---|
| gsm8k | ~15-25% | ~45% | ~35% |
| IFEval | ~30-40% | ~60% | ~50% |
| MMLU | ~25-30% | ~45% | ~30% |
| HumanEval | ~5-15% | ~30% | ~15% |
| 12-prompt OOD | ~6-8/12 | ~9/12 | ~8/12 |

We're **not going to beat Gemma 3 270M** on benchmarks at this budget.
What we WILL have:

- A working recursive MoE pipeline that scales to OSRT-600M when budget
  allows
- All the training infrastructure (Muon recipe, OOD probe, strict
  rewards, system-prompt SFT, HRA-only GRPO) battle-tested at small
  scale
- A research artifact demonstrating the architecture works
- No dependency on external base models (vs the Gemma-3-270M-base
  fallback)

---

## 11. v6 → v5 path (if we use existing nano-osrt as base)

**Alternative use of $280 if from-scratch is too risky:**

Use the existing nano-osrt v5 363M model as base, apply OSRT-600M
training lessons:

| stage | $ | what it does |
|---|---|---|
| System-prompt MOPD on v5 | $30 | Already collecting data; train 500 steps |
| HRA-only GRPO from system-MOPD ckpt | $50 | Multi-env (math + IFEval + maybe MBPP) |
| Tool-use SFT + GRPO | $80 | Add calculator tool support |
| Quick vision SFT (LLaVA-tiny) | $100 | Projector-only, frozen base |
| Eval suite | $20 | Full benchmark sweep |
| **Total** | **$280** | v6-fixed nano-osrt with tools + vision + system prompts |

This trade: **keep the trained-knowledge of v5's 363M** (which cost ~$2-5K
to build originally) and ADD what we now know how to do. Skips
pretraining a smaller new model from scratch. Lower risk; relies on
v5's existing baseline being good enough to build on.

**Recommendation:** I'd actually pick **track B (build on v5)** over
track A (new from-scratch 100M) for the $280 spend. Reasoning:

1. v5 already has 12+ months of trained knowledge — throwing that
   away to build a smaller model from scratch is wasteful
2. The training-pipeline lessons (system prompts, HRA-only, strict
   rewards) are what we mainly want to apply — those can be applied
   ON TOP of v5
3. From-scratch with 100B tokens is genuinely under-trained; v5 has
   much more pretraining behind it
4. v6 with tools + vision + system prompts is a complete, deployable
   model; 100M from-scratch is a research artifact

The recursive MoE architecture thesis is ALREADY VALIDATED by v5. We
don't need to re-validate by training a smaller version. We need to
add the missing capabilities (tools, vision, system prompts) and
deploy.

---

## 12. Sources

Same as OSRT_600M.md — see that doc's source list. This doc adds
nothing new beyond the scale-down arithmetic.

---

## 13. What this document is NOT

- **Not a recommendation to build OSRT-100M from scratch.** Read §11
  — the existing v5 base is a better foundation for $280 than a
  fresh small model.
- **Not a critique of OSRT_600M.md.** That doc is the right answer
  for $15K. This doc is the right answer for $280. Same lessons, very
  different cost ceilings.
- **Not committed.** Track A and Track B are both viable; pick based
  on whether you want to:
  - **Track A:** demonstrate recursive MoE works at small from-scratch
    scale (research-oriented)
  - **Track B:** ship the most capable model possible on $280
    (deployment-oriented)
