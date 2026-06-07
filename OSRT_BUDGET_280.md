# OSRT-50M (budget edition) — same plan as OSRT_600M, $280 ceiling

**Parent doc:** [`OSRT_600M.md`](OSRT_600M.md) — same plan, no budget cap
**Date:** 2026-06-07
**Constraint:** $280 Modal budget (build-small workspace), ~70 H100-hours total

Same 14-section plan as OSRT_600M.md, every number scaled to fit $280.
Same recursive MoE architecture. Same Muon optimizer. Same system-prompt-
from-day-1 tokenizer. Same multi-stage pipeline (pretrain → SFT → GRPO →
tool use → vision → eval). All 10 codified lessons preserved.

**Headline scale-down:** 600M → 50M total params (12×), 3T → 30B
pretraining tokens (100×). Every other stage scales proportionally to
fit its share of the $280.

---

## 0. The single biggest meta-lesson

> **"What do you want to measure?" should drive what you build, not the
> other way around.**

Unchanged from OSRT_600M.md. The budget cut doesn't change the lesson —
it makes following it MORE important. At $280 we cannot afford to
discover problems late.

---

## 1. Architecture — recursive MoE, scaled down

Same recursive-MoE thesis (3 physical blocks × 6 loops + 1 shared + N
routed top-2 experts + HRA adapters). Just smaller dim and fewer
routed experts.

### 1.1 Parameter budget

```
Embedding (32K × 512, tied with LM head)        : 16,777,216   (always active)
Attention × 3 blocks (qkv + out_proj)           :  3,145,728   (always active)
Shared experts × 3 (SwiGLU h=1536)              :  7,077,888   (always active)
Routed experts: 3 × 4 × (SwiGLU h=768)          :  7,077,888 total
  → with top-2 of 4, active per token           :  3,538,944
HRA adapters (rank 64, injected day 1)          :  4,718,592   (trainable)
Router + loop_emb + adapters + norms            :  ~0.3 M

Total physical params                           : ~38 M (call it "50M class")
Active per token                                : ~31 M (~82 %)
Effective compute per token (× 6 loops)         : ~370 M FLOPs-equivalent
```

### 1.2 What's different from OSRT-600M (600M → 50M)

| change | OSRT-600M | **OSRT-50M (budget)** | rationale |
|---|---|---|---|
| Hidden dim | 1792 | **512** | 3.5× smaller; pretraining fits in budget |
| Routed experts per block | 12 | **4** | Less specialisation capacity needed at this scale |
| Routed expert hidden | 2304 | **768** | Match dim |
| Shared expert hidden | 4608 | **1536** | Match dim |
| HRA adapter rank | 256 | **64** | Smaller base, smaller adapter |
| Vocab | 48K | **32K** | Save the embedding tax |
| Loops | 6 | **6** (kept) | The architecture IS the architecture |
| Physical blocks | 3 | **3** (kept) | Same |

The 6-loop recurrence still gives us **~370M-FLOPs-equivalent compute
per token from a 38M parameter budget** — proportionally the same
recursive multiplier as OSRT-600M. Recursive MoE thesis preserved.

### 1.3 Stability fixes — kept verbatim (they're free)

These are not optional and they cost nothing. Same list as OSRT-600M:

- **QK-norm + QK-clip** on every attention block (Muon stability)
- **Sandwich RMSNorm** (pre + post) in recurrent block (recursion stability)
- **Per-loop expert-load logging** (catches routing collapse)
- **Aux-loss-free balancing** (DeepSeek-V3 bias-update method)
- **Loop embeddings** capped at `min(r, 7)`
- **GQA** (Grouped Query Attention) with 8 query / 2 KV heads — standard
  across Qwen3 / Phi-4-Mini / Gemma 3 / MiniCPM5-1B; cuts KV cache to
  25 % at minimal quality cost
- **Local / global attention layout** (Gemma 3 style: 5 local : 1 global,
  sliding window 1024) — when the model is mostly local-attention,
  KV cache scales O(n × window) not O(n²), enabling cheap long-context
  later

### 1.4 Global-batch load balancing (Qwen3 add-on)

In addition to DeepSeek-V3's per-expert bias updates (which run per
micro-batch), accumulate routing statistics ACROSS the global batch
when computing balance signal. Qwen3 explicitly uses this; it
substantially reduces noise in the balance penalty at small batch
sizes — which we always operate at on small models.

### 1.4 MTP heads via aux_loop_loss — kept

Same as OSRT-600M. The intermediate-loop LM heads are trained via
`aux_loop_loss_weight=0.05` already. Adds zero training cost.
Inference-time speculative decoding via this signal is optional
(engineering cost not in budget; revisit post-training if useful).

---

## 2. Optimizer — Muon (kept verbatim from OSRT-600M)

**No scale-down here.** Muon is free — same memory as AdamW, same
compute per step, but ~2× sample efficiency. Our $150 pretraining
budget effectively buys $300 of training quality vs Lion/AdamW.

Configuration identical to OSRT-600M.md §2:

- Muon for all 2D hidden matrices
- AdamW for embedding, LM head, RMSNorm gains, biases, router bias
- Weight decay 0.01-0.10 on Muon params
- Update-RMS alignment across Muon/AdamW
- Gram Newton-Schulz (5 iterations, reset at step 2-3)
- WSD schedule: 500-step warmup → stable at peak LR 3e-4 → 10% decay
  - (warmup shorter than 600M's 1000-step because total training is shorter)

### 2.1 Aux loss — same as OSRT-600M

DeepSeek-V3 loss-free balancing as default (γ=0.001 bias update).
Backup to α=0.01 aux + z-loss 0.001 if loss-free shows pathologies.

### 2.2 μP / μTransfer — DROPPED

Tuning HPs on a width-256 proxy and transferring is one of the cuts.
At width 512 we're close enough to known-good Muon defaults that the
HP search isn't worth a separate proxy run. Use modded-nanoGPT
defaults directly.

---

## 3. Tokenizer — chat template from day 1 (kept verbatim)

**No scale-down here.** The token reservations cost essentially zero
extra parameters. The pretraining data discipline (10% chat-formatted,
5% tool-using, 5% reasoning mixed in) is also free at this scale —
maybe even MORE important because we have so few tokens to spend.

Same single-token reservations as OSRT-600M.md §3:

| token | id | purpose |
|---|---|---|
| `<|system|>` | reserved | system prompt opener |
| `<|user|>` | reserved | user turn |
| `<|assistant|>` | reserved | assistant turn |
| `<|end_turn|>` | reserved | turn separator |
| `<|think|>` / `<|/think|>` | reserved | reasoning block |
| `<|answer|>` / `<|/answer|>` | reserved | answer block |
| `<|tool_call|>` / `<|/tool_call|>` | reserved | tool invocation |
| `<|tool_result|>` / `<|/tool_result|>` | reserved | tool result |

### 3.1 Vocab size — 32K (vs 600M's 48K)

Save 8M params on the tied embedding. English-centric is enough at
this scale; we don't need multilingual headroom we can't afford to
train on.

### 3.2 Pretraining text mixed in (same percentages)

- ~10% chat-formatted with full `<|system|>...<|/end_turn|>` chains
- ~5% tool-using with `<|tool_call|>...<|tool_result|>` patterns
- ~5% reasoning with `<|think|>...<|/answer|>` patterns

The percentages don't change — they're what makes the format native
rather than retrofitted. We just have fewer total tokens in absolute
terms.

---

## 4. Pretraining — 30B tokens, single-stage WSD

### 4.1 Token budget — scaled to fit

30B tokens at 38M total params = **~790 tokens/param**. Below the
6,500× ratio of SmolLM2 but above Chinchilla-optimal (20×). Comparable
to early-2024 SLMs at this scale.

This is the dominant cost of the budget. ~$150 of the $280 goes here.

Throughput math: H100 at ~150K tokens/sec on a 38M recursive MoE →
30B tokens = 55 H100-hours. We allocate **40 H100-hours = $160**, which
buys ~22B tokens. The extra 8B (to reach 30B target) comes from using
**Muon's 2× sample efficiency** vs the AdamW baseline.

### 4.2 Data mixture — single-stage (compressed from 3-stage)

The 3-stage WSD with annealing is one of the bigger budget cuts. At
30B tokens, a single high-quality mix captures most of the gain:

- 65% FineWeb-Edu (quality-classified web)
- 10% chat-formatted (OpenHermes system-prompt rows — we have them)
- 5% reasoning (OpenThoughts + R1-traces sample)
- 5% math (Nemotron-CC-Math-v1 subset — we have access)
- 5% code (Stack v2 filtered subset)
- 5% Wikipedia + RedPajama (diversity)
- 5% tool-using synthetic

Stream from HF directly. Single epoch.

### 4.3 WSD schedule — compressed

- Warmup: 500 steps to peak LR 3e-4
- Stable: bulk at peak LR
- Decay: final 10% linear-decay to 3e-5

### 4.3a Context length progression (Gemma 3 / Qwen3 lesson)

DON'T pretrain at max context day 1. That bankrupts small models.
Instead:

- **Stable phase: 4K context** (covers ~95 % of training tokens
  effectively; far cheaper attention cost)
- **Decay phase: extend to 8K** via RoPE θ scaling (free architectural
  knob)
- **Post-training context extension to 16K-32K** if needed (cheap with
  YaRN-style position scaling at SFT time)

Gemma 3 explicitly pretrains at 32K and extends to 128K near the end.
For OSRT-50M we don't need long context; 8K final is plenty.

### 4.4 Distillation IN pretraining (not just at end)

This is the 2026 frontier move. Gemma 3, Ministral 3, Qwen3, MiniCPM5-1B
all use teacher distillation during PRETRAINING, not just at SFT/instruct
time. Especially valuable for small models because the teacher provides
denser supervision than next-token CE alone.

Implementation:

- For the **final 20 % of pretraining tokens** (~6B tokens in our 30B
  budget), pull top-K logits (K=8) from DeepSeek v4-flash for each
  token
- KL divergence loss between student logits and teacher top-K
- Mixed with standard CE loss at 0.3 KD-weight
- API cost for ~6B tokens at $0.14/1M input = **~$840** for a complete
  KD pass — too expensive
- **Compromise:** apply ONLY to a high-value subset (~500M tokens of
  curated math/code/reasoning) → ~$70 → fits if we trim elsewhere

For the $280 budget, **defer KD-pretraining** unless we find we have
headroom. The simpler MOPD/SFT distillation in §5 captures most of
the benefit at lower cost.

### 4.4 Validation cadence — kept

Every 3B tokens (~10 checkpoints over the run):

- **gsm8k 200-subset** (full 1319 too expensive per ckpt)
- **MMLU 1K-subset**
- **Per-loop CE loss** (Test 3 — depth utilization)
- **Per-loop expert balance** (router collapse detection)
- **12-prompt OOD probe** (down from 600M's 50)

Auto-pipelined via Modal volume-commit hook. **The v5 lesson —
benchmarks at every ckpt — is non-negotiable.** This costs ~$5 of
the budget across 10 ckpts; we keep it.

---

## 5. SFT / MOPD — system prompts + multi-turn

### 5.1 Data — OpenHermes + tiny DeepSeek top-up

- **5K rollouts** from OpenHermes-2.5 filtered for system-prompt-bearing
  rows (we already collected this for v5; reuse)
- **+500 DeepSeek v4-flash rollouts** with varied system prompts from
  our `system_prompts.py` pool (~$2 API cost)
- Filter: must have system prompt; multi-turn captured if teacher returns

### 5.2 Training

Same as OSRT-600M but compressed:

- **300 steps** (vs 600M's 1500)
- Peak LR 1e-6 → cosine 1e-7
- Format: `<|system|>{sys}<|user|>{q}<|assistant|>{response}<|end_turn|>`
- Loss masked on prefix
- aux_loop_loss_weight 0.05, loop_dropout 0.10 (preserve depth fix)

**7 H100-hours = $28**

### 5.3 Multi-turn — kept

When teacher returns multi-turn, train on full chain. ~30% of data
is multi-turn; rest is single-turn. Same mix as 600M plan.

### 5.4 Controllable inference (think / no-think + variable loops)

Qwen3, MiniCPM5-1B, and Gemma 4 all expose **single-checkpoint
controllable reasoning** — the same model can be told to think briefly
or extensively at inference. For us, this is a NATURAL fit because:

1. We already train with `<|think|>...<|/think|>` tags in pretraining
2. Our 6 recursive loops are a literal "thinking budget" knob — we
   can run inference with fewer loops for fast responses, more loops
   for deeper reasoning

**Implementation (free, just plumbing):**

- SFT data 50/50 mix of:
  - "/think" examples — verbose `<|think|>` block before answer
  - "/nothink" examples — direct `<|answer|>` (skip think block)
- System-prompt vocab includes `/think` and `/nothink` directives
- Inference: `generate(loops=K)` exposes K ∈ {3, 4, 5, 6} as an API
  knob (we trained 6 loops; running fewer at inference is safe because
  the architecture fix ensures intermediate-loop outputs are coherent)

This is a **product differentiator** for OSRT-50M. Other 50M models
don't have a "depth dial" because they're not recursive. We do.

---

## 6. RL / GRPO — HRA-only, multi-env, strict rewards

### 6.1 HRA-only training (kept verbatim from OSRT-600M)

Freeze the 38M base weights. Train only the 4.7M HRA adapters. Same
reasoning as 600M: prevents base-weight drift that caused the v5
regression at step 75.

### 6.2 Multi-env (compressed but still multi-env)

- **Math (gsm8k)** — 70% weight (up from 600M's 60%; consolidate signal)
- **IFEval** — 30% weight (up from 600M's 30%; merged from MBPP slot)
- **MBPP** — **CUT** (the v5 lesson — mbpp was 0/180 pass rate, pure format hacking; not worth $5 of the budget)

### 6.3 Strict reward extraction (kept verbatim)

`extract_numeric_answer_strict()` with confidence tiers, ambiguous
penalty −0.5. Already built and tested in v5 post-mortem.

### 6.4 Regurgitation penalty (kept verbatim)

Word-level 5-gram overlap, free under 10%, linear penalty to −5.0 at
40%. Already built in v5 post-mortem.

### 6.5 OOD probe (compressed)

- **12 prompts** at T=0.3 (down from 600M's 50)
- Every 25 steps
- Auto-stop if OOD drops 2× in a row while reward EMA climbs

### 6.6 Per-env hit-rate logging (kept)

`math.exact_rate`, `ifeval.constraint_hit_rate`. Already built in v5.

### 6.7 Schedule

- **100 steps** (vs 600M's 500)
- Peak LR 5e-6 → cosine 5e-7
- kl_coeff 0.15
- Group size 6 (vs 600M's 8; saves rollout compute)
- max_gen_len 384
- ckpt at step 50 and 100

**5 H100-hours = $20**

---

## 7. Tool use — first-class commitment (kept, compressed)

Same lesson as OSRT-600M: tools are the highest-value capability
extension for a small model. Don't make this Stage 6 / optional.
We compress to fit.

### 7.1 Native tool tokens (already reserved in §3)

Same tokens as 600M. They're already in the tokenizer; they get
trained in pretraining via the 5% tool-using mix.

### 7.2 Tool registry — minimum viable

- **Calculator** — `numexpr` sandbox. The 17×23=391 failure dies the
  moment we wire this in.
- **Python exec** — sandboxed subprocess (we already built the v5
  hardening; reuse)
- Web search — **CUT** (would need API integration + budget)

### 7.3 Tool-use SFT + GRPO

- 1K synthetic tool-using examples (cheap to generate from any teacher)
- 200 steps SFT on top of GRPO checkpoint
- + 200 steps GRPO with tool-use env (math problems requiring calc/exec)
- Reward: `tool_call_format` (+1), `tool_result_useful` (+1),
  `tool_call_unnecessary` (-0.5), `tool_call_malformed` (-1)

**4 H100-hours = $16** (combined SFT + GRPO)

### 7.4 Expected impact

Multi-digit arithmetic → tool-callable → ~100%. The 17×23 hallucinations
disappear. Even though our 50M model doesn't "know" arithmetic, it can
INVOKE arithmetic.

---

## 8. Vision retrofit — kept, compressed to projector-only

Same Stage from OSRT-600M, compressed.

### 8.1 Encoder choice — frozen pretrained vision encoder

- **CLIP-ViT-B/16** (frozen, ~86M params) — small, fast, good
  quality. We don't train it; just use as a feature extractor.

### 8.2 Projector

- 2-layer MLP from CLIP's 512-dim output → our 512-dim LM hidden
- ~1M params, fully trainable
- Trains alongside HRA adapters; base LM frozen

### 8.3 Vision SFT

- 50K image-text pairs from LLaVA-Instruct-150K subset
- 500 steps SFT (projector + HRA only; base frozen)
- Format: `<|user|>[<|image|> tokens][text]<|assistant|>...`
- System prompts include "You can see images. Describe what you see."

### 8.4 Eval

- MMBench-100 subset (cheap)
- ScienceQA-100 subset

**12 H100-hours = $48**

---

## 9. Evaluation — built BEFORE training (kept)

Same lesson as OSRT-600M: build eval harness FIRST. The cost is small
and the value is enormous.

### 9.1 Benchmarks (one full pass at end)

- **gsm8k full (1319)** — math
- **IFEval full (541)** — instruction following
- **MMLU-Pro 200-subset** — knowledge
- **HumanEval (164)** — code
- **MMBench-100** — vision (post-retrofit)
- **MT-Bench short** — chat quality
- **12-prompt OOD** — generalisation
- **Per-loop CE** — depth utilization
- **Per-loop expert balance** — router health

### 9.2 Per-ckpt micro-eval (during training)

- Every 3B tokens pretrain: gsm8k 200 + MMLU 1K + OOD 12
- Every 25 GRPO steps: OOD 12 + per-env hit rate
- ~$5 across all ckpts

### 9.3 Final pass

- ~$5 for full sweep on final ckpt
- Comparison table: OSRT-50M vs Gemma 3 270M / Qwen3-0.6B / SmolLM2
- All scored at T=0.3, bare format

**3 H100-hours = $12** total eval (mid-training + final)

---

## 10. The 10 v5 lessons — codified (kept verbatim)

Same table as OSRT_600M.md §10. Every lesson applies regardless of
scale. The budget cuts WHAT we build, not HOW we build it.

| # | lesson | how OSRT-50M applies it |
|---|---|---|
| 1 | System prompts in pretraining | 10% chat-formatted pretrain text |
| 2 | Per-loop CE + OOD probe from day 1 | Built into every logging |
| 3 | Don't train base weights during RL | HRA-only GRPO default |
| 4 | Strict reward design from day 1 | strict extraction + hit-rate + OOD probe |
| 5 | Pretrain text includes inference chat template | think/answer tags in pretraining |
| 6 | Real benchmarks every ckpt | gsm8k/IFEval/MMLU auto-pipelined |
| 7 | Tool use first-class | Tokens day 1 + pretraining data + GRPO env |
| 8 | Pick the right size | 50M is OUR sweet spot for $280; recursive multiplier still applies |
| 9 | Faster feedback loops | 12-prompt OOD probe, frequent micro-eval |
| 10 | Tighter scoping | Math + IFEval + tools; not chasing multilingual or coding excellence |

---

## 11. Research nuggets to apply (kept verbatim from 600M)

All apply at 50M scale — none of the research findings are
size-dependent:

- Muon optimizer with Moonlight recipe (§2)
- Aux-loss-free expert balancing (DeepSeek-V3)
- Gram Newton-Schulz for Muon orthogonalization
- WSD schedule (vs cosine)
- Loop embeddings (Universal Transformer / Huginn / Ouro lineage)
- Frontier convergence defaults (decoder-only, RMSNorm pre+post,
  GQA, RoPE, SwiGLU, tied emb, QK-norm)

See OSRT_600M.md §11 for details — they're identical.

---

## 12. Cost estimate

| stage | tokens / steps | H100-hr | $ |
|---|---|---|---|
| Pretraining | 30B tokens, ~22K steps | 40 | **$160** |
| MOPD-style SFT | 5K rollouts × 300 steps | 7 | **$28** |
| HRA-only GRPO | 100 steps × 6 group | 5 | **$20** |
| Tool-use SFT + GRPO | 200 + 200 steps | 4 | **$16** |
| Vision retrofit | 500 steps SFT + projector | 12 | **$48** |
| Eval (mid + final) | per-ckpt + final sweep | 3 | **$12** |
| **Subtotal** | | **71** | **$284** |
| **Buffer / reruns** | | (folded in) | **−$4** |
| **Total** | | **70** | **$280** |

(API costs: ~$2 DeepSeek top-up, negligible.)

For comparison: OSRT-600M plan was ~$15,940. We deliver the same
pipeline at ~1.7% of that cost by going to a 50M model and cutting
token volume 100×.

---

## 13. Open questions / unknowns (same as 600M)

These are unchanged — scale doesn't resolve them:

1. **Per-loop routing accounting** — should each loop have its own
   router bias? Novel territory for sparse MoE + depth recurrence.
2. **Loops × layers trade-off** — is 3 blocks × 6 loops still optimal
   at 50M? Could ablate at 30M scale (proxy) — though we don't have
   budget for the ablation, the answer probably holds.
3. **Loss-free balancing under recurrence** — proven non-recurrent;
   needs validation when same router fires 6× per forward.
4. **Speculative decoding via aux-loop heads** — measure actual
   acceptance rate post-training.
5. **Vision via projector vs encoder-free** — we pick projector (CLIP
   frozen) for parameter efficiency at this scale; Gemma-4-12B
   encoder-free needs more params than we have.

---

## 14. Tooling commitments (kept verbatim)

- **Modal** for all training (volumes, spawn, rescue)
- **W&B dashboards from day 1**
- **modded-nanoGPT-style speedrun stack** as codebase reference
- **Auto-eval after every ckpt** via Modal `@app.function`
- **Structured experiment tracking** with run-ids, ckpt benchmarks,
  wandb URLs

## 14a. Deployment-first design choices

Small models live or die by inference cost. The 2026 frontier
(especially Gemma 4) treats QAT and speculative decoding as
first-class. We commit to these design choices now even if we don't
implement them during the $280 build:

- **Speculative decoding via aux-loop heads** — already trained for
  free via `aux_loop_loss_weight`. Inference path uses loop-3 output
  as draft prediction, loop-6 as verifier. Expected accept rate
  60-75 %. ~2× faster generation. Engineering: ~1 day post-training.
- **Quantization-aware training (QAT) friendliness** — use symmetric
  per-channel int8 for FFN weights; ensure RMSNorm placement doesn't
  block int8 fusion. Free at training; enables int8 deployment with
  minimal quality loss.
- **MTP draft model as deliverable** — ship the loop-3 sub-model as
  a separate artifact for speculative decoding clients that prefer
  an explicit draft.
- **TurboQuant KV-cache compression** — Google Research's
  random-rotation + per-block quantization scheme; compresses KV
  cache 4-8× with near-lossless quality. For our 6-loop recurrent
  arch this matters DOUBLY because each effective layer caches its
  own K/V, so cache memory is the dominant inference cost. Apply
  TurboQuant at int4 to the K/V projections in every block:
  - Cache footprint: **8× reduction** (bf16 → int4)
  - Long-context (8K) decode: now fits on consumer GPUs (4 GB VRAM)
  - Quality: ~0.01 perplexity delta per Google's results
  - Implementation: post-training, no retraining needed. Pairs well
    with QJL (Quantized Johnson-Lindenstrauss) for the routing
    matrices to keep the whole inference path int4.
  - **Engineering:** ~2 days, can use Google's reference implementation.

Combined with speculative decoding via aux-loop heads, this stack
gives:

| component | improvement |
|---|---|
| TurboQuant KV int4 | 8× cache, ~4× memory bandwidth |
| Speculative decoding (loop-3 draft) | ~2× generation speed |
| int8 weights (QAT) | 2× model memory |
| Combined | **fits on a phone / Raspberry Pi 5 at usable speed** |

This is the deployment story that makes a 50M recursive MoE a
deployable PRODUCT, not just a research artifact. The 2026 frontier
is all about deployment economics; we lean into it.

---

## 15. Expected outcomes (honest)

Calibrated against v5 results + research literature scaling laws +
the 100× token reduction:

| benchmark | OSRT-50M target | Gemma 3 270M | Qwen3-0.6B |
|---|---|---|---|
| gsm8k | ~10-18% | ~35% | ~45% |
| IFEval | ~25-35% | ~50% | ~60% |
| MMLU | ~22-28% | ~30% | ~45% |
| HumanEval | ~5-12% | ~15% | ~30% |
| MMBench (vision) | ~30-40% | n/a | n/a |
| 12-prompt OOD | ~5-7/12 | ~8/12 | ~9/12 |

**We do NOT beat Gemma 3 270M.** At 50M params and 790 tokens/param
(vs Gemma's 22,000) the gap is mostly trained-token count, not
architecture. With tools active, multi-digit arithmetic + counting +
conversions go from ~0% to ~100%, which can shift the perceived
capability dramatically on "everyday tasks" even though benchmark
numbers stay similar.

The PIPELINE works. The MODEL is small. That's the budget trade.

---

## 16. What this document is NOT

- **Not a recommendation to actually do this from scratch.** This
  documents what the $280 plan looks like done cleanly with all the
  v5 lessons applied. It does NOT make the case that pretraining a
  50M model from scratch is the best use of $280 — that's a separate
  judgement call (e.g. "polish v5 on top instead" — see git history
  for an earlier version of this doc that discussed that path).
- **Not a critique of OSRT_600M.md.** That's the right answer for
  $15K. This is the right answer for $280 IF we want a from-scratch
  build of the OSRT-600M pipeline at a smaller scale.
- **Not exhaustive.** Same architecture-level decisions are pinned;
  many smaller knobs (exact GQA ratios, batch sizes per stage, init
  schemes) follow the modded-nanoGPT speedrun defaults unless
  otherwise specified.

---

## 17. LFM2 integration (December 2025)

The Liquid AI LFM2 technical report (arXiv 2511.23404) shipped a
complete edge-first SLM family (350M-8.3B) on Dec 1, 2025. Several of
their findings directly upgrade our plan; the most impactful five
are integrated below.

### 17.1 Gated short convolutions instead of most attention

**LFM2's central architectural finding** — under realistic on-device
budgets, a minimal hybrid of gated short convolutions for most layers
+ a small minority of GQA blocks beats SSM/linear-attention hybrids,
beats attention-heavy stacks, and runs ~2× faster on CPUs. Their
hardware-in-the-loop search converged on this repeatedly across scales.

**The gated short conv block** (specific formula):
```
(B, C, h̃) = Linear(h)        # 3-way split along feature dim
y = B ⊙ h̃                    # input-aware gate
z = Conv_k(y)                 # depthwise 1D conv, kernel k=3
o = Linear_out(C ⊙ z)         # output gate + projection
```

For OSRT-50M, this is a **major architectural change** we should test.
Current plan: 3 attention blocks × 6 loops. LFM2 plan: most layers
gated-short-conv, minority GQA.

**Proposed OSRT-50M v2 layout:**
- 12 effective layers (2 physical blocks × 6 loops)
  - Block 0: gated short conv (kernel k=3)
  - Block 1: GQA (8q/2kv, our existing config)
- Pattern within each loop: GSC → GQA
- 3rd "block" becomes a SwiGLU FFN as before (the MoE part stays)

**Expected impact:** 1.5-2× faster CPU inference (LFM2's measured
result vs attention-heavy baselines), at similar or better quality.
**Engineering cost:** ~1-2 days; we already have the gated short
conv primitive (similar to SiLU-based gating used in v5).

**This is the single biggest update.** For an edge-deployed model
(which is what tiny recursive MoE wants to be), gated conv > attention
for most layers. Adopt.

### 17.2 Decoupled Top-K Knowledge Distillation

Our MOPD currently uses plain cross-entropy on the full teacher
response. LFM2 uses a much better objective — **decoupled, tempered
Top-K KL** that solves support mismatch:

```
L_DTK = KL(Bern(p_T(T)) || Bern(p_S(T)))           # binary mass term
      + p_T(T) · KL_τ(p_T(·|T) || p_S(·|T))        # conditional Top-K
```

Where T is the teacher's Top-K=32 token set. Temperature applies
only to the conditional term (avoids the OOD blow-up that naive
truncated tempering causes).

**Why it matters for us:** during MOPD we use ~5K rollouts. With
plain CE, each rollout teaches just the chosen response token. With
Top-K KD, each rollout teaches the **full teacher distribution** over
the top 32 alternatives — ~32× denser supervision per token.

**Cost:** ~$0 if teacher API returns logits (DeepSeek v4-flash does
via `logprobs=true, top_logprobs=20`). Just collect richer rollouts
during data collection. ~+10% data file size; negligible.

**Engineering:** ~half day to implement the decoupled-KL loss; the
math from §3.3 / Appendix A of the LFM2 paper is clean.

### 17.3 Curriculum learning via ensemble difficulty scoring

LFM2 uses **12 different LLMs** to score each SFT example's difficulty
(easy if most models get it right, hard if few do), then trains in
easy → hard order. Free quality gain.

For us with OpenHermes + DeepSeek rollouts: score each prompt with
e.g. 4-5 small open models (Qwen3-0.6B, SmolLM2-360M, Gemma 3 270M,
Phi-3-mini, our own MOPD ckpt). Records empirical `p_i` (prob of
success). Train sorted by `p_i` ascending.

**Cost:** ~$5 in API/compute for one-time difficulty scoring of 5K
examples. **Engineering:** ~half day.

### 17.4 Length-normalized preference optimization (LNPO)

When we eventually add preference data (Stage 5 in some future plan):
use LFM2's length-normalized DPO instead of plain DPO. Avoids the
"longer responses win because more tokens accumulate reward" pathology
that small models are especially vulnerable to.

```
Δ(x, yw, yl) = r_θ(x, yw)/|yw| - r_θ(x, yl)/|yl|
```

With **CLAIR refinement** (Contrastive Learning from AI Revisions):
take chosen responses from on-policy SFT outputs, refine through a
larger model into stronger "chosen" responses. They use this for the
preference dataset and report it's central to their IFEval gains
(LFM2-2.6B hits 79.56%).

For OSRT-50M: we're not budgeted for preference learning. Note for
OSRT_600M plan.

### 17.5 Model merging at end of post-training

LFM2 trains **multiple SFT/alignment variants in parallel** (different
data mixes, different curriculum schedules, different LR), evaluates
each, then **merges the best 3-5 via parameter-space techniques**
(model soup, task arithmetic, TIES, DARE, DELLA). The merged model
inherits balanced strengths.

For OSRT-50M: launch 3 SFT variants at different data weights, merge
at end. Costs ~$10 extra Modal (each variant is short). Pure win.

### 17.6 Other LFM2 findings worth noting (not integrated)

- **65,536 vocab BPE** with FIM + tool-calling + ChatML tokens. Our
  48K is fine; 65K would add ~16M params on the tied embedding — too
  expensive at 50M, fine at 600M.
- **10-12T token pretrain + 1T mid-training at 32K context** —
  validates our OSRT_600M plan to overtrain. We do 30B at $280
  because that's the budget; LFM2 confirms the direction.
- **LFM2-2.6B numbers as targets**: 82.41% GSM8K, 79.56% IFEval at
  2.6B / 11T tokens. **This is what to beat for OSRT_600M.** At our
  50M / 30B scale, beating this is not the goal — beating 270M-class
  baselines (Gemma 3 270M, ~35% GSM8K) is.
- **Hardware-in-the-loop search methodology** — they profile EVERY
  architecture candidate on real Snapdragon + Ryzen CPUs and discard
  failures. We should do this for OSRT_600M; out of scope for $280.
- **LFM2-Audio architecture** — separated continuous-in / discrete-out
  with RQ-Transformer for code generation. Bookmark for if we ever
  add audio.
- **Multi-stage VLM training: connector-only → joint with 5:5:1 LR
  ratio (text:connector:encoder) → multimodal SFT**. Use this exact
  recipe when we do vision retrofit.

### 17.7 Updated budget impact

The 5 integrations cost:
- Gated conv blocks: ~$0 (architectural; bake in to pretrain)
- Decoupled Top-K KD: ~$0 (data collection format change)
- Curriculum scoring: ~$5 (one-time API/compute)
- Length-norm DPO: out of $280 scope
- Model merging: ~$10 (parallel SFT variants)

**Net added cost: ~$15.** Fits in the $8 buffer + a slight pretrain
trim. Doable within $280.

**Expected quality lift:** 5-15% on benchmarks based on LFM2's
ablations (Top-K KD: dense supervision; curriculum: easier loss
landscape; merging: ensemble effect; gated conv: same quality at
higher throughput).

## Sources

Same as OSRT_600M.md plus:
- Liquid AI (Dec 2025) "LFM2 Technical Report" arXiv:2511.23404 —
  source for §17 integrations
