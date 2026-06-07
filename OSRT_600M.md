# OSRT-600M — "If we started again" design doc

**Author:** post-mortem of nano-osrt-100M (363M params, ~12 months of iteration)
**Date:** 2026-06-07
**Status:** design proposal — codifies every lesson from v5 + 2025-2026 SLM research frontier

This document captures **everything we would do differently** to build the
next-generation OSRT model from scratch, given full hindsight on the v5
training run. It pulls in:

1. Lessons from the v5 lineage (PLAN.md, ARCHITECTURE.md, TRAINING.md)
2. The recursive-MoE + Muon research synthesis (June 2026 SLM frontier)
3. Multimodal design notes (MULTIMODAL.md)
4. Specific gaps discovered DURING v5 training (system prompts,
   reward hacking, base-weight drift, etc.)

The target is a **600M-parameter recursive MoE** — bigger than v5 (363M)
where it matters (more routed experts, more knowledge), with EVERY
training-time decision driven by what we now know works.

---

## 0. The single biggest meta-lesson

> **"What do you want to measure?" should drive what you build, not the
> other way around.**

In v5 we built capability we hoped existed; we should have built
*measurement* of capability we knew we needed. Most of our pain came
from discovering problems late — loop collapse at 4 stages in, reward
hacking after a full run, missing system-prompt support after a year of
training. **Every architectural decision in OSRT-600M is anchored to
the metric that proves it works.**

---

## 1. Architecture — recursive MoE, refined

The recursive-MoE core (3 physical blocks applied 6 times via depth
recurrence + 1 shared + 8 routed top-2 experts) IS the differentiator.
Every released frontier model uses one of (a) dense, (b) MoE, or (c)
recurrent — none combine sparse MoE with depth recurrence. We keep
this. The refinements below address the v5 pain points.

### 1.1 Parameter budget

```
Embedding (32K × 1792, tied with LM head)      :  58,720,256   (4.2 % active)
Attention × 3 blocks (qkv + out_proj)          :  38,535,168   (2.7 %)
Shared experts × 3 (SwiGLU h=4608)             :  74,317,824   (5.3 %)
Routed experts: 3 × 12 × SwiGLU h=2304         : 366,051,328   (8.7 % active = top-2 of 12)
HRA adapters (rank 256, injected day 1)        :  86,114,304   (always trained)
Router + loop_emb + adapters + norms           :  ~1.5 M

Total physical params                          : ~625 M
Active per token                               : ~210 M (33.6 %)
Effective compute per token (× 6 loops)        : ~2.5 B FLOPs-equivalent
```

### 1.2 What's different from v5 (363M → 625M)

| change | v5 | OSRT-600M | rationale |
|---|---|---|---|
| Hidden dim | 1536 | **1792** | More representational headroom for math/code facts |
| Routed experts per block | 8 | **12** | Trend (Qwen3 128, OLMoE 64, DeepSeek-V3 256) — finer-grained specialisation |
| Routed expert hidden | 2048 | **2304** | Match dim increase |
| Shared expert hidden | 4096 | **4608** | Match dim increase |
| HRA adapter rank | 256 | **256** (kept) | Day-1 injection, not retrofit |
| Vocab | 32K | **48K** | English-centric but slightly more code/multilingual headroom; tying makes it cheap |
| Loops | 6 | **6** (kept) | Ouro found R4 sweet spot; v5 worked at R6; do NOT chase higher (Ouro reports instability at high recursion) |

The 625M total / 210M active is in the **same compute class as Phi-3-mini
(3.8B dense)** thanks to the 6× recurrence multiplier — so we punch
heavier than the dense-equivalent baseline.

### 1.3 Stability fixes for the recursive + MoE + Muon stack

These are not optional. They prevent the failure modes the v5 lineage
hit (loop collapse, router collapse, attention-logit explosion under
Muon):

- **QK-norm** on every attention block — Qwen3 standard, prevents the
  attention-logit blow-up that Muon (Moonshot found) and depth
  recurrence (Huginn found) both amplify. Kimi K2 added "QK-clip" on
  top; we should too.
- **Sandwich RMSNorm** (pre AND post) in the recurrent block — Huginn
  used four norms per block specifically to survive deep unrolling;
  Gemma 3 uses pre+post on every layer. v5 used pre-only and ate
  representation collapse twice.
- **Per-loop routing accounting** — this is novel territory. When the
  same router fires 6 times per forward, we must check whether per-loop
  expert load balances independently. If loop 1 always picks experts
  {0,1} but loop 6 always picks {3,4}, the global balance metric looks
  fine but the model has implicitly serialised the work. **Log per-loop
  expert-load distribution from step 1.**
- **Aux-loss-free balancing** (DeepSeek-V3 bias-update method) — see §2.2.

### 1.4 The "should we add MTP heads" question

YES. v5 had `aux_loop_loss_weight` train every intermediate-loop output
to predict the next token. This IS the MTP training recipe. At
inference we never use it as a draft model — that's leaving free 2-3×
inference speedup on the table. For OSRT-600M:

- Add a `generate_speculative()` path that uses loop-3 output as a
  draft (already trained to predict via aux loss), verifies with the
  loop-6 output. Expected accept rate 60-75%.
- Net: **~2× faster inference + rollout, ~$50-100 saved per GRPO run**.

---

## 2. Optimizer — Muon (production) + AdamW (embeddings)

This is the **largest single training-quality lever** we missed in v5.
The v5 lineage used Lion. The June 2026 research is unambiguous:

| optimizer | final loss (1200 steps, this arch) | balance | clean_emin |
|---|---|---|---|
| Lion + aux 0.10 | ~7.4 | ~1.2 | ~0.002 (collapsing) |
| Lion + aux 0.00 | ~7.6 | ~3.9 | 0.000 (full collapse by step 500) |
| **Muon + aux 0.10** | **3.43** | **1.02** | **0.105** (perfect) |
| Muon + aux 0.00 | ~4.7 | ~2.3 | ~0.001 |

A **4-point cross-entropy gap** (Muon vs Lion) on the same architecture.

### 2.1 Muon configuration (Moonlight recipe)

- **Muon for all 2D hidden matrices** (attention QKV/out, FFN gates/up/down, router projection).
- **AdamW for everything else** — embedding (tied), LM head, RMSNorm gains, biases, router bias accumulator.
- **Weight decay 0.01-0.10** on Muon params (Moonlight: required for scale; v5 + Lion didn't need this so we missed it).
- **Update-RMS alignment** across Muon and AdamW param groups (Moonlight specific).
- **Gram Newton-Schulz** (Dao-AILab) instead of standard Newton-Schulz for non-square matrices like attention QKV — up to 2× FLOP reduction on the orthogonalisation step using symmetric GEMM kernels (Hopper/Blackwell-optimised). Configure for 5 iterations with reset at iteration 2-3.
- **Sample efficiency**: Muon achieves comparable perplexity to AdamW at ~52% of training FLOPs in undertrained regime (where SLM pretraining lives). Memory: ~50% less than AdamW because no second-moment vector.

### 2.2 Aux loss — calibrate carefully

The research notes v5's aux=0.10 was likely **too high** (Switch
default 0.01, OLMoE 0.01, DeepSeekMoE 0.001). At α=0.10 the auxiliary
loss can distort the language-model gradient.

**OSRT-600M plan:** start with **DeepSeek-V3's loss-free bias balancing**
(`b_i` per-expert, nudged ±γ=0.001 by deviation from average load, NOT
in the gradient). With top-2-of-12 sparsity this is safely far from the
"super-high sparsity → lower-layer imbalance" regime that breaks the
loss-free method.

Backup: if loss-free shows pathologies, fall back to **α=0.01 aux + z-loss
0.001**. NOT α=0.10 again.

### 2.3 μP / μTransfer for HP search

Tune LR, init, multipliers on a **width-256 proxy**, zero-shot transfer
to width-1792. Saves orders of magnitude on HP search. Caveat: vanilla
μTransfer is "half-aligned" with Muon (Muon controls update spectral
norm but doesn't prevent weight drift); apply **strong decoupled weight
decay** to recover full feature-learning stability.

### 2.4 WSD schedule (not cosine)

- **Warmup-Stable-Decay** — MiniCPM popularised; SmolLM3 uses; Gemma uses.
- v5 used cosine for everything, which forced full-run commitments.
  WSD lets us branch decay phases at any point and inject high-quality
  data during decay. Massive flexibility win.
- 2000-step warmup → long stable phase at peak LR (2e-4) → 10% decay
  to high-quality math/code mix.

---

## 3. Tokenizer — chat template from day 1

The **single biggest v5 oversight**: `<|system|>` token id 13 existed
in the tokenizer but was never trained on. We discovered this in
month 12 and had to retrofit via SFT, with all the OOD risk that
implies.

### 3.1 Required tokens (single-token, baked into pretraining)

| token | id | purpose |
|---|---|---|
| `<|system|>` | reserved | system prompt opener |
| `<|user|>` | reserved | user turn opener |
| `<|assistant|>` | reserved | assistant turn opener |
| `<|end_turn|>` | reserved | turn separator (new!) — fixes the "no close tag" ambiguity v5 had |
| `<|think|>` / `<|/think|>` | reserved | reasoning block (kept from v5) |
| `<|answer|>` / `<|/answer|>` | reserved | answer block (kept from v5) |
| `<|tool_call|>` / `<|/tool_call|>` | reserved | tool invocation (single-token, NOT multi-piece like v5) |
| `<|tool_result|>` / `<|/tool_result|>` | reserved | tool result |

**Why `<|end_turn|>` is new:** v5 used open-only role tags (`<|user|>`...
`<|assistant|>`...`<|user|>`...) which forced the model to use the NEXT
role's open tag as the implicit close of the previous turn. This
breaks down when generating multi-turn responses because the model
has no clear "I'm done with this turn" signal. ChatML and Llama 3
templates use explicit end-of-turn markers; we should too.

### 3.2 Vocab size — 48K, not 32K

48K (between SmolLM2 49,152 and LFM2 65,536) balances:

- English-centric is the focus, so we don't need Llama 3's 128K or
  Gemma's 256K
- Slightly better code tokenisation than 32K (function names,
  operators)
- ~58M params for the tied embedding vs v5's 50M — modest extra cost
- Future fine-tuning to new domains has more room (Gemma team's
  rationale for 256K)

### 3.3 Pretraining text includes chat templates

This is critical. Mix into the pretraining corpus:

- **~10% chat-formatted text** with full `<|system|>...<|user|>...
  <|assistant|>...<|end_turn|>` structure (sources: OpenHermes,
  WildChat, ShareGPT, UltraChat-with-system, no_robots)
- **~5% tool-using text** with `<|tool_call|>calculator(2+2)<|/tool_call|>
  <|tool_result|>4<|/tool_result|>` patterns (synthetic + Glaive +
  ToolBench + Hermes-function-calling)
- **~5% reasoning text** with explicit `<|think|>...<|/think|>
  <|answer|>...<|/answer|>` (OpenThoughts, NuminaMath, DeepSeek-R1
  traces)

By the end of pretraining the model has seen each chat token tens of
thousands of times in natural context. SFT then becomes about ALIGNING
behaviours, not teaching FORMAT — a much easier learning problem.

---

## 4. Pretraining

### 4.1 Token budget — over-train heavily

Chinchilla = 20 tokens/param. Inference-efficient SLMs use **thousands**:

| model | total params | tokens | ratio |
|---|---|---|---|
| Llama-3-8B | 8B | 15T | 1,875× |
| SmolLM2-1.7B | 1.7B | 11T | 6,500× |
| **Gemma 3 270M** | 270M | **6T** | **22,000×** |
| Qwen2.5-1.5B | 1.5B | 18T | 12,000× |
| Ouro-2.6B | 2.6B | 7.7T | 3,000× |

For OSRT-600M (210M active), target **≥1.5T tokens** minimum, **3T+ if
budget allows**. MoE scaling tracks active params for loss but total
for capacity, so 1.5T-3T is the right band.

### 4.2 Data mixture — multi-stage with annealing

Use **3-stage WSD** schedule (SmolLM3 template):

**Stage 1 (0 → 70%):** Diverse base
- 60% FineWeb-Edu + DCLM (both, ~10% overlap — complementary)
- 12% Stack v2 / CodeParrot (code)
- 8% chat-formatted (OpenHermes, WildChat, UltraChat)
- 5% tool-using synthetic
- 5% reasoning (OpenThoughts, NuminaMath, R1-traces)
- 5% math (Nemotron-CC-Math-v1 — we have access; Open-Web-Math)
- 5% Wikipedia + RedPajama

**Stage 2 (70% → 90%):** Higher quality, more reasoning
- 30% FineMath, OpenMath
- 25% reasoning traces
- 20% code (filtered)
- 15% chat (curated for system-prompt + multi-turn)
- 10% Cosmopedia v2

**Stage 3 (90% → 100%, decay phase):** Anneal to best
- Highest-quality math/code only
- Domain-specific (chem, bio, physics) if relevant
- Heavy chat-format reinforcement
- Tool-using examples upweighted (system + tools = downstream win)

### 4.3 Avoid token-starvation

At 3T-token horizon, naive FineWeb-Edu filtering (which drops 90%) starves
us of unique tokens. **Use Nemotron-CC ensemble + synthetic rephrasing**
strategy: classifier-ensembled web + targeted synthetic upsampling.
Nemotron-CC's 15T-token Llama-3-8B beat Llama-3.1-8B by +5 MMLU using
exactly this recipe.

### 4.4 Validation — run benchmarks at each ckpt

This is the v5 anti-pattern we MUST fix. Every 500B-tokens-trained (or
every 5K steps), run:

- **gsm8k** (math reasoning)
- **MATH-500** (harder math)
- **MMLU-Pro subset** (knowledge)
- **IFEval** (instruction following)
- **HumanEval** (code)
- **Per-loop CE loss** (Test 3 — depth utilization probe)
- **OOD probe** (the 12-prompt diverse capability check we built late
  in v5)
- **Active-routed-expert distribution** (per loop, per stage)

Auto-pipelined. We had 20 v5 checkpoints and ran gsm8k on zero of
them. That's the project blindspot.

---

## 5. SFT / MOPD

### 5.1 Data — system prompts from day 1, multi-turn from day 1

v5 MOPD used `<|user|>{q}<|assistant|>{response}` — no system prompts,
no multi-turn. Even the rollout collection script didn't capture
system roles from the teacher API.

**OSRT-600M rollout collection:**

1. **Vary system prompts** across requests (~50 templates ranging from
   minimal-format to 3-shot embedded-examples)
2. **Capture multi-turn** when teacher supports it (DeepSeek v4 does)
3. **Format on training-side** as full
   `<|system|>{sys}<|user|>{q1}<|assistant|>{a1}<|end_turn|><|user|>{q2}
   <|assistant|>{a2}` chains
4. **Loss-mask the prefix** so the model attends to but doesn't
   regurgitate the system prompt
5. **Mix system+no-system 70/30** so the model can also handle bare
   user input (the format we currently support)

### 5.2 Teachers — DeepSeek v4 + Gemini 3.5 + open-source fallback

What v5 used:
- Gemini 3.5 Flash ($1.50/$9.00 per 1M) — too expensive
- DeepSeek v4-flash ($0.14/$0.28 per 1M) — winner, 2500 concurrent

What OSRT-600M should add:
- **Self-hosted Gemma 3 4B** on Modal — we have gated access; zero
  API cost; ~$0.005/rollout amortised; supports system prompts
  natively
- Use mix-of-teachers (DeepSeek + Gemma + maybe Llama 3) for diversity

### 5.3 Training format

```
<|system|>{system}<|user|>{question}<|assistant|><|think|>{thinking}
<|/think|><|answer|>{answer}<|/answer|><|end_turn|>
```

Loss only on assistant turn (everything before `<|assistant|>` is
masked with -100). Critical: the model learns to ATTEND to the
system prompt without gradient pulling it to ECHO. The
regurgitation-penalty in §6.3 is the GRPO-time backstop.

---

## 6. RL / GRPO — HRA-only, strict rewards, OOD probe

### 6.1 HRA-only training (LoRA-only RL)

v5 GRPO trained both base weights and HRA adapters. Result: base weight
drift caused inference 4/6 → 2/6 regression at step 150 — the exact
failure mode the entire "tighter knobs" debate in v5 was unable to fix
(because the knobs were the wrong tool — we needed to STOP TRAINING THE
BASE).

**OSRT-600M GRPO:** freeze the 540M base weights, train only the 86M HRA
adapters. Standard LoRA-only RL pattern (DPO/PPO/GRPO papers all do
this). Benefits:

- MOPD/SFT capability anchor structurally preserved
- KL drift bounded by construction (base contribution to logits stays
  fixed; only the additive HRA contribution changes)
- ~4× fewer params getting Adam state (memory + speed)
- Lower risk of catastrophic forgetting

### 6.2 Strict reward extraction (no last-number-wins)

v5 used `extract_numeric_answer(text)` which returns the LAST number in
the answer block. GRPO learned to hack this by dumping multiple
candidate numbers ("I tried 50, then 32, but actually 18") — last-token
match → +3 reward. We measured this AFTER damage was done.

**OSRT-600M reward stack:**

- `extract_numeric_answer_strict()` — returns answer only at high
  confidence (single number, boxed, concluding phrase)
- Ambiguous answers get `-0.5` penalty (not 0)
- Loose extractor kept for inference scoring only

### 6.3 Regurgitation penalty for system-prompt era

When the model can be trained to USE system prompts, it can also be
trained to PARROT them back. We need a heavy penalty:

- Word-level 5-gram overlap between system prompt and completion
- Free under 10% overlap
- Linear penalty to **−5.0** max at 40% overlap
- Tested in v5 post-mortem; should be on by default in any GRPO run
  that uses system prompts

### 6.4 OOD probe baked into the training loop

v5 built this AFTER the regression that motivated it. OSRT-600M:

- 20-50 held-out prompts NOT in the training distribution
- Run at low temp (T=0.3) every N steps
- Logged to wandb as `grpo/ood_score`
- **Auto-stop if OOD drops 2× in a row while training reward
  climbs** — that's the reward-hacking signature

### 6.5 Per-env capability hit rate (not just reward)

v5 added hit-rate logging after the first GRPO run already
demonstrated mbpp pure-format-hacking at 0/180 tests passing. From
day 1 for OSRT-600M:

- `math.exact_rate`, `math.partial_rate`
- `ifeval.constraint_hit_rate`
- `code.test_pass_rate`
- Whatever env we add, its CAPABILITY metric (not just reward EMA)

Without this, "reward EMA climbing" is meaningless — it could be
hacking, learning, or both.

### 6.6 Sandboxed code exec (carry from v5)

The v5 `mbpp_test_reward` security hardening (subprocess with stripped
env, tempdir cwd, process-group kill on timeout, absolute python path)
is good — keep it. Add Modal Sandbox for true isolation when budget
allows.

---

## 7. Tool use — Stage 6 made first-class

For a 600M model, tools are the single highest-value capability
extension. v5 treated this as "Stage 6, optional." OSRT-600M makes it
a day-1 architectural commitment.

### 7.1 Native tool tokens (in tokenizer + pretraining)

```
<|tool_call|>calculator("17 * 23")<|/tool_call|>
<|tool_result|>391<|/tool_result|>
<|answer|>The answer is 391.<|/answer|>
```

Tokens reserved in tokenizer; pretrained on synthetic tool-using
examples + curated open data (Hermes-function-calling, Glaive,
ToolBench).

### 7.2 Tool registry — minimum viable

1. **Calculator** — `numexpr` sandbox, evaluates arithmetic. The
   17×23=391 failure that defined v5 dies the moment the model can
   call this.
2. **Python exec** — sandboxed (Modal Sandbox, not in-process), runs
   short snippets for multi-step problems.
3. **Web search** (optional, post-launch) — Brave Search API + result
   summarisation.

### 7.3 Tool-aware GRPO

Adds tool-use env to the multi-env GRPO mix:

- Reward `tool_call_format` (+1 if parseable expression)
- Reward `tool_result_useful` (+1 if tool's result appears in final
  answer)
- Penalty `tool_call_unnecessary` (-0.5 for calling calculator on 1+1)
- Penalty `tool_call_malformed` (-1)
- Plus all existing format + correctness rewards

### 7.4 Expected impact

Multi-digit arithmetic / large-number division → ~100% with calculator
tool. Date math, conversions, counting → tool-callable. We'd never see
"17 × 23 = 459" again.

---

## 8. Vision retrofit (pull from MULTIMODAL.md)

Encoder-free Gemma-4-12B style or LLaVA-style projector — both
documented in MULTIMODAL.md. Key v5 → OSRT-600M change:

- **Vision SFT data must include system prompts** ("You can see
  images. Describe what you see.") — same lesson as text training
- **Vision GRPO uses HRA-only** — same lesson as text GRPO
- **MMBench + ScienceQA eval at every ckpt**

Vision is a Stage 5 task; ~$120-160 Modal. Same architecture decisions
as v5 multimodal plan but with the system-prompt + HRA-only fixes.

---

## 9. Evaluation — built BEFORE training

This is the v5 anti-pattern most worth killing. Build the eval harness
FIRST, train SECOND. Concrete:

| benchmark | what it measures | when to run |
|---|---|---|
| **gsm8k** (full 1319) | grade-school math | every 500B tokens pretrain, every 100 GRPO steps |
| **MATH-500** | competition math | every 1T tokens |
| **MMLU-Pro** (1K subset) | knowledge | every 500B |
| **IFEval** (541) | instruction-following | every 100 GRPO steps |
| **HumanEval** (164) | code generation | every 500B |
| **MMBench** (vision) | visual reasoning | post-vision retrofit only |
| **GPQA** (diamond, 198) | hard reasoning | end of training |
| **MT-Bench** | chat quality | end of SFT |
| **OOD probe** (50 diverse) | generalisation | every 25 GRPO steps |
| **Per-loop CE** | depth utilisation | every ckpt |
| **Per-loop expert balance** | router collapse | every ckpt |

Auto-pipelined to a wandb dashboard. Cost: ~$5-10 per full eval pass
on a single H100, negligible against training cost.

---

## 10. The 10 v5 lessons, codified

| # | lesson | how OSRT-600M fixes it |
|---|---|---|
| 1 | **System prompts must be in pretraining**, not retrofitted | ~10% chat-formatted pretraining text with full `<\|system\|>` blocks |
| 2 | **Per-loop CE + OOD probe from day 1**, not retrofitted | Both built into every training stage's logging |
| 3 | **Don't train base weights during RL** | HRA-only GRPO is default |
| 4 | **Strict reward design from day 1** | `extract_numeric_answer_strict`, ambiguous penalty, hit-rate logging, OOD probe — all defaults |
| 5 | **Pretrain text includes inference chat template** | `<\|think\|>`/`<\|answer\|>` tokens trained in pretraining, not just at SFT |
| 6 | **Run real benchmarks every ckpt**, not "at the end" | gsm8k + MMLU + IFEval auto-pipelined per ckpt |
| 7 | **Tool use is a first-class commitment**, not Stage 6 | Tokens reserved + pretraining data + GRPO env from day 1 |
| 8 | **600M is the sweet spot**, not 363M | More routed experts → more facts can be stored without proportional compute increase |
| 9 | **Faster feedback loops** (MTP retrofit + benchmarks + faster sanity) | MTP-aware generate(), 10-step sanities, parallel eval |
| 10 | **Tighter scoping** | "best small reasoning-with-tools model" as headline focus; chat/multilingual are secondary |

---

## 11. Research nuggets to apply

### 11.1 Muon optimizer (the headline 2026 frontier)

- **2× compute efficiency vs AdamW** at compute-optimal (Moonshot 3B/16B
  MoE, 5.7T tokens)
- **~50% memory savings** (no second moment vector)
- **4-point cross-entropy gap** vs Lion in ablation (this exact arch)
- Apply only to 2D hidden matrices; AdamW for embeddings + LM head +
  norms + biases
- Required additions: **weight decay**, **update-RMS alignment**,
  **QK-norm + QK-clip** (Muon makes attention logits more prone to
  blow-up)

### 11.2 Aux-loss-free expert balancing (DeepSeek-V3)

- Per-expert bias `b_i`, nudged ±γ=0.001 by load deviation
- NOT in the gradient (heuristic-only)
- Removes the gradient distortion of α=0.10 aux loss
- Safe at low-to-medium sparsity (top-2 of 12 is fine; super-high
  sparsity = lower-layer imbalance)

### 11.3 Gram Newton-Schulz (Dao-AILab)

- Drop-in for standard Newton-Schulz in Muon
- Iterates on small Gram matrix XX^T instead of full X
- ~2× FLOP reduction for non-square matrices (attention QKV, gates)
- Symmetric GEMM kernels = Hopper/Blackwell optimised
- 5 iterations, reset at step 2-3

### 11.4 WSD schedule

- Warmup → Stable → Decay (vs cosine)
- Branch decay at any point → inject high-quality data
- SmolLM3 / MiniCPM / Gemma 3 use this
- v5 used cosine and forced full-run commitments

### 11.5 μP / μTransfer

- Tune HPs on width-256 proxy → transfer to 1792
- Saves 10-100× on HP search
- Half-aligned with Muon — add strong weight decay

### 11.6 Loop embeddings (LIE) — kept from v5

- Per-loop bias vector added before attention block
- `min(r, 7)` cap means we can't safely loop past R=8 — kept at R=6
- Critical for letting shared weights act as virtual distinct layers
- Already in v5 — keep, don't change

### 11.7 Ouro / Huginn validation

- Looped 1.4B (R4) beats dense Qwen3-4B on GSM8K (78.92 vs 72.86)
- Validates the recursive-MoE thesis: parameter efficiency via
  weight tying with depth recurrence works
- 6 loops × 3 blocks is a sweet spot — Ouro reports instability
  beyond R6, Huginn at R32 needs heavy stabilisation

### 11.8 Frontier convergence (use defaults that work)

- Decoder-only, RMSNorm pre-norm, GQA, RoPE, SwiGLU — universal
- Tied embeddings at ≤4B — Qwen3 standard
- QK-norm — Qwen3, Gemma 3, modded-nanoGPT
- WSD schedule — SmolLM3, MiniCPM, Gemma 3
- AdamW baseline → migrate to Muon

---

## 12. Cost estimate (single-pass, no reruns)

| stage | tokens / steps | cost |
|---|---|---|
| Pretraining (3T tokens, 600M model) | 3T tokens, ~50K H100-hours | **~$15K** |
| MOPD-style SFT (5K rollouts × 500 steps) | 5K rollouts (~$10 API) + 1.5K H100-hours | **~$700** |
| HRA-only GRPO (multi-env, 500 steps) | 500 steps × ~30 sec compiled | **~$15** |
| Vision retrofit (LLaVA-style) | 2K steps SFT + 200K image-text pairs | **~$200** |
| Tool-use GRPO (500 steps) | 500 × ~30 sec | **~$15** |
| Eval suite (full sweep) | 1× through gsm8k + MMLU + IFEval + etc | **~$10** |
| **Total** | | **~$15,940** |

For comparison: SmolLM2 (Allal et al.) total cost was ~$250K at 11T
tokens on a 1.7B dense model. OSRT-600M at 3T tokens / 600M MoE = ~6%
of that. We'd ship a meaningfully more capable model for ~$16K because
of the recursive + sparse efficiency.

---

## 13. Open questions / unknowns

These are genuinely unresolved and worth ablating:

1. **Per-loop routing accounting** — should each loop have its own
   router bias / aux loss, or one global? (Novel territory — no published
   precedent for sparse MoE + depth recurrence.)
2. **Loops × layers trade-off** — is 3 blocks × 6 loops better or
   worse than 4 × 4 or 2 × 8 at fixed compute? Should ablate at
   600M scale.
3. **Loss-free balancing under recurrence** — DeepSeek-V3's method is
   proven in non-recurrent MoE. Does the same bias-update converge
   when the same router fires 6× per forward?
4. **Speculative decoding via aux-loop heads** — what acceptance rate
   do we actually get? Need to measure.
5. **Vision via projector vs encoder-free** — Gemma-4-12B claims
   encoder-free works; LLaVA proves projector works. Smaller model
   may bias toward projector for parameter efficiency.

---

## 14. Tooling commitments

- **Modal** for all training (volumes, spawn, rescue) — same as v5
- **W&B dashboards from day 1** (not retrofitted)
- **modded-nanoGPT-style speedrun stack** as the codebase reference
  (Muon + QK-norm + ReLU²/SwiGLU + FP8 head + μP-like init)
- **Auto-eval after every ckpt** via Modal `@app.function` triggered
  on volume commit
- **Structured experiment tracking** — every run gets a run-id, every
  ckpt gets benchmarks, every benchmark gets a wandb URL

---

## 15. What this document is NOT

- **Not a commitment to build OSRT-600M.** It's a design doc capturing
  the lessons. Whether we use it is a separate decision.
- **Not a critique of v5.** v5 was the path that taught us this. The
  368M model exists; the lessons exist. Both are valuable.
- **Not exhaustive.** There are 100+ smaller decisions (init schemes,
  exact GQA ratios, batch sizes per stage) that are TBD. This doc
  fixes the architecture-level choices.

---

## Sources

- v5 lineage: `PLAN.md`, `ARCHITECTURE.md`, `TRAINING.md`,
  `MULTIMODAL.md`
- Muon: Moonshot "Muon is Scalable for LLM Training" (arXiv 2502.16982);
  KellerJordan/Muon repo
- Gram Newton-Schulz: Dao-AILab repo
- AdaMuon / Newton-Muon: arXiv 2025 variants
- Ouro / LoopLM: Zhu et al. arXiv 2510.25741
- Huginn / recurrent depth: Geiping et al. arXiv 2502.05171
- DeepSeekMoE: Dai et al. arXiv 2401.06066
- DeepSeek-V3: technical report + Hugging Face card
- OLMoE: Muennighoff et al. arXiv 2409.02060
- SmolLM2/3: Allal et al. arXiv 2502.02737
- Gemma 3: official model card
- Nemotron-CC: Su et al. arXiv 2412.02595
- modded-nanoGPT: speedrun records, GitHub
- μP/μTransfer: Yang et al. Tensor Programs V (arXiv 2203.03466)
- WSD: MiniCPM paper, SmolLM3 docs
- Frontier comparators: Qwen3, LFM2, SmolLM, Gemma 3, OLMoE, DeepSeek-V3
- Cell A/B/C/D ablation: the v5 1200-step probe + supplied research
  synthesis
