# nano-osrt — Post-extend3 Plan

**Owner:** Daniel · **Last update:** 2026-06-06 · **Current state:** MOPD step 200/1000 (in progress)

Living document — update inline as we progress. Per-stage cost reconciliation
and probe results captured at the bottom of each stage block.

---

## Where we are

| asset | status |
|---|---|
| Architecture | ✅ Fixed (loopfixv2 + extend3 with aux_loop_loss_weight + loop dropout) |
| Best ckpt | `extend3_merged.pt` (Test 3 + inference verified, no `<\|answer\|>` yet) |
| Rollouts | ✅ 13,368 done (Gemini 5,440 + DeepSeek 7,924 + ~10 unknown) |
| MOPD | 🟢 in progress on codhe-hugging-mcp (step 200/1000) |
| Budget — Modal | ~$19 codhe-hugging-mcp + $15 danielhalwell = $34 |
| Budget — API | ~$210 Gemini + DeepSeek used ~$5 = plenty left |

### Teacher provider catalog (in scripts/collect_rollouts.py)

| key | provider | $/1M in | $/1M out | notes |
|---|---|---|---|---|
| `gemini-3.5-flash` | Gemini API | $1.50 | $9.00 | expensive, used for initial 5440 |
| `gemini-2.5-flash` | Gemini API | $0.075 | $0.30 | cheap fallback |
| `nemotron-3-ultra-free` | OpenRouter | $0 | $0 | top quality, ~100s/rollout (too slow) |
| `deepseek-or-v3.1` | OpenRouter | $0.27 | $1.10 | OpenRouter routed |
| `deepseek-v4-flash` | DeepSeek direct | $0.14 | $0.28 | **winner** — 13-15s/rollout, 2500 concurrent |
| `deepseek-v4-pro` | DeepSeek direct | $0.435 | $0.87 | reasoning-heavy variant |

---

## Pipeline overview

```
extend3_final.pt
   ↓ stage 1: probe + merge        (free)
extend3_merged.pt
   ↓ stage 2: MOPD distillation   ($5-7 Modal)
mopd_final.pt
   ↓ stage 3: multi-env GRPO      ($25 Modal — needs top-up)
grpo_final.pt
   ↓ stage 4: rejection-sample SFT polish  (optional, $5-10)
polished_final.pt
   ↓ stage 5: vision retrofit + multimodal SFT  ($120-160 Modal)
multimodal_final.pt
   ↓ stage 6: full eval suite     ($10-20 Modal)
```

Total Modal: $165-225 (Gemini API: $30-50).

---

## Stage 1 — Probe + checkpoint merge

**Status:** ✅ complete
**Cost:** $0
**Duration:** ~45 min

### Actions
- [x] Downloaded `extend3_final.pt` (3.2 GB)
- [x] Probed extend3_final → Test 3 OK, depth utilization preserved
- [x] Inference test on extend3_final → 1/6 (regression of 1 prompt vs loopfixv2_merged)
- [x] Sliding-window merge of step 1800, 2100, 2400, 2700, final → `extend3_merged.pt`
- [x] Probed merged → marginal improvement at n=6
- [x] Inference test on merged → 2/6 (recovered)

### Results

**Test 3 — Loop ablation CE loss (n=1 → n=6 across all ckpts):**

| n | extend2 (broken) | loopfixv2_merged | extend3_final | **extend3_merged** 🏆 |
|---|---|---|---|---|
| 1 | 9.58 | 4.14 | 4.00 | **4.02** |
| 2 | 9.31 | 3.85 | 3.69 | **3.69** |
| 3 | 9.06 | 3.73 | 3.66 | **3.61** |
| 4 | 8.83 | 3.65 | 3.65 | **3.61** |
| 5 | 9.21 | 3.60 | 3.66 | **3.63** |
| 6 | 3.24 | 3.46 | 3.58 | **3.56** |

**Inference test (8 prompts, 6 scoreable):**
- extend2 (broken): 1/6 (17%)
- loopfixv2_merged: 2/6 (33%)
- extend3_final: 1/6 (regressed by 1 prompt)
- **extend3_merged: 2/6 (33%)** — merge recovered the regression

**Adapter compression** continued throughout extend3 (L0: 7.10 peak → 5.95 final).

**Key insight:** extend3 with loop dropout produced a flatter performance curve (better at shallow depths, slight cost at full depth). Merging across the converging steps recovers most of the peak full-depth ability.

**Canonical going forward:** `extend3_merged.pt`. Still no `<|answer|>` tag in generated text — that's the gap MOPD fixes.

---

## Stage 2 — MOPD distillation

**Status:** 🟢 in progress (step 200/1000 at time of last update)
**Cost estimate:** ~$5-7 Modal · spent so far: ~$1 + $108 API
**Duration:** ~2.5 hr

### Actions
- [x] Build pipeline (RolloutDataset + MOPDConfig + mopd Modal stage + dispatcher)
- [x] Collect Gemini 3.5 Flash math (4000) + reasoning (~1440) — $49
- [x] Add OpenRouter + DeepSeek direct teachers to collector
- [x] Collect DeepSeek v4-flash diverse (reasoning 1560 + chat 3000 + science 3000 + code 374) — $4.29 in 15.5 min at concurrency 2000
- [x] Verify local RolloutDataset loads 13,368 (filtered 6 empty responses)
- [x] Upload to codhe-hugging-mcp `osrt-rollouts` volume
- [x] Launch `mopd_sanity` (30 steps) → task 2.21 → 1.37, format pipeline OK
- [x] Launch `mopd` full (1000 steps from `extend3_merged.pt`)
- [ ] Re-run probe + inference test on `mopd_final.pt`

### Config (already built — `MOPDConfig` in train_config.py)
- 1000 steps, batch=4 × accum=16, seq_len=1024
- peak_lr 1.5e-6 → cosine 1.5e-7
- aux_loop_loss_weight=0.05, loop_dropout=0.10 (keeps depth fix active)
- Resume from extend3_merged

### Success criteria
- Inference test: model produces `<|answer|>X<|/answer|>` reliably (≥5 of 6 scorable prompts)
- Task loss settles below extend3_merged baseline
- Math primitives improve: at least 3/6 correct on inference test
- Loop utilization preserved (probe Test 3 still flat)

### Results

**Rollout dataset (`rollouts/mopd_v1.jsonl`, 13,368 valid records):**

| source | count | teachers |
|---|---|---|
| math | 4,000 | Gemini 3.5 Flash |
| reasoning | 3,000 | Gemini (~1440) + DeepSeek v4-flash (~1560) |
| chat | 3,000 | DeepSeek v4-flash |
| science | 3,000 | DeepSeek v4-flash |
| code | 374 | DeepSeek v4-flash (all of MBPP train) |

Total API spend: ~$108 (Gemini $103 + DeepSeek $4.29).

**Sanity (30 steps):** task 2.21 → 1.37 in 30 steps, rollout loader confirmed, format alignment trajectory established.

**Full run trajectory (complete, 1.6 hr, 1000 steps):**
- step 0: 2.38 → step 100: 1.79 → step 200: 1.79 → step 300: 1.35
- step 400: 1.40 → step 500: 0.85 → step 600: 0.76 → step 700: 0.39
- step 800: 0.37 → step 900: 0.15 → final saved

**Post-MOPD probe (held-out CE, lower=better):**

| n_loops | extend3_merged | mopd_final | Δ |
|---|---|---|---|
| 1 | 4.02 | 3.63 | -0.39 |
| 2 | 3.69 | 3.32 | -0.37 |
| 3 | 3.61 | 3.15 | -0.46 |
| 4 | 3.61 | 3.09 | -0.52 |
| 5 | 3.63 | 3.08 | -0.55 |
| 6 | 3.56 | **3.00** | **-0.56** |

**Post-MOPD inference test:** 3/6 (50%) — up from 2/6 (33%). FIRST `<|answer|>` tag appeared. Janet eggs word problem now correct. Multi-digit arithmetic (17×23) still failing — that's a GRPO target.

**Verdict:** MOPD delivered without memorization collapse. Format alignment underway. Depth utilization preserved. Canonical: `mopd_final.pt`.

---

## Stage 3 — Multi-env GRPO

**Status:** ⏳ pending (needs Modal top-up before launch)
**Cost estimate:** ~$25 Modal · ~3-4 hr
**Blocker:** Need additional Modal credits before launch

### Actions
- [ ] Top up Modal workspace (codhe-hugging-mcp or new workspace)
- [ ] Build `MultiEnvGRPOConfig` (extending existing GRPOConfig)
  - Math env: gsm8k-style problems with numeric verifier
  - Code env: HumanEval-style with test-passing verifier (subprocess sandbox)
  - IFEval env: instruction-following constraints (regex verifier)
- [ ] Add per-env reward functions to `rewards.py`
- [ ] Launch GRPO from `mopd_final.pt`, ~2000 steps
- [ ] Probe + inference + reward-per-env tracking

### Config
- Resume from `mopd_final.pt`
- Total steps: 2000 (cosine warm 50 → peak 5e-6 → cool 5e-7)
- aux_loop_loss_weight kept at 0.03 (preserve depth during RL)
- Group size 8, KL coeff 0.01 (low — we want capability gain, not just stability)
- Multi-env: round-robin or weighted sampling across math/code/IFEval

### Success criteria
- gsm8k accuracy crosses 50% (current peak ~43% on broken architecture)
- IFEval pass rate ≥ 60% on a 100-prompt subset
- HumanEval-test ≥ 15% (low bar for 363M model)
- Loop utilization still preserved

### Results
_TBD_

---

## Stage 4 — Rejection-sample SFT polish (optional)

**Status:** 🤔 conditional (only if eval after GRPO shows stylistic failures)
**Cost estimate:** ~$5-10 Modal · ~1 hr

### Trigger conditions
Run this if post-GRPO eval shows:
- Rambling / second-guessing in reasoning chains
- Format errors (missing answer tag, wrong delimiters)
- Inconsistent answer style across categories

### Actions (if triggered)
- [ ] Sample 4-8 completions per prompt at T=0.7 from `grpo_final.pt`
- [ ] Score with verifiable reward functions
- [ ] Filter: take top-1 per prompt
- [ ] SFT-train on filtered completions, 300-500 steps, low LR (1e-6)

### Success criteria
- Specific failure modes from triggering eval are reduced
- No regression on existing capabilities

### Results
_TBD_

---

## Stage 5 — Vision retrofit (encoder-free, Gemma 4 12B style)

**Status:** ⏳ pending (post-text-pipeline)
**Cost estimate:** ~$120-160 Modal · ~6-8 hr training
**Architectural change:** new module + multimodal training stage

### Design — encoder-free architecture
Following the Gemma 4 12B paper: bypass ViT entirely, project raw 48×48 RGB patches directly into the LLM hidden dim with a single matmul.

```
image (224×224 RGB)
  ↓ split into 16 patches of 48×48 (5×5 grid w/ slight overlap, or 4×4 = 16 patches at 56×56)
patches (16, 48, 48, 3) → flatten to (16, 6912)
  ↓ vision_proj: Linear(6912 → 1536)   # ~10.6M params at hidden=1536
patch tokens (16, 1536)
  ↓ + factorized X/Y coordinate lookup (Gemma 4 12B trick)
positioned patch tokens (16, 1536)
  ↓ prepend with <|image|> ... <|/image|> markers
hidden state for transformer
```

Coordinate factorization (saves params vs full 2D positional embedding):
- `x_embed: Embedding(grid_size, hidden_dim // 2)` — ~6K params
- `y_embed: Embedding(grid_size, hidden_dim // 2)` — ~6K params
- Concatenate `[x_embed(i), y_embed(j)]` per patch

Total vision params: ~10.6M (vs 35M at 12B → scales reasonably to 363M backbone).

### Actions
- [ ] Add `VisionProjector` module in `src/nano_osrt/vision.py`
- [ ] Add `<|image|>` / `<|/image|>` special tokens (extend tokenizer or repurpose unused IDs)
- [ ] Hook into `NanoOSRTForCausalLM.forward()` — when image_patches given, project and inject
- [ ] Add `MultimodalSFTConfig` in train_config.py
- [ ] Build multimodal dataloader (image + prompt + response)
- [ ] Multimodal datasets to use:
  - LLaVA-Instruct (158K image-instruction pairs) — chat-style
  - AI2D (5K diagram QA) — diagram reasoning
  - ChartQA (28K chart QA) — quantitative
  - DocVQA subset (10K) — document understanding
  - Total target: ~50-100K examples
- [ ] Two-stage training:
  - **Stage 5a:** projector-only (freeze backbone), 500 steps, peak_lr 5e-5 — learn projection
  - **Stage 5b:** unfreeze backbone, full SFT 2000 steps, peak_lr 1e-6 — capability merge
- [ ] Probe + inference test (text-only) — verify no text regression
- [ ] Inference test (image+text) — sample visual capability

### Success criteria
- Text-only inference doesn't regress vs pre-vision baseline (within 0.1 CE)
- Model can describe simple images coherently
- Can answer questions about charts/diagrams at chance+ level on AI2D test split

### Risk
At 363M params, vision-text capacity competition is real. Gemma 4 12B has 12B params to share between modalities; we have 25× less. Expectation: modest visual capability, primarily proving the architecture works rather than competitive VLM performance.

### Results
_TBD_

---

## Stage 6 — Full eval suite

**Status:** ⏳ pending (final stage)
**Cost estimate:** ~$10-20 Modal · ~1-2 hr

### Benchmarks
| benchmark | what it measures | size |
|---|---|---|
| gsm8k full | grade-school math word problems | 1319 test |
| MATH-500 | competition math subset | 500 test |
| MMLU-Pro subset | knowledge | 1000 test |
| IFEval | instruction following | 541 test |
| HumanEval | code generation | 164 test |
| MMBench (if vision done) | visual reasoning | 4327 test |
| ScienceQA (if vision done) | multimodal science QA | 4241 test |

### Actions
- [ ] Build `eval_full.py` script (extends existing probe infrastructure)
- [ ] Run sequentially, save per-benchmark JSON results
- [ ] Compare to:
  - Original extend2_final (broken architecture baseline)
  - Architecture-fix-only (loopfixv2_merged)
  - Final model
- [ ] Write up final report

### Results
_TBD_

---

## Decision points

### After stage 1 (probe extend3)
- If extend3 task CE > loopfixv2_merged → something regressed, investigate before MOPD
- If extend3 inference test = 2/6 (same as loopfixv2) → expected, MOPD will provide format gains

### After stage 2 (MOPD)
- If inference test shows `<|answer|>` reliably → proceed to GRPO
- If inference test still misses `<|answer|>` → brief format-only SFT first (~$3, 200 steps)

### After stage 3 (GRPO)
- If gsm8k > 50% → skip stage 4 (polish) unless other failure modes
- If gsm8k 30-50% → consider stage 4 polish, possibly more GRPO
- If gsm8k < 30% → diagnose; likely architecture or training issue, not RL hyperparams

### After stage 5 (vision)
- If text regressed > 0.2 CE → reduce vision training, retry with smaller projector
- If vision capability emerged → write up, demo

---

## Out of scope (for now)

- MTP head retrofit (engineering-heavy, marginal gain at 363M)
- 1M-context extension (unnecessary for this use case)
- Mamba-Attention hybrid (architectural rewrite)
- Multi-teacher distillation beyond Gemini (Claude/GPT-OSS pairs could be future v2)
- Pretrain restart (sunk cost — architecture fix unlocks most of the lost capability cheaper)
- Production deployment / inference optimization (TurboQuant, QJL KV-cache quantization) — defer until model is good enough to be worth optimizing

---

## Stage status legend
- ⏳ pending (not started)
- 🟢 in progress
- ✅ complete
- 🤔 conditional / waiting on trigger
- ❌ blocked
- ⏭️ skipped (by design)
