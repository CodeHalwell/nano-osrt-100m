# nano-osrt — Post-extend3 Plan

**Owner:** Daniel · **Last update:** 2026-06-06 · **Current state:** extend3 in progress (step 2500/3000)

Living document — update inline as we progress. Per-stage cost reconciliation
and probe results captured at the bottom of each stage block.

---

## Where we are

| asset | status |
|---|---|
| Architecture | ✅ Fixed (loopfixv2 + extend3 with aux_loop_loss_weight + loop dropout) |
| Best ckpt | extend3 step 2400 (task CE 1.48, best of run); final pending at step 3000 |
| Rollouts | 4,406 done, ~12K more in flight via Gemini 3.5 Flash; target ~16K total |
| Budget — Modal | $12 codhe-hugging-mcp + $15 danielhalwell = $27 (top-up needed before GRPO) |
| Budget — Gemini API | ~$270 remaining of £300 grant |
| Plus | Claude Max + ChatGPT Pro subs (not currently used in pipeline) |

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

**Status:** ⏳ pending (waiting for extend3 to finish)
**Cost:** $0 (all local)
**Duration:** ~30 min

### Actions
- [ ] Download `extend3_final.pt` from codhe volume
- [ ] Run `probe_recursion.py` against extend3_final
- [ ] Run 8-prompt inference test (`infer_local.py`) on extend3_final
- [ ] Compare to baselines:
  - extend2_final: Test 3 cliff 5.97 / inference 1/6
  - loopfixv2_merged: Test 3 cliff 0.13 / inference 2/6
  - extend3_final: ?
- [ ] Sliding-window ckpt merge over extend3 steps 1800, 2100, 2400, 2700, 3000
- [ ] Probe merged ckpt → confirm marginal improvement
- [ ] Update canonical: `extend3_merged.pt`

### Success criteria
- Test 3 (loop ablation) at least as flat as loopfixv2_merged (no regression to cliff)
- Inference test ≥ 2/6 correct (likely 2-3/6 — meaningful improvement here only happens post-MOPD)
- Adapter trajectory in probe shows depth utilization preserved

### Results
_TBD_

---

## Stage 2 — MOPD distillation

**Status:** ⏳ pending (waiting for stage 1 + rollouts)
**Cost estimate:** ~$5-7 Modal
**Duration:** ~2.5 hr

### Actions
- [ ] Resume `collect_rollouts.py` if still in progress (target ~16K final)
- [ ] Upload rollouts: `modal volume put osrt-rollouts rollouts/mopd_v1.jsonl mopd_v1.jsonl`
- [ ] Launch `mopd_sanity` (30 steps, no compile/wandb) → validate end-to-end
- [ ] Launch `mopd` full (1000 steps from `extend3_merged.pt`)
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
_TBD_

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
