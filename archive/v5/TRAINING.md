# Training Process — NanoOSRT v5

> **Status:** draft. Pretrain + base SFT + SFT-long are complete (or in
> progress). SFT-ultralong and GRPO numbers are projected from the
> measured throughput of earlier phases. Energy and cost figures are
> rounded H100-equivalent estimates — actual cloud invoices may differ
> by 5-10 %.

---

## 1. Model at a glance

NanoOSRT is a recursive Mixtral-style mixture-of-experts transformer.
Three physical decoder blocks are reused six times (recursive weight
sharing) to give 18 effective transformer layers, with eight routed
experts per block plus one always-on shared expert (no dense FFN).

| Dimension | Value |
|-----------|-------|
| `dim` (hidden) | 1536 |
| Attention heads | 24 (head_dim 64) |
| Physical decoder blocks | 3 |
| Recursive loops | 6 |
| Effective transformer layers | **18** |
| Routed experts per block | 8 (top-2) |
| Shared experts per block | 1 (hidden 4096) |
| Routed expert hidden | 2048 |
| Vocabulary | 32,768 (BPE, single-token structural tags) |
| Max position embeddings | 8192 |
| RoPE θ | 10,000 |

Architectural decisions and the v4 → v5 lessons that justify them are
documented separately in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 2. Parameter accounting

There are three useful ways to count parameters in a recursive MoE.
Each answers a different question.

### 2.1 Physical parameters (storage)

What's actually on disk and in VRAM. Measured directly from the
`NanoOSRTForCausalLM` graph:

| Component | Params | Share |
|-----------|--------:|------:|
| Routed experts (3 blocks × 8 experts × SwiGLU h=2048) | 226,492,416 | 62.4 % |
| Shared experts (3 blocks × SwiGLU h=4096) | 56,623,104 | 15.6 % |
| Token embedding (32 768 × 1536, tied with LM head) | 50,331,648 | 13.9 % |
| Attention QKV + out_proj (3 blocks) | 28,311,552 | 7.8 % |
| Loop adapters (rank-16 × 18 effective layers) | 884,736 | 0.24 % |
| Router weight matrices | 36,864 | 0.01 % |
| Loop embeddings, layer norms, scalars | ~40,000 | < 0.01 % |
| **Total** | **362,720,259** | **100 %** |

Storage: **~726 MB at bf16**, ~1.45 GB at fp32.

### 2.2 Active parameters per token, per forward pass

Parameters that actually multiply against a token's hidden state in a
single pass through the architecture (one trip through all 3 blocks).
Top-2-of-8 sparsity reduces routed-expert activation by 4×; the shared
expert and attention are always live.

| Component | Active per token | Notes |
|-----------|------------------:|-------|
| Token embedding | 50,331,648 | always |
| Attention (3 blocks) | 28,311,552 | always |
| Shared expert (3 blocks) | 56,623,104 | always |
| Routed top-2 (3 blocks × 2 of 8) | 56,623,104 | sparse |
| Misc (norms, gates, adapters, router) | ~1,000,000 | always |
| **Active per token** | **~192,000,000** | **52.9 % of physical** |

### 2.3 Compute-equivalent ("effective") parameters

Each token traverses the 3 physical blocks **six times** via recursion.
Recursion adds compute without parameters, so a fair comparison to a
dense (non-recursive) model multiplies the active-per-pass figure by
the loop count.

| Quantity | Value |
|----------|-------|
| Physical params | 363 M |
| Active per pass | 192 M |
| **Effective compute per token (×6 loops)** | **~1.15 B FLOPs-equivalent** |
| **Effective params if fully unrolled (no sharing, no MoE sparsity)** | **~2.18 B** |

The 2.18 B figure is the "if we replaced the recursive MoE block with
18 independent dense transformer blocks of the same per-block size"
upper bound. The 1.15 B figure is the more meaningful one for runtime
cost: it reflects what each token actually pays for, given that top-2
sparsity is real and recursion really does cost FLOPs.

---

## 3. Training pipeline

Five sequential stages, each producing a checkpoint that seeds the
next. Pretrain is autoregressive next-token prediction; SFT phases
introduce instruction-following with native-token chat tags
(`<|user|>`, `<|assistant|>`, `<|think|>`, `<|/think|>`, `<|answer|>`,
`<|/answer|>`); GRPO is RL on top of SFT with verifiable rewards.

```
   Pretrain      SFT base     SFT long      SFT ultralong      GRPO
  (seq 2048→     (seq 2048,   (seq 4096,    (seq 8192,        (seq 8192,
   4096, 17 k    2.5 k        1 k steps,    200 steps,         2 k steps,
   steps)        steps)       Nemotron-     Nemotron-          verifiable
                              heavy)        heavy)             math reward)
   ─────►        ─────►       ─────►        ─────►             ─────►
```

### 3.1 Pretrain — `--stage pretrain`

Two phases of a planned three; phase 3 was deferred to SFT-ultralong
once the eval-loss curve flatlined and Chinchilla-optimal exposure
was met. **Stopped at step 17 000** (eval loss 3.48, perplexity 32.4).

| Phase | Steps | Seq len | Batch × accum | Datasets | Tokens / step |
|-------|------:|--------:|--------------:|----------|--------------:|
| 1 — Foundation | 0 → 9 500 | 2048 | 8 × 8 | FineWeb-Edu (60 %) + CodeParrot-Clean (40 %) | 131 k |
| 2 — Knowledge | 9 500 → 17 000 | 4096 | 6 × 11 | FineWeb-Edu (50 %) + CodeParrot (30 %) + Wikipedia (20 %) | 270 k |
| 3 — Instruction *(deferred)* | 17 000 → 25 000 | 8192 | 2 × 32 | SmolTalk + Evol-Code + OpenHermes + Nemotron math/stem | 524 k (planned) |

**Optimizer:** Hybrid Muon (Newton-Schulz orthogonalised update for 2D
matrix params) + AdamW for embeddings, norms, scalars, and router. LR
schedule: linear warmup over 3 000 steps to peak 6e-4, then cosine to
6e-5 over 25 000 steps total horizon.

**Routing health:** Switch balance loss (coeff 0.10) on raw pre-bias
router logits + DeepSeek-style per-expert bias controller for clean
deployed load. Z-loss (1e-3) for numerical safety. Annealed Gumbel
noise on top-k (τ 0.5 → 0 over 4 000 steps) to prevent first-20-step
expert death.

**Token totals:**
- Phase 1: 9 500 × 131 k = **1.25 B tokens**
- Phase 2: 7 500 × 270 k = **2.03 B tokens**
- **Pretrain total: ~3.27 B tokens** (≈ 85 % of Chinchilla-optimal for
  192 M active params, which is 192 M × 20 ≈ 3.84 B)

### 3.2 SFT base — `--stage sft`

| | |
|--|--|
| Seq length | 2048 |
| Steps | 2 500 (stopped early; original plan 5 000) |
| Batch × accum | 8 × 8 (effective batch 64) |
| Optimizer | AdamW, peak LR 1.5e-5 → min 1.5e-6 cosine, warmup 250 |
| HRA injection | rank 256, +86.1 M params (frozen base, only HRA trained on init then unfrozen via `hra_freeze_pretrained=False`) |
| Datasets | Math 25 % (GSM8K + NuminaMath-CoT), Code 25 % (Evol-Code + Alpaca-Code), STEM 20 % (Orca-Math + MathInstruct), General 20 % (Alpaca + OpenHermes), IF 10 % (IFEval + LongForm) |
| Loss masking | prompt + `<\|user\|>...<\|assistant\|>` masked (`IGNORE_INDEX`); `<\|think\|>...<\|/answer\|>` trained |
| Token utilisation | ~70 % response tokens (rest is masked prompt) |

**Token totals:** 2 500 × 131 k × 0.70 = **~230 M trained-on response
tokens**. Loss trajectory: 2.90 → 1.02 over 2 500 steps.

### 3.3 SFT long — `--stage sft_long` *(in progress at time of writing)*

Long-context fine-tune at seq 4096 with Nemotron-heavy data mix. Same
HRA adapters carried forward (`hra_before_load=True`).

| | |
|--|--|
| Seq length | 4096 |
| Steps | 1 000 |
| Batch × accum | 4 × 16 (effective batch 64) |
| Peak LR | 5e-6 → 5e-7 cosine, warmup 50 |
| Datasets | Nemotron Post-Training (60 %: math 30, stem 20, code 15, tool_calling 10) + diversity (NuminaMath 10, Evol-Code 10, LongForm 5) |
| Token utilisation | 69.4 % (measured) |

**Token totals:** 1 000 × 270 k × 0.694 = **~187 M trained-on
response tokens**. Loss at step 125 of 1 000: **1.43**.

### 3.4 SFT ultralong — `--stage sft_ultralong` *(planned)*

Short polish pass at seq 8192 to anchor long-context behaviour before
GRPO (which itself runs at seq 8192).

| | |
|--|--|
| Seq length | 8192 |
| Steps | 200 |
| Batch × accum | 2 × 32 (effective batch 64) |
| Peak LR | 3e-6 → 3e-7 cosine, warmup 10 |
| Datasets | Same Nemotron-heavy + diversity mix as SFT-long (inherited) |

**Token totals:** 200 × 524 k × 0.70 = **~73 M trained-on response
tokens** (projected).

### 3.5 GRPO — `--stage grpo` *(planned)*

Group Relative Policy Optimization with verifiable math rewards on
GSM8K. Reference model is the SFT-ultralong checkpoint, frozen.

| | |
|--|--|
| Seq length | 8192 |
| Steps | 2 000 |
| Group size (rollouts per prompt) | 16 |
| Max generation | 512 tokens |
| Temperature / top-p | 0.8 / 0.95 |
| KL coefficient | 0.05 |
| Clip range | 0.2 |
| Rewards | correctness 1.0, format 0.2, reasoning bonus 0.3, truncation penalty −0.5, empty-think penalty −0.1 |
| Optimizer | AdamW, peak LR 3e-6 → 3e-7 |

Reward parsing leverages the native single-token tags learned in SFT:
`<\|/think\|>` → `<\|answer\|>` boundary is a single token-id check;
the answer span is everything between `<\|answer\|>` and `<\|/answer\|>`.

---

## 4. Token totals across the full pipeline

| Stage | Tokens trained on | Cumulative |
|-------|------------------:|-----------:|
| Pretrain Phase 1 | 1.25 B | 1.25 B |
| Pretrain Phase 2 | 2.03 B | 3.27 B |
| SFT base | 0.23 B | 3.50 B |
| SFT long *(in progress)* | 0.19 B | 3.69 B |
| SFT ultralong *(planned)* | 0.07 B | 3.76 B |
| GRPO *(planned, generation tokens)* | ~0.06 B | ~3.82 B |
| **Total (projected at end of GRPO)** | **~3.82 B** | |

---

## 5. Wall-clock time and compute cost

Both Modal (H100 SXM 80 GB at $3.95 / hr) and Lightning AI Studios
(Nebius H100 at 3.01 credits / hr) were used. The hardware is
equivalent for our purposes; we report combined H100-hours.

Pretrain throughput (Modal H100, after raising the gradient
checkpointing threshold from 4096 → 8192 in commit `57513a9`) is
~40 k tok/s sustained at Phase 2 sizes. SFT-long throughput is
~22 sec/step (~12 k tok/s effective on response tokens).

| Stage | H100-hours | Cost (cloud equivalent) |
|-------|-----------:|------------------------:|
| Sanity + ablation (4 cells × 1200 steps) | ~10 | ~$40 |
| Pretrain Phase 1 (9 500 steps) | ~12 | ~$48 |
| Pretrain Phase 2 (7 500 steps) | ~75 | ~$295 |
| SFT base (2 500 steps) | ~5 | ~$20 |
| SFT long (1 000 steps) | ~6 | ~$24 |
| SFT ultralong (200 steps, planned) | ~2.5 | ~$10 |
| GRPO (2 000 steps, planned) | ~13 | ~$50 |
| **Total** | **~123 H100-hours** | **~$487** |

The Phase 2 figure looks high because two effects compound: (1) some
Lightning AI runs hit transient HF Hub throttling that wasted credits
on retries before `_open_stream` was retry-aware, and (2) early Phase
2 steps before commit `57513a9` ran at ~21 k tok/s (over-aggressive
gradient checkpointing) instead of ~40 k tok/s.

The actual user spend is lower than $487 because Lightning AI credits
were promotional ($240 worth bundled with the Studio).

---

## 6. Energy and carbon

H100 SXM has a 700 W TDP. Under sustained training the average draw
sits around 600-650 W (we measured ~60 GB VRAM and 90 % SM
utilisation during SFT-long at 60.1 GB / 69.4 % token utilisation).
Use **650 W average** for the GPU itself.

Datacenter overhead (cooling, networking, power conversion) adds
PUE 1.2-1.4 on top of GPU draw. Assuming PUE 1.3:

| Quantity | Value |
|----------|------:|
| GPU energy | 123 h × 0.65 kW ≈ **80 kWh** |
| Including PUE 1.3 | **~104 kWh** |

For grounding:
- **Equivalent to** ≈ 3–4 days of an average US household's
  electricity consumption (US avg ~30 kWh/day).
- **Equivalent to** ≈ 360 miles in a Tesla Model 3 (~3.5 miles/kWh).
- **Equivalent to** ≈ 1 round-trip economy flight London–Edinburgh
  (~110 kWh aviation fuel-equivalent per passenger, very rough).

### Carbon footprint

Depends entirely on datacenter location. Three scenarios:

| Grid | g CO₂eq / kWh | Total CO₂eq |
|------|--------------:|------------:|
| Coal-heavy (e.g. Poland) | ~770 | ~80 kg |
| Global average | ~430 | ~45 kg |
| AWS us-west-2 (mostly hydro) | ~280 | ~29 kg |
| Iceland / Nordic (geothermal/hydro) | ~50 | ~5 kg |

Modal does not publish its datacenter mix; Lightning AI's Nebius
Studios run primarily in Finland and the Netherlands, both of which
are below the global average grid intensity. A reasonable single-point
estimate is **~30-40 kg CO₂eq** for the full pipeline.

For comparison, that's roughly:
- Driving an average ICE car ~100 miles (~380 g/mile).
- ~10 hours of streaming HD video on a 55-inch TV.
- Producing ~70 cheeseburgers (~500 g CO₂eq each, very rough).

---

## 7. Summary numbers

| | |
|--|--|
| Physical parameters | **363 M** (726 MB at bf16) |
| Active parameters per token | **192 M** (52.9 % of physical) |
| Effective parameters (compute-equivalent, ×6 loops) | **1.15 B** |
| Effective parameters (unrolled 18-block dense) | **2.18 B** |
| Total tokens trained on (full pipeline) | **~3.82 B** |
| Total wall-clock | **~123 H100-hours** |
| Total cloud cost equivalent | **~$487** |
| Total energy (incl. PUE 1.3) | **~104 kWh** |
| Total carbon footprint | **~30-40 kg CO₂eq** |

This is roughly the carbon cost of a single transatlantic flight
divided by 200 — small at the scale of one model, but worth knowing
for the full picture.
