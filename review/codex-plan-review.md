# Codex Review: Root OSRT-600M Plans

**Date:** 2026-06-07  
**Reviewer:** Codex  
**Reviewed files:** `README.md`, `ARCHITECTURE.md`, `RESEARCH.md`, `LEARNINGS.md`  
**Verification run:** `uv run --group dev pytest -q`

## Executive Summary

The root Markdown plans contain several strong engineering lessons from v5:
measurement-first training, day-1 system-prompt exposure, strict reward
extraction, per-env capability metrics, HRA-only RL, and per-loop diagnostics.
Those are the best parts of the plan and should be preserved.

The plan is not yet implementation-ready. The largest problems are not taste or
model-design disagreement. They are hard contract failures: tokenizer IDs do not
match the spec, parameter and FLOP accounting disagree across files, the root
package is missing, several forward-pass tensor operations are dimensionally
wrong, DeepSeek/LFM2 source attribution is incorrect in places, and the cost and
deployment memory budgets do not close.

Before training or implementation starts, freeze four contracts:

1. Tokenizer IDs and chat/tool template.
2. Exact parameter/FLOP/memory accounting.
3. Recurrence plus mHC tensor shapes and cache semantics.
4. Tier-specific training plan: $280 build vs ship-ready build.

## Verified Repo State

`uv run --group dev pytest -q` currently fails during collection:

```text
ModuleNotFoundError: No module named 'nano_osrt'
```

The failure is consistent with the repository layout:

- `pyproject.toml:39-40` packages `src/nano_osrt`.
- `app.py:66` uploads `src/nano_osrt` to Modal.
- The root tree has no `src/` directory.
- The old implementation exists under `archive/v5/src/nano_osrt`.

This is a blocking repo readiness issue. The root docs describe an OSRT-600M
future plan, but the root project metadata and tests still expect a live
`nano_osrt` package.

## Blocking Findings

### 1. Root Package Is Missing

**Evidence**

- `pyproject.toml:39-40`: wheel packaging points at `src/nano_osrt`.
- `app.py:66`: Modal image adds local directory `src/nano_osrt`.
- `tests/test_model.py:7`: imports `nano_osrt.config`.
- `find src` fails with `No such file or directory`.

**Impact**

No root test, package build validation, or Modal run can be trusted until the
package location is resolved. This also weakens the claim in `ARCHITECTURE.md`
that someone could implement from the doc alone, because the local project does
not have an implementation target.

**Recommendation**

Choose one:

- Restore or move `archive/v5/src/nano_osrt` to `src/nano_osrt` as the current
  v5 baseline, then branch OSRT-600M from there.
- Change root metadata/tests/app to explicitly target `archive/v5` and document
  that OSRT-600M is design-only.
- Create a new `src/nano_osrt_600m` package and update tests to match.

Do this before model changes.

### 2. Tokenizer IDs Do Not Match The Plan

**Evidence**

- `ARCHITECTURE.md:151-169` reserves:
  - BOS 0, EOS 1, PAD 2
  - `<|end_turn|>` 14
  - tool call/result tokens 15-18
  - image/audio 19-20
- `scripts/train_tokenizer.py:240-255` defines:
  - PAD 0, BOS 1, EOS 2
  - no `<|end_turn|>`
  - no tool call/result tokens
  - no image/audio tokens
- `tokenizer/tokenizer.json:237-251` confirms current ID 14 is `"!"`.
- `ARCHITECTURE.md:809-810` uses stop IDs `[10, 14, 18]`.

**Impact**

This is the exact class of failure the v5 post-mortem warns about. If the
current tokenizer is reused, OSRT-600M will not have the planned end-turn,
tool-result, image, or audio IDs. Worse, generation would treat `!` as
`<|end_turn|>` because the plan says ID 14 is a stop token.

**Recommendation**

Create a tokenizer contract file before training:

```text
0  <|padding|> or <|begin_of_text|>  pick one and never change it
1  ...
...
14 <|end_turn|>
15 <|tool_call|>
16 <|/tool_call|>
17 <|tool_result|>
18 <|/tool_result|>
19 <|image|>
20 <|audio|>
```

Then update:

- tokenizer training script
- existing tokenizer artifact or regenerate it
- config defaults
- generation stop IDs
- SFT/MOPD formatting code
- reward parsers

Do not start pretraining until this is validated by an encode/decode test.

### 3. Parameter Accounting Is Internally Inconsistent

**Evidence**

- `ARCHITECTURE.md:58` says `32,768 x 1,536` but gives
  `100,663,296`, which is `65,536 x 1,536`.
- `ARCHITECTURE.md:62-65` says attention is `28,311,552` params and describes
  `4 x 1536^2` per block.
- `ARCHITECTURE.md:311-318` specifies attention as:
  - `W_Q`: 2.36M
  - `W_K_DOWN`: 0.79M
  - `W_V_FROM_K`: 0.26M
  - `W_O`: 2.36M
  - total ~5.76M per block, ~17.3M across 3 blocks.
- `README.md:67` says active per token is ~206M.
- `ARCHITECTURE.md:94` says active per token is ~232.5M.
- `README.md:107` says 625M total / 210M active.

**Impact**

The plan cannot be used for capacity planning, cost estimation, memory
estimation, optimizer state allocation, or active-parameter comparisons. The
numbers are close enough to look plausible but different enough to cause wrong
budget decisions.

**Recommendation**

Add a small deterministic accounting script and make the docs consume its
output. It should compute:

- physical params
- trainable params by stage
- active params for learned-routing layers
- active params for hash-routing layers
- BF16 weight memory
- optimizer memory by param group
- quantized deployment memory
- forward FLOPs per token with and without LM head

Then replace hand-written totals in `README.md` and `ARCHITECTURE.md`.

### 4. Tier 1 Cost Estimate Is Off By About 10x

**Evidence**

- `README.md:831-836` assumes H100 at ~$4/hr.
- `README.md:836` says `50K H100-hours` costs ~$15,000.
- At $4/hr, 50K H100-hours costs ~$200,000.
- `README.md:837` says `1.5K H100-hours` costs ~$700.
- At $4/hr, 1.5K H100-hours costs ~$6,000.

**Impact**

The Tier 1 plan is not a $16K ship-ready build under the stated assumptions.
It is closer to a six-figure pretraining plan if the H100-hour estimate is
correct. If the $15K number is correct, then the H100-hour estimate is wrong.

**Recommendation**

Separate:

- H100-hours.
- Wall-clock hours.
- Number of GPUs.
- $/GPU-hour.
- Expected utilization.

Then recompute Tier 1 and Tier 2 from the same formula.

### 5. Research Source Attribution Has Hard Errors

**Evidence**

- `RESEARCH.md:93` labels arXiv `2511.23404` as DeepSeek-V4.
- arXiv `2511.23404` is the LFM2 Technical Report.
- Official DeepSeek docs place DeepSeek-V4 Preview on 2026-04-24.
- `RESEARCH.md:608` and `RESEARCH.md:671` cite "Nov 2026" reports in a repo
  dated 2026-06-07.

**Impact**

The plan mixes accurate ideas with incorrect provenance. That makes it hard to
decide what is verified, what is speculative, and what is user-supplied
synthesis.

**Recommendation**

Refactor `RESEARCH.md` into:

- `Primary sources`: official papers/model cards/docs.
- `Secondary summaries`: blogs, reports, synthesis docs.
- `User-supplied internal evidence`: v5 results, post-mortems, ablations.
- `Unverified claims`: useful ideas that still need source confirmation.

Correct the DeepSeek-V4 and LFM2 citation blocks.

## High Severity Findings

### 6. mHC Forward Pseudocode Is Dimensionally Broken

**Evidence**

- `ARCHITECTURE.md:514-516` defines:
  - `A_l` as `1 x 4`
  - `B_l` as `4 x 4`
  - `C_l` as `4 x 1`
- `ARCHITECTURE.md:668` computes:

```python
x_view = (A_l @ X.reshape(B, L, 4, 1536).transpose(2, 3)).squeeze(-1)
```

After transpose, the tensor shape is `[B, L, 1536, 4]`. A normal
`1 x 4` matrix multiply against this shape does not produce the intended
`[B, L, 1536]` view.

- `ARCHITECTURE.md:677-678` and `ARCHITECTURE.md:694` have the same class of
  issue for `B_l` and `C_l`.

**Impact**

An engineer implementing this literally will hit shape errors or accidentally
write broadcasting logic that does not represent the intended mHC update.

**Recommendation**

Use explicit einsum notation in the spec. For example:

```python
# X: [B, L, H, D], H = n_hc
# A: [B, L, H]
# Bmix: [B, L, H, H]
# C: [B, L, H]
x_view = torch.einsum("blh,blhd->bld", A, X)
x_layer = F(x_view)
X_next = (
    torch.einsum("blij,bljd->blid", Bmix, X)
    + torch.einsum("bli,bld->blid", C, x_layer)
)
```

Then write tests for tiny shapes.

### 7. mHC Residual Stream Initialization Uses Aliasing

**Evidence**

- `ARCHITECTURE.md:655` initializes with:

```python
X = x.unsqueeze(-2).expand(-1, -1, 4, -1)
```

- `ARCHITECTURE.md:662` mutates channel 0:

```python
X[:, :, 0, :] = X[:, :, 0, :] + loop_bias
```

`expand()` creates a view with shared underlying storage along expanded
dimensions. In-place writes against one expanded channel can alias other
channels or be rejected by PyTorch.

**Impact**

This can destroy the intended four-stream residual representation at the first
loop embedding update.

**Recommendation**

Use `repeat` or `expand(...).clone()`:

```python
X = x.unsqueeze(2).repeat(1, 1, n_hc, 1)
```

Then test that changing channel 0 does not change channels 1-3.

### 8. Final mHC Collapse Is Undefined

**Evidence**

- `ARCHITECTURE.md:701-702` extracts final hidden state with the last `A_l`:

```python
x_final = (A_l @ X.transpose(-1, -2)).squeeze(-1)
```

But `A_l` is generated inside the last attention sub-block, not as a final
collapse head. It also excludes the final FFN `A_l_ffn`.

**Impact**

Final logits depend on a stale intermediate mixing vector. That is an
architecture bug, not just pseudocode looseness.

**Recommendation**

Add a final hyper-head or explicit collapse parameter:

```python
x_final = HyperHead(X)  # [B, L, 1536]
```

Specify whether it is static, dynamic, Sinkhorn-constrained, or a simple
learned nonnegative weighted sum.

### 9. KV Cache Compression Is Double-Counted

**Evidence**

- `ARCHITECTURE.md:895-903` says cache stores only `K_DOWN`, not V.
- `ARCHITECTURE.md:912-915` computes raw cache size from that K-only layout:
  `18 x 512 x 2 = 18KB/token`, 72MB at 4K.
- `ARCHITECTURE.md:922-924` then applies `+ V-from-K (cache K only)` again to
  reduce 72MB to 36MB.

**Impact**

The KV deployment table overstates cache savings by 2x at that step.

**Recommendation**

Choose one baseline:

- Standard GQA K+V baseline: `18 x 1024 x 2` bytes/token.
- OSRT K-only baseline: `18 x 512 x 2` bytes/token.

Then apply int4/TurboQuant once.

### 10. Deployment Memory Budget Does Not Close

**Evidence**

- `ARCHITECTURE.md:968` says routed experts are `319 MB -> 80 MB FP4`.
  318.5M parameters at 4 bits is about 159MB before metadata, not 80MB.
- `ARCHITECTURE.md:969` says HRA bf16 is 172MB.
- `ARCHITECTURE.md:972-973` says total disk is ~390MB and active inference is
  ~150MB.

**Impact**

The model likely does not fit the claimed 150-250MB deployment envelope while
keeping 86M HRA params in bf16.

**Recommendation**

Recompute memory under explicit assumptions:

- decimal MB vs MiB
- whether HRA is merged, quantized, offloaded, or active
- whether all routed experts must be resident
- per-tensor quantization metadata
- runtime allocator overhead
- embedding/LM head tied storage

### 11. Hash Routing Conflicts With Top-2 Routing And Recurrence

**Evidence**

- `ARCHITECTURE.md:105` says routed experts use top-2.
- `ARCHITECTURE.md:441-445` says hash routing selects one fixed expert.
- `README.md:708-710` says replacing first 1-2 physical block routers with
  hash routing affects only the first loop iteration's first blocks.

In a recurrent model, physical blocks 0 and 1 are reused on every loop unless
there is an explicit loop-dependent condition.

**Impact**

The plan is ambiguous on active params, load balancing, and specialization:

- If hash layers select one expert, active FLOPs are lower than stated.
- If they select top-2 fixed experts, the hash function must produce two IDs.
- If hash routing applies every loop, early blocks cannot learn
  representation-dependent routing at any depth.
- If hash routing only applies loop 0, the spec must say so.

**Recommendation**

Define one of:

```python
if r == 0 and b < 2:
    routing = "hash"
else:
    routing = "learned"
```

or:

```python
if b < 2:
    expert_ids = hash(input_id, loop_idx, block_idx, k=2)
```

Then update active parameter accounting.

### 12. HRA Injection Count Is Ambiguous

**Evidence**

- `ARCHITECTURE.md:284` says 87 HRA injection points.
- `ARCHITECTURE.md:291-292` says HRA is injected into Q/K/V projections,
  attention output, gate/up/down of each expert, and router projection.

Naively counting per block:

- attention: 4
- shared expert: 3
- routed experts: `12 x 3 = 36`
- router: 1
- total per block: 44
- across 3 physical blocks: 132

That does not match 87.

**Impact**

HRA parameter count, active trainable params, RL memory, and optimizer grouping
are all uncertain.

**Recommendation**

Add a table enumerating every HRA module:

```text
block_id | submodule | shared_across_experts | count | params
```

Then recompute the 86.1M total.

### 13. Training Loss Defaults Conflict

**Evidence**

- `README.md:176-183` says start with DeepSeek-V3 loss-free bias balancing
  and only fall back to aux loss 0.01.
- `ARCHITECTURE.md:748-750` sets `aux_loop_loss_weight` to 0.05 for
  pretrain/MOPD/SFT and 0.03 for GRPO.
- `LEARNINGS.md:72-75` says v5 tightened aux loop loss to 0.10 during GRPO
  and v6 should make per-loop aux losses permanent.
- `RESEARCH.md:655-656` says Cell C uses Muon + 0.10 aux loss as recipe while
  trialing 0.01 and loss-free.

These refer to two different aux concepts:

- router/load-balancing aux loss
- per-loop LM-head aux loss

The docs often discuss them near each other without disambiguating.

**Impact**

Implementation could accidentally remove per-loop LM aux loss while trying to
remove router aux loss, reintroducing loop collapse.

**Recommendation**

Rename explicitly:

- `loop_lm_aux_loss_weight`
- `router_balance_loss_weight`
- `router_z_loss_weight`
- `loss_free_router_bias_enabled`

Then state defaults by stage.

### 14. Speculative Decoding Algorithm Is Not Distribution-Preserving

**Evidence**

- `ARCHITECTURE.md:862-866` drafts tokens greedily from loop 3.
- `ARCHITECTURE.md:869-880` verifies by greedy full-loop predictions and
  accepts matching prefix.
- `ARCHITECTURE.md:806-827` normal generation uses temperature and top-p
  sampling.

**Impact**

The speculative path does not preserve the same distribution as top-p sampling.
It is a greedy acceleration heuristic. That might be fine, but it cannot be
presented as standard speculative sampling with no behavior change.

**Recommendation**

Document two modes:

- `generate_greedy_loop_speculative`: match greedy verifier exactly.
- `generate_speculative_sampling`: implement acceptance/rejection using draft
  and target probabilities.

Measure acceptance rate and output quality separately.

### 15. Forward And Generation Signatures Do Not Match

**Evidence**

- `ARCHITECTURE.md:650` defines `forward(input_ids, kv_cache=None, training=False)`.
- `ARCHITECTURE.md:833` calls `forward(next_token, kv_cache=kv_cache, loops=loops)`.
- `ARCHITECTURE.md:674` indexes `kv_cache[r, b]` even when `kv_cache=None`
  during prefill.

**Impact**

The inference pseudocode is not executable and hides cache lifecycle decisions.

**Recommendation**

Define:

```python
forward(input_ids, kv_cache=None, loops=6, use_cache=False)
```

and specify:

- prefill cache creation
- decode cache append
- cache entries for skipped loops when `loops < 6`
- whether aux outputs are returned during inference

## Medium Severity Findings

### 16. Gated Short Convolutions Are Claimed But Not Specified

**Evidence**

- `ARCHITECTURE.md:43-45` says the architecture uses gated short
  convolutions plus GQA attention.
- No convolution sub-block is specified in `ARCHITECTURE.md`.
- `README.md:570-571` and `RESEARCH.md:176-182` discuss LFM2 gated short
  convolutions as research context.

**Impact**

The one-sentence architecture overclaims. The technical spec is attention-only
plus MoE.

**Recommendation**

Either remove "gated short convolutions" from the OSRT-600M architecture or add
an explicit conv block, placement schedule, params, FLOPs, and cache behavior.

### 17. HCA Is Listed As Required For Deployment But Missing From The Spec

**Evidence**

- `README.md:762-763` says HCA sequence compression must be baked into
  pretraining and is not retrofittable.
- `ARCHITECTURE.md` only specifies K-only cache, TurboQuant, and optional
  sliding window.

**Impact**

If HCA is truly needed for the deployment story, it must be part of the model
architecture before pretraining. If it is optional, the deployment memory
claims should not depend on it.

**Recommendation**

Mark HCA as either:

- `in scope for OSRT-600M`: add full architecture and training details.
- `v7/deferred`: remove it from OSRT-600M deployment claims.

### 18. Evaluation Cadence Does Not Fit Tier 2

**Evidence**

- `README.md:319-320` says run benchmarks every 500B tokens or every 5K steps.
- `README.md:857` says Tier 2 pretraining is only 12B tokens.

**Impact**

The token-based benchmark trigger never fires in the $280 plan. The plan says
measurement-first, but the current cadence is for the large tier.

**Recommendation**

Define separate cadence:

- Tier 2: every 1B or fixed step interval, plus start/mid/final full eval.
- Tier 1: every 500B, plus stage boundaries.

### 19. OOD Probe Size Is Inconsistent

**Evidence**

- `README.md:433` says 20-50 held-out prompts.
- `README.md:539` says 50 diverse prompts.
- `README.md:869` says Tier 2 uses the 12-prompt subset.
- `LEARNINGS.md:309-314` says the validated v5 OOD probe is 12 prompts.

**Impact**

Instrumentation and budget planning will diverge. A 12-prompt auto-stop signal
has different variance than a 50-prompt signal.

**Recommendation**

Define:

- `ood_probe_smoke_12`: every 25 RL steps, auto-stop capable.
- `ood_probe_full_50`: stage boundary and final reporting only.

### 20. Tool Use Is Both Day-1 And Deferred

**Evidence**

- `README.md:462-466` says tool use is a day-1 architectural commitment.
- `README.md:476-478` says tool tokens are reserved and pretrained.
- `README.md:866` says Tier 2 defers tool-use SFT/GRPO.

**Impact**

The tokenizer/pretraining part is day-1, but tool behavior training is tier
dependent. The docs should separate these commitments.

**Recommendation**

Use three levels:

- Tier 2 required: reserve tokens and include synthetic tool text in
  pretraining.
- Tier 2 optional: small SFT-only tool formatting.
- Tier 3/Tier 1: tool-use GRPO.

### 21. mHC Runtime Claim Needs Kernel Assumptions

**Evidence**

- `ARCHITECTURE.md:581-588` says mHC adds ~720K params and ~6.7% wall-clock
  overhead.
- `ARCHITECTURE.md:538-540` uses up to 20 Sinkhorn iterations per generated
  matrix.
- `ARCHITECTURE.md:551-567` generates `A/B/C` dynamically per token.

**Impact**

A naive PyTorch implementation could be far slower than 6.7%, especially if
implemented with Python loops and dynamic small matmuls.

**Recommendation**

Make the spec conditional:

- naive PyTorch: prototype only
- `torch.compile`: required for training
- Triton/custom kernel: required before long runs if profiling shows overhead

## External Source Verification

The review checked current public sources because several claims concern
recent or future-looking model reports.

- LFM2 Technical Report: https://arxiv.org/abs/2511.23404
  - Confirms LFM2 is arXiv 2511.23404, not DeepSeek-V4.
  - Confirms 65,536 byte-level BPE, LFM2-700M hidden size 1536, 24/8/64 GQA,
    16 layers, FF dim 6912, and hybrid short-conv/GQA architecture.
- Official DeepSeek V4 release: https://api-docs.deepseek.com/news/news260424
  - Confirms DeepSeek-V4 Preview release on 2026-04-24.
  - Confirms V4-Pro 1.6T/49B active and V4-Flash 284B/13B active.
- DeepSeek V4 Hugging Face card:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
  - Confirms CSA/HCA, mHC, Muon, 32T+ pretraining tokens, and two-stage
    post-training with specialist cultivation plus OPD.
- Hugging Face Transformers DeepSeek-V4 docs:
  https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md
  - Confirms mHC stream shape, final hyper-head, hash MoE behavior,
    Sqrt(Softplus) affinity, clamped SwiGLU, and DeepSeek-V4 implementation
    details.
- AlphaQ OpenReview: https://openreview.net/forum?id=rbE8Pxs8vx
  - Confirms calibration-free MoE bit allocation and near full-precision
    Qwen1.5-MoE result at 3.5 average expert bits with >4x expert compression.
- Gemma 4 developer guide:
  https://developers.googleblog.com/gemma-4-12b-the-developer-guide/
  - Confirms Gemma 4 12B was announced on 2026-06-03, has an encoder-free
    multimodal architecture, and has a dedicated MTP model for local inference.

## Recommended Remediation Sequence

### Phase 0: Make The Repository Coherent

1. Restore or intentionally replace `src/nano_osrt`.
2. Make `pytest` collect.
3. Add a minimal doc-check test for tokenizer IDs.
4. Decide whether root docs are active implementation docs or design-only docs.

### Phase 1: Freeze Contracts

1. Tokenizer ID table.
2. Chat, tool, and stop-token template.
3. Exact model config.
4. Param/FLOP/memory accounting generated by script.
5. Stage-specific training defaults.

### Phase 2: Reduce Architecture Ambiguity

1. Rewrite mHC section with executable tensor notation.
2. Define final mHC collapse head.
3. Decide hash routing semantics under recurrence.
4. Decide whether HCA and gated short convolutions are in scope.
5. Define MTP vs loop-draft decoding precisely.

### Phase 3: Split Plans By Budget Tier

The current docs mix three different projects:

- $280 constrained build.
- $2K-$3K mid-budget build.
- full ship-ready pretraining build.

Create one table per tier with:

- data volume
- context schedule
- benchmark cadence
- SFT/MOPD/OPD/RL stages
- tool-use scope
- vision scope
- expected artifacts
- hard stop criteria

### Phase 4: Only Then Implement

Start with a tiny config and tests:

- tokenizer contract test
- mHC shape test
- mHC stream aliasing test
- hash routing top-k test
- tied embedding identity test
- loop aux output count test
- cache prefill/decode shape test
- active parameter accounting test

Do not launch any training job until those pass.

## Bottom Line

The v5 lessons are valuable and should drive OSRT-600M. The current root plans
are not yet a training launch plan. They are a strong research/design notebook
that needs contract hardening.

The highest-leverage edits are:

1. Fix tokenizer contract.
2. Fix repo package layout.
3. Generate all accounting from code.
4. Rewrite mHC pseudocode into executable shape-safe math.
5. Correct research provenance.
6. Split optional frontier ideas from the actual $280 build plan.
