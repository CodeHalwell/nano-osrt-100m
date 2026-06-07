# NanoOSRT v5 — Design Plan

## Lessons from v4

### What worked
- **Capacity-capped routing** prevented training crashes (but at the cost of making the router's decisions irrelevant)
- **Recursive weight sharing** (3 blocks × 6 loops) — compute budget was fine, no instability
- **Resilient HF streaming** with retries + reconnects — essential for Modal serverless
- **Packing for SFT** — 5× token utilisation improvement
- **HRA adapters for SFT** — +90M capacity without retraining
- **BPE 32K vocabulary** with single-token structural tags
- **KV cache** for inference

### What failed — the MoE never actually routed
1. **Router stuck at `prob_entropy = ln(11)` for 10,000+ pretrain steps** — perfectly uniform softmax, zero learned preferences
2. **Capacity cap hid router decisions** — experts all got uniform token mixes, so they couldn't specialise, so router had no signal to learn from, so it stayed uniform. Self-reinforcing degeneracy.
3. **Importance loss actively hurt** — its minimum is uniform distribution. Router converged to the minimum and stayed there.
4. **Dense FFN dominated** — with `dense_hidden=4096` next to experts at `expert_hidden=1024`, the dense path did 70% of the work. Gates for blocks 0 and 1 went *negative* on MoE output (model actively subtracting expert contributions).
5. **Recovery experiment failed** — freezing everything except router+experts and training with diversity loss for 500 steps produced no task loss recovery. Confirms experts can't recover from this initialisation.
6. **Soft warmup + blend didn't prevent collapse** — hard routing phase still collapsed once the blend ended.
7. **Balance bias controller oscillated** — proportional controller on hard-assignment fractions, too noisy a signal.

### Training outcomes
- Pretrain loss floor: ~7.3 on FineWeb-Edu (plateaued after 9k steps)
- SFT loss floor: ~7.3 on balanced instruct data (3500 steps)
- Gate b2 was the only block where MoE contributed positively

---

## v5 Design — Kill the Crutches

### Core architectural changes

| Component | v4 | v5 | Rationale |
|-----------|-----|-----|-----------|
| Dense FFN | hidden=4096 (always on) | **removed** | Was doing all the work and letting MoE stay optional |
| Shared expert | hidden=1024 | **hidden=4096** | Absorbs the dense FFN's role as "common knowledge" path |
| Routed experts | 11 × hidden=1024 | **8 × hidden=2048** | Fewer, larger experts — each has real capacity |
| Routing | top-2 (sigmoid) | **top-2 (softmax, renormalised)** | Mixtral-style; fix was balance loss + no dense, not top-k |
| Aux loss | importance (enforces uniform) | **Switch balance (f_i × p_i)** | Penalises imbalance without enforcing uniformity |
| Capacity factor | 1.25 (tight) | **2.0** | Loose enough that router preferences actually matter |
| Soft warmup / blend | yes | **removed** | Didn't prevent collapse |
| Balance bias controller | yes | **reintroduced, slower** | Three v5 sanities showed aux loss delays but does not prevent LR-ramp collapse |
| Gumbel noise | yes | **annealed early only** | Prevents first-20-step expert death, then decays before health gate |
| Expert init | random | **orthogonal per-expert** | Break symmetry structurally |

### Parameter budget (measured, not estimated)

```
Embedding (32K × 1536, tied with LM head) : 50,331,648   (always active)
Attention × 3 blocks (qkv + out_proj)     : 28,311,552   (always active)
Shared expert × 3 (SwiGLU h=4096)         : 56,623,104   (always active — replaces dense FFN)
Routed experts: 3 × 8 × (SwiGLU h=2048)   : 226,492,416 total
  → with top-2 of 8, active per token     : 56,623,104
Router + loop_emb + adapters + norms      : ~1M

Total physical params                     : 362,720,259 (~363M)
Active per token                          : ~191,889,408 (~192M, 52.9%)
Effective compute per token (×6 loops)    : ~1.15B FLOPs-equivalent
```

LM head is weight-tied to the embedding (matches v4; saves ~50M vs untied at 32K vocab × 1536 dim).

vs v4: 306M total, 215M active (70%). **The MoE path carries substantially more real capacity (227M vs 155M routed in v4), top-2 gives stronger per-token capacity (57M routed active vs v4's 28M), and there's no dense-FFN crutch.**

### Block forward (simplified)

```python
def forward(self, x, loop_idx):
    x = x + self.attn(self.ln1(x), loop_idx)
    # MoELayer returns (shared_out, routed_out) as a tuple so the Block
    # can gate them independently. Shared expert is always full weight;
    # only the routed experts pass through moe_gate.
    shared_out, routed_out = self.moe(self.norm_moe(x), loop_idx)
    x = x + shared_out + self.moe_gate * routed_out
    return x
```

No `dense_gate`, no parallel dense FFN. `moe_gate` initialises at 1.0 (no dense crutch to hide behind — we want the full MoE branch online from step 0). `moe_gate` is trainable and can drift if the model finds routed experts net-harmful, but the shared expert contribution is never gated.

### Routing algorithm

```
router_logits       = router(x + loop_embedding[loop_idx])   # (N, E=8)
expert_probs        = softmax(router_logits, dim=-1)         # (N, E)

# Mixtral-style top-k (K=2) with renormalised gates
raw_top_probs, top_idx = probs.topk(K, dim=-1)               # (N, K)
top_probs              = raw_top_probs / raw_top_probs.sum(-1, keepdim=True)  # sums to 1

# Switch balance loss extended to top-k
one_hot      = F.one_hot(top_idx, E).to(probs.dtype)         # (N, K, E)
f            = one_hot.sum((0, 1)) / (N * K)                 # sum(f) = 1
p            = probs.mean(0)                                 # sum(p) = 1
balance_loss = E * sum(f * p)                                # min = 1.0 at uniform

# Per-expert capacity depends on top_k because each token picks K experts
capacity     = ceil(capacity_factor * K * N / E)             # default 2.0 * 2 * N / 8 = N/2

# Tokens exceeding capacity for a given expert are dropped FROM THAT EXPERT's
# branch in training only. Eval mode sets capacity to N*K so drops never
# occur — generation is chunk-stable by construction.
```

Aux loss coefficient: **0.03**. The original Switch default (`0.01`) was too
weak for the 8-expert v5 sanity run: by step 190 the router was sharp
(`per_token_entropy=0.656`, `raw_max_prob=0.647`) but globally collapsed to
roughly two active experts (`marginal_entropy=0.712`, `drop_rate=0.44`). Raising
to `0.03` improved balance without hurting task loss (`marginal_entropy=1.08`,
`drop_rate=0.276`), but still left dead experts. Raising to `0.1` produced
diminishing returns at step 100, so v5 keeps `0.03` and adds early Gumbel
top-k exploration to prevent experts from going cold during warmup.
Aux loss is added to total loss only when `model.training` is True; eval loss is pure task CE so perplexity isn't polluted by a hyperparameter choice.

Gumbel top-k exploration: training starts with `router_gumbel_tau=0.5` and
anneals it to `0.0` over the first 4,000 steps. The 1,000-step anneal survived
in extended sanity, but failed in Foundation because noise expired while the
3,000-step LR warmup was still increasing task-gradient pressure
(`clean_marginal_entropy` fell from 1.79 at step 800 to 0.75 at step 1000).
The production schedule keeps exploration through peak LR and still gives the
clean router 1,000 no-noise steps before the 5k health gate. Training logs both
dispatch/noisy telemetry and `clean_*` telemetry so the run can distinguish
noise-assisted routing from the learned clean router.

### Expert initialisation

Orthogonal init per expert to break symmetry:
```python
# For each expert's w_gate, w_up, w_down:
# - Generate random matrix
# - QR decomposition for orthogonal columns
# - Scale to match standard init variance
# - Add small per-expert unique rotation (different seed per expert)
```

This gives each expert a different feature subspace to start from. Gradients push them in different directions organically.

### Kept from v4 (no changes)

- Tokenizer (32K BPE with native structural tags)
- Recursive weight sharing (3 physical blocks × 6 loops = 18 effective layers)
- KV cache infrastructure
- Resilient HF streaming data loader
- Packing for SFT
- HRA for SFT
- Modal deployment, W&B logging, checkpoint/resume logic
- Progressive seq_len curriculum (2048 → 4096 → 8192)
- Dataset choices (FineWeb-Edu + CodeParrot + Wikipedia for pretrain, balanced instruct for SFT)

---

## Training plan

### Phase 1 — Foundation (10k steps, seq_len 2048, B=8, accum=8)
- FineWeb-Edu (60%) + CodeParrot (40%)
- peak_lr 6e-4 (AdamW side), 0.02 (Muon side), both cosine
- **Muon hybrid optimiser** (Muon for 2D matrix weights, AdamW for
  embeddings/norms/router/scalars). w/d 0.3 on Muon-managed weights.
- ckpt every 1000 steps
- **Success criteria** (v4 would have failed all four by step 5k):
  - `moe/clean_per_token_entropy` drops below 1.5 from initial ~2.08 (ln 8) — router
    actually differentiating per-token, not just per-batch.
  - `moe/clean_raw_max_prob` rises above 0.30 from initial ~0.14 (1/8 + ε) — router
    has a clear primary pick for most tokens.
  - `moe/clean_top_margin` rises above 0.10 — meaningful gap between top-1 and top-2.
  - `moe/clean_marginal_entropy` stays above 1.8 — globally balanced (no dead experts).

  If all four don't hold by step 5k, the architecture has deeper issues and
  we stop the run.

### Phase 2 — Knowledge (30-60k steps, seq_len 4096, B=4, accum=16)
- FineWeb-Edu (50%) + CodeParrot (30%) + Wikipedia (20%)
- peak_lr continues cosine from phase 1
- **Success criterion**: training loss below 5.0 by step 20k (v4 never got below 7.3)

### Phase 3 — Instruction (10-20k steps, seq_len 8192)
- SmolTalk + Evol-Code + OpenHermes

### Budget estimate
- Phase 1: ~4h H100 = ~$16
- Phase 2: ~24h H100 = ~$96
- Phase 3: ~8h H100 = ~$32
- **Total pretrain: ~$140** (realistic for a clean run)
- Add SFT + GRPO: ~$40

---

## Optimizer × routing ablation (settled)

A 2×2 ablation at 1200 Foundation-matched steps drove the v5 production
defaults. Run it again with `modal run app.py --stage ablate`.

| Cell | Optimizer | Aux  | Final task | clean emin | bal  | Verdict                                 |
|------|-----------|-----:|-----------:|-----------:|-----:|-----------------------------------------|
| A    | Lion      | 0.10 |       ~7.4 |     ~0.002 | ~1.2 | Baseline; partial late-warmup collapse  |
| B    | Lion      | 0.0  |       ~7.6 |      0.000 | ~3.9 | Bias-only collapses by step 500         |
| C    | Muon      | 0.10 |   **3.43** |  **0.105** | 1.02 | **Production recipe**                   |
| D    | Muon      | 0.0  |    ~4.7    |     ~0.001 | ~2.3 | Same loss as C, B-style collapsed router |

Three load-bearing conclusions:

1. **Muon is a ~4-nat task-loss win** at this scale, regardless of
   routing scheme. Cells C and D both crossed `task < 5.0` by step
   450, while A and B were still around 7.2 there. The Newton-Schulz
   orthogonalisation of the momentum update equalises step sizes
   across singular directions, which is exactly what helps cold MoE
   experts get useful gradient instead of being shrunk into oblivion
   by Adam-style per-coordinate variance scaling.
2. **Gradient aux loss is necessary for router health**, regardless
   of optimiser. The bias controller alone collapses the raw router
   under both Lion and Muon. The DeepSeek-V3 "auxiliary-loss-free"
   claim is reportedly stable at 671B but does not hold at 363M on
   this curriculum.
3. **C is the production recipe.** Best loss, best balance, best emin,
   best margin. D buys C's loss while letting the raw router degenerate
   to 2-3 active experts — fine at Phase 1 difficulty, but Phase 2/3
   will need the unused expert capacity that D throws away.

The earlier v4 plan called for Lion (`w/d 0.3`); v5 keeps Lion as a
fallback (`optimizer_name="lion"` in `train_config.PretrainConfig`) for
A/B comparison runs, but the Muon hybrid is the default.

---

## Implementation order

1. **`v5_config.py`** — new config class, remove dense_hidden, change expert counts
2. **`v5_model.py`** — fork from v4_model.py:
   - Remove parallel dense FFN from Block
   - Simplify MoE to top-k softmax (Mixtral-style) with renormalised gates
   - Add Switch-style balance loss (generalised to top-k)
   - Add orthogonal per-expert init (applied after HF post_init)
   - Return (shared_out, routed_out) so only the routed branch is gated
   - Keep recursive loop, attention, KV cache
3. **`v5_train.py`** — fork from v4_train.py:
   - Remove soft warmup / blend / bias controller
   - Add Switch balance loss integration
   - Keep everything else (data, ckpt, telemetry)
4. **`v5_train_config.py`** — new config with phase definitions
5. **`app_v5.py`** — new Modal app (can live alongside v4)
6. **Unit tests** — fork v4 tests, add top-k routing + orthogonal-init-survives assertions
7. **Sanity run**: 500 steps, verify router differentiates

---

## What we're betting on

- **Top-k forces commitment via renormalised gates**: each picked expert carries a real weight in the output, so the router gets direct task gradient through the gate on the chosen path
- **No dense FFN crutch**: the model can't ignore the MoE path — routed experts are on the critical path for every token
- **Switch balance loss**: penalises imbalance without forcing uniform softmax (the v4 importance-loss failure mode)
- **Bigger experts**: 2048 hidden means each expert can represent useful structure, not just a rank-16 perturbation
- **Orthogonal init**: experts start in different feature subspaces, not identical random noise (and survives HF post_init)
- **Loose capacity (2.0)**: router preferences actually determine routing in training; eval mode disables drops entirely so generation is chunk-stable
- **Annealed Gumbel top-k**: keeps all experts sampled through LR warmup, then decays before the 5k router-health gate
- **Per-loop/per-expert balance bias**: non-gradient controller counter-rotates clean top-k load imbalance once per optimizer step; bias is persistent and part of the deployed routing path. The controller is loop-specific because capacity/drop is enforced per MoE call.

The success check at step 5k uses four clean-router metrics together (`clean_per_token_entropy`, `clean_raw_max_prob`, `clean_top_margin`, `clean_marginal_entropy`) — see the Phase 1 criteria above. A single-metric check (e.g. marginal entropy alone) would have missed v4's failure because batch-marginal entropy stayed high while per-token entropy also stayed high (router never committed).

If those four don't move in the right direction by step 5k, the issue is deeper than routing mechanics — likely the recursive weight sharing compounding across loops, or an expert-capacity vs task-diversity mismatch.

---

## What to keep an eye on

1. **Early expert collapse** — router finding a small expert subset that works for almost everything. The 1200-step v5 sanities showed `router_aux_loss_coeff=0.03`, `0.10` on noisy routing, and `0.10` on clean routing all eventually collapsed during LR warmup. Current fix is raw-router Switch aux plus a loop-specific DeepSeek-style per-expert balance-bias controller.
2. **Expert capacity vs. task diversity** — 8 experts is the right number for this model size, but if specialisation doesn't emerge, 4 experts × hidden=4096 is a fallback.
3. **Shared expert dominating** — if the shared expert does 95% of the work and routed experts remain weak, shrink shared expert hidden to 2048.
4. **Cross-loop routing consistency** — same router reused across 6 loops; check if it learns loop-specific preferences via loop_embedding.
