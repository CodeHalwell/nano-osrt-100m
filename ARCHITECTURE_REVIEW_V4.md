# NanoOSRT v4 Architecture Review — Recursive MoE

**Reviewer:** Claude (Opus 4.6)
**Date:** 2026-03-16
**Scope:** Full architectural review of v4 Recursive Mixture-of-Experts model against current best practices.

---

## Executive Summary

The v4 architecture is a **well-designed novel model** that combines recursive weight sharing with sparse mixture-of-experts. The core idea — 3 physical blocks × 6 recursive loops with loop-aware MoE routing — is architecturally sound and genuinely novel. Several design choices align well with best practices (parallel residual paths, shared expert, SwiGLU, RoPE, weight-tied head). However, there are **11 actionable issues** ranging from correctness bugs to missed optimization opportunities that should be addressed before training.

**Rating: Strong foundation with issues to fix before training.**

---

## 1. CRITICAL ISSUES (Fix Before Training)

### 1.1 MoE Expert Dispatch is O(B×S×K×E) — Use Scatter/Gather Instead

**File:** `v4_model.py:146-155`

In the pre-optimization baseline, the expert dispatch used a nested Python loop over `top_k` × `num_routed` experts. For 11 experts with top-2 routing, this is **22 sequential expert forward passes per token**, each with masking overhead. This was the single biggest performance bottleneck and would make training extremely slow.

**Baseline (pre-optimization, slow):**
```python
for k in range(self.top_k):           # 2 iterations
    for expert_idx in range(self.num_routed):  # 11 iterations
        mask = expert_indices == expert_idx
        if mask.any():
            expert_output = self.experts[expert_idx](expert_input)
```

**Current in `v4_model.py` — batched scatter/gather dispatch (replaces nested-loop baseline):**
```python
# Option A: Megablocks / grouped GEMM (fastest, requires triton kernel)
# Option B: Scatter-gather pattern (no external dependency)
def _dispatch_experts(self, x_flat, indices, weights):
    """Batch tokens by expert, run each expert once, scatter back."""
    B_S, D = x_flat.shape
    output = torch.zeros_like(x_flat)

    # Flatten across top-k dimension
    flat_indices = indices.reshape(-1)        # (B*S*top_k,)
    flat_weights = weights.reshape(-1, 1)     # (B*S*top_k, 1)
    # Repeat x for each top-k selection
    x_rep = x_flat.repeat_interleave(self.top_k, dim=0)  # (B*S*top_k, D)

    # Sort by expert for batched execution
    sorted_idx = flat_indices.argsort()
    sorted_experts = flat_indices[sorted_idx]
    sorted_x = x_rep[sorted_idx]
    sorted_weights = flat_weights[sorted_idx]

    # Find boundaries per expert
    expert_counts = torch.bincount(sorted_experts, minlength=self.num_routed)

    offset = 0
    for eid in range(self.num_routed):
        count = expert_counts[eid].item()
        if count == 0:
            continue
        expert_out = self.experts[eid](sorted_x[offset:offset+count])
        sorted_x[offset:offset+count] = expert_out * sorted_weights[offset:offset+count]
        offset += count

    # Unsort and reduce
    result = torch.zeros_like(x_rep)
    result[sorted_idx] = sorted_x
    # Sum across top-k
    result = result.view(-1, self.top_k, D).sum(dim=1)
    return result
```

This batched scatter/gather implementation runs each expert exactly **once** per forward pass (batching all its assigned tokens), instead of the 22-iteration nested-loop baseline.

### 1.2 `torch.tensor(loop_idx)` Creates a CPU Tensor Every Forward Pass

**File:** `v4_model.py:122-124`

```python
loop_emb = self.loop_embeddings(
    torch.tensor(loop_idx, device=x.device)
)
```

`torch.tensor()` is called **18 times per forward pass** (3 blocks × 6 loops). This creates a new tensor each time. With `torch.compile`, this may be traced and fused, but it's still bad practice and may cause recompilation with different graph shapes.

**Fix:** Pre-compute loop index tensors once:
```python
# In __init__:
self.register_buffer("loop_indices",
    torch.arange(config.recursive_loops), persistent=False)

# In forward:
loop_emb = self.loop_embeddings(self.loop_indices[loop_idx])
```

### 1.3 Adapter Residual Connection Placement May Cause Information Loss

**File:** `v4_model.py:232`

```python
x_mod = x + adapter_scale * (x @ adapter_a @ adapter_b)
# ... attention writes into x_mod ...
x = x_mod + self.out_proj(attn_out)  # Residual from x_mod, not x
```

The adapter modifies `x` into `x_mod`, and then the attention residual is added to `x_mod`. This means the **adapter shift is permanent** — it cannot be undone by the attention mechanism. This is intentional (per-pass differentiation), but has a subtle issue: the adapter output depends on `x`, but the residual stream now carries the adapter shift through **all subsequent layers in this loop**. If the adapter produces a bad direction early in training (before B learns useful structure), it could destabilize training.

**Recommendation:** Consider applying the adapter **after** the attention residual as a separate residual path:
```python
# Safer: adapter as post-attention modulation
h = self.norm_attn(x)
# ... attention ...
x = x + self.out_proj(attn_out)
x = x + adapter_scale * (x @ adapter_a @ adapter_b)  # Adapter after attention
```

This is more stable because the adapter modifies a richer representation (post-attention), and the residual stream is preserved cleanly through attention.

---

## 2. SIGNIFICANT ISSUES (Should Fix)

### 2.1 No Expert Capacity Factor — Risk of Token Dropping at Scale

The MoE layer has no capacity factor limiting how many tokens each expert processes. In the current loop-based dispatch this doesn't matter (all tokens are processed), but if you switch to a proper scatter-gather dispatch, you should add a capacity factor to prevent OOM from load imbalance.

**Recommendation:** Add `expert_capacity_factor: float = 1.25` to config. Cap tokens per expert at `capacity = ceil(capacity_factor * B * S * top_k / num_routed)`.

### 2.2 Router Should Use Sigmoid + Top-K (Not Softmax)

**File:** `v4_model.py:128`

Recent MoE research (DeepSeek-V2, Mixtral follow-ups, GROK) has moved away from softmax routing toward **sigmoid-based routing** with top-k selection. Softmax creates competition between experts that can lead to routing collapse (where a few experts dominate).

```python
# Current:
router_probs = F.softmax(router_logits, dim=-1)

# Recommended (DeepSeek-V2 style):
router_probs = torch.sigmoid(router_logits)
top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
# No renormalization needed — each expert gate is independent
```

With sigmoid routing, each expert's gate is independent, which:
- Reduces routing collapse
- Allows the model to express "no expert is very relevant" (all low) or "many experts are relevant" (all high)
- Makes the auxiliary loss less critical

### 2.3 Missing Router Z-Loss for Stability

Beyond the auxiliary load-balancing loss, modern MoE models (ST-MoE, PaLM) add a **router z-loss** that penalizes large router logits. This prevents the router from becoming overconfident, which causes training instability.

```python
# Add to _compute_aux_loss:
z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
self.aux_loss = load_balance_loss + self.z_loss_coeff * z_loss
```

**Recommended z_loss_coeff:** 0.001

### 2.4 RoPE Implementation Has Redundant Concatenation

**File:** `v4_model.py:40`

```python
cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
```

This duplicates the cos/sin tensors along the feature dimension. The standard implementation (used by Llama, Mistral, etc.) applies rotation to interleaved pairs without duplication. The current approach works but uses **2x the memory** for RoPE buffers and has a non-standard rotation pattern.

The `apply_rope` function at line 46-49 splits into halves and rotates, which matches the "GPT-NeoX style" rotation. This works correctly but differs from the "Llama style" interleaved rotation that most modern models use. Either works — just be aware that this is the NeoX convention.

### 2.5 Lion Optimizer with MoE Requires Careful Tuning

**File:** `v4_train_config.py:28`

Lion optimizer with weight_decay=0.3 is aggressive. For MoE models, the router weights are extremely sensitive to weight decay — too much decay pushes all router weights toward zero, causing uniform routing (defeating the purpose of MoE).

**Recommendation:** Use differential weight decay:
```python
# Router parameters: no weight decay
# Expert parameters: standard weight decay
# Dense parameters: standard weight decay
router_params = [p for n, p in model.named_parameters() if "router" in n]
other_params = [p for n, p in model.named_parameters() if "router" not in n]
optimizer = Lion([
    {"params": router_params, "weight_decay": 0.0},
    {"params": other_params, "weight_decay": 0.3},
], lr=6e-4)
```

### 2.6 Parallel Dense + MoE Residual Scaling

**File:** `v4_model.py:253-255`

```python
h_dense = self.ffn_dense(self.norm_dense(x))
h_moe = self.moe(self.norm_moe(x), loop_idx)
x = x + h_dense + h_moe
```

Both the dense FFN and MoE output contribute to the residual stream at full scale. This effectively **doubles the residual update magnitude** compared to a standard transformer block. Over 18 effective layers, this can cause the residual stream to grow unboundedly.

**Recommendation:** Scale the contributions:
```python
# Option A: Fixed scaling (simple, used by some Mixtral variants)
x = x + 0.5 * h_dense + 0.5 * h_moe

# Option B: Learnable gate (more flexible)
self.dense_gate = nn.Parameter(torch.tensor(0.5))
self.moe_gate = nn.Parameter(torch.tensor(0.5))
x = x + self.dense_gate * h_dense + self.moe_gate * h_moe
```

---

## 3. MODERATE ISSUES (Nice to Have)

### 3.1 No KV-Cache for Inference — Generate is O(n²) per Token

**File:** `v4_model.py:416-418`

The `generate()` method re-processes the full context at every token. For seq_len=8192, this means the 512th new token processes 8704 tokens from scratch. This makes inference extremely slow.

**Impact:** Only matters for inference, not training. But if you plan to use this model for GRPO (which generates 16 completions per prompt), this will be a major bottleneck.

**Recommendation:** Add `past_key_values` support to the forward pass. This is complex with recursive weight sharing (you'd need KV caches per loop × per block = 18 cache slots), but essential for practical inference.

### 3.2 Inter-Loop Normalization May Constrain Representational Growth

**File:** `v4_model.py:346-347`

```python
if loop < self.config.recursive_loops - 1:
    x = self.norm_loop(x)
```

Applying RMSNorm between loops constrains the hidden state magnitude. This is good for stability but may prevent the model from gradually building up representational complexity across loops. Universal Transformers and similar recursive architectures often benefit from **not normalizing** between passes, or using a learnable interpolation.

**Alternative:** Consider a learnable residual gate per loop:
```python
# gate_i controls how much of the new representation to keep vs the old
x = gate_i * norm_loop(x) + (1 - gate_i) * x
```

### 3.3 `num_workers=0` in DataLoader

**File:** `v4_data.py:178`

With `num_workers=0`, all data loading happens in the main process, blocking the GPU. For streaming datasets, even `num_workers=2` would help overlap data fetching with computation.

**Caveat:** With streaming HF datasets, multi-worker can cause duplication. Use `num_workers=2` with proper worker sharding (already partially handled by the seed logic).

### 3.4 Gradient Checkpointing Configuration & Tradeoffs

**File:** `v4_model.py:268`

The config declares `supports_gradient_checkpointing = True`, and `v4_model.py` now includes gradient checkpointing logic that wraps the recursive block when this flag is enabled. This is appropriate for a 305M parameter model with 8192 seq_len, where activation memory would otherwise be substantial.

**Recommendation:** Keep gradient checkpointing enabled for long-context training runs, and ensure tests cover both configurations (with and without gradient checkpointing) so that any future refactors do not silently break the checkpointed path.

---

## 4. ARCHITECTURAL STRENGTHS (What's Done Well)

### 4.1 Loop-Aware Router — Novel and Well-Motivated
The decision to concatenate learned loop embeddings with hidden states for routing is excellent. This allows the MoE to specialize experts per recursive depth — e.g., early loops might route to "syntax" experts while later loops route to "reasoning" experts. This is a genuinely novel contribution that differentiates this from simply stacking MoE layers.

### 4.2 Shared Expert Architecture
Having 1 always-active shared expert alongside 11 routed experts follows the DeepSeek-MoE pattern. The shared expert acts as a "safety net" ensuring every token gets meaningful processing regardless of routing decisions. This is especially important during early training when the router hasn't learned useful patterns yet.

### 4.3 Parallel Dense + MoE Residual Paths
Running dense FFN and MoE FFN in parallel (rather than sequential) is a good choice for recursive architectures. It provides a guaranteed dense processing path while allowing the MoE to add specialized capacity. This reduces the risk of the recursive model degrading if MoE routing is poor.

### 4.4 Per-Pass Adapter Design
The low-rank adapter pairs (A_i, B_i) with zero-init B are well-designed. Starting as identity (no adapter effect) and letting gradients differentiate each pass is sound. The rank-16 choice is conservative enough to not overparameterize.

### 4.5 Weight-Tied LM Head
Using the embedding matrix as the LM head (`F.linear(hidden, self.model.embedding.weight)`) saves ~197M parameters (128256 × 1536) and is standard best practice.

### 4.6 Progressive Curriculum
The 2048→4096→8192 sequence length curriculum with shifting dataset mixtures (syntax→knowledge→instruction) is a well-proven approach. The TinyStories warmup in phase 1 is smart for establishing basic language modeling before scaling up.

### 4.7 HRA for Post-Training
High Rank Adaptation (rank 256-512) for SFT/GRPO is well-suited to this architecture. The recursive weight sharing means each adapter injection point is reused 6×, so the effective adaptation capacity is much higher than the parameter count suggests.

---

## 5. PARAMETER BUDGET ANALYSIS

| Component | Params | % of Total |
|-----------|--------|-----------|
| Embedding (128256 × 1536) | 196.9M | 64.6% |
| 3× Attention (QKV + out_proj) | 28.3M | 9.3% |
| 3× Dense SwiGLU (1536→4096) | 56.6M | 18.6% |
| 3× Shared Expert (1536→1024) | 14.2M | 4.7% |
| 3× 11 Routed Experts | 155.7M | ~51% of non-embedding |
| 18× Per-pass Adapters (rank 16) | 0.88M | 0.3% |
| Router + Loop Embeddings | ~0.1M | <0.1% |
| **Total Physical** | **~305M** | |
| **Active per token** | **~155M** | ~51% sparsity |

**Observation:** This parameter budget assumes a 128,256-token vocabulary, which yields a ~196.9M-parameter embedding layer and a total of ~305M parameters. The embedding layer dominates at 64.6% of parameters, which is typical for small models with large vocabularies. In contrast, the current v4 default configs use a 65,536-token vocabulary and are ~208M parameters in total (primarily due to a smaller embedding matrix). Consider whether a 128K vocabulary is necessary — e.g., dropping to a 32K vocabulary in this regime would save on the order of ~147M parameters that could be reallocated to more experts or a larger hidden dim.

---

## 6. COMPARISON WITH CURRENT BEST PRACTICES

| Aspect | v4 Design | Best Practice (2025-2026) | Assessment |
|--------|-----------|--------------------------|------------|
| Expert routing | Softmax + top-2 | Sigmoid + top-2 (DeepSeek-V3) | Update recommended |
| Load balancing | Switch-style aux loss | Aux loss + z-loss + expert capacity | Partially aligned |
| Expert dispatch | Python loop | Grouped GEMM / Megablocks | Must fix for speed |
| Shared expert | 1 shared + 11 routed | Yes, shared experts are standard | Aligned |
| FFN activation | SwiGLU | SwiGLU | Aligned |
| Position encoding | RoPE (NeoX style) | RoPE (Llama style, with NTK scaling) | Acceptable |
| Normalization | RMSNorm (pre-norm) | RMSNorm (pre-norm) | Aligned |
| Attention | SDPA (FlashAttention) | FlashAttention-2/3 | Aligned |
| Weight tying | Embedding ↔ LM head | Standard practice | Aligned |
| Recursive sharing | Per-pass adapters | Novel (no standard yet) | Innovative |
| KV-cache | Not implemented | Essential for inference | Gap |
| Gradient checkpointing | Declared but not implemented | Should implement | Gap |

---

## 7. PRIORITIZED RECOMMENDATIONS

### Must-Fix (Before Training)
1. **Replace expert dispatch loop** with batched scatter-gather (Section 1.1) — 5-10x speedup expected
2. **Cache loop index tensors** (Section 1.2) — trivial fix, prevents torch.compile issues
3. **Add router z-loss** (Section 2.3) — critical for MoE training stability

### Should-Fix (Before Long Training Runs)
4. **Switch to sigmoid routing** (Section 2.2) — reduces routing collapse risk
5. **Differential weight decay for router** (Section 2.5) — prevents router weight decay issues with Lion
6. **Scale parallel dense + MoE contributions** (Section 2.6) — prevents residual stream explosion

### Nice-to-Have (Before Inference/GRPO)
7. **Implement KV-cache** (Section 3.1) — essential for GRPO (16 completions/prompt)
8. **Implement gradient checkpointing** (Section 3.4) — helps with 8192 seq_len phase
9. **Consider adapter placement** (Section 1.3) — experiment with post-attention placement
10. **Add num_workers to DataLoader** (Section 3.3) — simple throughput improvement

---

## 8. OVERALL ASSESSMENT

This is a **well-conceived novel architecture** with strong foundational choices. The combination of recursive weight sharing + MoE + loop-aware routing is genuinely innovative — I'm not aware of another model that does exactly this. The key risks are:

1. **Training speed** — the expert dispatch loop will make training painfully slow without the scatter-gather fix
2. **MoE stability** — without z-loss and proper router weight decay, the model may experience routing collapse during the 300K-step pre-training
3. **Inference speed** — no KV-cache means GRPO will be extremely slow

The HuggingFace compatibility, progressive curriculum, and post-training pipeline (SFT → GRPO with HRA) are all well-designed and production-ready. The codebase is clean, well-documented, and follows good engineering practices.

**Bottom line:** Fix the 3 must-fix issues, and this architecture is ready for an ambitious training run.
