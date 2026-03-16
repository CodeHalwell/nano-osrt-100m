# Nano-OSRT v4 — Recursive MoE Architecture Plan

## Overview

v4 combines recursive weight sharing (proven in v3) with Mixture of Experts
to dramatically increase model capacity while keeping active compute manageable.

| Property | v3 | v4 (proposed) |
|----------|-----|--------------|
| Physical blocks | 2 | 3 |
| Recursive loops | 6 | 6 |
| Effective layers | 12 | 18 |
| Dense FFN | Yes | Yes (kept) |
| MoE FFN | No | 12 experts (1 shared + 11 routed, top-2) |
| Physical params | 104.5M | ~356M |
| Active params/token | 104.5M | ~180M |
| Effective params (recursive) | ~302M | ~2.1B |
| HRA (post-training) | +11M | +15-20M |
| Hidden dim | 1280 | 1536 |
| Attention heads | 20 | 24 |
| Head dim | 64 | 64 |
| Tokenizer | GPT-NeoX (50K) | Custom 64K BPE |
| Seq len (training) | 2048 → 4096 | 2048 → 4096 → 8192 |
| Seq len (inference) | 4096 | 16K+ (RoPE scaling) |

---

## 1. Architecture Details

### 1.1 Block Structure

Each of the 3 physical `RecursiveBlock` modules contains:

```
Input
  ├── RMSNorm → Per-pass adapter → Causal Attention (RoPE) → Residual
  ├── RMSNorm → Dense SwiGLU FFN → Residual
  └── RMSNorm → MoE FFN (shared + top-2 routed) → Residual
```

The dense FFN and MoE FFN run in **parallel residual** paths:

```python
# Dense path (always active, full capacity)
h_dense = self.ffn_dense(self.norm_dense(x))

# Sparse MoE path (shared expert + top-2 routed)
h_moe = self.moe(self.norm_moe(x))

# Combined residual
x = x + h_dense + h_moe
```

### 1.2 MoE Layer

```
                    ┌─────────────┐
                    │   Router    │ (dim → 11, softmax)
                    │  + loop     │
                    │  embedding  │
                    └──────┬──────┘
                           │ top-2 indices + weights
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌──────────┐ ┌──────────┐
     │   Shared    │ │ Expert i │ │ Expert j │  (top-2 of 11)
     │   Expert    │ │          │ │          │
     │ (always on) │ │          │ │          │
     └──────┬──────┘ └─────┬────┘ └─────┬────┘
            │              │            │
            └──────────────┼────────────┘
                           ▼
                    weighted sum + shared
```

**Router design:**
- Input: hidden state + learned loop embedding (loop-aware routing)
- Output: softmax over 11 routed experts
- Top-2 selection with load balancing auxiliary loss
- Shared expert output added unconditionally

**Expert architecture:**
- Each expert is a small SwiGLU: dim → expert_hidden → dim
- Expert hidden = dim × 2/3 ≈ 1024 (half the dense FFN hidden)
- All experts share the same architecture, different weights

### 1.3 Parameter Budget

**Dimension: 1536, Heads: 24, Expert hidden: 1024, Dense hidden: 4096**

| Component | Per Block | × 3 Blocks |
|-----------|----------|------------|
| QKV projection (1536 → 4608) | 7.08M | 21.2M |
| Output projection (1536 → 1536) | 2.36M | 7.1M |
| Dense SwiGLU (1536 → 4096 → 1536) | 18.87M | 56.6M |
| Shared expert (1536 → 1024 → 1536) | 4.72M | 14.2M |
| 11 routed experts × 4.72M | 51.9M | 155.7M |
| RMSNorm ×3 | 4.6K | 13.8K |
| **Block total** | **84.9M** | **254.8M** |

| Component | Params |
|-----------|--------|
| Token embedding (32K × 1536) | 49.2M |
| Blocks (3) | 254.8M |
| Per-pass adapters (18 × rank 16) | 0.88M |
| Norm layers (loop + output) | 3.1K |
| Router weights (3 blocks × 11) | ~50K |
| **Total physical** | **~305M** |
| **Active per token** | **~155M** |
| **Effective (recursive ×6)** | **~930M** |

> Note: Physical params are higher than 200M target. Options to reduce:
> - Reduce to 8 experts (saves ~60M)
> - Reduce expert hidden to 768 (saves ~40M)
> - Reduce dim to 1408 (saves ~30M across everything)
> - Drop to 2 blocks (back to v3 structure, saves ~85M)

**Recommended trim: 8 routed experts + 1 shared = 9 total, expert hidden 896:**

| | 12 experts | 9 experts (trimmed) |
|--|-----------|-------------------|
| Physical | ~305M | ~220M |
| Active/token | ~155M | ~135M |
| Effective | ~930M | ~810M |

### 1.4 Tokenizer

**Recommendation: Train a custom 32K BPE tokenizer** or use Llama 3's tokenizer.

| Tokenizer | Vocab | Embedding params (at 1536 dim) |
|-----------|-------|-------------------------------|
| GPT-NeoX (current) | 50,277 | 77.2M (35% of 220M) |
| Llama 3 | 128,256 | 197M (too big!) |
| Custom 32K BPE | 32,000 | 49.2M (22% of 220M) |
| Mistral v0.3 | 32,768 | 50.3M |

Llama 3's 128K vocab is too large — it would eat most of the parameter budget.
A 32K tokenizer is the sweet spot for this model size. We can train one on our
pre-training data mix using SentencePiece, or use Mistral's.

**Practical recommendation: Use Mistral's tokenizer (32,768 vocab).** It's
well-optimized for code + multilingual, Apache 2.0 licensed, and widely supported.

### 1.5 RoPE Scaling

For inference beyond training seq_len:

```python
# NTK-aware scaling (simple, effective)
def compute_rope_freqs_ntk(seq_len, dim, theta=10000.0, scale_factor=4.0):
    # Scale theta by factor to extend context
    scaled_theta = theta * scale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (scaled_theta ** (torch.arange(0, dim, 2) / dim))
    ...
```

Train at progressive lengths: 2048 → 4096 → 8192
Inference: NTK scaling to 16K-32K without additional training.

### 1.6 Per-Pass Adapters

Same as v3 but more pairs: 3 blocks × 6 loops = 18 adapter pairs.
Rank 16, activation-space modulation: `x_mod = x + scale * (x @ A @ B)`.

Inter-loop normalization (RMSNorm) between each recursive pass.

---

## 2. Pre-training

### 2.1 Data Mix

Code mixed in from the start (unlike v3 which was text-only):

| Phase | Steps | Seq len | Dataset | Purpose |
|-------|-------|---------|---------|---------|
| 1: Foundation | 0 → 10K | 2048 | TinyStories + CodeParrot-clean | Syntax for text + code |
| 2: Knowledge | 10K → 120K | 4096 | FineWeb-Edu (60%) + StarCoder-data (25%) + Wikipedia (15%) | Dense knowledge + code patterns |
| 3: Instruction | 120K → 140K | 8192 | SmolTalk (50%) + Evol-Instruct-Code (30%) + OpenHermes (20%) | Chat + code instruction format |

**Target tokens:** ~30-50B (budget dependent)
- At 70K tok/s on H100: 50B tokens ≈ 200h ≈ $800 (over budget)
- At 70K tok/s on A100: 30B tokens ≈ 120h ≈ $334 (fits budget)
- **Realistic target: 25-35B tokens on A100**

### 2.2 Optimizer

| Option | Pros | Cons |
|--------|------|------|
| Lion | Half optimizer VRAM | Needs careful tuning |
| AdamW | Robust, well-understood | 2× optimizer VRAM |
| Muon | Strong recent results | Less proven at scale |

**Recommendation: Lion** for pre-training (VRAM savings critical with 305M params + MoE).
AdamW for post-training (smaller, differential LR needed).

### 2.3 Compute Budget

| Resource | A100 80GB | H100 |
|----------|----------|------|
| Cost/hr | $2.78 | $3.95 |
| Budget $300 | 108h | 76h |
| Budget $400 | 144h | 101h |
| Est. tok/s (220M model) | 50-70K | 80-120K |
| Est. tokens (108h A100) | 19-27B | — |
| Est. tokens (101h H100) | 29-44B | — |

**Recommendation: H100 at $400 budget** — more tokens for similar cost.

### 2.4 MoE-specific Training Considerations

**Load balancing loss:**
```python
# Auxiliary loss to prevent expert collapse
aux_loss = num_experts * (fraction_routed_to_expert * avg_gate_prob_for_expert).sum()
total_loss = language_model_loss + 0.01 * aux_loss
```

**Expert parallelism:** Not needed at this scale — all experts fit in H100 80GB.

**Gradient scaling:** Each expert sees fewer tokens (top-2 of 12 = ~17% of tokens).
May need slightly higher LR for expert FFNs.

---

## 3. Post-training Pipeline

### 3.1 SFT (Balanced)

Mixed domains from the start:

| Domain | Weight | Datasets |
|--------|--------|----------|
| Math/reasoning | 25% | GSM8K, NuminaMath-CoT, MathInstruct |
| Code | 25% | Evol-Instruct-Code, CodeAlpaca, Python-instructions |
| STEM | 20% | MegaScience, OpenThoughts3 (science subset) |
| General | 20% | Alpaca, OpenHermes, SlimOrca |
| Instruction following | 10% | IFEval-like |

With `<think>...</think>` chain-of-thought format across all domains.

### 3.2 GRPO (Retry)

At 220M+ params with better math SFT, the model should generate correct
GSM8K answers often enough for GRPO to work (v3 failed at 115M — 0-6% accuracy).

Target: 10-20%+ accuracy before GRPO starts → meaningful reward signal.

Updated reward functions (from v3 lessons):
- Correctness: +1.0
- Format: +0.2
- Reasoning bonus: +0.3 (multi-step thinking when correct)
- Truncation penalty: -0.5 (degenerate loops)
- No length penalty (reasoning needs room)

### 3.3 HRA

Rank 256-512 depending on VRAM. At 220M base, rank 256 adds ~15-20M.

---

## 4. Evaluation & Release

### 4.1 Benchmark Suite

| Benchmark | Tests | Comparison |
|-----------|-------|-----------|
| IFEval | Instruction following | Gemma 270M: 51.2% |
| GSM8K | Math reasoning | — |
| HellaSwag | Commonsense | — |
| HumanEval | Code generation | — |
| ARC-Challenge | Science reasoning | — |
| MMLU (5-shot) | General knowledge | — |

### 4.2 HuggingFace Hub

- Model card with architecture description + benchmarks
- Safetensors format
- HF-compatible wrapper (from v3, extended for MoE)
- Inference code + examples

---

## 5. Open Questions

1. **Physical params vs budget:** 305M is over the 200M target. Trim experts (9 instead of 12) or reduce dim?
2. **Tokenizer:** Train custom 32K BPE, or use Mistral's off-the-shelf?
3. **Expert granularity:** Fewer bigger experts (9 × 5M) vs more smaller experts (12 × 3.5M)?
4. **Parallel residual:** Dense + MoE as parallel residual, or sequential (dense → MoE)?
5. **Loop-aware routing:** Should the router see a loop embedding, or just the hidden state?
