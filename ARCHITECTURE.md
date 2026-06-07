# ARCHITECTURE.md — OSRT-600M technical specification

**Scope:** the complete, implementation-ready specification of the
OSRT-600M model. Every layer, every dimension, every formula,
every connection.

**Companion docs:**
- [`README.md`](README.md) — design philosophy, why each choice was made
- [`LEARNINGS.md`](LEARNINGS.md) — v5 lessons that shaped these choices
- [`RESEARCH.md`](RESEARCH.md) — external research cited

**Reading order:** read README.md first for context, then this doc for
the technical details. Someone could implement the model from
ARCHITECTURE.md alone.

---

## Table of contents

1. [One-sentence overview](#1-one-sentence-overview)
2. [Parameter budget](#2-parameter-budget)
3. [Tokenizer specification](#3-tokenizer-specification)
4. [Embedding layer](#4-embedding-layer)
5. [Recursive transformer block](#5-recursive-transformer-block)
6. [Attention sub-block](#6-attention-sub-block)
7. [MoE sub-block](#7-moe-sub-block)
8. [Manifold-Constrained Hyper-Connections (mHC)](#8-manifold-constrained-hyper-connections-mhc)
9. [LM head and auxiliary heads](#9-lm-head-and-auxiliary-heads)
10. [Forward pass walkthrough](#10-forward-pass-walkthrough)
11. [Training losses](#11-training-losses)
12. [Inference path](#12-inference-path)
13. [KV cache structure](#13-kv-cache-structure)
14. [Quantization for deployment](#14-quantization-for-deployment)
15. [Total compute and memory math](#15-total-compute-and-memory-math)
16. [Architectural invariants](#16-architectural-invariants)

---

## 1. One-sentence overview

OSRT-600M is a **recursive Mixtral-style sparse MoE transformer** with
**3 physical decoder blocks applied 6 times via depth recurrence**
(giving 18 effective layers), using **HRA adapters**, **gated short
convolutions + GQA attention**, **manifold-constrained
hyper-connections**, **Muon-optimized weights**, and **MLA-style KV
cache compression** — totaling ~599M physical params, ~206M active
per token, ~2.5B FLOPs equivalent per token.

---

## 2. Parameter budget

### 2.1 Exact accounting

```
COMPONENT                                          PHYSICAL          ACTIVE PER TOKEN
─────────────────────────────────────────────────────────────────────────────────────
Embedding (32,768 × 1,536, tied with LM head)      100,663,296       ≈ same
  -- one row used per token at embedding lookup
  -- full matrix touched at LM head computation

Attention × 3 physical blocks                       28,311,552       28,311,552
  -- per block: 4 × (1,536²) = 9,437,184
  -- W_Q, W_K_DOWN, W_V_DOWN, W_O projections
  -- (W_K, W_V are partial — see §6 MLA-style design)

Shared experts × 3 (SwiGLU, h=4,608)                63,700,992       63,700,992
  -- per block: 3 × 1,536 × 4,608 = 21,233,664
  -- always active on every token

Routed experts: 3 × 12 × (SwiGLU, h=1,920)         318,504,960       53,084,160 (top-2 of 12)
  -- per expert: 3 × 1,536 × 1,920 = 8,847,360
  -- 12 experts per physical block, top-2 active

HRA adapters (rank 256, 87 injection points)        86,114,304       86,114,304
  -- adapter_a (1,536 × 256) + adapter_b (256 × 1,536)
  -- per injection: 786,432 params
  -- trainable during all stages

mHC residual transforms (n_hc = 4)                    ~720,000        ~720,000

Router projections (per block, sigmoid(W·x))           ~55,000          ~55,000

Loop embeddings (6, 1,536)                              9,216           9,216

LayerNorms (sandwich, pre + post per block × 6)         9,000           9,000

V-from-K transformations (per attention block × 6)     ~98,000         ~98,000

Attention sink logits (24 query heads, learnable)         576              576

─────────────────────────────────────────────────────────────────────────────────────
TOTAL PHYSICAL                                    ~598,800,000
TOTAL ACTIVE PER TOKEN                                              ~232,500,000
ACTIVE FRACTION                                                      ≈ 38.8%
```

### 2.2 At-a-glance

- **Hidden dimension `d_model`**: 1,536
- **Vocab size**: 65,536 (BPE)
- **Physical transformer blocks**: 3
- **Recursive loops**: 6 → 18 effective layers
- **Attention**: GQA 24 query heads / 8 KV heads / head_dim 64
- **MoE**: 1 shared expert (h=4,608) + 12 routed (h=1,920), top-2
- **HRA adapter rank**: 256
- **mHC expansion**: 4× residual stream width
- **Position encoding**: Partial RoPE (last 64 dims of Q and K)
- **Activation**: SwiGLU (FFN), Sqrt(Softplus) (routing affinity)
- **Norm**: RMSNorm pre + post sandwich

### 2.3 FLOP count per token (forward pass, one inference)

```
6 loops × (
    1 attention pass: ~2 × (28M / 3) = ~19M FLOPs
  + 1 shared expert: ~2 × 21M = 42M FLOPs
  + 1 routed expert pair (top-2): ~2 × 2 × 8.8M = 35M FLOPs
  + HRA adapter contribution: ~2 × (86M / 18) = ~9M FLOPs
  + mHC + norms: ~5M FLOPs
)
= 6 × 110M = ~660M FLOPs per forward pass
+ embedding lookup: ~1.5M
+ LM head: ~150M

TOTAL: ~810M FLOPs per token (forward)
With backward: ~2.4 BFLOPs per token

(Numbers are approximate; FLOP definitions vary across sources.
Use these as ratios, not absolutes.)
```

---

## 3. Tokenizer specification

### 3.1 BPE configuration

- **Algorithm**: byte-level BPE (sentencepiece or HuggingFace
  tokenizers)
- **Vocab size**: 65,536
- **Encoding focus**: English + 6 multilingual (Arabic, Japanese,
  Korean, Spanish, French, German) + code (Python, JS, Rust, C++)
- **Pre-tokenization**: GPT-2 style regex (handles contractions,
  numbers, punctuation)

### 3.2 Special tokens (reserved IDs)

| token | id | role |
|---|---|---|
| `<|begin_of_text|>` | 0 | BOS |
| `<|end_of_text|>` | 1 | EOS |
| `<|padding|>` | 2 | PAD |
| `<|unknown|>` | 3 | unk |
| `<|fim_prefix|>` | 4 | FIM prefix marker |
| `<|fim_middle|>` | 5 | FIM middle marker |
| `<|fim_suffix|>` | 6 | FIM suffix marker |
| `<|think|>` | 7 | reasoning block open |
| `<|/think|>` | 8 | reasoning block close |
| `<|answer|>` | 9 | answer block open |
| `<|/answer|>` | 10 | answer block close |
| `<|user|>` | 11 | user turn open |
| `<|assistant|>` | 12 | assistant turn open |
| `<|system|>` | 13 | system prompt open |
| `<|end_turn|>` | 14 | turn separator (ChatML style) |
| `<|tool_call|>` | 15 | tool invocation open |
| `<|/tool_call|>` | 16 | tool invocation close |
| `<|tool_result|>` | 17 | tool result open |
| `<|/tool_result|>` | 18 | tool result close |
| `<|image|>` | 19 | reserved for vision retrofit |
| `<|audio|>` | 20 | reserved for future audio |

IDs 21-31 reserved for future expansion. Real vocab begins at id 32.

### 3.3 Chat template

```
<|system|>{system_message}
<|user|>{user_question}
<|assistant|><|think|>{reasoning}<|/think|><|answer|>{final_answer}<|/answer|>
<|end_turn|>
```

Multi-turn:
```
<|system|>{system}
<|user|>{q1}<|assistant|>{a1}<|end_turn|>
<|user|>{q2}<|assistant|>{a2}<|end_turn|>
```

Tool use:
```
<|user|>{question_needing_calc}<|assistant|>
<|think|>I need to compute 17 × 23.<|/think|>
<|tool_call|>calculator("17 * 23")<|/tool_call|>
<|tool_result|>391<|/tool_result|>
<|answer|>The answer is 391.<|/answer|><|end_turn|>
```

---

## 4. Embedding layer

### 4.1 Shape and tying

- `embedding_matrix ∈ ℝ^(65536 × 1536)`
- **Tied with LM head**: `lm_head.weight = embedding.weight`
- Total params: 100,663,296 (16.9% of model)

### 4.2 Initialization

- Truncated normal, std = 1 / √(1536) ≈ 0.0255
- LM head logits scale: divide by √(1536) at output for μP
  compatibility

### 4.3 Optimizer routing

- **AdamW** (not Muon — embedding is special, see §11.2)
- No weight decay on embedding (preserve representation norms per
  SmolLM3 convention)

---

## 5. Recursive transformer block

### 5.1 Structure

```
For each loop r ∈ {0, 1, 2, 3, 4, 5}:
    For each physical block b ∈ {0, 1, 2}:

        # Add loop conditioning (broken symmetry per-iteration)
        x = x + loop_emb[min(r, 7)]    # if b == 0 (start of loop)

        # mHC pre-block mixing (replaces standard residual)
        residual = x
        x_normed = RMSNorm_pre[b](x)

        # Attention sub-block
        x_attn = AttentionBlock[b](x_normed, cache=kv_cache[b])

        # mHC post-attention residual mixing
        x = mHC_mix(residual, x_attn, b)

        # mHC pre-FFN mixing
        residual_ffn = x
        x_normed = RMSNorm_post[b](x)

        # MoE FFN sub-block
        x_ffn = MoEBlock[b](x_normed)

        # mHC post-FFN residual mixing
        x = mHC_mix(residual_ffn, x_ffn, b)
```

### 5.2 Loop embeddings

```
loop_emb ∈ ℝ^(6 × 1536)
```

Added BEFORE the first physical block at each loop. This is the
**only parameter that differs across loop iterations** — the bias
that tells the model "you're on iteration r of 6."

Capping at `min(r, 7)` means hard wall at R=8. Model trained for R=6
will function (with quality degradation) at R=3-5; cannot safely
extend beyond R=6 without retraining loop embeddings.

### 5.3 Sandwich RMSNorm

Two RMSNorm layers per physical block per sub-block:
- `RMSNorm_pre[b]` before attention
- `RMSNorm_post[b]` before MoE FFN

Each is `RMSNorm(d_model=1536, eps=1e-6)` with learnable scale, no
bias. Total norm params per block: 2 × 1536 = 3,072.

Gemma 3's "sandwich" placement validated for deep stacks; Huginn used
similar to survive 32+ recursive iterations.

### 5.4 HRA injection

87 HRA injection points across the model. At each point:
```
adapter_a ∈ ℝ^(1536 × 256)
adapter_b ∈ ℝ^(256 × 1536)
HRA_output(x) = x + adapter_b(adapter_a(x))    # low-rank residual
```

Injected into: Q/K/V projections, attention output, gate/up/down of
each expert, router projection. Trainable in all stages, especially
during RL (HRA-only training in GRPO stage).

---

## 6. Attention sub-block

### 6.1 GQA configuration

- **Query heads**: 24
- **Key/Value heads**: 8 (groups of 3 queries share a KV head)
- **Head dimension**: 64
- **Total Q dim**: 24 × 64 = 1,536
- **Total K dim**: 8 × 64 = 512
- **Total V dim**: 8 × 64 = 512

### 6.2 Projections

```
W_Q ∈ ℝ^(1536 × 1536)        # 2.36M params
W_K_DOWN ∈ ℝ^(1536 × 512)    # 0.79M params — to latent K
W_V_FROM_K ∈ ℝ^(512 × 512)   # 0.26M params — derive V from K
b_V ∈ ℝ^(512)                # bias for V derivation
W_O ∈ ℝ^(1536 × 1536)        # 2.36M params
```

Per block: ~5.76M params; across 3 blocks: ~17.3M. Plus HRA adapters.

### 6.3 V derived from K (MLA-inspired)

```
K = W_K_DOWN @ x_normed          # [batch, seq, 512]
V = W_V_FROM_K @ K + b_V         # [batch, seq, 512] — derived from K
```

**Cache only K** (not K and V separately) to halve KV cache. V is
recomputed at decode time via the learnable transform.

### 6.4 QK-Norm

Apply RMSNorm to each Q and K head independently before scaled dot-
product:
```
Q_head = RMSNorm(Q.view(batch, seq, 24, 64), dim=-1)
K_head = RMSNorm(K.view(batch, seq, 8, 64), dim=-1)
```

Prevents attention-logit explosion (Muon-trained models are
particularly prone; Kimi K2 added "QK-Clip" on top — we use just
QK-Norm and rely on Muon stability).

### 6.5 Partial RoPE

Apply RoPE to the **last 64 dimensions only** of Q and K head vectors:
- First 0 dims: position-free (content-only matching)
- Last 64 dims: rotary-encoded

Base θ = 10,000 (standard). Will be scaled via YaRN-style for context
extension in mid-training.

### 6.6 Attention sink

Learnable per-head sink logits:
```
sink_logits ∈ ℝ^(24)
```

Added to softmax denominator:
```
s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(sink_logits[h]))
```

Allows attention scores per head to sum to <1. Prevents the model
from being forced to attend somewhere when no relevant context exists.

### 6.7 Scaled dot-product attention

Standard formulation with the modifications above:
```
scores = (Q_head @ K_head.T) / √64
scores += causal_mask  # -inf above diagonal
attn_weights = softmax_with_sink(scores)
attn_output = attn_weights @ V_head  # V grouped to match heads
```

### 6.8 Output projection

```
attn_output_concat = attn_output.view(batch, seq, 1536)
attn_block_output = W_O @ attn_output_concat
```

HRA adapter applied to `W_O` output additively.

---

## 7. MoE sub-block

### 7.1 Structure

Each MoE block has:
- 1 always-active shared expert (large: h=4,608)
- 12 routed experts (small: h=1,920), top-2 active per token
- 1 router (linear projection + sigmoid + sqrt-softplus)

### 7.2 Shared expert (SwiGLU)

```
w_gate ∈ ℝ^(1536 × 4608)       # 7.08M params
w_up ∈ ℝ^(1536 × 4608)         # 7.08M params
w_down ∈ ℝ^(4608 × 1536)       # 7.08M params

shared_output(x) = w_down @ (SiLU(w_gate @ x) ⊙ (w_up @ x))
```

Per shared expert: 21.23M params. Across 3 blocks: 63.7M.

### 7.3 Routed experts (SwiGLU)

Per routed expert (smaller hidden):
```
w_gate ∈ ℝ^(1536 × 1920)       # 2.95M params
w_up ∈ ℝ^(1536 × 1920)         # 2.95M params
w_down ∈ ℝ^(1920 × 1536)       # 2.95M params
```

Per expert: 8.85M. Per block (12 experts): 106.2M. Across 3 blocks:
318.5M (39.1% of MoE-relevant params, ~53% of total physical).

### 7.4 Router

```
W_route ∈ ℝ^(1536 × 12)        # 18,432 params per block
b_route_bias ∈ ℝ^(12)          # per-expert bias for load balancing
                                # (not in gradient; nudged by load deviation)
```

Affinity score:
```
affinity = sqrt(softplus(W_route @ x))      # sqrt(softplus) — DeepSeek-V4
balanced_affinity = affinity + b_route_bias  # static bias for balancing
top_2_indices = argmax(balanced_affinity, k=2)

normalized_weights = softmax(balanced_affinity[top_2_indices])
# (DeepSeek-style: bias only in TOP-K selection, not in gating weights)
```

### 7.5 Hash routing for blocks 0 and 1

For physical blocks 0 and 1 (first 2 of 3), routing is HASH-based,
not learned:
```
expert_id = hash(token_id) mod 12
# Always select this fixed expert, no learned router
```

Stabilizes early training (prevents collapse before router learns).
Block 2 uses normal learned routing.

### 7.6 Aux-loss-free load balancing

Per-expert balancing bias `b_route_bias[i]` accumulates per training
step:
```
mean_load = (1/12) × total_tokens_in_batch
for i in range(12):
    deviation = expert_load[i] - mean_load
    if deviation > 0:
        b_route_bias[i] -= γ              # nudge down
    else:
        b_route_bias[i] += γ              # nudge up
# γ = 0.001 (per DeepSeek-V3)
# This bias is HEURISTIC — not in the gradient
```

Combined with a small sequence-balance loss (weight 0.0001) to prevent
extreme imbalance within single sequences.

### 7.7 MoE output

```
moe_output(x) = shared_output(x) + Σ_{i ∈ top2} weight_i × routed_output_i(x)
```

### 7.8 SwiGLU Clamping (stability)

Inside every SwiGLU (shared and routed):
```
gate_pre = w_gate @ x
up_pre = w_up @ x

# Apply DeepSeek-V4 stability clamps
gate_clamped = torch.clamp(gate_pre, max=10.0)         # cap upper
linear_clamped = torch.clamp(up_pre, min=-10.0, max=10.0)  # clamp both

output = w_down @ (SiLU(gate_clamped) ⊙ linear_clamped)
```

---

## 8. Manifold-Constrained Hyper-Connections (mHC)

### 8.1 Residual stream expansion

Standard transformers: residual stream is `[batch, seq, d_model]`
(1536 channels).

mHC expands by factor `n_hc = 4`:
```
X ∈ ℝ^(batch × seq × n_hc × d_model)    # 4 channels × 1,536 = 6,144 total
```

Per channel still operates on `d_model=1536`; the inner layers
(attention, MoE) consume one 1,536-dim view and produce one 1,536-dim
output. The 4× expansion is in the **residual** stream only.

### 8.2 Mixing matrices (dynamic)

Per mHC application (one per attention sub-block, one per MoE
sub-block), three mixing matrices:

```
A_l ∈ ℝ^(1 × 4)       # input mapping: residual → layer input
B_l ∈ ℝ^(4 × 4)       # residual transformation (the constrained matrix)
C_l ∈ ℝ^(4 × 1)       # output mapping: layer output → residual
```

Update rule:
```
X_{l+1} = B_l @ X_l + C_l @ F_l(A_l @ X_l)
        = (residual mixing) + (layer contribution)
```

Where `F_l` is either Attention or MoE depending on which sub-block.

### 8.3 Birkhoff polytope constraint on B_l

`B_l` is constrained to be **doubly stochastic** (rows and columns
sum to 1, all entries ≥ 0). This is the Birkhoff polytope of n×n
matrices.

Constraint enforcement:
```
# Start with unconstrained ~B_l (computed as in §8.4)
# Project onto manifold via Sinkhorn-Knopp iteration
M_0 = exp(~B_l)
for t in range(t_max=20):
    M = T_r(T_c(M))     # alternating row/column normalization
B_l = M
```

The doubly-stochastic constraint guarantees `||B_l||_2 ≤ 1` (spectral
norm bounded). This means the residual transformation is
**non-expansive** — guaranteed numerical stability across the
forward pass and backprop. Closed under multiplication, so deep stacks
(our 18 effective layers) stay stable.

### 8.4 Dynamic parameter generation

`A_l`, `B_l`, `C_l` are dynamically generated per token:

```
# Flatten + normalize the residual stream
X_flat = vec(X_l) ∈ ℝ^(1 × (4 × 1536))      # (1 × 6144)
X_normed = RMSNorm(X_flat)

# Generate raw (unconstrained) parameters
~A_l = α_pre × (X_normed @ W_pre) + S_pre        # (1 × 4)
~B_l = α_res × Mat(X_normed @ W_res) + S_res     # (4 × 4)
~C_l = α_post × (X_normed @ W_post)^T + S_post   # (4 × 1)
```

Where:
- `W_pre ∈ ℝ^(6144 × 4)` (dynamic component for A_l)
- `W_res ∈ ℝ^(6144 × 16)` (dynamic component for B_l, reshaped to 4×4)
- `W_post ∈ ℝ^(6144 × 4)` (dynamic component for C_l)
- `S_pre, S_res, S_post`: static learnable biases
- `α_pre, α_res, α_post`: learnable gating factors, initialized small

### 8.5 Sigmoid bounds on A_l and C_l

```
A_l = σ(~A_l)         # bounded [0, 1]
C_l = 2σ(~C_l)        # bounded [0, 2]
```

Prevents signal cancellation. The factor 2 on C_l preserves the
ability to scale layer contributions.

### 8.6 Cost (~720K params, ~6.7% overhead)

Total mHC params per injection point (one for attn, one for FFN, per
block, per loop ITERATION):
- 18 effective layers × 2 sub-blocks = 36 mHC applications
- But the dynamic generation matrices are SHARED across the loop
  iterations (per physical block)
- Net added: ~720K trainable params + ~6.7% wall-clock overhead

---

## 9. LM head and auxiliary heads

### 9.1 Main LM head

Tied with embedding:
```
logits = embedding.weight @ x_final.transpose(-1, -2)
# Shape: [batch, seq, 65536]
```

Final hidden state `x_final` comes from the LAST physical block of
the LAST loop iteration.

### 9.2 Auxiliary per-loop LM heads (architecture-fix knob)

The **same** LM head (tied with embedding) is applied to intermediate
loop outputs:
```
for r in range(6):
    if aux_loop_loss_weight > 0:
        x_loop_r = output of physical block 2 at loop r
        logits_loop_r = embedding.weight @ x_loop_r.transpose(-1, -2)
        # Use these for auxiliary cross-entropy losses
```

**Key insight:** the LM head is SHARED across all loop outputs (it
IS the embedding). No additional parameters. The per-loop training
signal alone makes intermediate loops produce coherent predictions.

This is what enables:
1. Architecture-fix: loops 1-5 actually contribute to predictions
2. Speculative decoding at inference: loop-3 output is a draft
   prediction, loop-6 verifies (~60-75% accept rate expected)

### 9.3 MTP (multi-token prediction) heads

Per DeepSeek-V3 / V4: predict tokens at offsets +1, +2 via separate
small heads on the FINAL loop output:
```
MTP_head_1 ∈ ℝ^(1536 × 65536)     # tied with embedding
MTP_head_2 ∈ ℝ^(1536 × 65536)     # tied with embedding
```

(Heads are tied with embedding too — no separate params, just
multiple uses of the LM head with different small projection layers
in between if needed.)

Used during training for the MTP loss; helps the main model learn
longer-range structure.

---

## 10. Forward pass walkthrough

Detailed pseudocode for one forward pass on a batch of `B` sequences
of length `L`:

```python
def forward(input_ids, kv_cache=None, training=False):
    # Step 1: Embedding lookup
    x = embedding(input_ids)              # [B, L, 1536]

    # Step 2: Initialize mHC residual stream (4× width)
    X = x.unsqueeze(-2).expand(-1, -1, 4, -1)    # [B, L, 4, 1536]

    # Step 3: Recursive loop
    per_loop_outputs = []
    for r in range(6):
        # Add loop bias to channel 0 (or to all channels — design choice)
        loop_bias = loop_emb[min(r, 7)]
        X[:, :, 0, :] = X[:, :, 0, :] + loop_bias

        for b in range(3):
            # mHC pre-attention
            X = X.reshape(B, L, 6144)
            A_l, B_l, C_l = generate_mHC_params(X, layer_id=(r, b, 'attn'))
            x_view = (A_l @ X.reshape(B, L, 4, 1536).transpose(2, 3)).squeeze(-1)

            # Pre-norm
            x_normed = RMSNorm_pre[b](x_view)

            # Attention sub-block
            x_attn = AttentionBlock[b](x_normed, kv_cache=kv_cache[r, b])

            # mHC update of residual
            X = (B_l @ X.reshape(B, L, 4, 1536).transpose(2, 3)).squeeze(-1) \
                + (C_l @ x_attn.unsqueeze(-2)).squeeze(-2)

            # mHC pre-FFN
            A_l_ffn, B_l_ffn, C_l_ffn = generate_mHC_params(X, layer_id=(r, b, 'ffn'))
            x_view = (A_l_ffn @ X.transpose(-1, -2)).squeeze(-1)

            # Post-norm
            x_normed = RMSNorm_post[b](x_view)

            # MoE sub-block
            if b < 2:
                x_moe = MoEBlock[b](x_normed, routing='hash')
            else:
                x_moe = MoEBlock[b](x_normed, routing='learned')

            # mHC update
            X = (B_l_ffn @ X) + (C_l_ffn @ x_moe.unsqueeze(-2)).squeeze(-2)

        # End of loop r — capture output for aux LM head
        if training and aux_loop_loss_weight > 0:
            x_end_of_loop_r = (A_l @ X.transpose(-1, -2)).squeeze(-1)
            per_loop_outputs.append(x_end_of_loop_r)

    # Step 4: Final output — extract from mHC residual stream
    x_final = (A_l @ X.transpose(-1, -2)).squeeze(-1)    # [B, L, 1536]

    # Step 5: LM head (tied with embedding)
    logits = x_final @ embedding.weight.T               # [B, L, 65536]

    return {
        'logits': logits,
        'per_loop_outputs': per_loop_outputs,           # for aux losses
        'kv_cache': kv_cache,
    }
```

The pseudocode is illustrative; the actual implementation should:
- Use efficient batched matrix ops
- Fuse RMSNorm + Linear where possible
- Use Flash Attention or similar for the attention block
- Cache mHC parameter generations across the recursion (they only
  depend on the residual state, which evolves)

---

## 11. Training losses

### 11.1 Main loss

Standard next-token cross-entropy on the final logits:
```
L_main = CrossEntropy(logits, targets, ignore_index=-100)
```

Label masking: -100 on the prefix (system + user prefix during SFT),
real token IDs on the assistant target.

### 11.2 Aux per-loop LM-head loss

For each intermediate loop output:
```
L_aux_loop_r = CrossEntropy(per_loop_logits_r, targets) × aux_loop_loss_weight
```

Total aux loss:
```
L_aux_total = Σ_{r=1}^{5} L_aux_loop_r
            # Note: loop 6 IS the main loss; we add r=1..5
```

`aux_loop_loss_weight = 0.05` during pretrain/MOPD/SFT.
`aux_loop_loss_weight = 0.03` during GRPO (preserve training but
don't dominate policy gradient).

### 11.3 MoE auxiliary balance loss (small)

Sequence-wise balance loss to prevent extreme intra-sequence
imbalance:
```
L_balance = α_balance × Σ_blocks Σ_experts f_i × p_i
# α_balance = 0.0001 (small; main balancing comes from b_route_bias)
```

### 11.4 MTP loss

```
L_MTP_1 = CrossEntropy(mtp_logits_1, targets_shifted_by_1) × β_mtp
L_MTP_2 = CrossEntropy(mtp_logits_2, targets_shifted_by_2) × β_mtp
# β_mtp = 0.3 most of training, decayed to 0.1 at LR decay
```

### 11.5 Router z-loss (insurance against logit blow-up)

```
L_z = mean(logsumexp(router_logits) ** 2) × γ_z
# γ_z = 0.001
```

### 11.6 Total training loss

```
L_total = L_main + L_aux_total + L_balance + L_MTP_1 + L_MTP_2 + L_z
```

### 11.7 Decoupled Top-K KD (during MOPD)

For knowledge distillation from teacher (LFM2 method):
```
L_DTK_per_token = KL(Bern(P_T(T)) || Bern(P_S(T)))                         # binary mass
                + P_T(T) × KL_τ(P_T(·|T) || P_S(·|T))                       # top-K conditional

# where T = teacher's top-K (K=32) token set, τ = temperature
# applied only to the conditional term
```

Replaces standard CE on teacher response during MOPD distillation.
Provides ~32× denser supervision per token.

---

## 12. Inference path

### 12.1 Generation modes

```python
def generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.95,
    loops=6,                      # adjustable: 3-6
    eos_token_id=1,
    stop_token_ids=[10, 14, 18],  # </answer>, end_turn, /tool_result
):
    # Prefill phase: full forward pass over the prompt
    output = forward(input_ids, kv_cache=None)
    kv_cache = output['kv_cache']

    # Speculative decoding via loop-3 draft head (optional)
    if speculative_decoding_enabled:
        return generate_speculative(input_ids, kv_cache, loops)

    # Standard autoregressive decode
    generated = []
    for step in range(max_new_tokens):
        next_logits = output['logits'][:, -1, :]
        next_logits = next_logits / temperature
        # top-p sampling
        probs = top_p_filter(softmax(next_logits), p=top_p)
        next_token = sample(probs)

        if next_token in stop_token_ids or next_token == eos_token_id:
            break

        generated.append(next_token)
        output = forward(next_token, kv_cache=kv_cache, loops=loops)

    return generated
```

### 12.2 Variable loop count (controllable inference compute)

`generate(loops=K)` runs only K of the 6 trained loops. Trained for
6, but the aux per-loop LM head training makes loops 3-5 also
produce coherent outputs. Quality vs speed trade-off:

| loops | speed | quality |
|---|---|---|
| 3 | 2× faster than 6 | ~85% of full quality |
| 4 | 1.5× faster | ~93% of full quality |
| 5 | 1.2× faster | ~98% of full quality |
| 6 | baseline | full quality |

### 12.3 Speculative decoding via loop-3 draft

```python
def generate_speculative(input_ids, kv_cache, K_draft=4):
    """
    Draft K tokens using loop-3 output of main model.
    Verify all K with single forward pass at loop-6.
    Commit accepted prefix.
    """
    # Draft K tokens cheaply
    drafts = []
    for k in range(K_draft):
        x_loop_3 = forward_partial(loops=3, cache=kv_cache)
        logits = embedding.weight @ x_loop_3.T
        draft_token = greedy(logits)
        drafts.append(draft_token)

    # Verify with full forward
    full_logits = forward(drafts, loops=6, cache=kv_cache)
    full_predictions = greedy(full_logits)

    # Accept matching prefix
    accept_prefix = []
    for k in range(K_draft):
        if drafts[k] == full_predictions[k]:
            accept_prefix.append(drafts[k])
        else:
            # Reject from here; emit the verifier's prediction for position k
            accept_prefix.append(full_predictions[k])
            break

    return accept_prefix
```

Expected accept rate: 60-75% per draft (the loop-3 head is trained to
predict the same thing the loop-6 head predicts via the aux loss).
Net speedup: ~1.8-2.4× on generation.

---

## 13. KV cache structure

### 13.1 Per-token cache contents

For OSRT-600M, **we cache only the K_DOWN (latent K) output**, not
full K or V:

```
cache_per_token_per_effective_layer = K_DOWN ∈ ℝ^512    # 8 KV heads × 64
```

V is recomputed at decode time via `V = W_V_FROM_K @ K + b_V`. This
halves the cache size vs caching both K and V.

### 13.2 Cache layout

```
kv_cache: dict
    keys: (loop_idx, block_idx) ∈ {0..5} × {0..2}
    values: tensor of shape [batch, seq, 512]

# Total cache entries: 18 effective layers × 512 floats per token
# Per token, BF16: 18 × 512 × 2 = 18,432 bytes = 18 KB
# At 4K context: 4096 × 18 KB = 72 MB raw
# At 8K context: 8192 × 18 KB = 144 MB raw
```

### 13.3 Compression stack at deployment

| step | reduction | size at 4K |
|---|---|---|
| Raw cache (BF16) | 1× | 72 MB |
| + V-from-K (cache K only) | 2× | 36 MB |
| + TurboQuant int4 | 4-8× | 4.5-9 MB |
| + Sliding window (if applicable) | 2-4× | 1-4 MB |

Final deployment cache: **< 5 MB at 4K context**.

### 13.4 Cache update during decode

After each new token, append to all 18 caches:
```
for r in range(6):
    for b in range(3):
        new_K = W_K_DOWN[b] @ new_x_in_loop_r_block_b
        kv_cache[(r, b)] = concat(kv_cache[(r, b)], new_K, axis=seq)
```

Each loop iteration computes FRESH K at that loop (the input differs
from loop r-1's output). No K sharing across loops — they're
genuinely different representations.

---

## 14. Quantization for deployment

### 14.1 Per-component quantization plan

| component | format | method |
|---|---|---|
| Embedding (tied) | int8 | symmetric per-channel QAT |
| Attention W_Q, W_K_DOWN, W_O | int8 | symmetric per-channel QAT |
| Shared experts | int8 | symmetric per-channel QAT |
| Routed experts | **FP4 (MXFP4)** | AlphaQ-allocated bit budget |
| HRA adapters | bf16 | kept full precision (small, sensitive) |
| Router projections | bf16 | kept full precision |
| mHC matrices | bf16 | kept full precision |
| Loop embeddings | bf16 | kept full precision |
| LayerNorms / biases | bf16 | always bf16 |
| K cache (per layer) | **int4** | TurboQuant random-rotation + per-block |

### 14.2 Memory budget at deployment

```
Embedding (int8):           100 MB
Attention (int8 × 3):        17 MB
Shared experts (int8 × 3):   64 MB → 16 MB int8
Routed experts (FP4):       319 MB → 80 MB FP4
HRA (bf16):                  86 MB → 172 MB bf16
Misc (bf16):                ~5 MB

TOTAL on disk:              ~390 MB (int8 mixed)
At inference (active):       ~150 MB (only active routed experts loaded)
```

### 14.3 AlphaQ bit allocation (routed experts)

Per AlphaQ:
- Compute PL Alpha Hill metric for each routed expert weight matrix
- ILP solver allocates bits ∈ {2, 3, 4} per layer per expert under
  global budget of 3.5 bits average
- Heavy-tailed experts (high importance) → 4 bits
- Light-tailed experts (less critical) → 2 bits
- Layer-wise allocation (each up/gate/down independently)

Expected quality: near-lossless at 3.5-bit average (per AlphaQ
results on Qwen1.5-MoE; our 12-experts-per-block is similar regime).

---

## 15. Total compute and memory math

### 15.1 Training compute

Per token, per forward pass: ~810M FLOPs (§2.3)
Backward is ~2× forward: ~1.6B FLOPs
Total per token per training step: ~2.4 BFLOPs

For 12B training tokens (the $280 budget):
```
total_train_flops = 12e9 × 2.4e9 = 2.88e19 FLOPs
on H100 (~989 TFLOPs/s effective for bf16):
total_time = 2.88e19 / 9.89e14 = ~29,000 seconds
              = ~8 hours
```

Plus overhead (data loading, gradient accumulation, etc): **~50
H100-hours** to pretrain 12B tokens. Matches the §12 cost estimate.

### 15.2 Inference compute per token (generation)

Just the forward pass: ~810M FLOPs
On H100 (consumer-equivalent at int8): ~200 GFLOPs effective
→ ~250K tokens/sec single-token decode (unrealistic — bandwidth bound)
→ Realistic on CPU (Snapdragon 8 Elite, int8): ~50-100 tokens/sec

### 15.3 Memory at inference (full deployment)

Weights: ~150-200 MB
KV cache (4K context, full stack): ~5 MB
Activations (transient): ~50 MB
Total: **~250 MB** — fits comfortably on phones / Raspberry Pi 5.

---

## 16. Architectural invariants

These are the design properties that MUST hold for the architecture
to function as designed. Violating any of these is a bug.

### 16.1 Recursion correctness

- `loop_embeddings.shape[0] >= 6` — must have a bias for each loop
- Recursive forward MUST apply the SAME 3 physical blocks 6 times
  (not 18 different block instances)
- `aux_loop_loss_weight > 0` during training keeps the recursion
  meaningful; if 0, training MUST monitor for loop collapse

### 16.2 mHC stability

- `B_l` MUST satisfy `||B_l||_2 ≤ 1` at every step (doubly stochastic)
- Sinkhorn-Knopp MUST converge within `t_max=20` iterations
- `A_l, C_l` MUST be non-negative (sigmoid-bounded)

### 16.3 Routing correctness

- Per training step, `Σ_i b_route_bias[i]` MUST remain bounded
- Block 0 and Block 1 MUST use hash routing (not learned)
- Aux-loss-free bias `b_route_bias` MUST NOT receive gradient
- `affinity = sqrt(softplus(W_route @ x))` — NEVER negative

### 16.4 Attention correctness

- QK-Norm MUST apply per-head, not flattened
- Partial RoPE applies to LAST 64 dims only
- V derivation MUST use `V = W_V_FROM_K @ K + b_V` (not from x)
- Attention sink logits added to denominator, not numerator

### 16.5 KV cache correctness

- Cache stores only K_DOWN (the latent), not V
- 18 separate cache entries per token (one per effective layer)
- Each loop's K is computed FRESH from that loop's input
- TurboQuant int4 applied to cached entries, not to the live forward
  pass

### 16.6 Tied LM head correctness

- `lm_head.weight = embedding.weight` (literal reference, not copy)
- Auxiliary per-loop heads use the SAME tied weight
- Logits computation: `x_final @ embedding.weight.T`

### 16.7 Gradient routing correctness

- Muon optimizer handles ALL 2D matrices in attention, experts, HRA,
  mHC W_pre/W_res/W_post
- AdamW handles: embedding, LM head (tied), RMSNorm gains, biases,
  router_bias accumulator
- Weight decay applied via decoupled scheme (Muon paper)
- `b_route_bias` NEVER in optimizer (heuristic update only)

---

## Document changelog

- **2026-06-07** — initial creation, captures complete OSRT-600M
  architecture spec as planned in README.md
