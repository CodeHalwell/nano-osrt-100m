# Nano-OSRT

**Omni-Sparse Recursive Titan** -- Recursive weight-sharing transformer language models trained from scratch on Modal serverless GPUs. Two versions: v3 (completed, 104.5M params) and v4 (in development, ~306M params with MoE).

---

## Models

### v3: Recursive Transformer (Complete)

104.5M physical parameters achieving 302M effective via recursive weight sharing. 2 physical blocks looped 6 times = 12 effective layers, each with unique per-pass residual adapters.

| Property | Value |
|----------|-------|
| Physical params | 104.5M (115.7M with HRA) |
| Effective params | ~302M |
| Architecture | 2 blocks x 6 loops = 12 effective layers |
| Hidden dim | 1280 |
| Attention heads | 20 (head_dim=64) |
| Tokenizer | EleutherAI/gpt-neox-20b (50K vocab) |
| Context length | 4096 (SFT/inference) |
| Training data | ~20B tokens |
| IFEval score | 26.7% (instruction-level strict) |

**Training pipeline:** Pretrain (150K steps) -> Math SFT (3K) -> GRPO (1.5K) -> Code SFT (7K)

### v4: Recursive MoE (In Development)

306M physical parameters with Mixture of Experts. 3 physical blocks x 6 loops = 18 effective layers. Dense FFN + MoE (1 shared + 11 routed experts, top-2) in parallel residual with capacity-capped routing.

| Property | Value |
|----------|-------|
| Physical params | ~306M |
| Active params/token (transformer body) | ~130M |
| Total compute (recursive x6 block applications) | ~1.8B |
| Architecture | 3 blocks x 6 loops = 18 effective layers |
| MoE | 12 experts (1 shared + 11 routed, top-2 capacity-capped) |
| Hidden dim | 1536 |
| Attention heads | 24 (head_dim=64) |
| Tokenizer | Custom 32K BPE (trained on 2GB of text + code + Wikipedia) |
| Context length | 8192 (progressive: 2048 -> 4096 -> 8192) |
| Target training data | ~90B tokens if full 300K-step schedule completes |
| RoPE scaling | NTK-aware for inference beyond training length |
| KV cache | O(1) per-token inference with incremental decoding |

**Key v4 features:**
- **Capacity-capped routing:** each expert has a hard per-batch token cap (capacity_factor=1.25), structurally preventing any expert from dominating. Tokens scan candidates in descending sigmoid score and are assigned to the first top-k experts with remaining capacity. Same path in training and inference.
- Loop-aware router with learned loop embeddings (additive, not concat -- halves router params)
- Row-normalized router init (equal expert norms at step 0)
- Learnable dense/MoE gates for parallel residual scaling (dense_gate=1.0, moe_gate=0.01 init)
- Soft warmup (steps 0-500) -> blend (500-1000) -> hard capacity-capped dispatch (1000+)
- Vectorised capacity assignment (sort + segment-cumsum, race-free)
- KV cache for O(1) per-token autoregressive generation
- Gradient checkpointing (auto-enabled during soft/blend phases and for seq_len >= 4096)
- Resilient HF streaming data loader (auto-reconnect on shard failures)
- Native single-token tags: `<|think|>`, `<|/think|>`, `<|answer|>`, `<|/answer|>`, `<|user|>`, `<|assistant|>`, `<|system|>`, FIM tokens
- HuggingFace `PreTrainedModel` compatible from day one

---

## Architecture

### v3: Recursive Transformer

2 physical blocks are looped 6 times, producing 12 effective layers. Each virtual layer gets a unique per-pass residual adapter to prevent representational collapse.

```
          Input Token IDs (B, S)
                   |
          +--------v--------+
          |  Token Embedding |  (50304 x 1280, weight-tied with LM head)
          +---------+-------+
                    |
                    |     +-------------------------------------------+
                    |     |  RoPE Buffers (precomputed, non-persistent) |
                    |     |  cos, sin: (1, seq_len, 1, head_dim=64)    |
                    |     +---------------------+---------------------+
                    |                           |
   +================v===========================v======================+
   ||                 Recursive Loop (x6)                              ||
   ||                                                                  ||
   ||  +------------------------------------------------------------+  ||
   ||  |  Block 0 (physical)                                        |  ||
   ||  |                                                            |  ||
   ||  |  x_mod = x + scale * (x @ adapter_a[i] @ adapter_b[i])    |  ||
   ||  |                    ^-- unique per (block, loop) pair       |  ||
   ||  |                                                            |  ||
   ||  |  +-- RMSNorm --> QKV Proj --> RoPE --> Causal SDPA --+     |  ||
   ||  |  |                        (FlashAttention-2 backend)  |     |  ||
   ||  |  +-- Out Proj + Residual (connects to x_mod) --------+     |  ||
   ||  |                                                            |  ||
   ||  |  +-- RMSNorm --> SwiGLU FFN (dim=1280, hidden=3456) -+     |  ||
   ||  |  +-- Residual ----------------------------------------+     |  ||
   ||  +------------------------------+-----------------------------+  ||
   ||                                 |                                ||
   ||  +------------------------------v-----------------------------+  ||
   ||  |  Block 1 (physical)                                        |  ||
   ||  |  (same structure as Block 0, different adapter pair)       |  ||
   ||  +------------------------------+-----------------------------+  ||
   ||                                 |                                ||
   ||                          Loop RMS measurement                    ||
   ||                          Inter-loop RMSNorm (loops 0-4)         ||
   +==================================+===============================+
                                      |  (x6 = 12 effective layers)
                    +-----------------v-----------------+
                    |            RMSNorm (final)         |
                    +-----------------+-----------------+
                                      |
                    +-----------------v-----------------+
                    |   LM Head (weight-tied embedding)  |
                    +-----------------+-----------------+
                                      |
                              Logits (B, S, vocab)
```

**Parameter budget (v3):**

```
  Token Embedding    64.4M  [=====================================]  61.6%
  SwiGLU FFN x2     26.5M  [===============]                        25.4%
  QKV Proj x2        9.8M  [=====]                                   9.4%
  Out Proj x2        3.3M  [==]                                      3.1%
  Adapters x12       0.5M  []                                        0.5%
  ─────────────────────────────────────────────────────────────────
  Total            104.5M   Physical params
                   302M     Effective (recursive x6)
```

### v4: Recursive MoE

3 physical blocks x 6 loops = 18 effective layers. Each block has causal attention + parallel dense FFN and MoE FFN with learnable gating.

```
          Input Token IDs (B, S)
                   |
          +--------v--------+
          |  Token Embedding |  (32768 x 1536, weight-tied with LM head)
          +---------+-------+
                    |
                    |     +-------------------------------------------+
                    |     |  RoPE Buffers (max 8192 positions)         |
                    |     |  NTK-aware scaling for inference >8K      |
                    |     +---------------------+---------------------+
                    |                           |
   +================v===========================v======================+
   ||                 Recursive Loop (x6)                              ||
   ||                                                                  ||
   ||  For each of 3 physical blocks:                                  ||
   ||                                                                  ||
   ||  +------------------------------------------------------------+  ||
   ||  |  RecursiveBlockV4                                          |  ||
   ||  |                                                            |  ||
   ||  |  adapter_out = scale * (x @ adapter_a[i] @ adapter_b[i])  |  ||
   ||  |                                                            |  ||
   ||  |  +== ATTENTION ========================================+   |  ||
   ||  |  | RMSNorm -> QKV (1536->4608) -> RoPE -> Causal SDPA |   |  ||
   ||  |  | -> Out Proj + adapter_out + Residual (parallel)     |   |  ||
   ||  |  +=====================================================+   |  ||
   ||  |                          |                                 |  ||
   ||  |           +--------------+---------------+                 |  ||
   ||  |           |                              |                 |  ||
   ||  |  +========v==========+  +================v==============+  |  ||
   ||  |  | DENSE FFN         |  | MoE FFN                      |  |  ||
   ||  |  |                   |  |                               |  |  ||
   ||  |  | RMSNorm           |  | RMSNorm                      |  |  ||
   ||  |  | SwiGLU            |  |     +---------------------+  |  |  ||
   ||  |  | (1536->4096->1536)|  |     | Loop-Aware Router    |  |  |  ||
   ||  |  |                   |  |     | h + loop_emb -> sig  |  |  |  ||
   ||  |  |                   |  |     +----------+----------+  |  |  ||
   ||  |  |                   |  |                |             |  |  ||
   ||  |  |                   |  |    capacity-capped top-2     |  |  ||
   ||  |  |                   |  |    (scan candidates, cap     |  |  ||
   ||  |  |                   |  |     per-expert token limit)  |  |  ||
   ||  |  |                   |  |        /       |       \     |  |  ||
   ||  |  |                   |  |  +--------+ +-----+ +-----+ |  |  ||
   ||  |  |                   |  |  | Shared | | E_i | | E_j | |  |  ||
   ||  |  |                   |  |  | Expert | | (r) | | (r) | |  |  ||
   ||  |  |                   |  |  | (1024) | |(1024)| |(1024)||  |  ||
   ||  |  |                   |  |  +---+----+ +--+--+ +--+--+ |  |  ||
   ||  |  |                   |  |      |         |       |     |  |  ||
   ||  |  |                   |  |      +----+----+-------+     |  |  ||
   ||  |  |                   |  |           | weighted sum      |  |  ||
   ||  |  +=========+=========+  +===========+=================+  |  ||
   ||  |            |                        |                     |  ||
   ||  |            |    gate_d              |    gate_m           |  ||
   ||  |            +--------+    +----------+                     |  ||
   ||  |                     |    |                                |  ||
   ||  |           x = x + gate_d * dense + gate_m * moe          |  ||
   ||  |                                                            |  ||
   ||  +------------------------------------------------------------+  ||
   ||                                                                  ||
   ||  (repeated for all 3 blocks per loop)                            ||
   ||                                                                  ||
   ||  Loop RMS measurement + Inter-loop RMSNorm                       ||
   +===================================================================+
                                      |  (x6 = 18 effective layers)
                    +-----------------v-----------------+
                    |            RMSNorm (final)         |
                    +-----------------+-----------------+
                                      |
                    +-----------------v-----------------+
                    |   LM Head (weight-tied embedding)  |
                    +-----------------+-----------------+
                                      |
                              Logits (B, S, vocab)
```

**MoE routing detail (capacity-capped):**

```
  Hidden state (B, S, 1536)
         |
         +---> add loop_embedding[loop_idx]  --> (B, S, 1536)  [additive, not concat]
         |
         +---> Router linear (1536 -> 11)  --> sigmoid scores
         |     (row-normalised init: all expert rows same norm)
         |
         +---> Capacity-capped candidate scan:
         |     1. Sort tokens' top-11 candidates by sigmoid score
         |     2. For each candidate rank, assign tokens to experts
         |        that still have capacity (vectorised sort + cumcount)
         |     3. Per-expert cap = ceil(1.25 * N * 2 / 11)
         |     4. Each token gets exactly 2 experts (overflow ~0%)
         |     5. Weights from clean sigmoid, renormalised
         |
         |     Shared expert: SwiGLU(1536 -> 1024 -> 1536)  [always active]
         |     Routed expert IDs 0-10: SwiGLU(1536 -> 1024 -> 1536)
         |       [2 selected per token]
         |
         +---> output = shared_out + weighted_sum(capped_expert_outputs)
```

The capacity cap is a structural guarantee -- no expert can receive more than
~11.4% of assignments regardless of the router's preferences. This solves the
deterministic top-k winner lock-in problem that caused routing collapse in all
soft-regularisation approaches (z-loss, Gumbel noise, proportional bias).
The router is free to learn token-dependent preferences; the cap just prevents
those preferences from producing imbalanced load.

**Parameter budget (v4, 32K vocab):**

```
  MoE Routed (11 experts) 156M  [================================]      50.9%
  Dense FFN x3             57M  [============]                          18.5%
  Token Embedding          50M  [==========]                            16.4%
  Attention x3             28M  [======]                                 9.3%
  Shared Expert x3         14M  [===]                                    4.6%
  Adapters x18              1M  []                                       0.3%
  Router + Loop Emb       <1M  []                                        0.0%
  ─────────────────────────────────────────────────────────────────
  Total Physical         ~306M
  Active transformer body ~130M   per-token compute (excl. embedding lookup)
  Block applications      6x     via recursive weight sharing
```

Note on "effective params": recursion gives 18 block applications, not 18
independent sets of weights. The model has 306M **physical** parameters;
each forward runs the body ~6 times with unique per-pass adapters and
loop-conditioned routing, so the compute budget is comparable to a model
of roughly ~1.8B FLOPs per token — but with much tighter memorisation
capacity than a dense 1.8B model. Treat this as iterative refinement on
a budget, not a 1:1 parameter substitute.

### Mathematical Formulation

#### Notation

Let $B$ denote batch size, $S$ sequence length, $d$ hidden dimension (1536), $H$ number of attention heads (24), $d_h$ head dimension (64), $N_b$ number of physical blocks (3), $L$ number of recursive loops (6), $E$ total experts (12), $E_s$ shared experts (1), $E_r$ routed experts (11), and $k$ experts selected per token (2).

#### 1. Recursive Forward Pass

The model applies $N_b$ physical blocks $L$ times each, yielding $N_b \times L = 18$ effective layers from 3 sets of weights. For loop $\ell \in \{0, \ldots, L-1\}$ and block $b \in \{0, \ldots, N_b-1\}$:

$$\mathbf{x}^{(\ell, b)} = \text{Block}_b\!\left(\mathbf{x}^{(\ell, b-1)},\; \mathbf{A}_{\ell N_b + b},\; \mathbf{B}_{\ell N_b + b},\; \ell\right)$$

where $\mathbf{x}^{(\ell, 0)} = \text{RMSNorm}(\mathbf{x}^{(\ell-1, N_b-1)})$ for $\ell > 0$ (inter-loop normalisation), and $\mathbf{x}^{(0,0)} = \text{Embed}(\text{input\_ids})$. Each block reuses the same parameters $\text{Block}_b$ across all $L$ loops; only the adapter pairs $(\mathbf{A}_i, \mathbf{B}_i)$ and the loop embedding index $\ell$ vary.

#### 2. Per-Pass Residual Adapters

Each of the $N_b \times L = 18$ effective layers has a unique adapter pair $(\mathbf{A}_i, \mathbf{B}_i)$ with $\mathbf{A}_i \in \mathbb{R}^{d \times r}$, $\mathbf{B}_i \in \mathbb{R}^{r \times d}$ (rank $r = 16$). Unlike weight-space LoRA (Hu et al., 2021), these operate in activation space as a parallel residual branch:

$$\mathbf{x}_{\text{adapted}} = \mathbf{x} + \frac{\alpha}{r}\left(\mathbf{x}\,\mathbf{A}_i\,\mathbf{B}_i\right)$$

where $\alpha = 16$, $r = 16$, giving scale $= 1.0$. At initialisation $\mathbf{B}_i = \mathbf{0}$, so the adapter is a no-op and all loops begin identically. Differentiation emerges through gradient flow during training.

#### 3. Attention with RoPE

Within each block, causal multi-head attention with Rotary Position Embeddings (Su et al., 2021):

$$\mathbf{q}, \mathbf{k}, \mathbf{v} = \text{split}\!\left(\mathbf{W}_{qkv}\,\text{RMSNorm}(\mathbf{x})\right)$$

$$\mathbf{q}' = \text{RoPE}(\mathbf{q}, \cos, \sin), \quad \mathbf{k}' = \text{RoPE}(\mathbf{k}, \cos, \sin)$$

$$\text{Attn}(\mathbf{q}', \mathbf{k}', \mathbf{v}) = \text{softmax}\!\left(\frac{\mathbf{q}'\mathbf{k}'^{\!\top}}{\sqrt{d_h}} + \mathbf{M}_{\text{causal}}\right)\mathbf{v}$$

$$\mathbf{x} \leftarrow \mathbf{x} + \mathbf{W}_o\,\text{Attn}(\mathbf{q}', \mathbf{k}', \mathbf{v}) + \mathbf{x}_{\text{adapted}}$$

where the adapter output is added as a parallel branch alongside the attention output. Optional NTK-aware RoPE scaling (Bloc97, 2023) extends context beyond training length by rescaling $\theta_{\text{eff}} = \theta \cdot f^{d/(d-2)}$ for extension factor $f$.

#### 4. Parallel Dense + MoE FFN

The FFN stage runs a dense SwiGLU network and a Mixture of Experts network in parallel, with learnable scalar gates:

$$\mathbf{h}_{\text{dense}} = \text{SwiGLU}_{\text{dense}}(\text{RMSNorm}(\mathbf{x}))$$

$$\mathbf{h}_{\text{moe}} = \text{MoE}(\text{RMSNorm}(\mathbf{x}), \ell)$$

$$\mathbf{x} \leftarrow \mathbf{x} + g_d \cdot \mathbf{h}_{\text{dense}} + g_m \cdot \mathbf{h}_{\text{moe}}$$

where $g_d, g_m$ are learnable scalars initialised to $g_d = 1.0$, $g_m = 0.01$. This "dense-first" initialisation means the model starts as a nearly pure dense network and gradually incorporates MoE output as the router learns meaningful routing.

Each SwiGLU network computes $\text{SwiGLU}(\mathbf{x}) = \mathbf{W}_{\text{down}}(\text{SiLU}(\mathbf{W}_{\text{gate}}\mathbf{x}) \odot \mathbf{W}_{\text{up}}\mathbf{x})$.

#### 5. Loop-Aware Routing

The router produces per-expert scores conditioned on both the hidden state and the current recursive loop index. A learned embedding $\mathbf{e}_\ell \in \mathbb{R}^d$ is added (not concatenated) to the hidden state before projection:

$$\mathbf{s}_{b,t} = \sigma\!\left(\mathbf{W}_R\,(\mathbf{h}_t + \mathbf{e}_\ell)\right) \in \mathbb{R}^{E_r}$$

where $\sigma$ is the element-wise sigmoid function, $\mathbf{W}_R \in \mathbb{R}^{E_r \times d}$ is the router weight matrix, and $\mathbf{h}_t$ is the normalised hidden state for token $t$. Addition (not concatenation) keeps router parameters at $E_r \times d$ rather than $E_r \times 2d$, since $\mathbf{W}_R[\mathbf{h}; \mathbf{e}] = \mathbf{W}_R^{(1)}\mathbf{h} + \mathbf{W}_R^{(2)}\mathbf{e}$ is equivalent under a linear projection.

**Router initialisation.** $\mathbf{W}_R$ is initialised from $\mathcal{N}(0, 0.02)$ then row-normalised so every expert row has equal norm. This prevents any expert from starting with a larger projection magnitude and winning deterministic top-$k$ tie-breaking at step 0. Loop embeddings are initialised from $\mathcal{N}(0, 0.1)$ (5x the default $0.02$) so that routing is loop-dependent from the first step.

#### 6. Capacity-Capped Expert Selection

Standard deterministic top-$k$ selection suffers from self-reinforcing winner lock-in: small initial logit differences cause the same experts to be selected repeatedly, strengthening their weights through gradient, which increases their logit advantage, creating a positive feedback loop that collapses routing to $k$ out of $E_r$ experts within 30-50 training steps. We verified this failure empirically across six approaches: raw top-$k$, additive Gaussian noise, soft all-expert warmup, Gumbel-top-$k$, proportional bias controller, and importance-weighted regularisation. All failed to prevent collapse.

We adopt **capacity-capped candidate routing**, which provides a structural guarantee on load balance. Define the per-expert capacity:

$$C = \left\lceil \gamma \cdot \frac{N \cdot k}{E_r} \right\rceil$$

where $N = B \times S$ is the total number of tokens in the batch, $k = 2$ is the number of experts per token, and $\gamma = 1.25$ is the capacity factor providing 25% slack above the uniform allocation $N \cdot k / E_r$.

**Algorithm.** For each token, we scan its $E_r$ candidate experts in descending order of sigmoid score $\mathbf{s}_{b,t}$ and assign the first $k$ experts that have remaining capacity. Let $\pi_t$ be the permutation sorting token $t$'s scores in descending order, and let $n_j$ be the running count of tokens assigned to expert $j$. Token $t$ is assigned expert $\pi_t(r)$ at candidate rank $r$ if and only if:

$$n_{\pi_t(r)} < C \quad \text{and} \quad |\{j : t \to j\}| < k$$

After assignment, the dispatch weights for token $t$'s selected experts $\{j_1, j_2\}$ are:

$$w_{t,j} = \frac{\sigma((\mathbf{W}_R(\mathbf{h}_t + \mathbf{e}_\ell))_j)}{\sum_{j' \in \{j_1, j_2\}} \sigma((\mathbf{W}_R(\mathbf{h}_t + \mathbf{e}_\ell))_{j'})}$$

Note that the weights come from the clean sigmoid scores, not from the capacity-assignment order. This preserves meaningful gradient flow to the router: the selected experts receive gradient proportional to their sigmoid score, incentivising the router to produce high scores for experts that reduce loss.

**Vectorised implementation.** To avoid a sequential loop over tokens, we process all tokens in parallel per candidate rank using a sort-and-segment-cumsum approach:

1. For candidate rank $r$: gather each eligible token's desired expert ID.
2. Sort tokens by expert ID (stable sort preserves token order within groups).
3. Compute within-group position via segment cumsum: detect group boundaries where sorted expert IDs change, compute cumulative count within each group.
4. Token at within-group position $p$ fits if $p < C - n_j$ (remaining capacity for expert $j$).
5. Unsort the fit mask back to original token order and apply assignments.
6. Update per-expert counts via `scatter_add_`.

This runs in $O(K_c \cdot M \log M)$ where $K_c$ is the candidate pool size (default $K_c = E_r = 11$) and $M$ is the number of eligible tokens, shrinking each iteration as tokens fill their $k$ slots. It avoids token-by-token Python assignment while preserving the same deterministic choices as the reference loop.

**Structural guarantees.** With $\gamma = 1.25$ and candidate pool size $= E_r$ (all experts), the capacity cap ensures:

$$\max_j \frac{n_j}{N \cdot k} \leq \frac{C}{N \cdot k} = \frac{\gamma}{E_r} \approx 0.114$$

This bound holds by construction regardless of the router's learned preferences. The overflow rate (fraction of tokens receiving fewer than $k$ experts) is empirically $0.000$ with $\gamma = 1.25$ and full candidate pool.

#### 7. MoE Output

The MoE layer output combines the shared expert (always active) with the capacity-capped routed output:

$$\text{MoE}(\mathbf{h}, \ell) = \text{FFN}_{\text{shared}}(\mathbf{h}) + \sum_{j \in \text{capped}(t)} w_{t,j} \cdot \text{FFN}_j(\mathbf{h}_t)$$

where $\text{capped}(t)$ denotes the set of (at most $k$) experts assigned to token $t$ by the capacity-capped algorithm.

#### 8. Training Loss

The training objective combines cross-entropy with a differentiable importance balance regulariser:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{imp}} \cdot \frac{1}{N_b \cdot L} \sum_{\ell, b} \mathcal{L}_{\text{imp}}^{(\ell, b)}$$

where the importance loss for a single MoE application is:

$$\mathcal{L}_{\text{imp}} = E_r \cdot \sum_{j=1}^{E_r} \left(\bar{p}_j\right)^2, \quad \bar{p}_j = \frac{1}{N} \sum_{t=1}^{N} \text{softmax}\!\left(\frac{\mathbf{W}_R(\mathbf{h}_t + \mathbf{e}_\ell)}{\tau}\right)_j$$

This is minimised at $\mathcal{L}_{\text{imp}} = 1.0$ when the softmax marginal is uniform ($\bar{p}_j = 1/E_r$ for all $j$), and grows quadratically with concentration. The default coefficient is $\lambda_{\text{imp}} = 0.05$ with temperature $\tau = 1.0$. Unlike the capacity cap (which constrains hard assignments), the importance loss provides differentiable gradient pressure on the soft probability distribution, encouraging the router to maintain balanced preferences even though the cap prevents imbalanced outcomes.

#### 9. Routing Schedule

Training proceeds through three routing phases:

| Phase | Steps | Routing | Grad Checkpointing |
|---|---|---|---|
| Soft warmup | $[0, 500)$ | All $E_r$ experts run on every token, weighted by $\text{softmax}(\mathbf{s}/\tau)$ | On |
| Blend | $[500, 1000)$ | $\alpha \cdot \text{capped\_hard} + (1-\alpha) \cdot \text{soft}$, $\alpha$ linear $0 \to 1$ | On |
| Hard | $[1000, \infty)$ | Capacity-capped top-$k$ only | Off (seq < 4096) |

The soft warmup ensures every expert receives gradient from step 0, preventing dead experts before the router has learned any token-dependent preferences. The blend phase provides a smooth transition so the model's loss trajectory is not disrupted by the switch from dense to sparse expert selection. In the hard phase, only the $k$ capacity-capped experts run per token, reducing compute to $\sim$$130$M active parameters per token.

### Per-Pass Residual Adapters

Not weight-LoRA (Hu et al.) -- modulates hidden states directly:

```
                      +---------------------------------------------+
                      |        Per-Pass Residual Adapter              |
                      |                                               |
   x ────────+────────v────────+                                      |
             |     x @ A       |   A: (dim, rank) -- N(0, 0.01) init  |
             |       |         |   B: (rank, dim) -- zero init        |
             |     x @ A @ B   |   scale = alpha / rank = 1.0        |
             |       |         |                                      |
             |   scale * (...) |                                      |
             |       |         |                                      |
             +-------+---------+                                      |
             |                                                        |
   x_mod  <--+  x_mod = x + scale * (x @ A @ B)                      |
                      |                                               |
                      |  v3: 12 pairs (rank 16), v4: 18 pairs         |
                      +---------------------------------------------+

  At step 0, B=0 so adapter is no-op. All loops start identical.
  Differentiation emerges organically through gradient flow.
```

### High Rank Adaptation (HRA)

Post-training capacity expansion. Injected alongside each linear layer after loading pretrained weights:

```
                  +------------------------------------------+
                  |        HRA Linear Wrapper                 |
                  |                                           |
   x ─────+──────v──────+                                    |
           |  Original   |                                    |
           |  Linear(x)  |   W: (in, out) -- pretrained       |
           |      |       |                                    |
           +------+       |   A: (in, 256) -- Kaiming init    |
           |              |   B: (256, out) -- zero init       |
           |  (x @ A @ B) |   scale = 1.0                     |
           |      |       |                                    |
           +------+-------+                                    |
           |                                                   |
   y  <----+  y = Linear(x) + scale * (x @ A @ B)             |
                  |                                           |
                  |  +11.2M params (v3) / +15-20M (v4)        |
                  +------------------------------------------+

  Differential LR: pretrained weights at 2e-5, HRA at 1e-4 (5x)
```

### v4 Chat Format (Native Token Tags)

Each tag is a single token in the v4 tokenizer -- no multi-token string matching:

```
  +---------------------------------------------------------------------+
  |  <|begin_of_text|>                                                  |
  |  <|system|> You are a helpful coding assistant.                     |
  |  <|user|> Write a function to check if a number is prime.          |
  |  <|assistant|>                                                      |
  |  <|think|>                                                          |
  |  I need to check divisibility from 2 to sqrt(n).                   |
  |  For each potential divisor, if n is evenly divisible, it's not     |
  |  prime. Otherwise, after checking all divisors, it is prime.        |
  |  <|/think|>                                                         |
  |  <|answer|>                                                         |
  |  def is_prime(n):                                                   |
  |      if n < 2:                                                      |
  |          return False                                               |
  |      for i in range(2, int(n**0.5) + 1):                           |
  |          if n % i == 0:                                             |
  |              return False                                           |
  |      return True                                                    |
  |  <|/answer|>                                                        |
  |  <|end_of_text|>                                                    |
  +---------------------------------------------------------------------+

  Loss masking (SFT):
    IGNORE: <|begin_of_text|> ... <|assistant|>  (system + user prompt)
    TRAIN:  <|think|> ... <|/answer|>            (reasoning + answer)
    IGNORE: <|end_of_text|> padding              (EOS + pad)
```

### Progressive Context Length (v4)

```
  seq_len
    8192 |                                              +============+
         |                                              |  Phase 3   |
    4096 |                     +========================+  Instruct  |
         |                     |       Phase 2          |   8192     |
    2048 |  +=================+   Knowledge             |            |
         |  |    Phase 1      |     4096                |            |
         |  |   Foundation    |                         |            |
         |  |     2048        |                         |            |
    ─────+--+-----------------+-------------------------+------------+---> steps
         0              10K                        250K          300K
```

### Training Pipeline Overview

```
  +===========+     +==========+     +========+     +==========+     +======+
  | Tokenizer |---->| Pretrain |---->|  SFT   |---->|  GRPO    |---->| Eval |
  | (32K BPE) |     | (300K    |     | (5K    |     | (2K      |     |      |
  |  2GB      |     |  steps)  |     |  steps)|     |  steps)  |     |      |
  +===========+     +==========+     +========+     +==========+     +======+
       |                 |               |               |               |
    Custom          Progressive      Balanced        Verifiable      IFEval
    vocab           2048->8192       math+code       math rewards    GSM8K
    code+text       Lion optimizer   +STEM+general   group_size=16   HumanEval
    +wiki           ~90B tokens      HRA adapters    KL penalty      HellaSwag
```

---

## Training Pipeline

### v3 Pipeline (Complete)

```
Stage 1: Pretrain     -> 150K steps, Lion optimizer, seq_len 2048
  TinyStories (8K) -> FineWeb-Edu (132K) -> SmolTalk (10K)

Stage 2: Math SFT     -> 3K steps, AdamW + HRA, seq_len 4096
  GSM8K + Orca-Math + NuminaMath-CoT + MathInstruct + LongForm

Stage 3: GRPO         -> 1.5K steps (accuracy 0-6%, model too small)
  GSM8K prompts, verifiable math rewards

Stage 4: Code SFT     -> 7K steps (2 epochs), AdamW + HRA, seq_len 4096
  Evol-Instruct-Code + CodeAlpaca + Python-instructions + GSM8K + LongForm
```

### v4 Pipeline (Planned)

```
Step 0: Tokenizer     -> Train custom 32K BPE on 2GB text+code+wiki
Step 1: Pretrain      -> 300K steps, progressive seq_len 2048->4096->8192
  Foundation: TinyStories + CodeParrot (10K steps)
  Knowledge:  FineWeb-Edu + CodeParrot + Wikipedia (240K steps)
  Instruction: SmolTalk + Evol-Code + OpenHermes (50K steps)

Step 2: Balanced SFT  -> 5K steps, math + code + STEM + general
Step 3: GRPO          -> 2K steps (retry with larger model)
Step 4: Eval          -> IFEval, GSM8K, HellaSwag, HumanEval, ARC, MMLU
```

---

## Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Modal](https://modal.com/) account for cloud training

### v3 Inference (Trained Model)

```bash
# Export checkpoint to HF format
uv run python export_model.py

# Single prompt
uv run python inference.py --model ./nano-osrt-model --prompt "Write a Python function to reverse a string"

# Interactive chat
uv run python inference.py --model ./nano-osrt-model --interactive

# Adjust generation
uv run python inference.py --model ./nano-osrt-model --prompt "..." --temperature 0.2 --repetition-penalty 1.3
```

### v3 Training

```bash
# Pre-training
uv run modal run --detach app.py --stage pretrain

# Math SFT
uv run modal run --detach app.py --stage sft

# GRPO
uv run modal run --detach app.py --stage grpo

# Code SFT
uv run modal run --detach app.py --stage code

# Benchmarks
uv run modal run app.py --stage eval
```

### v4 Training

```bash
# Train custom 32K tokenizer on H100 (2GB sample)
uv run modal run --detach app_v4.py --stage tokenizer

# Pre-training (progressive seq_len)
uv run modal run --detach app_v4.py --stage pretrain

# Balanced SFT
uv run modal run --detach app_v4.py --stage sft

# GRPO
uv run modal run --detach app_v4.py --stage grpo

# Benchmarks
uv run modal run app_v4.py --stage eval
```

**Recommended first run: sanity check, not full send.** Launch the pretrain
stage, watch the first ~100-500 steps in W&B, and confirm the following
signals before committing more GPU hours:

| What to watch | Healthy signal |
|---|---|
| `train/loss` | Drops from ~10.7 (ln 32768) toward ~7 by step 100 |
| `train/tok_per_sec` | 12-15K in soft phase, 18-20K in hard phase (H100) |
| `moe/clean_expert_max_mean` | Pinned at ~0.114 (= capacity cap bound) |
| `moe/overflow_rate_mean` | 0.000 (every token gets 2 experts) |
| `moe/assigned_per_token_mean` | 2.00 |
| `moe/candidate_rank_mean` | < 4.0 (tokens finding capacity without going deep) |
| `moe/moe_gate_b{0,1,2}` | Starts at 0.01, climbs steadily (MoE becoming useful) |
| `moe/dense_gate_b{0,1,2}` | Starts at 1.0, gradually decreases (shifting to MoE) |
| `moe/raw_assign_entropy_mean` | Diagnostic only -- may collapse, that's OK |
| `eval/loss` | Slowly improving on the held-out FineWeb-Edu stream |

The capacity cap makes routing collapse structurally impossible -- `expert_max`
cannot exceed `ceil(capacity_factor * N * top_k / num_routed) / (N * top_k)`.
If you see `overflow_rate > 0.01` or `assigned_per_token < 1.9`, increase
`router_candidate_k` (default 11 = num_routed) or `router_capacity_factor`.

---

## Project Structure

```
nano-osrt-100m/
├── README.md
├── app.py                          # v3 Modal entrypoint
├── app_v4.py                       # v4 Modal entrypoint
├── inference.py                    # v3 inference script
├── export_model.py                 # Export checkpoint to HF format
├── eval_model.py                   # Local evaluation harness
│
├── src/nano_osrt/
│   ├── # v3 (complete)
│   ├── recursive_model.py          # v3 architecture (2 blocks x 6 loops)
│   ├── modal_config.py             # v3 pre-training config
│   ├── modal_train.py              # v3 pre-training loop
│   ├── modal_data.py               # v3 streaming data pipeline
│   ├── sft_config.py               # v3 SFT + Code SFT configs
│   ├── sft_train.py                # v3 SFT training loop
│   ├── sft_data.py                 # v3 SFT data with format functions
│   ├── grpo_config.py              # v3 GRPO config
│   ├── grpo_train.py               # v3 GRPO training loop
│   ├── rewards.py                  # Verifiable reward functions
│   ├── hra.py                      # High Rank Adaptation
│   ├── hf_model.py                 # v3 HuggingFace wrapper
│   ├── rope.py                     # Rotary Position Embeddings
│   │
│   ├── # v4 (in development)
│   ├── v4_config.py                # HF PretrainedConfig (MoE + native tags)
│   ├── v4_model.py                 # Recursive MoE architecture
│   ├── v4_data.py                  # Pre-training data (progressive seq_len)
│   ├── v4_train.py                 # Pre-training loop (phase transitions)
│   ├── v4_train_config.py          # Pretrain + SFT + GRPO configs
│   ├── v4_sft_data.py              # SFT data with native token tags
│   └── v4_sft_train.py             # SFT training loop
│
├── scripts/
│   └── train_tokenizer.py          # Custom 32K BPE tokenizer training
│
├── docs/
│   ├── training-report.md          # v3 end-to-end training report
│   ├── dataset-reference.md        # Curated dataset catalog
│   └── v4-architecture-plan.md     # v4 design document
│
├── checkpoints/                    # Downloaded model checkpoints
└── tests/                          # Model and utility tests
```

---

## Benchmarks (v3)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| IFEval (instruction-level strict) | 26.7% | Gemma 270M: 51.2%, SmolLM2 135M: 38% |

Additional benchmarks (GSM8K, HellaSwag) pending.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Recursive weight sharing | 6x parameter compression on transformer body; reasoning depth without knowledge capacity cost |
| Per-pass residual adapters | Prevents representational collapse in recursive loops; each virtual layer gets unique identity |
| HRA (not LoRA) for post-training | Adds substantial capacity (11-20M params) rather than parameter-efficient fine-tuning |
| Lion optimizer (pre-training) | Halves optimizer VRAM; competitive perplexity at this scale |
| MoE alongside dense FFN (v4) | Dense path maintains baseline quality; MoE adds specialist capacity |
| Capacity-capped routing (v4) | Structural load balance guarantee; deterministic top-k collapses without it (confirmed across 6+ sanity runs with soft warmup, Gumbel noise, and proportional bias -- all failed) |
| Loop-aware routing (v4) | Additive loop embeddings; router learns to dispatch differently at each recursive pass |
| Soft warmup schedule (v4) | Steps 0-500 use all-expert soft dispatch so every expert gets gradient before hard selection starts |
| Custom tokenizer (v4) | Optimized for code+text distribution; native single-token tags |
| Progressive seq_len (v4) | Start short (fast), extend long (capability); RoPE naturally supports this |

---

## Observability

Training logs to Weights & Biases with per-stage metrics:

- **Pre-training:** loss, lr, vram, tok/s, phase, loop RMS, adapter similarity
- **MoE routing (v4):** capped/raw assignment entropy, expert max/min fractions, overflow rate, assigned experts per token, candidate rank mean, gate values, importance loss -- all per-block per-loop
- **SFT:** loss, lr, vram, token utilization
- **GRPO:** loss, mean reward, accuracy, KL divergence

W&B project: [nano-osrt-100m](https://wandb.ai/codhe-synextra/nano-osrt-100m) | [nano-osrt-v4](https://wandb.ai/codhe-synextra/nano-osrt-v4)

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v4.0** (dev) | 3 blocks, MoE (12 experts, capacity-capped top-2), 32K custom tokenizer, parallel adapter residual, dense-first MoE gating, additive loop-aware router with row-norm init, soft warmup + blend + hard routing schedule, vectorised capacity-capped dispatch, KV cache, resilient HF data streaming, progressive seq_len, native tags, HF-native |
| **v3.3** | Code SFT (7K steps, 2 epochs), HRA adapters (+11.2M params), HF inference wrapper, IFEval benchmark |
| **v3.2** | Post-training pipeline: Math SFT + GRPO + improved reward functions |
| **v3.1** | RoPE, FP32 master weights, dynamic vocab, SmolTalk formatting |
| **v3.0** | Renamed adapters, gradient accumulation, dataset shuffle |
| **v2.0** | Scaled 46M -> 104.5M, inter/intra-block telemetry |
| **v1.0** | Initial: 1 block x 6 loops, 46M params, Lion optimizer |

---

## License

MIT. See [pyproject.toml](pyproject.toml) for details.
