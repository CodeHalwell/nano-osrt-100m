# Nano-OSRT

**Omni-Sparse Recursive Titan** -- Recursive weight-sharing transformer language models trained from scratch on Modal serverless GPUs. Two versions: v3 (completed, 104.5M params) and v4 (in development, 208M+ params with MoE).

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

208M+ physical parameters with Mixture of Experts. 3 physical blocks x 6 loops = 18 effective layers. Dense FFN + MoE (1 shared + 11 routed experts, top-2) in parallel residual.

| Property | Value |
|----------|-------|
| Physical params | ~208M |
| Active params/token | ~131M |
| Effective params | ~790M+ |
| Architecture | 3 blocks x 6 loops = 18 effective layers |
| MoE | 12 experts (1 shared + 11 routed, top-2) |
| Hidden dim | 1536 |
| Attention heads | 24 (head_dim=64) |
| Tokenizer | Custom 64K BPE (trained on code + text + Wikipedia) |
| Context length | 8192 (progressive: 2048 -> 4096 -> 8192) |
| Target training data | 50B tokens |
| RoPE scaling | NTK-aware for inference beyond training length |

**Key v4 features:**
- Loop-aware router with learned loop embeddings (sigmoid gating, DeepSeek-V3 style)
- Learnable dense/MoE gates for parallel residual scaling
- Router z-loss (ST-MoE/PaLM) for training stability
- Batched scatter-gather expert dispatch
- Gradient checkpointing for long sequences
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
          |  Token Embedding |  (65536 x 1536)
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
   ||  |  x_mod = x + scale * (x @ adapter_a[i] @ adapter_b[i])    |  ||
   ||  |                                                            |  ||
   ||  |  +== ATTENTION ========================================+   |  ||
   ||  |  | RMSNorm -> QKV (1536->4608) -> RoPE -> Causal SDPA |   |  ||
   ||  |  | -> Out Proj (1536->1536) + Residual                 |   |  ||
   ||  |  +=====================================================+   |  ||
   ||  |                          |                                 |  ||
   ||  |           +--------------+---------------+                 |  ||
   ||  |           |                              |                 |  ||
   ||  |  +========v==========+  +================v==============+  |  ||
   ||  |  | DENSE FFN         |  | MoE FFN                      |  |  ||
   ||  |  |                   |  |                               |  |  ||
   ||  |  | RMSNorm           |  | RMSNorm                      |  |  ||
   ||  |  | SwiGLU            |  |     +---------------------+  |  |  ||
   ||  |  | (1536->4096->1536)|  |     |  Loop-Aware Router   |  |  |  ||
   ||  |  |                   |  |     |  h + loop_emb -> sig  |  |  |  ||
   ||  |  |                   |  |     +----------+----------+  |  |  ||
   ||  |  |                   |  |                |             |  |  ||
   ||  |  |                   |  |         top-2 selection      |  |  ||
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

**MoE routing detail:**

```
  Hidden state (B, S, 1536)
         |
         +---> concat with loop_embedding[loop_idx]  --> (B, S, 3072)
         |
         +---> Router linear (3072 -> 11)  --> sigmoid --> top-2 selection
         |
         |     Expert 0: SwiGLU(1536 -> 1024 -> 1536)  [always active = shared]
         |     Expert 1: SwiGLU(1536 -> 1024 -> 1536)  \
         |     Expert 2: SwiGLU(1536 -> 1024 -> 1536)   |
         |     ...                                       +-- top-2 selected
         |     Expert 11: SwiGLU(1536 -> 1024 -> 1536)  /
         |
         +---> output = shared_out + weighted_sum(selected_expert_outputs)
```

**Parameter budget (v4):**

```
  Token Embedding       101M  [===========================]             32.3%
  MoE Routed (11)       119M  [================================]        38.1%
  Dense FFN x3           57M  [===============]                         18.2%
  Attention x3           28M  [========]                                 9.0%
  Shared Expert x3        5M  [=]                                        1.5%
  Adapters x18            1M  []                                         0.3%
  Router + Loop Emb      <1M  []                                         0.1%
  ─────────────────────────────────────────────────────────────────
  Total Physical       ~208M
  Active / Token       ~131M   (shared + 2 of 11 routed)
  Effective            ~790M   (recursive x6)
```

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
         0              15K                        250K          300K
```

### Training Pipeline Overview

```
  +===========+     +==========+     +========+     +==========+     +======+
  | Tokenizer |---->| Pretrain |---->|  SFT   |---->|  GRPO    |---->| Eval |
  | (64K BPE) |     | (300K    |     | (5K    |     | (2K      |     |      |
  |  10GB     |     |  steps)  |     |  steps)|     |  steps)  |     |      |
  +===========+     +==========+     +========+     +==========+     +======+
       |                 |               |               |               |
    Custom          Progressive      Balanced        Verifiable      IFEval
    vocab           2048->8192       math+code       math rewards    GSM8K
    code+text       Lion optimizer   +STEM+general   group_size=16   HumanEval
    +wiki           50B tokens       HRA adapters    KL penalty      HellaSwag
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
Step 0: Tokenizer     -> Train custom 64K BPE on 10GB text+code+wiki
Step 1: Pretrain      -> 300K steps, progressive seq_len 2048->4096->8192
  Foundation: TinyStories + CodeParrot (15K steps)
  Knowledge:  FineWeb-Edu + CodeParrot + Wikipedia (235K steps)
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
# Train custom 64K tokenizer
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
│   └── train_tokenizer.py          # Custom 64K BPE tokenizer training
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
| Loop-aware routing (v4) | Router learns to dispatch differently at each recursive pass |
| Custom tokenizer (v4) | Optimized for code+text distribution; native single-token tags |
| Progressive seq_len (v4) | Start short (fast), extend long (capability); RoPE naturally supports this |

---

## Observability

Training logs to Weights & Biases with per-stage metrics:

- **Pre-training:** loss, lr, vram, tok/s, phase, loop RMS, adapter similarity
- **SFT:** loss, lr, vram, token utilization
- **GRPO:** loss, mean reward, accuracy, KL divergence

W&B project: [nano-osrt-100m](https://wandb.ai/codhe-synextra/nano-osrt-100m) | [nano-osrt-v4](https://wandb.ai/codhe-synextra/nano-osrt-v4)

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v4.0** (dev) | 3 blocks, MoE (12 experts, top-2), 64K custom tokenizer, progressive seq_len, native tags, HF-native |
| **v3.3** | Code SFT (7K steps, 2 epochs), HRA adapters (+11.2M params), HF inference wrapper, IFEval benchmark |
| **v3.2** | Post-training pipeline: Math SFT + GRPO + improved reward functions |
| **v3.1** | RoPE, FP32 master weights, dynamic vocab, SmolTalk formatting |
| **v3.0** | Renamed adapters, gradient accumulation, dataset shuffle |
| **v2.0** | Scaled 46M -> 104.5M, inter/intra-block telemetry |
| **v1.0** | Initial: 1 block x 6 loops, 46M params, Lion optimizer |

---

## License

MIT. See [pyproject.toml](pyproject.toml) for details.
