# Nano-OSRT 100M

**Omni-Sparse Recursive Titan** -- A 104.5M parameter recursive transformer language model that achieves the reasoning depth of a 12-layer network using only 2 physical transformer blocks, looped 6 times each with unique per-pass residual adapters. Designed to train from scratch on a single NVIDIA H100 GPU within a \$280 compute budget on [Modal](https://modal.com/) serverless infrastructure.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Model Variants](#model-variants)
- [Recursive Weight Sharing](#recursive-weight-sharing)
- [Per-Pass Residual Adapters](#per-pass-residual-adapters)
- [Data Pipeline](#data-pipeline)
- [Training Curriculum](#training-curriculum)
- [Training Configuration](#training-configuration)
- [Infrastructure](#infrastructure)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Telemetry and Monitoring](#telemetry-and-monitoring)
- [Checkpointing and Resume](#checkpointing-and-resume)
- [VRAM Budget](#vram-budget)
- [Design Decisions](#design-decisions)
- [Testing](#testing)
- [Version History](#version-history)
- [License](#license)

---

## Architecture Overview

The repository contains two model variants sharing a common codebase: a **standard GPT-2 style model** for local development and a **recursive weight-sharing model** for production Modal cloud training.

```
                        ┌─────────────────────────────────────────────┐
                        │            Nano-OSRT 100M Repo              │
                        └────────────────────┬────────────────────────┘
                                             │
                     ┌───────────────────────┼────────────────────────┐
                     │                       │                        │
              ┌──────▼──────┐        ┌───────▼───────┐       ┌───────▼───────┐
              │   Standard  │        │   Recursive   │       │   Shared      │
              │   Model     │        │   Model       │       │   Utilities   │
              │  (GPT-2)    │        │  (Production) │       │               │
              ├─────────────┤        ├───────────────┤       ├───────────────┤
              │ model.py    │        │ recursive_    │       │ rope.py       │
              │ config.py   │        │   model.py    │       │ modal_data.py │
              │ data.py     │        │ modal_config  │       │ modal_train   │
              │ train.py    │        │   .py         │       │   .py         │
              └──────┬──────┘        └───────┬───────┘       └───────────────┘
                     │                       │
              ┌──────▼──────┐        ┌───────▼───────┐
              │ scripts/    │        │   app.py      │
              │  train.py   │        │ (Modal entry) │
              │ (CLI entry) │        │               │
              └─────────────┘        └───────────────┘
```

---

## Model Variants

### Standard Model (NanoOSRT -- GPT-2 Style)

Used for local development and experimentation. A decoder-only causal LM modelled after GPT-2 small with modern improvements.

```
          Input Token IDs (B, T)
                   │
          ┌────────▼────────┐
          │  Token Embedding │  (vocab_size × 768)
          │  + Pos Embedding │  (1024 × 768)
          │  + Dropout       │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │   Block × 12    │──┐
          │  ┌─────────────┐│  │  Pre-norm residual blocks:
          │  │  LayerNorm  ││  │    x = x + Attn(LN(x))
          │  │  CausalAttn ││  │    x = x + MLP(LN(x))
          │  │  LayerNorm  ││  │
          │  │  MLP (GELU) ││  │  Attention: SDPA / FlashAttention
          │  └─────────────┘│  │  MLP: Linear → GELU → Linear
          └────────┬────────┘──┘
                   │
          ┌────────▼────────┐
          │   LayerNorm     │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │    LM Head      │  (weight-tied with token embedding)
          └────────┬────────┘
                   │
              Logits (B, T, vocab_size)
```

| Property          | Value             |
|-------------------|-------------------|
| Parameters        | ~117M             |
| Layers            | 12                |
| Heads             | 12                |
| Embedding dim     | 768               |
| FFN hidden        | 3072 (4 x 768)   |
| Context length    | 1024              |
| Positional enc.   | Learned absolute  |
| Activation        | GELU              |
| Normalization     | LayerNorm (pre-norm) |

### Recursive Model (RecursiveNanoOSRT -- Production)

The production variant used for Modal cloud training. Uses recursive weight sharing to achieve 12 effective layers from only 2 physical blocks.

```
          Input Token IDs (B, S)
                   │
          ┌────────▼────────┐
          │  Token Embedding │  (vocab_size × 1280)
          └────────┬────────┘
                   │
                   │    ┌──────────────────────────────────────────┐
                   │    │  Precomputed RoPE Buffers                │
                   │    │  cos, sin: (1, seq_len, 1, head_dim)    │
                   │    └──────────────────┬───────────────────────┘
                   │                       │
          ┌────────▼───────────────────────▼─────────┐
          │          Recursive Loop × 6               │
          │  ┌─────────────────────────────────────┐  │
          │  │  Block A  (physical block 0)        │  │
          │  │  ┌───────────────────────────────┐  │  │
          │  │  │ Adapter(A_i, B_i) on x        │  │  │
          │  │  │ RMSNorm → QKV → RoPE → SDPA  │  │  │
          │  │  │ Output Projection + Residual  │  │  │
          │  │  │ RMSNorm → SwiGLU + Residual   │  │  │
          │  │  └───────────────────────────────┘  │  │
          │  └──────────────────┬───────────────────┘  │
          │                    │                       │
          │  ┌─────────────────▼───────────────────┐  │
          │  │  Block B  (physical block 1)        │  │
          │  │  ┌───────────────────────────────┐  │  │
          │  │  │ Adapter(A_j, B_j) on x        │  │  │
          │  │  │ RMSNorm → QKV → RoPE → SDPA  │  │  │
          │  │  │ Output Projection + Residual  │  │  │
          │  │  │ RMSNorm → SwiGLU + Residual   │  │  │
          │  │  └───────────────────────────────┘  │  │
          │  └──────────────────┬───────────────────┘  │
          │                    │                       │
          │             Loop RMS measurement           │
          └────────────────────┬───────────────────────┘
                               │  (× 6 loops = 12 effective layers)
                   ┌───────────▼───────────┐
                   │       RMSNorm         │
                   └───────────┬───────────┘
                               │
                   ┌───────────▼───────────┐
                   │ Linear (weight-tied   │
                   │  with embedding)      │
                   └───────────┬───────────┘
                               │
                    Logits (B, S, vocab_size)
```

| Property              | Value                                          |
|-----------------------|------------------------------------------------|
| Physical parameters   | 104,538,880 (104.5M)                           |
| Effective depth       | 12 layers (2 blocks x 6 loops)                 |
| Embedding dimension   | 1280                                           |
| Attention heads       | 20 (head_dim = 64)                             |
| FFN hidden dimension  | 3456 (SwiGLU, 8/3 x dim, 64-aligned)          |
| Vocabulary            | Dynamic from tokenizer, padded to multiple of 64 |
| Positional encoding   | Rotary (RoPE), theta=10000                     |
| Attention             | Causal SDPA (FlashAttention-2 backend)         |
| Sequence length       | 2048                                           |
| Weight tying          | Input embedding = output projection            |
| Normalization         | RMSNorm                                        |
| Activation            | SwiGLU (SiLU-gated)                            |

### Parameter Breakdown (Recursive Model)

| Component                           | Parameters | % of Total |
|-------------------------------------|------------|------------|
| Token Embedding (vocab x 1280)      | ~64.4M     | ~61.6%     |
| QKV Projection x2 (1280 x 3840)    | 9.83M      | 9.4%       |
| Output Projection x2 (1280 x 1280) | 3.28M      | 3.1%       |
| SwiGLU x2 (gate + up + down)       | 26.54M     | 25.4%      |
| RMSNorm x6                         | ~7.7K      | <0.01%     |
| Adapters x12 (A + B matrices)      | 491K       | 0.47%      |
| **Total**                           | **104.5M** | **100%**   |

---

## Recursive Weight Sharing

The model stores 2 physical `RecursiveBlock` modules. During the forward pass, the input passes through Block A then Block B, and this pair is looped 6 times, producing 12 effective layers.

```
                    ┌─────────────────────────────────────────────────┐
                    │            Recursive Unrolling                  │
                    │                                                 │
  Input ──►  Loop 0:  [Block A] ──► [Block B]  ──► RMS              │
             Loop 1:  [Block A] ──► [Block B]  ──► RMS              │
             Loop 2:  [Block A] ──► [Block B]  ──► RMS              │
             Loop 3:  [Block A] ──► [Block B]  ──► RMS              │
             Loop 4:  [Block A] ──► [Block B]  ──► RMS              │
             Loop 5:  [Block A] ──► [Block B]  ──► RMS  ──► Output  │
                    │                                                 │
                    │  Same weights reused at each loop, but each    │
                    │  pass gets a unique adapter (A_i, B_i)         │
                    └─────────────────────────────────────────────────┘

  Adapter indices:   A0  A1    A2  A3    A4  A5    A6  A7    A8  A9   A10 A11
                     B0  B1    B2  B3    B4  B5    B6  B7    B8  B9   B10 B11
```

This approach is grounded in Universal Transformers, ALBERT, and LoopLM/Ouro (2025), which demonstrated that looped models with 2-3x fewer parameters can match dense models on reasoning tasks.

The key limitation of pure weight sharing is **representational collapse**: repeated application of identical weights homogenises the hidden states. The model addresses this with per-pass residual adapters.

---

## Per-Pass Residual Adapters

Each of the 12 (block, loop) combinations receives a unique pair of low-rank matrices A and B:

```
                      ┌─────────────────────────────────────────┐
                      │        Per-Pass Residual Adapter         │
                      │                                         │
   x ────────┬────────▼────────┐                                │
             │     x @ A       │   A: (dim, rank) = (1280, 16)  │
             │       │         │   B: (rank, dim) = (16, 1280)  │
             │     x @ A @ B  │   scale = alpha / rank = 1.0   │
             │       │         │                                │
             │   scale * (...)│                                 │
             │       │         │                                │
             ├───────+─────────┘                                │
             │                                                  │
   x_mod  ◄──┘  x_mod = x + scale * (x @ A @ B)               │
                      │                                         │
                      └─────────────────────────────────────────┘
```

This is **not** weight-LoRA (Hu et al., 2022), which modifies weight matrices. This operates directly on the activation stream, applying a learned low-rank residual transformation that gives each recursive pass a distinct identity.

Key design choices:

- **B is zero-initialised.** At step 0, the adapter is a no-op and all loops are mathematically identical. Differentiation emerges organically through gradient flow.
- **Scale = alpha/rank = 16/16 = 1.0.** Tunable knob for controlling adapter influence.
- **Rank 16** is conservative (LoRA literature shows r=4-16 suffices).
- **Residual connects to x_mod, not x.** The adapter's geometric shift survives through attention and into the FFN.

---

## Data Pipeline

The repository has two data pipelines: a **local pipeline** for pre-tokenised binary files and a **streaming pipeline** for Modal cloud training.

### Local Pipeline (data.py)

```
  Pre-tokenised .bin file (uint16)
           │
     ┌─────▼─────┐
     │  np.memmap │  Memory-mapped for out-of-core access
     └─────┬─────┘
           │
     ┌─────▼──────────────────┐
     │  TokenDataset          │  Map-style: random access by index
     │  StreamingTokenDataset │  Iterable: infinite random sampling
     │  get_batch()           │  Direct batch sampling from memmap
     └─────┬──────────────────┘
           │
     (input_ids, labels) with 1-token causal shift
```

### Streaming Pipeline (modal_data.py -- Production)

```
  HuggingFace Streaming Dataset
           │
     ┌─────▼──────────────────────────────────────────────┐
     │  TokenStream (IterableDataset)                     │
     │                                                    │
     │  ┌────────────────────────────────────────────┐    │
     │  │  1. Load streaming dataset                 │    │
     │  │  2. Shuffle (buffer=10K, seed=42+step)     │    │
     │  │  3. Shard across DataLoader workers        │    │
     │  │  4. Tokenise on-the-fly                    │    │
     │  │  5. Concatenate into continuous buffer      │    │
     │  │  6. Append EOS at document boundaries      │    │
     │  │  7. Chunk into (seq_len+1) windows         │    │
     │  │  8. Yield (input_ids, labels) pairs        │    │
     │  └────────────────────────────────────────────┘    │
     └─────┬──────────────────────────────────────────────┘
           │
     ┌─────▼──────────────────────────────────────────────┐
     │  DataLoader                                        │
     │  - 2 background workers                            │
     │  - pin_memory=True                                 │
     │  - prefetch_factor=4                               │
     │  - persistent_workers=True                         │
     └─────┬──────────────────────────────────────────────┘
           │
     ┌─────▼──────┐
     │  GPU        │  non_blocking transfer
     └────────────┘
```

Every tensor is 100% dense -- no padding tokens, no wasted FLOPs. Token packing concatenates documents into a continuous stream with EOS markers at boundaries.

For instruction-tuning data (Phase 3, SmolTalk), the pipeline detects `messages` columns and formats them with `role: content` pairs, injecting EOS tokens after assistant turns.

---

## Training Curriculum

Training proceeds through three phases. Phase transitions are step-based and handled automatically.

```
   Step:  0          8,000                              140,000    150,000
          │◄─ Phase 1 ─►│◄──────── Phase 2 ───────────────►│◄─ Phase 3─►│
          │             │                                   │            │
          │ TinyStories │         FineWeb-Edu               │  SmolTalk  │
          │  (~1.05B    │         (~17.3B tokens)            │  (~1.3B    │
          │   tokens)   │         Filtered CommonCrawl      │   tokens)  │
          │             │         Educational quality        │  Instruct  │
          │  Syntax     │         Knowledge                  │  Alignment │
          │  Acquisition│         Integration                │            │
```

| Phase | Steps | Dataset | Purpose |
|-------|-------|---------|---------|
| 1 - Syntax Acquisition | 0 - 8,000 | roneneldan/TinyStories | Simplified synthetic stories (~3K word vocab). Establishes clean grammar patterns and basic attention structure. |
| 2 - Knowledge Integration | 8,000 - 140,000 | HuggingFaceFW/fineweb-edu | Filtered CommonCrawl scored by Llama-3-70B for educational quality. The bulk of training (~17.3B tokens). |
| 3 - Instruction Alignment | 140,000 - 150,000 | HuggingFaceTB/smoltalk | Instruction-tuning via MagPie pipeline (Llama-3.1-405B-Instruct). Teaches conversational formatting. |

Total token budget: ~19.7B tokens across all phases.

---

## Training Configuration

### Training Loop Architecture

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                       Modal Training Loop                           │
  │                                                                     │
  │   app.py (Modal entrypoint)                                        │
  │     │                                                               │
  │     ├── Provision H100 container (Debian Slim + Python 3.11)       │
  │     ├── Load tokenizer (EleutherAI/gpt-neox-20b)                   │
  │     ├── Align vocab to 64-byte boundary                            │
  │     └── Call run_training(cfg, vol, tokenizer_name)                │
  │           │                                                         │
  │           ├── Build RecursiveNanoOSRT → FP32 master weights         │
  │           ├── torch.compile(mode="max-autotune")                   │
  │           ├── Initialise Lion optimizer (or AdamW fallback)        │
  │           ├── Resume from rescue checkpoint if exists              │
  │           │                                                         │
  │           └── for step in range(total_steps):                      │
  │                 ├── Determine curriculum phase                      │
  │                 ├── Update LR (linear warmup → cosine decay)       │
  │                 ├── Gradient accumulation (4 micro-batches):       │
  │                 │     ├── Forward pass (BF16 autocast)             │
  │                 │     ├── CE loss on real_vocab_size logits (FP32) │
  │                 │     └── Backward pass                            │
  │                 ├── Gradient clipping (max_norm=1.0)               │
  │                 ├── Optimizer step                                  │
  │                 ├── Log every 50 steps (loss, LR, VRAM, tok/s,     │
  │                 │     adapter divergence, loop RMS)                 │
  │                 ├── Checkpoint every 2,000 steps                   │
  │                 └── 23h rescue checkpoint (Modal 24h limit)        │
  └──────────────────────────────────────────────────────────────────────┘
```

### Hyperparameters

| Hyperparameter        | Value          | Rationale                                         |
|-----------------------|----------------|---------------------------------------------------|
| Micro-batch size      | 16             | Fits activation memory on H100                    |
| Gradient accumulation | 4              | Effective batch = 64, tokens/step = 131,072       |
| Total steps           | 150,000        | ~19.7B tokens total                               |
| Warmup steps          | 2,000          | Lion requires longer warmup (sign-based updates)  |
| Peak learning rate    | 1e-4           | Lion paper LM setting (3-10x lower than AdamW)    |
| Min learning rate     | 1e-5           | Cosine decay floor                                |
| Weight decay          | 0.3            | Lion paper: 3-10x higher than AdamW's 0.1         |
| Gradient clipping     | 1.0            | Safety net against NaN gradients                  |
| LR schedule           | Warmup + cosine| Standard for pretraining                          |

### Optimizer

The default optimizer is **Lion** (EvoLved Sign Momentum). Lion uses only the sign of the momentum for updates, halving optimizer state VRAM compared to AdamW (no variance tracking). An AdamW fallback is available by setting `optimizer_name = "adamw"` in `ModalConfig`.

### Precision Strategy

```
  ┌────────────────────────────────────────────────────────┐
  │                 Precision Pipeline                     │
  │                                                       │
  │  Master Weights ──► FP32  (prevents BF16 underflow)   │
  │         │                                              │
  │  Forward Pass  ──► BF16  (torch.amp.autocast)         │
  │         │                                              │
  │  Backward Pass ──► BF16  (autocast)                   │
  │         │                                              │
  │  CE Loss       ──► FP32  (logits cast before loss)    │
  │         │                                              │
  │  Optimizer     ──► FP32  (Lion sign updates on        │
  │                          master weights)               │
  │                                                       │
  │  No GradScaler needed -- BF16 shares FP32's           │
  │  8-bit exponent and dynamic range.                    │
  └────────────────────────────────────────────────────────┘
```

### Compute Budget

| Property     | Value                              |
|--------------|------------------------------------|
| GPU          | NVIDIA H100 80GB SXM (Hopper)      |
| Provider     | Modal (serverless)                 |
| Cost         | \$3.95/hr                          |
| Budget       | \$280                              |
| Runtime      | ~71 hours                          |
| 24h boundary | Rescue checkpoint at 23h (82,800s) |

---

## Infrastructure

### Modal Deployment Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                     Local Machine                               │
  │                                                                 │
  │   $ modal run app.py                                           │
  │       │                                                         │
  │       ├── Modal CLI parses app.py locally                      │
  │       │   (torch imports inside train() to avoid crash)        │
  │       └── Sends to Modal cloud                                 │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │                     Modal Cloud                                 │
  │                                                                 │
  │   ┌─────────────────────────────────────────────────────────┐   │
  │   │  Container (Debian Slim + Python 3.11)                  │   │
  │   │  ┌──────────────────────────────────────────────────┐   │   │
  │   │  │  Dependencies:                                   │   │   │
  │   │  │  - PyTorch 2.10 + CUDA 12.8                     │   │   │
  │   │  │  - transformers, datasets                        │   │   │
  │   │  │  - lion-pytorch, triton                          │   │   │
  │   │  │  TORCH_LOGS=perf_hints                           │   │   │
  │   │  └──────────────────────────────────────────────────┘   │   │
  │   │                                                         │   │
  │   │  ┌───────────┐    ┌────────────────────────────────┐    │   │
  │   │  │  H100 GPU │◄──►│  train() function              │    │   │
  │   │  │  80GB SXM │    │  - 24h timeout                 │    │   │
  │   │  └───────────┘    └────────────────────────────────┘    │   │
  │   └──────────────────────────┬──────────────────────────────┘   │
  │                              │                                  │
  │   ┌──────────────────────────▼──────────────────────────────┐   │
  │   │  Persistent Volume: osrt-checkpoints                    │   │
  │   │  /vol/checkpoints/                                      │   │
  │   │    ├── osrt100m_step_2000.pt                            │   │
  │   │    ├── osrt100m_step_4000.pt                            │   │
  │   │    ├── ...                                              │   │
  │   │    ├── osrt100m_rescue.pt   (23h auto-save)             │   │
  │   │    └── osrt100m_final.pt    (training complete)         │   │
  │   └─────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
nano-osrt-100m/
├── README.md                       # This file
├── overview.md                     # Detailed technical design document
├── pyproject.toml                  # Project config, deps, entry points
├── uv.lock                        # Locked dependencies
├── app.py                         # Modal cloud deployment entry-point
│
├── src/nano_osrt/
│   ├── __init__.py                # Package exports (NanoOSRT, ModelConfig)
│   ├── config.py                  # ModelConfig & TrainConfig dataclasses
│   │                                (GPT-2 style: 768d, 12L, 12H)
│   ├── model.py                   # NanoOSRT: standard GPT-2 style model
│   │                                (CausalSelfAttention, MLP, Block)
│   ├── data.py                    # Local data: memory-mapped token datasets
│   │                                (TokenDataset, StreamingTokenDataset,
│   │                                 get_batch, load_data_split)
│   ├── train.py                   # Local training loop (AdamW, cosine LR,
│   │                                eval, checkpointing)
│   ├── modal_config.py            # ModalConfig for recursive-block deployment
│   │                                (1280d, 20H, 2 blocks x 6 loops)
│   ├── rope.py                    # Rotary Position Embedding utilities
│   │                                (compute_rope_freqs, apply_rope)
│   ├── recursive_model.py         # RecursiveNanoOSRT: production model
│   │                                (SwiGLU, RecursiveBlock, adapters, RoPE)
│   ├── modal_data.py              # Streaming HuggingFace data pipeline
│   │                                (TokenStream, make_loader)
│   └── modal_train.py             # Modal training loop & helpers
│                                    (run_training, get_lr, get_phase,
│                                     save/load_checkpoint, log_step)
│
├── scripts/
│   └── train.py                   # CLI entry-point for local training
│                                    (argparse → TrainConfig → train())
│
├── tests/
│   ├── test_model.py              # Tests for GPT-2 style model
│   │                                (forward, generate, weight tying, etc.)
│   └── test_recursive_model.py    # Tests for recursive model & utilities
│                                    (RoPE, SwiGLU, adapters, LR schedule,
│                                     phase selection)
│
├── checkpoints/                   # Local training checkpoints
└── data/                          # Local pre-tokenised data (train.bin, val.bin)
```

### Module Dependency Graph

```
  app.py
    ├── nano_osrt.modal_config.ModalConfig
    └── nano_osrt.modal_train.run_training
          ├── nano_osrt.modal_config.ModalConfig
          ├── nano_osrt.modal_data.make_loader
          │     └── nano_osrt.modal_data.TokenStream
          └── nano_osrt.recursive_model.RecursiveNanoOSRT
                ├── nano_osrt.modal_config.ModalConfig
                └── nano_osrt.rope.{compute_rope_freqs, apply_rope}

  scripts/train.py
    ├── nano_osrt.config.{ModelConfig, TrainConfig}
    └── nano_osrt.train.train
          ├── nano_osrt.config.TrainConfig
          ├── nano_osrt.data.{get_batch, load_data_split}
          └── nano_osrt.model.NanoOSRT
                └── nano_osrt.config.ModelConfig
```

---

## Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- For cloud training: [Modal](https://modal.com/) account with GPU access

### Install and Test

```bash
# Install dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Lint
uv run ruff check src/ scripts/ tests/
```

### Local Training

Requires pre-tokenised data in `data/train.bin` and `data/val.bin` (uint16 numpy arrays):

```bash
# Using the CLI entry-point
uv run train

# With custom options
uv run train --n-layer 6 --n-head 6 --n-embd 384 --batch-size 8
```

### Modal Cloud Training

```bash
# Install Modal CLI
pip install modal
modal token new

# Launch training on H100
modal run app.py

# Resume after 24h timeout (automatic checkpoint detection)
modal run app.py
```

---

## Telemetry and Monitoring

Every 50 steps, the trainer logs:

```
step    1000 | loss 7.2341 | lr 5.00e-05 | vram 24.3GB | tok/s 89,432 | phase tinystories
           intra-block (b0: L0 vs L1..5): [0.95, 0.82, 0.71, 0.63, 0.54]
           inter-block (b0 vs b1 per loop): [0.88, 0.79, 0.71, 0.65, 0.58, 0.52]
           loop RMS: [1.024, 1.031, 1.028, 1.035, 1.029, 1.033]
```

### Key Metrics

| Metric | What it measures | Healthy range |
|--------|-----------------|---------------|
| **Loss** | Cross-entropy on next-token prediction | Start ~10.8, below 7.0 in Phase 1 |
| **tok/s** | Training throughput | 80K-150K+ on H100 after compilation |
| **Intra-block similarity** | Cosine sim of Block 0's adapter_b across loops | 1.0 initially, spreading to 0.2-0.5 |
| **Inter-block similarity** | Block A vs Block B adapter_b at each loop | Should diverge independently |
| **Loop RMS** | Activation magnitude after each loop | 0.5-2.0 (exponential growth = trouble) |

### Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss plateaus at ~10.8 | Model not learning | Check data pipeline is yielding real tokens |
| Loss spikes to NaN | Gradient explosion | Reduce lr, increase warmup |
| Loop RMS explodes | Adapter norms too large | Reduce adapter_alpha |
| Intra-block sims stay at 1.0 | Loops not differentiating | Increase adapter_rank or lr |
| tok/s much lower than expected | torch.compile fallback | Check TORCH_LOGS for perf_hints warnings |
| Loss flatlines after phase transition | Dataset format mismatch | Verify smoltalk messages column handling |

---

## Checkpointing and Resume

```
  ┌───────────────────────────────────────────────────────────┐
  │                 Checkpoint Strategy                        │
  │                                                           │
  │   Every 2,000 steps:                                     │
  │     /vol/checkpoints/osrt100m_step_{N}.pt                │
  │     Contains: step, model_state_dict, optimizer_state_dict│
  │                                                           │
  │   At 82,800s (23h mark):                                 │
  │     /vol/checkpoints/osrt100m_rescue.pt                  │
  │     Auto-detected on next `modal run app.py`             │
  │                                                           │
  │   On completion:                                          │
  │     /vol/checkpoints/osrt100m_final.pt                   │
  │     Model weights only (no optimizer state)              │
  │                                                           │
  │   torch.compile compatibility:                           │
  │     Save/load uses model._orig_mod to unwrap compiled    │
  │     model, ensuring clean checkpoint semantics.          │
  └───────────────────────────────────────────────────────────┘
```

---

## VRAM Budget

Estimated memory usage on H100 80GB (micro-batch = 16):

| Component                                | GB     |
|------------------------------------------|--------|
| Model parameters (FP32)                  | 0.42   |
| Optimizer state (Lion, FP32 momentum)    | 0.42   |
| Gradients (FP32)                         | 0.42   |
| Activations (BF16, 12 effective layers)  | ~22.7  |
| CUDA context + compile workspace         | ~3.0   |
| **Total**                                | **~27 GB** |
| **Headroom**                             | **~53 GB** |

Activation memory dominates because recursive weight sharing stores weights once but must retain activations at all 12 effective passes for backpropagation. The single largest allocation is the logits tensor: `16 x 2048 x vocab x 2 bytes ~ 3.1 GB`.

---

## Design Decisions

### Why recursive weight sharing?

2 physical blocks x 6 loops gives 12 effective layers of iterative refinement with only ~104.5M parameters. The model gets reasoning depth (12 unrolled layers) without the knowledge capacity cost of 12 unique parameter sets. Research shows weight-shared models store ~2 bits/parameter regardless of loop count.

### Why not sparse attention at 2048 tokens?

DeepSeek's NSA benchmarks (ACL 2025) show sparse attention is 2.4x *slower* than FlashAttention at 2048 tokens. The full N x N attention matrix at this length fits entirely in GPU SRAM. Sparse machinery becomes beneficial at ~8K+ tokens.

### Why Lion over AdamW?

Lion halves optimizer VRAM by eliminating variance tracking (~400MB savings). Lion also showed competitive validation perplexity on 110M parameter models in the original paper. The sign-based update mechanism requires higher weight decay (0.3 vs 0.1) and longer warmup (2000 steps).

### Why "adapters" not "LoRA"?

The mechanism modulates activations directly (`x + scale * (x @ A @ B)`), not weight matrices. Standard LoRA applies low-rank updates to W_q, W_k, etc. Calling this "LoRA" would be technically inaccurate.

### Why GPT-NeoX tokenizer?

The GPT-NeoX tokenizer (50K+ tokens) dedicates ~61% of parameters to the embedding matrix. A 32K vocabulary would free ~14M parameters. The current choice prioritises ecosystem compatibility with the Pythia model suite over parameter efficiency.

---

## Testing

The test suite covers both model variants and all shared utilities:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=nano_osrt
```

**test_model.py** -- Standard GPT-2 model:
- Config validation (head_dim computation)
- Model instantiation and parameter counting
- Forward pass with/without targets
- Autoregressive generation
- Weight tying between embedding and LM head
- Block size boundary enforcement

**test_recursive_model.py** -- Recursive model and utilities:
- ModalConfig defaults and phase structure
- RoPE: frequency shape, application shape, norm preservation (rotation invariant)
- SwiGLU: output shape, tensor-core alignment (hidden dim % 64 == 0)
- RecursiveBlock: output shape with adapters
- RecursiveNanoOSRT: instantiation, forward shape, loop RMS values, adapter count
- Training helpers: LR warmup/decay schedules, curriculum phase selection

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v3.2** (current) | Lion wd 0.01 -> 0.3, warmup 500 -> 2000 steps, sparse NSA -> standard SDPA, PyTorch 2.10 + CUDA 12.8 |
| **v3.1** | RoPE added, FP32 master weights, dynamic vocab, SmolTalk formatting, async DataLoader, orphaned adapter residual fix |
| **v3.0** | Renamed "LoRA" to "adapters", adapter_scale, gradient accumulation, dataset shuffle, AdamW fallback |
| **v2.0** | Scaled 46M -> 104.5M (dim 768->1280, 1->2 blocks), inter/intra-block telemetry, phantom vocab slice |
| **v1.0** | Initial baseline: 1 block x 6 loops, 46M params, Lion optimizer, Modal checkpoint resume |

---

## License

MIT. See [pyproject.toml](pyproject.toml) for details.
