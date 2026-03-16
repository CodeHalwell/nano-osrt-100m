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

### Recursive Weight Sharing

The core innovation: physical transformer blocks are reused across multiple loops, simulating a much deeper network. Each loop gets unique per-pass residual adapters to prevent representational collapse.

```
Input -> Embedding -> [Block0, Block1, ...] x N loops -> RMSNorm -> LM Head
                           ^          ^
                      adapter_a[i]  adapter_b[i]   (unique per virtual layer)
```

**v3:** 2 blocks x 6 loops = 12 effective layers
**v4:** 3 blocks x 6 loops = 18 effective layers + MoE

### v4 Block Structure

Each v4 block contains attention + parallel dense/MoE FFN:

```
x = x + attention(norm(x))           # causal attention with RoPE
x = x + gate_d * dense(norm(x))      # dense SwiGLU FFN
      + gate_m * moe(norm(x), loop)  # MoE: shared expert + top-2 routed
```

### Per-Pass Residual Adapters

Not weight-LoRA -- modulates hidden states directly:

```python
x_mod = x + scale * (x @ A @ B)   # A: (dim, rank), B: (rank, dim)
```

- Zero-initialized B: adapter starts as no-op, differentiation emerges through training
- Each (block, loop) pair gets a unique adapter

### High Rank Adaptation (HRA)

Post-training capacity expansion. Injects rank-256 adapter matrices alongside linear layers:

```python
y = Linear(x) + scale * (x @ A @ B)   # A: (in, 256), B: (256, out)
```

Adds ~11-20M trainable parameters for SFT without disrupting pretrained weights.

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
