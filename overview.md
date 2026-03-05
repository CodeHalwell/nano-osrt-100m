# Nano-OSRT 100M

**Omni-Sparse Recursive Titan — A 104.5M Parameter Foundation Model for $280**

Version 3.2 | March 2026

-----

## Overview

Nano-OSRT 100M is a recursive transformer language model that achieves the reasoning depth of a 12-layer network using only 2 physical transformer blocks, looped 6 times each with unique per-pass residual adapters. It is designed to train from scratch on a single NVIDIA H100 GPU within a $280 compute budget on Modal serverless infrastructure.

The model is a research baseline for validating whether recursive weight sharing with activation-space adapters can produce meaningful language modelling capabilities at sub-billion parameter scales.

### Quick Start

```bash
pip install modal
modal run train_osrt_baseline.py
```

The script handles everything: container provisioning, GPU allocation, tokenizer loading, data streaming, training, and checkpoint saving. Re-running the same command after a 24-hour timeout automatically resumes from the last checkpoint.

-----

## Architecture

### Model Summary

|Property            |Value                                                   |
|--------------------|--------------------------------------------------------|
|Physical parameters |104,538,880 (104.5M)                                    |
|Effective depth     |12 layers (2 blocks × 6 loops)                          |
|Embedding dimension |1280                                                    |
|Attention heads     |20 (head_dim = 64)                                      |
|FFN hidden dimension|3456 (SwiGLU, 8/3 × dim, 64-aligned)                    |
|Vocabulary          |Dynamic from tokenizer, padded to nearest multiple of 64|
|Positional encoding |Rotary (RoPE), theta=10000                              |
|Attention           |Causal SDPA (FlashAttention-2 backend)                  |
|Sequence length     |2048                                                    |
|Weight tying        |Input embedding = output projection                     |

### Parameter Breakdown

|Component                         |Parameters|% of Total|
|----------------------------------|----------|----------|
|Token Embedding (vocab × 1280)    |~64.4M    |~61.6%    |
|QKV Projection ×2 (1280 × 3840)   |9.83M     |9.4%      |
|Output Projection ×2 (1280 × 1280)|3.28M     |3.1%      |
|SwiGLU ×2 (gate + up + down)      |26.54M    |25.4%     |
|RMSNorm ×6                        |~7.7K     |<0.01%    |
|Adapters ×12 (A + B matrices)     |491K      |0.47%     |
|**Total**                         |**104.5M**|**100%**  |

### Recursive Weight Sharing

The model stores 2 physical `RecursiveBlock` modules. During the forward pass, the input passes through Block A then Block B, and this pair is looped 6 times, producing 12 effective layers. This is grounded in work on Universal Transformers, ALBERT, and more recently LoopLM/Ouro (2025), which demonstrated that looped models with 2–3× fewer parameters can match dense models on reasoning tasks.

The key limitation of pure weight sharing is representational collapse: repeated application of identical weights homogenises the hidden states. The model addresses this with per-pass residual adapters.

### Per-Pass Residual Adapters

Each of the 12 (block, loop) combinations receives a unique pair of low-rank matrices A and B:

```
x_mod = x + (alpha/rank) * (x @ A @ B)
```

This is **not** weight-LoRA (Hu et al., 2022), which modifies weight matrices. This operates directly on the activation stream, applying a learned low-rank residual transformation that gives each recursive pass a distinct identity.

Key design choices:

- **B is zero-initialised.** At step 0, the adapter is a no-op and all loops are mathematically identical. Differentiation emerges organically through gradient flow.
- **Scale = alpha/rank = 16/16 = 1.0.** This provides a tunable knob: if adapter norms grow too large under Lion, reduce alpha without touching anything else.
- **Rank 16** is conservative. The LoRA literature shows r=4–16 suffices for most adaptations. RingFormer (2025) uses H/16 as default.
- **Residual connects to x_mod, not x.** The adapter’s geometric shift survives through attention and into the FFN. Earlier versions had an “orphaned residual” bug where the shift was discarded.

### Rotary Positional Encodings (RoPE)

RoPE is applied to query and key tensors before the head transpose, encoding relative position via rotation in the complex plane. Frequencies are precomputed and registered as non-persistent buffers (excluded from checkpoints). The implementation uses the standard LLaMA-style rotation with theta=10000, which is well-validated for 2048-token contexts.

### SwiGLU Feed-Forward Network

Each block uses a gated linear unit with SiLU activation:

```
output = W_down(SiLU(W_gate(x)) * W_up(x))
```

Hidden dimension = (8/3) × 1280 = 3413, rounded up to 3456 for 64-byte tensor core alignment. This is the exact formula used by LLaMA-1/2.

### Attention

Version 3.2 uses standard causal `F.scaled_dot_product_attention` with `is_causal=True`, which dispatches to FlashAttention-2 on the H100. Earlier versions used FlexAttention with a sparse NSA mask, which was removed after benchmarks showed NSA is 2.4× slower than full FlashAttention at 2048 tokens (DeepSeek, ACL 2025). The sparse mask machinery only becomes beneficial at 8K+ token contexts.

The per-head sparse offset pattern `((kv_idx // 64) + h) % 4` is preserved in code comments for future long-context work.

### Weight Initialisation

All `nn.Linear` and `nn.Embedding` modules are initialised with N(0, 0.02). Without this, PyTorch’s default N(0, 1) for embeddings produces initial logits with enormous variance at dim=1280, causing an initial CE loss spike to 100+ instead of the expected ~10.8. This gradient shock can permanently destabilise Lion’s momentum buffer.

-----

## Training Configuration

### Compute Budget

|Property    |Value                             |
|------------|----------------------------------|
|GPU         |NVIDIA H100 80GB SXM (Hopper)     |
|Provider    |Modal (serverless)                |
|Cost        |$3.95/hr                          |
|Budget      |$280                              |
|Runtime     |~71 hours                         |
|24h boundary|Rescue checkpoint at 23h (82,800s)|

### Hyperparameters

|Hyperparameter       |Value                       |Rationale                                       |
|---------------------|----------------------------|------------------------------------------------|
|Micro-batch size     |16                          |Fits activation memory on H100                  |
|Gradient accumulation|4                           |Effective batch = 64, tokens/step = 131,072     |
|Total steps          |150,000                     |~19.7B tokens total                             |
|Warmup steps         |2,000                       |Lion requires longer warmup (sign-based updates)|
|Peak learning rate   |1e-4                        |Lion paper LM setting (3–10× lower than AdamW)  |
|Min learning rate    |1e-5                        |Cosine decay floor                              |
|Weight decay         |0.3                         |Lion paper: 3–10× higher than AdamW’s 0.1       |
|Gradient clipping    |1.0                         |Safety net against NaN gradients                |
|LR schedule          |Linear warmup → cosine decay|Standard for pretraining                        |

### Optimizer

The default optimizer is **Lion** (EvoLved Sign Momentum). Lion uses only the sign of the momentum for updates, halving optimizer state VRAM compared to AdamW (no variance tracking). An AdamW fallback is available by setting `optimizer_name = "adamw"` in Config — the fallback uses its own `wd=0.1` rather than Lion’s 0.3.

### Precision Strategy

Parameters are stored in **FP32** (master weights). Forward and backward passes execute in **BF16** via `torch.amp.autocast`. This prevents the BF16 underflow problem where Lion’s ±1e-4 updates round to zero against weights of magnitude ~0.05 (BF16 has only 7 mantissa bits, epsilon ≈ 0.0078). No GradScaler is needed — BF16 shares FP32’s 8-bit exponent and dynamic range.

Cross-entropy loss is computed in FP32 (logits are cast before the loss function) for numerical stability.

### Compilation

The model is wrapped in `torch.compile(mode="max-autotune")`, which traces the forward pass and generates fused Triton/CUDA kernels. Expect a 10-20 minute compilation pause on the first step as the autotuner benchmarks kernel candidates. All `.item()` calls are kept outside the compiled region to avoid graph breaks.

-----

## Data Pipeline

### Architecture

The pipeline uses a `torch.utils.data.IterableDataset` backed by HuggingFace streaming datasets, wrapped in a `DataLoader` with 2 background workers, pinned memory, and prefetching.

```
HuggingFace Streaming → TokenStream (tokenise + buffer) → DataLoader (2 workers, pin_memory)
                                                            → GPU (non_blocking transfer)
```

### Token Packing

Documents are tokenised on-the-fly, concatenated into a continuous token buffer with EOS markers at document boundaries, and chunked into fixed-length (input_ids, labels) pairs with a 1-token causal shift. Every tensor is 100% dense — no padding tokens, no wasted FLOPs.

### Shuffle and Resume

Streaming datasets are shuffled with `buffer_size=10,000` and a dynamic seed of `42 + step_number`. This prevents the “Groundhog Day” bug where Modal container restarts replay the same data ordering. Each DataLoader worker additionally offsets the seed by its worker ID for inter-worker diversity.

### Instruction Format Handling

Phase 3 (SmolTalk) uses a `messages` column instead of `text`. The pipeline detects this and attempts `tokenizer.apply_chat_template()`. Since GPT-NeoX has no default chat template, a manual fallback constructs `role: content` pairs with EOS tokens injected after assistant turns, teaching the model when to stop generating.

-----

## Curriculum

Training proceeds through three phases. Phase transitions are step-based and handled automatically.

### Phase 1: Syntax Acquisition (Steps 0–8,000)

**Dataset:** roneneldan/TinyStories (~1.05B tokens)

Simplified synthetic stories with a restricted vocabulary of ~3,000 words. Establishes clean grammar patterns and basic attention structure without the noise of complex web text. This is a novel, unvalidated warmup strategy — low risk given its brevity but also unproven benefit.

### Phase 2: Knowledge Integration (Steps 8,000–140,000)

**Dataset:** HuggingFaceFW/fineweb-edu (~17.3B tokens)

The bulk of training. FineWeb-Edu is filtered CommonCrawl scored by Llama-3-70B for educational quality. It outperforms C4, Dolma, The Pile, and SlimPajama on aggregate benchmarks and is used by the SmolLM2 model family.

### Phase 3: Instruction Alignment (Steps 140,000–150,000)

**Dataset:** HuggingFaceTB/smoltalk (~1.3B tokens)

Instruction-tuning data created via the MagPie pipeline using Llama-3.1-405B-Instruct. Teaches conversational formatting and instruction following. EOS tokens after assistant turns provide a stop signal for generation.

-----

## Telemetry

Every 50 steps, the trainer logs:

```
step    1000 | loss 7.2341 | lr 5.00e-05 | vram 24.3GB | tok/s 89,432 | phase tinystories
           intra-block (b0: L0 vs L1..5): [0.95, 0.82, 0.71, 0.63, 0.54]
           inter-block (b0 vs b1 per loop): [0.88, 0.79, 0.71, 0.65, 0.58, 0.52]
           loop RMS: [1.024, 1.031, 1.028, 1.035, 1.029, 1.033]
```

### What to Watch

**Loss** should start at ~10.8 (ln(vocab_size)) and drop below 7.0 within the TinyStories phase. If it plateaus above 9.0, something is fundamentally wrong.

**tok/s** measures throughput. On an H100 with this model size and standard SDPA, expect 80K–150K+ tokens/second after compilation finishes.

**Intra-block similarity** tracks cosine similarity of Block 0’s adapter_b matrices across loops. This is the primary scientific metric: it measures whether the recursive loops are differentiating. Expected trajectory: ~1.0 initially, spreading to 0.2–0.5 as each loop specialises into a distinct “layer.”

**Inter-block similarity** tracks Block A vs Block B at each loop. Measures whether the two physical blocks develop complementary roles. Should diverge independently from intra-block similarity.

**Loop RMS** tracks activation root-mean-square after each complete loop. In a recursive architecture, any spectral norm > 1.0 compounds exponentially (1.1^6 ≈ 1.77, 1.5^6 ≈ 11.4). If later loops show RMS values growing unboundedly, reduce adapter_alpha or add activation clamping.

### Failure Modes

|Symptom                              |Likely Cause             |Fix                                        |
|-------------------------------------|-------------------------|-------------------------------------------|
|Loss plateaus at ~10.8               |Model not learning       |Check data pipeline is yielding real tokens|
|Loss spikes to NaN                   |Gradient explosion       |Reduce lr, increase warmup                 |
|Loop RMS explodes                    |Adapter norms too large  |Reduce adapter_alpha                       |
|Intra-block sims stay at 1.0         |Loops not differentiating|Increase adapter_rank or lr                |
|tok/s much lower than expected       |torch.compile fallback   |Check TORCH_LOGS for perf_hints warnings   |
|Loss flatlines after phase transition|Dataset format mismatch  |Verify smoltalk messages column handling   |

-----

## Checkpointing and Resume

### Periodic Checkpoints

Every 2,000 steps, a full checkpoint is saved to the Modal persistent volume:

```
/vol/checkpoints/osrt100m_step_2000.pt
/vol/checkpoints/osrt100m_step_4000.pt
...
```

Each checkpoint contains `step`, `model_state_dict`, and `optimizer_state_dict`.

### 24-Hour Rescue

Modal limits serverless functions to 24 hours. At 82,800 seconds (23 hours), the trainer saves a rescue checkpoint and exits cleanly:

```
/vol/checkpoints/osrt100m_rescue.pt
```

Re-running `modal run train_osrt_baseline.py` automatically detects and resumes from this checkpoint. The 1-hour buffer (82,800s vs 86,400s) ensures the volume commit completes before Modal kills the process.

### torch.compile Compatibility

`torch.compile` wraps the model, mangling the state dictionary keys. All checkpoint save/load operations access `model._orig_mod` to get the unwrapped module, ensuring clean checkpoint semantics across compile boundaries.

-----

## VRAM Budget

Estimated memory usage on H100 80GB (micro-batch = 16):

|Component                              |GB        |
|---------------------------------------|----------|
|Model parameters (FP32)                |0.42      |
|Optimizer state (Lion, FP32 momentum)  |0.42      |
|Gradients (FP32)                       |0.42      |
|Activations (BF16, 12 effective layers)|~22.7     |
|CUDA context + compile workspace       |~3.0      |
|**Total**                              |**~27 GB**|
|**Headroom**                           |**~53 GB**|

The activation memory dominates because recursive weight sharing stores weights once but must retain activations at all 12 effective passes for backpropagation. The single largest allocation is the logits tensor: 16 × 2048 × vocab × 2 bytes ≈ 3.1 GB.

-----

## Infrastructure

### Modal Configuration

The script provisions:

- 1× NVIDIA H100 GPU
- Debian Slim container with Python 3.11
- PyTorch 2.10 stable + CUDA 12.8
- Persistent volume `osrt-checkpoints` for checkpoint storage
- 24-hour function timeout
- `TORCH_LOGS=perf_hints` environment variable for compiler diagnostics

### Why Modal?

Modal provides on-demand H100 access at $3.95/hr without long-term commitments, persistent volumes that survive across function invocations, and a container model that handles dependency management. The 24-hour function limit requires the rescue checkpoint mechanism but is otherwise transparent.

### Dependencies

```
torch==2.10.0+cu128
transformers
datasets
lion-pytorch
triton
```

All torch imports are kept inside the `train()` function to prevent import errors on local machines that don’t have PyTorch installed. Modal parses the entire file locally before sending to the container.

-----

## Design Decisions and Trade-offs

### Why Not Sparse Attention at 2048?

DeepSeek’s NSA benchmarks (ACL 2025 Best Paper) show sparse attention is 2.4× *slower* than FlashAttention at 2048 tokens. The full N×N attention matrix at this length is only 4.2M elements per head and fits entirely in GPU SRAM. Block mask machinery adds pure overhead. NSA becomes beneficial at ~8K+ tokens. The sparse mask code is preserved in comments for future long-context scaling.

### Why Lion Over AdamW?

Lion halves optimizer VRAM by eliminating variance tracking. At 104.5M parameters, this saves ~400MB — marginal on an H100, but Lion also showed competitive or superior validation perplexity on 110M parameter models in the original paper. The sign-based update mechanism requires careful hyperparameter tuning (higher weight decay, longer warmup) but is well-validated at this scale.

### Why Not a Smaller Vocabulary?

The GPT-NeoX tokenizer (50K+ tokens) dedicates ~61% of parameters to the embedding matrix. A 32K vocabulary would free ~14M parameters for transformer layers. The current choice prioritises ecosystem compatibility with the Pythia model suite over parameter efficiency. This is a known trade-off, not an oversight.

### Why “Adapters” Not “LoRA”?

The mechanism modulates activations directly (`x + scale * (x @ A @ B)`), not weight matrices. Standard LoRA (Hu et al.) applies low-rank updates to W_q, W_k, etc. Calling this “LoRA” would be technically inaccurate and confuse anyone familiar with the literature. The term “per-pass residual adapters” is descriptively precise.

### What Does “302M Equivalent” Actually Mean?

Recursive weight sharing gives reasoning depth (12 unrolled layers of iterative refinement), not knowledge capacity. Research shows weight-shared models store ~2 bits/parameter regardless of loop count. The honest framing: “104.5M parameters with the reasoning depth of a 12-layer model, competitive with 2–3× larger dense models on reasoning tasks, but with 104.5M parameters of factual storage capacity.”

-----

## Version History

### v3.2 (Current)

- Lion weight decay: 0.01 → 0.3 (Lion paper 3–10× rule)
- Warmup: 500 → 2,000 steps (sign-based optimizer stability)
- Sparse NSA → standard causal SDPA (2.4× faster at 2048 tokens)
- AdamW fallback uses own wd=0.1
- PyTorch 2.10 stable + CUDA 12.8

### v3.1

- RoPE positional encodings added (model was position-blind)
- FP32 master weights (preventing BF16 underflow)
- Dynamic vocab size from tokenizer, TC-aligned
- SmolTalk instruction formatting with EOS injection
- Async DataLoader with dynamic shuffle seed
- `.item()` graph breaks removed from compiled forward
- Orphaned adapter residual fixed (`x_mod`, not `x`)
- GPT-standard weight init (std=0.02)
- Per-head sparse offset for future long-context work

### v3.0

- Renamed “LoRA” to “adapters” (semantic precision)
- Added adapter_scale = alpha/rank
- Gradient accumulation (effective batch 64)
- Dataset shuffle (buffer_size=10,000)
- Pinned memory transfers
- AdamW fallback option
- Conservative 23h rescue boundary

### v2.0

- Scaled from 46M to 104.5M (dim 768→1280, 1→2 blocks)
- 2-block model with inter/intra-block divergence telemetry
- Phantom vocab slice (logits[:, :real_vocab_size])
- Pinned nightly → pinned stable PyTorch version
- TORCH_LOGS=perf_hints for compiler diagnostics
- Curriculum acceleration (TinyStories 20k→8k steps)

### v1.0

- Initial baseline: 1 block × 6 loops, 46M parameters
- Real tokenization pipeline (replacing torch.randint)
- Removed GradScaler (unnecessary with BF16)
- Lion optimizer
- Checkpoint resume across Modal 24h boundary

-----

## Future Work

These additions are gated on the baseline loss curve converging first.

**Vectorised Micro-MoE.** Replace the dense SwiGLU with 256 tiny experts using `torch.scatter_add` or MegaBlocks for parallel dispatch. The Python loop from the original whitepaper was identified as a fatal bottleneck.

**Mamba Hybrid Layers.** Add selective state space model sub-layers for linear-time sequence processing. Requires building `mamba-ssm` from source in a multi-stage Dockerfile, with `torch.compiler.disable` to prevent CUDA stream races with `torch.compile`.

**Long-Context Scaling.** Increase seq_len to 4096–8192 with the NSA sparse mask reintroduced (where it actually provides speedup). Requires retuning RoPE theta and the sparse mask window/stride parameters.

**System-2 RL Phase.** Freeze base weights, unfreeze only adapters and potential MoE router, apply GRPO on math/logic datasets to carve reasoning pathways through the recursive layers.

-----

## Running the Training

### Prerequisites

- [Modal](https://modal.com/) account with GPU access
- Modal CLI installed (`pip install modal`)
- Modal token configured (`modal token new`)

### Deploy

```bash
modal run train_osrt_baseline.py
```

### Monitor

Training logs are streamed to your terminal. Look for:

1. **Compilation pause** (10–20 min): `torch.compile` autotuning. Do not kill the process.
1. **First loss print**: Should be ~10.8. If it’s 100+, weight init failed.
1. **Loss below 7.0**: Confirms the model is learning from TinyStories.
1. **Adapter divergence**: Intra-block sims spreading from 1.0 → 0.2–0.5 proves recursive depth.
1. **Loop RMS stability**: Values should stay in the 0.5–2.0 range. Exponential growth = trouble.

### Resume After Timeout

Simply re-run the same command. The script detects the rescue checkpoint and continues from the last step.

### Switch to AdamW

Edit Config:

```python
optimizer_name: str = "adamw"
```

The AdamW fallback uses its own hyperparameters (wd=0.1, betas=(0.9, 0.95)).

### Adjust Batch Size

If VRAM is tight (unlikely on H100), reduce micro-batch and increase accumulation:

```python
batch_size: int = 8
grad_accum_steps: int = 8  # effective batch stays at 64
```

This halves activation memory while maintaining the same effective batch size.

-----

## License

This is a research project. The training script, architecture, and documentation are provided as-is for educational and experimental purposes.
