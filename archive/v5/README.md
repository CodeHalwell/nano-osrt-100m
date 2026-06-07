# nano-osrt

Recursive Mixtral-style MoE transformer. 363M physical params, ~192M active per token (top-2 of 8 routed experts + 1 shared), ~1.15B effective compute via recursive weight sharing (3 physical blocks × 6 loops).

## Architecture

- **Dense FFN: removed.** Shared expert (SwiGLU, hidden=4096) replaces it as the always-on path.
- **Routed experts:** 8 × SwiGLU(hidden=2048), top-2 softmax with renormalised gates.
- **Balance loss:** Switch-style `E · Σ(f·p)` — penalises imbalance without forcing uniformity.
- **Balance bias controller:** persistent per-loop/per-expert routing bias pushes overloaded experts down and cold experts up.
- **Router exploration:** Gumbel top-k noise, tau 0.5 → 0 over 4000 steps (outlasts LR warmup so cold experts can't die during the peak-LR window).
- **Orthogonal per-expert init:** breaks symmetry at step 0, survives HF `post_init()`.
- **Eval-time drop-free capacity:** chunk-stable inference by construction.
- **KV-cached generate:** prefill + decode with per-effective-layer cache.

See `ARCHITECTURE.md` for full design rationale and the ablation sequence that led here.

## Layout

```
src/nano_osrt/
├── config.py         # NanoOSRTConfig (dims, routing, curriculum)
├── model.py          # NanoOSRTModel / NanoOSRTForCausalLM, MoE dispatch, generate
├── data.py           # Streaming pretrain loader (FineWeb-Edu, CodeParrot, Wikipedia)
├── sft_data.py       # SFT data with packing + structural-tag format
├── train.py          # Pretrain loop (Muon hybrid, Gumbel anneal, 5k early-stop gate)
├── train_config.py   # PretrainConfig / SFTConfig / GRPOConfig
├── sft_train.py      # HRA-adapted SFT loop
├── hra.py            # High-rank adapters (SFT capacity injection)
└── rewards.py        # GRPO reward functions (verifiable math)
app.py                # Modal entrypoint (sanity / sweep / pretrain / sft / grpo)
tests/test_model.py   # 83 unit tests (routing, cache, resume, gradient flow)
archive/              # v1/v2/v3/v4 code preserved for reference
```

## Stages

```bash
modal run --detach app.py --stage sanity     # 1200-step smoke test
modal run --detach app.py --stage sweep      # Gumbel schedule sweep
modal run --detach app.py --stage ablate     # Optimizer × routing 2x2 (cells A/B/C/D, 1200 steps each)
modal run --detach app.py --stage pretrain   # Full curriculum (Foundation → Knowledge → Instruction)
modal run --detach app.py --stage sft        # Balanced SFT on the pretrained checkpoint
modal run --detach app.py --stage grpo       # GRPO RL with verifiable math rewards
```

## Testing

```bash
uv run pytest tests/ -q
```

## Roadmap

The architectural decisions in v5 (Muon hybrid, MoE without dense FFN,
bias-controller routing, Z-loss + QK-Norm + softplus moe_gate) are
settled by the `--stage ablate` evidence. From here the plan is
deliberately small — pretrain → SFT → optional inference work — and
explicitly avoids chasing every new paper.

1. **Finish pretraining.** Currently mid-Phase 2 (Knowledge,
   seq_len 4096). Phase 3 (Instruction, seq_len 8192) afterwards.
   Resume contract handles partial-budget runs across providers
   (Modal, Lightning AI) — see `LIGHTNING.md`.
2. **SFT** the pretrained checkpoint with HRA adapters
   (`--stage sft`). The HRA path injects ~90M extra parameters of
   capacity without retraining the base. Balanced math + code +
   STEM + general mixture, native single-token tag chat format.
3. **(Optional) GRPO** on the SFT checkpoint with verifiable math
   rewards (`--stage grpo`). Single-GPU, group-normalised
   advantages, Schulman's unbiased KL approximation.
4. **(Optional) Multi-Token Prediction (MTP) head for speculative
   decoding.** A small extra Transformer block on the LM head trained
   to predict t+2/t+3 tokens. Doubles as a draft model at inference
   time, giving 2-3× decode speedup on greedy / low-temperature
   generation. Worth the complexity only if we're deploying inference
   at scale; otherwise skip.

### Explicitly NOT on the roadmap

The DeepSeek V4 paper (Apr 2026) introduced compressed sparse
attention (CSA / HCA), DSA top-k routing on compressed entries,
inverse-RoPE for shared KV, and Birkhoff-polytope-constrained
residual mixing. All of these solve "1M-token context is too
expensive" or "very deep residual stacks are unstable" — they are
real and clever, but our setup hits neither problem:

- max_position_embeddings is 8192. We never operate at long context.
  Implementing CSA/HCA would invalidate the pretrained checkpoint
  for no measurable gain.
- We don't share KV across heads or layers. Inverse RoPE is a
  no-op for our Q/K/V-per-head MHA.
- 18 effective layers (3 physical × 6 loops) with simple additive
  residuals show no instability through 11k+ measured steps
  (`bias_abs_max ≈ 0.25`, `clean_marg ≈ 2.07`, no loss spikes). Until
  we observe instability, mHC residuals add Sinkhorn cost for no
  benefit.

The general principle: at 363M params on a single H100, the wins
are in `tok/s` and `tokens/$` (gradient-checkpointing thresholds,
batch sizing, Muon LR tuning) — not in copying frontier-scale
architectural tricks that target problems we don't have.
