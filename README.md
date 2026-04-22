# nano-osrt

Recursive Mixtral-style MoE transformer. 363M physical params, ~192M active per token (top-2 of 8 routed experts + 1 shared), ~1.15B effective compute via recursive weight sharing (3 physical blocks × 6 loops).

## Architecture

- **Dense FFN: removed.** Shared expert (SwiGLU, hidden=4096) replaces it as the always-on path.
- **Routed experts:** 8 × SwiGLU(hidden=2048), top-2 softmax with renormalised gates.
- **Balance loss:** Switch-style `E · Σ(f·p)` — penalises imbalance without forcing uniformity.
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
├── train.py          # Pretrain loop (Lion, Gumbel anneal, 5k early-stop gate)
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
modal run --detach app.py --stage pretrain   # Full curriculum (Foundation → Knowledge → Instruction)
modal run --detach app.py --stage sft        # Balanced SFT on the pretrained checkpoint
modal run --detach app.py --stage grpo       # GRPO RL with verifiable math rewards
```

## Testing

```bash
uv run pytest tests/ -q
```
