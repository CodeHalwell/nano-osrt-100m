# Running on a local 3090 (or similar)

The training code is platform-agnostic — same `train_main.py` entry
point that works on Lightning AI runs locally too. This guide
documents what's actually viable on a single 24 GB consumer GPU and
what isn't.

## What fits on 24 GB

| Stage | Fits? | Notes |
|-------|-------|-------|
| Sanity (1200 steps, Phase-1 sizes) | ✓ | ~7 hours wall-clock |
| Ablate (4 cells × 1200 steps) | ✓ | ~28 hours wall-clock |
| Pretrain Phase 1 (seq 2048, batch 8) | ✓ | ~3 weeks for 10k steps |
| Pretrain Phase 2 (seq 4096, batch 4) | ✓ | ~16 months for full 240k |
| Pretrain Phase 3 (seq 8192, batch 2) | ✓ | needs grad checkpointing |
| SFT (seq 2048, batch 8 + HRA) | ✓ | ~1-2 days for 5k steps |
| GRPO (seq 8192, batch 4, ref model) | tight | ref-model copy + generation push it close to 24 GB |
| Inference / eval | ✓ | trivial — 363M bf16 = ~726 MB weights |

**Use the 3090 for sanity + SFT + inference. Use cloud for full
pretraining** (Modal H100 / Lightning Nebius — see `LIGHTNING.md`).
Phase 1 alone takes 3 weeks on the 3090 vs ~7 hours on H100.

## Why not just train everything locally

| Card | BF16 TFLOPs | tok/s on this workload |
|------|-------------|------------------------|
| H100 SXM | ~989 | ~40 k |
| RTX 3090 | ~71 (~14× slower) | ~3 k (Phase 2 sizes) |

You've already paid for the 3090 so per-run electricity is ~$2.50/day
at 350 W and $0.30/kWh. The bottleneck is wall-clock, not money.

## Setup

### 1. Verify GPU + CUDA

```bash
nvidia-smi                     # confirm 3090 visible, driver ≥ 525
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` returns `False`, install a CUDA-enabled
PyTorch wheel:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. Clone + deps

```bash
git clone https://github.com/CodeHalwell/nano-osrt-100m.git
cd nano-osrt-100m
uv sync
```

### 3. Sync the tokenizer + latest checkpoint from Modal

Same workflow as the Lightning Studio bootstrap — `modal volume get`
from the Modal volume:

```bash
mkdir -p tokenizer checkpoints/v5
uv run modal volume get osrt-v4-tokenizer / ./tokenizer/
uv run modal volume ls osrt-checkpoints v5/                           # see latest step
uv run modal volume get osrt-checkpoints v5/<latest>.pt ./checkpoints/v5/
```

### 4. Set secrets

`.env` in the repo root works (auto-loaded by `train_main.py`):

```
WANDB_API_KEY=...
HF_TOKEN=...
```

## What you can run today

### Sanity / ablate / pretrain (resumed)

```bash
uv run python -m nano_osrt.train_main \
    --tokenizer-path ./tokenizer \
    --ckpt-dir ./checkpoints/v5 \
    --total-steps 1200 \
    --wandb-run-name osrt-sanity-3090
```

Auto-resumes from the highest `osrt_v5_step_N.pt` in `--ckpt-dir`. To
start fresh, point at an empty directory.

For a longer pretrain segment (mostly useful for resumed runs that
just need a few thousand more steps):

```bash
uv run python -m nano_osrt.train_main \
    --tokenizer-path ./tokenizer \
    --ckpt-dir ./checkpoints/v5
```

### Tests

```bash
uv run pytest tests/ -q
```

83 tests should pass. Useful smoke check after pulling new commits.

### Inference

There's no dedicated inference script yet — drop in a quick loader
against the model class. PyTorch's `model.train(False)` is the
inference-mode toggle (functionally identical to the more common
`.eval()` method on `nn.Module`):

```python
import torch
from transformers import AutoTokenizer
from nano_osrt.config import NanoOSRTConfig
from nano_osrt.model import NanoOSRTForCausalLM

tok = AutoTokenizer.from_pretrained("./tokenizer")
cfg = NanoOSRTConfig(
    vocab_size=len(tok), real_vocab_size=len(tok),
    bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)

model = NanoOSRTForCausalLM(cfg).cuda()
ckpt = torch.load(
    "./checkpoints/v5/osrt_v5_step_<N>.pt",
    map_location="cuda", weights_only=True,
)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.train(False)             # disables MoE capacity drops, enables KV-cache path

prompt = "<|user|>What is the capital of France?<|assistant|>"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=128, temperature=0.7, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=False))
```

Decode speed: ~50-100 tok/s on 3090 at default sampling.

## What doesn't have a local entry point yet

The Modal app (`app.py`) defines `sft` and `grpo` functions that are
only callable via `modal run`. To run SFT or GRPO on the 3090, the
codebase would need a `train_main_sft.py` (and `train_main_grpo.py`)
mirror that wraps `nano_osrt.sft_train.run_sft` and
`nano_osrt.train_main` patterns the same way. ~30 lines each.

This isn't done yet. Worth adding when you actually need to run SFT
locally — the wrappers are mechanical, just argparse + `_LocalVol`
stub + a tokenizer load.

## Memory budget at common configs

Measured / estimated VRAM use (model weights bf16 + Muon momentum
fp32 + AdamW state + activations):

| Config | seq_len × batch × accum | Activations (no ckpt) | Total VRAM |
|--------|-------------------------|------------------------|------------|
| Phase 1 | 2048 × 8 × 8 | ~10 GB | ~15 GB |
| Phase 2 | 4096 × 4 × 16 | ~12 GB | ~17 GB |
| Phase 3 | 8192 × 2 × 32 | ~24 GB | **needs ckpt** |
| SFT (default) | 2048 × 8 × 8 | ~10 GB | ~14 GB (+22 MB HRA) |

Phase 3 on 24 GB needs gradient checkpointing. The threshold in
`train.py` was bumped to `seq_len ≥ 8192` in commit `57513a9` so it
auto-engages there.

## Tips

- Keep the room cool — 350 W under sustained load runs the GPU at
  75-80 °C even with a good cooler. Open the case if temps creep.
- `nvidia-smi dmon -s u` in another terminal lets you watch
  utilisation live; should sit at 90+ % during the inner training
  loop.
- If you see frequent `CUDA out of memory`, drop `batch_size` in
  `train_config.py::PretrainConfig.phases[<phase>]` rather than
  re-enabling grad checkpointing — smaller batch costs less throughput
  than the 50 % checkpointing penalty.
- Lightning W&B and HF datasets streaming both work the same as on
  the cloud — same `.env` vars, same dataset URLs. The first eval
  pays the ~15 min FineWeb skip; subsequent evals are cached.
