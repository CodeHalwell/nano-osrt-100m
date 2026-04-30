# Pretraining on Lightning AI

This guide gets you from "blank Lightning Studio" to "pretrain resumed
from the Modal step-3500 checkpoint" in under 15 minutes. The training
code itself (`src/nano_osrt/train.py`) is platform-agnostic; this doc
just walks the bootstrap.

## Why Lightning over Modal for this project

For our workload (single GPU, 49 GB VRAM, no multi-node all-reduce):

| Platform  | GPU       | Hourly  | tok/s est. | tokens / credit |
|-----------|-----------|---------|------------|-----------------|
| Modal     | H100 SXM  | $3.95   | 40 k       | ~10 k           |
| Lightning | H100 SXM  | ~$3.49  | 40 k       | ~11.5 k         |
| Lightning | A100 80GB | ~$1.49  | ~25 k      | **~17 k**       |

A100 80GB on Lightning is the cheapest credit/token for this exact
workload — Hopper's higher memory bandwidth doesn't help us at batch
8×8×2048 because we're compute-bound, not memory-bound.

## Step-by-step

### 1. Spin up a Studio

- New Studio → **A100 80GB** (or H100 if you want maximum throughput).
- Storage: default (Lightning Studios come with persistent disk).

### 2. Clone and install

```bash
git clone https://github.com/<your-fork>/nano-osrt-100m.git
cd nano-osrt-100m
git checkout v5/review-fixes  # or main, once merged

# Lightning Studios ship with `uv` pre-installed; if not:
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync                          # installs pinned deps from uv.lock
```

### 3. Set secrets

In the Studio's "Secrets" tab (or just export in shell):

```bash
export WANDB_API_KEY=...   # from https://wandb.ai/authorize
export HF_TOKEN=...        # if your HF datasets need auth (FineWeb-Edu doesn't)
```

### 4. Sync the tokenizer + checkpoint

The tokenizer (~3 MB, 3 files) and the latest checkpoint (~2.9 GB)
both live on the Modal Volume. Two options:

**Option A — Lightning Studio web upload (easiest for one-shot):**
- Drop your local `./tokenizer/` and `./checkpoints/v5/osrt_v5_step_3500.pt`
  into the Studio file browser. The 2.9 GB ckpt takes a few minutes
  over a residential connection.

**Option B — Modal CLI from inside the Studio:**

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>   # from modal.com/settings

mkdir -p tokenizer checkpoints/v5
modal volume get osrt-v4-tokenizer / ./tokenizer/
modal volume get osrt-checkpoints v5/osrt_v5_step_3500.pt ./checkpoints/v5/
```

Option B is faster end-to-end (Modal volumes serve from object storage
at multi-Gbps to the Studio) and avoids the round-trip through your
laptop.

### 5. Verify the bootstrap

```bash
uv run python -m nano_osrt.train_main --help
uv run pytest tests/ -q          # 83 tests should pass
```

### 6. Launch pretrain

```bash
uv run python -m nano_osrt.train_main \
    --tokenizer-path ./tokenizer \
    --ckpt-dir ./checkpoints/v5 \
    --wandb-run-name osrt-pretrain-muon-lightning
```

The script auto-resumes from the highest `osrt_v5_step_N.pt` in
`--ckpt-dir`, so the first thing you'll see in the log is
`Found checkpoint at step 3500 ... Resumed at step 3501`.

To start fresh (e.g. on a sanity dir), point `--ckpt-dir` somewhere
empty.

### 7. Run it in a tmux / screen session

Lightning Studios disconnect on browser close but the underlying VM
keeps your processes alive *only* if they're detached from the SSH
session. Easiest pattern:

```bash
tmux new -s pretrain
# inside tmux:
uv run python -m nano_osrt.train_main \
    --tokenizer-path ./tokenizer \
    --ckpt-dir ./checkpoints/v5 \
    --wandb-run-name osrt-pretrain-muon-lightning
# Ctrl-b d to detach. tmux ls / tmux a -t pretrain to reattach later.
```

Or use `nohup ... &` if you don't want tmux. The 23h Modal-style
rescue checkpoint loop in `train.py` still fires here, so even if the
Studio is killed at hour 23 the next launch resumes from the rescue
ckpt.

## What the entry point does NOT handle

- **Multi-GPU training.** `run_training` is single-GPU. Distributed
  setup would need `torchrun` + DDP wiring around `model` and the
  bias-controller buffer all-reduce flagged in the v5 review notes.
- **Auto-shutdown when out of credits.** Lightning bills per-second of
  GPU uptime; set a budget alert in the Studio settings if you want a
  hard stop.

## When the run finishes

Pull the final checkpoint back to your laptop:

```bash
# from your laptop:
lightning download <studio-name> /teamspace/studios/this_studio/.../osrt_v5_final.pt ./checkpoints/v5/
```

Or upload to HF Hub and `huggingface-cli download` from anywhere.
