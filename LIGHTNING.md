# Pretraining on Lightning AI

This guide gets you from "blank Lightning Studio" to "pretrain resumed
from the latest `osrt_v5_step_N.pt` checkpoint" in under 15 minutes.
The training code itself (`src/nano_osrt/train.py`) is platform-
agnostic; this doc just walks the bootstrap.

## GPU pick

Lightning brokers compute across providers (Lightning Cloud, Nebius,
GCP, AWS) and the per-credit price varies a lot. As of last check:

| Provider | GPU       | cr/hr  | tok/s est. | Notes                              |
|----------|-----------|--------|------------|------------------------------------|
| **Nebius** | **H100 80GB** | **3.01** | **~40 k**  | **Best value — pick this**           |
| GCP      | H100 80GB | ~5.17  | ~40 k      | Interruptible, more expensive      |
| AWS      | H100 80GB | ~1.99  | ~40 k      | Only sold in 8× packs (unusable)   |
| Lightning | A100 40GB | ~1.49  | ~25 k      | **Will OOM** — workload needs 49 GB |

**Pick H100 80GB on Nebius, on-demand, quantity 1.** The A100 40GB
looks cheap but our workload uses ~49 GB VRAM (forward + Muon momentum
buffers + AdamW moments) — it will OOM at the first batch. Don't be
tempted by the price.

For our workload (single GPU, 49 GB VRAM, no multi-node all-reduce),
Hopper's higher memory bandwidth doesn't help — we're compute-bound at
batch 8×8×2048, not memory-bound. Pick H100 over H200 unless H100 is
unavailable in your region.

## Step-by-step

### 1. Spin up a Studio

- New Studio → **AI development** template (ships with PyTorch + CUDA).
- Compute → **H100 80GB on Nebius**, quantity 1, on-demand.
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

Three ways, pick one:

**A. `.env` file in the repo root** (auto-loaded by `train_main.py`):
```bash
cat > .env <<'EOF'
WANDB_API_KEY=...   # from https://wandb.ai/authorize
HF_TOKEN=...        # lifts HF rate limits; required for gated datasets
EOF
```
Variable names matter — use `WANDB_API_KEY` exactly (wandb's library
reads that specific name). `.env` is gitignored.

**B. Shell exports** (per-session):
```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
```

**C. Lightning Studio Secrets manager** (survives Studio restarts):
- Studio → Settings → Secrets → add `WANDB_API_KEY` and `HF_TOKEN`
- Lightning auto-exposes them as env vars in every shell.

A is most convenient for one-off runs; C is best for repeat use.

### 4. Sync the tokenizer + checkpoint

The tokenizer (~3 MB, 3 files) and the latest checkpoint (~3 GB) both
live on the Modal Volume `osrt-checkpoints`. Use the Modal CLI from
inside the Studio — object-store to object-store at multi-Gbps,
typically ~3 min for the ckpt vs ~30 min uploading from your laptop.

```bash
uv pip install modal       # already in pyproject deps but keep this for clarity
modal token set --token-id <id> --token-secret <secret>   # from modal.com/settings/tokens

mkdir -p tokenizer checkpoints/v5
uv run modal volume get osrt-v4-tokenizer / ./tokenizer/

# Find the latest checkpoint and pull it. Listing first lets you skip
# stale step files if your last run produced several:
uv run modal volume ls osrt-checkpoints v5/
uv run modal volume get osrt-checkpoints v5/<latest-step-file>.pt ./checkpoints/v5/
```

The auto-resume logic in `train.py` picks the highest
`osrt_v5_step_N.pt` in `--ckpt-dir`, so you only need the most recent
file — older ones aren't required.

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
`--ckpt-dir`. The first non-W&B-noise log line will be
`Found checkpoint at step <N> ... Resumed at step <N+1>` where `<N>`
is whatever you synced down in step 4.

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

## Known issues + fixes (already in the codebase)

These bit during real Lightning runs and are now patched. Calling them
out so future readers don't re-debug:

- **`PyGILState_Release` at the foundation→knowledge phase boundary
  (step 9.5k–10k).** With `persistent_workers=True`,
  `multiprocessing_context="spawn"`, and live HF streaming connections
  (aiohttp/fsspec), the old `del current_loader` rebinding raced
  worker teardown against new-worker spawn. Fix lives at
  `train.py:907` — explicitly null `loader_iter` then `current_loader`
  and `gc.collect()` before building the new loader. Same race in
  `run_eval` is fixed at `train.py:255`.
- **`torch.compile` + `gradient_checkpoint(context_fn=...)` raises
  `NotImplementedError` from Dynamo's higher-order-op tracer.** Fixed
  at `model.py::_checkpoint_block` by wrapping the call in
  `@torch.compiler.disable` so just the checkpoint dispatch falls back
  to eager. The block_fn itself is still compiled.
- **Phase 1→2 transition recompiles the model** because seq_len jumps
  from 2048 → 4096. Expect a ~2-3 min compile freeze around
  step 9.5k. The first batch after compile takes longer than usual;
  step throughput recovers within ~50 steps.

## When the run finishes

Pull the final checkpoint back to your laptop:

```bash
# from your laptop:
lightning download <studio-name> /teamspace/studios/this_studio/.../osrt_v5_final.pt ./checkpoints/v5/
```

Or — easier — push it back to the Modal volume from inside the Studio
so all your checkpoints live in one place:

```bash
uv run modal volume put osrt-checkpoints \
    ./checkpoints/v5/osrt_v5_final.pt v5/osrt_v5_final.pt
```

Then `modal volume get` from anywhere later.
