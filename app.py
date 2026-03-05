# “””
Nano-OSRT 100M — Production Trainer (v3.2)

104.5M physical parameters simulating 302M equivalent dense via
recursive weight sharing. 2 physical blocks x 6 recursive loops = 12
effective layers, each with unique per-pass residual adapters.

Changes in v3.2:

- Lion weight decay 0.01 -> 0.3 (Lion paper: 3-10x higher than AdamW)
- Warmup 500 -> 2000 steps (sign-based optimizers need longer warmup)
- Sparse NSA replaced with standard causal SDPA (FlashAttention-2 is
  2.4x faster than NSA at 2048 tokens per DeepSeek benchmarks)
- AdamW fallback uses its own wd=0.1, not Lion’s 0.3
- PyTorch 2.10 stable + CUDA 12.8

Prior fixes (v3.0-3.1):

- RoPE positional encodings
- FP32 master weights (preventing BF16 underflow)
- Dynamic vocab size from tokenizer, TC-aligned
- Smoltalk instruction formatting with EOS injection
- Async DataLoader with dynamic shuffle seed
- .item() graph breaks removed from compiled forward
- Orphaned adapter residual fixed (x_mod, not x)
- GPT-standard weight init (std=0.02)
- Per-head sparse offset preserved in comments for future long-context

Deploy:
modal run train_osrt_baseline.py
Resume after 24h timeout:
(automatically resumes from latest rescue checkpoint)
“””

import os
import math
import time
import modal

# NOTE: torch imports are deliberately kept INSIDE train() because Modal

# parses this file locally before sending to the container. If torch isn’t

# installed on the local machine, top-level imports would crash.

# =============================================================================

# 1. MODAL INFRASTRUCTURE

# =============================================================================

app = modal.App(“nano-osrt-100m”)

image = (
modal.Image.debian_slim(python_version=“3.11”)
.apt_install(“git”, “build-essential”)
.env({“TORCH_LOGS”: “perf_hints”})
.pip_install(
“torch==2.10.0+cu128”,  # Stable release — no nightly roulette
“–index-url”, “https://download.pytorch.org/whl/cu128”,
)
.pip_install(“transformers”, “datasets”, “lion-pytorch”, “triton”)
)

vol = modal.Volume.from_name(“osrt-checkpoints”, create_if_missing=True)

# =============================================================================

# 2. CONFIGURATION

# =============================================================================

class Config:
dim: int = 1280
heads: int = 20
head_dim: int = 64
seq_len: int = 2048

```
# Overwritten dynamically at runtime from tokenizer
vocab_size: int = -1
real_vocab_size: int = -1

num_blocks: int = 2
recursive_loops: int = 6

adapter_rank: int = 16
adapter_alpha: float = 16.0

batch_size: int = 16
grad_accum_steps: int = 4
total_steps: int = 150_000
warmup_steps: int = 2000  # Lion needs longer warmup (sign-based updates are chaotic before momentum stabilises)
peak_lr: float = 1e-4
min_lr: float = 1e-5
weight_decay: float = 0.3  # Lion paper: 3-10x higher than AdamW's 0.1; effective wd = lr*lambda = 3e-5
grad_clip: float = 1.0
log_interval: int = 50
ckpt_interval: int = 2000
optimizer_name: str = "lion"

# Tokens/step = batch_size * seq_len * grad_accum = 16 * 2048 * 4 = 131,072
# Phase 1: 8k steps * 131,072 = ~1.05B tokens
phases = {
    "tinystories": {"start": 0, "end": 8_000, "dataset": "roneneldan/TinyStories"},
    "fineweb": {"start": 8_000, "end": 140_000, "dataset": "HuggingFaceFW/fineweb-edu"},
    "smoltalk": {"start": 140_000, "end": 150_000, "dataset": "HuggingFaceTB/smoltalk"},
}
```

# =============================================================================

# 3. MODEL & TRAINING (all torch imports inside for Modal compatibility)

# =============================================================================

@app.function(
gpu=modal.gpu.H100(count=1),
image=image,
volumes={”/vol/checkpoints”: vol},
timeout=86400,
)
def train():
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from lion_pytorch import Lion
from transformers import AutoTokenizer
import itertools

```
cfg = Config()
device = torch.device("cuda")

# ------------------------------------------------------------------
# Tokenizer + dynamic vocab alignment
# ------------------------------------------------------------------
tokenizer_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

cfg.real_vocab_size = len(tokenizer)
cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

# ------------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------------
class TokenStream(IterableDataset):
    def __init__(self, dataset_name, seq_len, tok_name, seed):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset

        tok = AutoTokenizer.from_pretrained(self.tok_name)
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id

        ds = load_dataset(self.dataset_name, split="train", streaming=True)
        ds = ds.shuffle(buffer_size=10_000, seed=seed)

        if worker_info is not None:
            try:
                ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            except Exception:
                ds = itertools.islice(ds, worker_info.id, None, worker_info.num_workers)

        buffer = []
        for example in ds:
            # Handle Phase 3 instruction-tuning format (messages column)
            if "messages" in example:
                try:
                    text = tok.apply_chat_template(
                        example["messages"], tokenize=False
                    )
                except Exception:
                    # GPT-NeoX has no default chat template; manual fallback.
                    # EOS after assistant turns teaches the model to stop.
                    parts = []
                    for m in example["messages"]:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        parts.append(f"{role}: {content}")
                        if role == "assistant":
                            parts.append(tok.eos_token)
                    text = "\n".join(parts)
            else:
                text = example.get("text", "")

            if not text or not text.strip():
                continue

            tokens = tok.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(tok.eos_token_id)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

def make_loader(dataset_name, step_num):
    ds = TokenStream(dataset_name, cfg.seq_len, tokenizer_name, seed=42 + step_num)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
        persistent_workers=True,
    )

# ------------------------------------------------------------------
# RoPE utilities
# ------------------------------------------------------------------
def compute_rope_freqs(seq_len, dim, theta=10000.0, device=None):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # Shape: (1, S, 1, head_dim) for broadcasting with (B, S, heads, head_dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2)
    return cos, sin

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin

# ------------------------------------------------------------------
# Note on attention: NSA sparse masks were removed in v3.2.
# At 2048 tokens, full causal FlashAttention (via F.scaled_dot_product_attention)
# is 2.4x FASTER than sparse NSA masks (DeepSeek benchmarks). The N×N matrix
# at 2048 is tiny (~4.2M ops/head), fits entirely in GPU SRAM, and the block
# mask machinery adds pure overhead. NSA only becomes beneficial at ~8K+ tokens.
# The per-head sparse offset trick (((kv_idx // 64) + h) % 4) is preserved
# in comments for future long-context work where it would be valuable.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Model components
# ------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(dim * 8 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class RecursiveBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_attn = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_ffn = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, adapter_a, adapter_b, adapter_scale, rope_cos, rope_sin):
        B, S, D = x.shape

        # Per-pass low-rank residual adapter on activations.
        # NOT weight-LoRA (Hu et al.) — modulates hidden state directly.
        x_mod = x + adapter_scale * (x @ adapter_a @ adapter_b)

        # --- Causal Attention with RoPE ---
        h = self.norm_attn(x_mod)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.heads, self.head_dim)
        k = k.view(B, S, self.heads, self.head_dim)
        v = v.view(B, S, self.heads, self.head_dim)

        # RoPE applied before head transpose
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard causal SDPA — dispatches to FlashAttention-2 on H100.
        # At 2048 tokens this is 2.4x faster than sparse NSA masks.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Residual connects to x_mod (not x) so the adapter's geometric
        # shift survives into the FFN, not orphaned.
        x = x_mod + self.out_proj(attn_out)

        x = x + self.ffn(self.norm_ffn(x))
        return x

class NanoOSRT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Precomputed RoPE buffers (non-persistent = won't bloat checkpoints)
        cos, sin = compute_rope_freqs(cfg.seq_len, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList([
            RecursiveBlock(cfg.dim, cfg.heads) for _ in range(cfg.num_blocks)
        ])

        total_pairs = cfg.num_blocks * cfg.recursive_loops
        self.adapters_a = nn.ParameterList(
            [nn.Parameter(torch.randn(cfg.dim, cfg.adapter_rank) * 0.01)
             for _ in range(total_pairs)]
        )
        self.adapters_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(cfg.adapter_rank, cfg.dim))
             for _ in range(total_pairs)]
        )
        self.adapter_scale = cfg.adapter_alpha / cfg.adapter_rank
        self.norm_out = nn.RMSNorm(cfg.dim)

        # GPT-standard init: N(0, 0.02) prevents logit variance explosion
        # at step 0. Without this, nn.Embedding defaults to N(0, 1) which
        # with dim=1280 produces initial CE loss of 100+ instead of ~10.8,
        # sending a gradient shock that can permanently destabilise Lion.
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        cos = self.rope_cos[:, :S, :, :]
        sin = self.rope_sin[:, :S, :, :]

        # Always collect loop RMS as raw tensors (no .item() graph breaks).
        # Cost: one float cast + pow + mean + sqrt per loop. Acceptable
        # trade-off vs dual-graph compilation from return_stats toggling.
        loop_rms = []

        for loop in range(self.cfg.recursive_loops):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.cfg.num_blocks + block_idx
                x = block(
                    x, self.adapters_a[idx], self.adapters_b[idx],
                    self.adapter_scale, cos, sin,
                )
            loop_rms.append(x.float().pow(2).mean().sqrt())

        x = self.norm_out(x)
        logits = F.linear(x, self.embedding.weight)
        return logits, loop_rms

# ==================================================================
# Training setup
# ==================================================================
print("=" * 60)
print("Nano-OSRT 100M Trainer (v3.1)")
print("=" * 60)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# FP32 master weights — autocast handles bf16 during forward/backward.
# This prevents the bf16 underflow where lr=1e-4 rounds to zero
# against weights of magnitude ~0.05 (bf16 epsilon ~0.0078).
model = NanoOSRT(cfg).to(device=device)

total_params = sum(p.numel() for p in model.parameters())
adapter_params = (
    sum(p.numel() for p in model.adapters_a)
    + sum(p.numel() for p in model.adapters_b)
)
eff_batch = cfg.batch_size * cfg.grad_accum_steps
tok_per_step = eff_batch * cfg.seq_len
adapter_scale_val = cfg.adapter_alpha / cfg.adapter_rank

print(f"Vocab size          : {cfg.real_vocab_size} (padded to {cfg.vocab_size})")
print(f"Physical parameters : {total_params:>12,}")
print(f"  of which adapters : {adapter_params:>12,} ({adapter_params/total_params*100:.2f}%)")
print(f"Adapter scale       : {cfg.adapter_alpha}/{cfg.adapter_rank} = {adapter_scale_val:.1f}")
print(f"Effective depth     : {cfg.recursive_loops} loops x {cfg.num_blocks} blocks = {cfg.recursive_loops * cfg.num_blocks} layers")
print(f"Micro-batch         : {cfg.batch_size}")
print(f"Grad accum steps    : {cfg.grad_accum_steps}")
print(f"Effective batch     : {eff_batch}")
print(f"Tokens per step     : {tok_per_step:,}")
print(f"Total token budget  : ~{cfg.total_steps * tok_per_step / 1e9:.1f}B")
print(f"Optimizer           : {cfg.optimizer_name}")
print(f"Precision           : FP32 master weights, BF16 compute (autocast)")
print()

model = torch.compile(model, mode="max-autotune")

if cfg.optimizer_name.lower() == "adamw":
    # AdamW uses standard wd=0.1 (Lion's 0.3 is 3x higher per Lion paper's rule)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.peak_lr,
        weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8,
    )
    print("Using AdamW (fallback, wd=0.1)")
else:
    optimizer = Lion(
        model.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay,
    )
    print(f"Using Lion (sign-based, wd={cfg.weight_decay})")

def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * progress)
    )

def get_phase(step):
    for name, p in cfg.phases.items():
        if p["start"] <= step < p["end"]:
            return name, p["dataset"]
    return "fineweb", cfg.phases["fineweb"]["dataset"]

# ------------------------------------------------------------------
# Checkpoint resume
# ------------------------------------------------------------------
start_step = 0
rescue_path = "/vol/checkpoints/osrt100m_rescue.pt"
if os.path.exists(rescue_path):
    print(f"Resuming from {rescue_path}...")
    ckpt = torch.load(rescue_path, map_location=device, weights_only=False)
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"Resumed at step {start_step}")

# ==================================================================
# Training loop
# ==================================================================
current_phase = None
loader_iter = None
current_loader = None
start_time = time.time()
step = start_step

while step < cfg.total_steps:
    phase_name, dataset_name = get_phase(step)

    if phase_name != current_phase:
        current_phase = phase_name
        print(f"\n>>> Phase: {current_phase} | Dataset: {dataset_name} | Step: {step}")
        # Clean up old loader workers before creating new ones
        if current_loader is not None:
            del current_loader
        current_loader = make_loader(dataset_name, step)
        loader_iter = iter(current_loader)

    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = torch.tensor(0.0, device=device)
    last_loop_rms = None

    for micro in range(cfg.grad_accum_steps):
        try:
            input_ids, labels = next(loader_iter)
        except StopIteration:
            _, ds_name = get_phase(step)
            if current_loader is not None:
                del current_loader
            current_loader = make_loader(ds_name, step)
            loader_iter = iter(current_loader)
            input_ids, labels = next(loader_iter)

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loop_rms_tensors = model(input_ids)
            # Cast logits to fp32 before CE for numerical stability
            active_logits = logits[..., :cfg.real_vocab_size].contiguous().float()
            loss = F.cross_entropy(
                active_logits.view(-1, cfg.real_vocab_size),
                labels.view(-1),
            )
            scaled_loss = loss / cfg.grad_accum_steps

        scaled_loss.backward()
        # Detach avoids CPU-GPU sync; .item() called once at log time
        accum_loss += loss.detach() / cfg.grad_accum_steps

        if micro == cfg.grad_accum_steps - 1:
            last_loop_rms = [r.detach() for r in loop_rms_tensors]

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    # --- Logging ---
    if step % cfg.log_interval == 0:
        elapsed = time.time() - start_time
        tokens_so_far = (step - start_step) * tok_per_step
        tok_per_sec = tokens_so_far / elapsed if elapsed > 0 else 0
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            b_mats = [b.flatten().float() for b in inner_model.adapters_b]
            if b_mats[0].norm() > 1e-6:
                b0_vecs = [b_mats[i] for i in range(0, len(b_mats), cfg.num_blocks)]
                intra_sims = [
                    F.cosine_similarity(b0_vecs[0], v, dim=0).item()
                    for v in b0_vecs[1:]
                ]
                intra_str = "[" + ", ".join(f"{s:.2f}" for s in intra_sims) + "]"
                inter_sims = [
                    F.cosine_similarity(
                        b_mats[loop * cfg.num_blocks],
                        b_mats[loop * cfg.num_blocks + 1], dim=0,
                    ).item()
                    for loop in range(cfg.recursive_loops)
                ]
                inter_str = "[" + ", ".join(f"{s:.2f}" for s in inter_sims) + "]"
            else:
                intra_str = "[waking up...]"
                inter_str = "[waking up...]"

        # .item() calls safely outside compiled region
        rms_str = (
            "[" + ", ".join(f"{r.item():.3f}" for r in last_loop_rms) + "]"
            if last_loop_rms else "[n/a]"
        )

        print(
            f"step {step:>7d} | loss {accum_loss.item():.4f} | "
            f"lr {lr:.2e} | vram {vram_gb:.1f}GB | "
            f"tok/s {tok_per_sec:,.0f} | phase {current_phase}\n"
            f"           intra-block (b0: L0 vs L1..5): {intra_str}\n"
            f"           inter-block (b0 vs b1 per loop): {inter_str}\n"
            f"           loop RMS: {rms_str}"
        )

    # --- Checkpoints ---
    if step > 0 and step % cfg.ckpt_interval == 0:
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        path = f"/vol/checkpoints/osrt100m_step_{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        vol.commit()
        print(f"  -> checkpoint saved: {path}")

    # --- 23h Modal safety ---
    if time.time() - start_time > 82_800:
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({
            "step": step,
            "model_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, rescue_path)
        vol.commit()
        print(f"\n23h boundary. Rescue checkpoint at step {step}. Re-run to resume.")
        return

    step += 1

# --- Final ---
inner = model._orig_mod if hasattr(model, "_orig_mod") else model
final_path = "/vol/checkpoints/osrt100m_final.pt"
torch.save(inner.state_dict(), final_path)
vol.commit()
elapsed_total = time.time() - start_time
print(f"\nTraining complete. {step:,} steps in {elapsed_total/3600:.1f}h")
print(f"Final model: {final_path}")
```

# =============================================================================

# ENTRYPOINT

# =============================================================================

@app.local_entrypoint()
def main():
train.remote()
