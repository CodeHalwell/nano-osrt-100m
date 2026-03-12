"""Nano-OSRT 100M — Modal deployment entrypoint (v3.2).

104.5M physical parameters simulating 302M equivalent dense via
recursive weight sharing. 2 physical blocks x 6 recursive loops = 12
effective layers, each with unique per-pass residual adapters.

Deploy::

    modal run app.py

Resume after 24h timeout (automatically resumes from latest rescue
checkpoint)::

    modal run app.py
"""

import modal

from nano_osrt.modal_config import ModalConfig

# NOTE: torch imports are deliberately kept INSIDE train() because Modal
# parses this file locally before sending to the container. If torch isn't
# installed on the local machine, top-level imports would crash.

# =============================================================================
# MODAL INFRASTRUCTURE
# =============================================================================

app = modal.App("nano-osrt-100m")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"TORCH_LOGS": "perf_hints"})
    .pip_install(
        "torch==2.10.0+cu128",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "lion-pytorch", "triton", "wandb")
)

vol = modal.Volume.from_name("osrt-checkpoints", create_if_missing=True)


# =============================================================================
# REMOTE TRAINING FUNCTION
# =============================================================================


@app.function(
    gpu=modal.gpu.H100(count=1),
    image=image,
    volumes={"/vol/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=86400,
)
def train():
    """Run the full training loop inside a Modal H100 container."""
    from transformers import AutoTokenizer

    from nano_osrt.modal_train import run_training

    cfg = ModalConfig()

    # Tokenizer + dynamic vocab alignment
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg.real_vocab_size = len(tokenizer)
    cfg.vocab_size = 64 * ((cfg.real_vocab_size + 63) // 64)

    run_training(cfg, vol, tokenizer_name)


# =============================================================================
# ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main():
    train.remote()
