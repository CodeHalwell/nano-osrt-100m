"""NanoOSRT — Modal deployment entrypoint.

~363M physical params (32K vocab + 1536 dim), ~192M active/token (top-2 of 8
routed experts + shared expert), ~1.15B effective via recursive weight sharing.
3 physical blocks × 6 loops = 18 effective layers.
Mixtral-style MoE: no dense FFN, 1 shared + 8 routed experts (top-2), Switch
balance loss, orthogonal per-expert init, eval-time drop-free capacity.

Reuses the v4 tokenizer volume (osrt-v4-tokenizer — same 32K BPE vocab and
structural tags). v5 keeps its own checkpoint volume (osrt-checkpoints)
so v4 and v5 can coexist during the transition.

Stages:
    modal run app_v5.py --stage sanity       200-step smoke test (~$1, ~20 min)
    modal run app_v5.py --stage pretrain     Full 300K-step pretrain
"""

import modal

# =============================================================================
# MODAL INFRASTRUCTURE
# =============================================================================

app = modal.App("nano-osrt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({
        "TORCH_LOGS": "perf_hints",
        "PYTHONUNBUFFERED": "1",
        # Disable HF tokenizers-rs thread pool before fork. Otherwise
        # DataLoader(num_workers=2) deadlocks when the child inherits a
        # locked mutex whose owning thread no longer exists. Confirmed
        # failure mode: sanity run stuck at "Fetching first batch..."
        # for 45 min with no output until manually stopped.
        "TOKENIZERS_PARALLELISM": "false",
    })
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers", "datasets", "lion-pytorch", "triton", "wandb",
        "tokenizers", "sentencepiece", "safetensors",
    )
    .add_local_dir("src/nano_osrt", remote_path="/root/nano_osrt")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

# v5 gets its own checkpoint volume so we can run v4 and v5 in parallel.
# Tokenizer volume is shared with v4 (same 32K BPE).
vol = modal.Volume.from_name("osrt-checkpoints", create_if_missing=True)
tokenizer_vol = modal.Volume.from_name(
    "osrt-v4-tokenizer", create_if_missing=True,
)


# =============================================================================
# PRE-TRAINING
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain():
    """Run v5 pre-training with progressive seq_len curriculum."""
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_training
    from nano_osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    expected_vocab = 32768
    if len(tok) != expected_vocab:
        print(f"WARNING: Expected {expected_vocab} vocab but got {len(tok)}!")
        print("  Retrain tokenizer: modal run app_v4.py --stage tokenizer")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    train_cfg = PretrainConfig()

    # Expected budget (measured, see V5_PLAN.md): ~363M total, ~192M active.
    # Effective compute via 6 recursive loops: ~1.15B FLOPs-equivalent.
    print("Expected v5 budget: ~363M total, ~192M active per token.")

    run_training(model_config, train_cfg, vol, tokenizer_name)


# =============================================================================
# SANITY (200-step smoke test)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=7200,
)
def sanity():
    """Short smoke test: 200 steps, verifies the full pipeline end-to-end.

    Purpose: before committing to a $30+ full pretrain run, prove that
      - torch.compile succeeds on the full model
      - Data streaming works for FineWeb-Edu + CodeParrot
      - Loss descends (sanity: should drop from ~ln(32768)=10.4 at step 0)
      - MoE telemetry populates sensibly (prob H near ln(8), balance loss ~1)
      - Eval path runs without errors (drops disabled, chunk-stable)
      - Checkpoint save + W&B logging work

    Overrides vs full pretrain:
      - total_steps 1200 (was 300k)
      - warmup_steps 3000 (same as Foundation)
      - eval / ckpt intervals 500
      - early_stop_check_step disabled (set past total_steps) — 1200 steps
        isn't enough for the 5k-calibrated gate to be meaningful.
      - W&B run name "osrt-extended-sanity" — keeps sanity separate from
        real pretrain runs in the dashboard.

    Uses a separate checkpoint dir (/vol/checkpoints/v5-sanity-gumbel1000) so
    this cold-expert experiment starts from step 0 and never collides with
    real pretrain checkpoints.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_training
    from nano_osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    class SanityConfig(PretrainConfig):
        total_steps = 1200
        warmup_steps = 3000
        log_interval = 50
        # Eval disabled for sanity: we only care about whether the new
        # architecture trains. Running eval would pay the ~15 min
        # 100M-record FineWeb skip that primes the held-out cache, and
        # sanity isn't long enough to collide with that offset anyway.
        eval_interval = 10_000_000
        ckpt_interval = 500
        # Foundation-matched schedule: LR warms for 3000 steps, and router
        # noise anneals over 4000 so exploration survives peak LR.
        router_gumbel_anneal_steps = 4000
        # Disabled: 1200 steps with Foundation LR warmup + cosine is not
        # enough for the 5k-calibrated gate to be meaningful.
        early_stop_check_step = 10_000_000
        wandb_run_name = "osrt-extended-sanity"

    train_cfg = SanityConfig()
    print("=" * 60)
    print("v5 EXTENDED SANITY — 1200 Foundation-matched steps")
    print("=" * 60)
    print(f"  total_steps         : {train_cfg.total_steps}")
    print(f"  warmup_steps        : {train_cfg.warmup_steps}")
    print(f"  ckpt_interval       : {train_cfg.ckpt_interval}")
    print(f"  eval_interval       : {train_cfg.eval_interval}")
    print(
        f"  router_gumbel_tau   : {train_cfg.router_gumbel_tau_init} -> "
        f"{train_cfg.router_gumbel_tau_final} over "
        f"{train_cfg.router_gumbel_anneal_steps} steps"
    )
    print(
        f"  early_stop_step     : {train_cfg.early_stop_check_step} "
        f"(disabled)"
    )
    print()

    run_training(
        model_config, train_cfg, vol, tokenizer_path,
        # Loop-level bias + raw-router aux validation. Bias is now
        # shaped recursive_loops × num_routed_experts (was block-level),
        # so loop-specific imbalances can't cancel in aggregate. Aux
        # regularizes pre-bias raw router probs, so bias can't mask
        # raw concentration. Fresh ckpt dir because bias buffer shape
        # changed — resume from the prior (block-level) ckpts would
        # fail the state_dict shape check.
        ckpt_dir="/vol/checkpoints/v5-sanity-biasloop",
    )


# =============================================================================
# GUMBEL SWEEP (runs B, C, D — A is the default sanity)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=14400,  # 4h max for 3 sequential runs
)
def sweep():
    """Run Gumbel schedule sweep configs B, C, D sequentially.

    A (tau=0.5, anneal=1000, aux=0.03) runs separately via --stage sanity.

    | Run | Aux  | Tau init | Anneal steps | Purpose                          |
    |-----|-----:|---------:|-------------:|----------------------------------|
    | B   | 0.03 | 0.8      | 1000         | Stronger early exploration       |
    | C   | 0.03 | 0.5      | 2000         | Same noise, slower decay         |
    | D   | 0.05 | 0.5      | 1000         | More balance pressure + explore  |
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_training
    from nano_osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config_kwargs = dict(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sweep_configs = [
        {
            "name": "B",
            "wandb_name": "osrt-sweep-B-tau0.8",
            "ckpt_dir": "/vol/checkpoints/v5-sweep-B",
            "aux_coeff": 0.03,
            "tau_init": 0.8,
            "anneal_steps": 1000,
        },
        {
            "name": "C",
            "wandb_name": "osrt-sweep-C-anneal2k",
            "ckpt_dir": "/vol/checkpoints/v5-sweep-C",
            "aux_coeff": 0.03,
            "tau_init": 0.5,
            "anneal_steps": 2000,
        },
        {
            "name": "D",
            "wandb_name": "osrt-sweep-D-aux0.05",
            "ckpt_dir": "/vol/checkpoints/v5-sweep-D",
            "aux_coeff": 0.05,
            "tau_init": 0.5,
            "anneal_steps": 1000,
        },
    ]

    for sc in sweep_configs:
        print("=" * 60)
        print(f"SWEEP RUN {sc['name']}: "
              f"aux={sc['aux_coeff']}, "
              f"tau={sc['tau_init']}→0 over {sc['anneal_steps']}")
        print("=" * 60)

        model_config = NanoOSRTConfig(
            router_aux_loss_coeff=sc["aux_coeff"],
            **model_config_kwargs,
        )

        class SweepConfig(PretrainConfig):
            total_steps = 200
            warmup_steps = 25
            log_interval = 10
            eval_interval = 100
            ckpt_interval = 100
            early_stop_check_step = 10_000_000

        cfg = SweepConfig()
        cfg.router_gumbel_tau_init = sc["tau_init"]
        cfg.router_gumbel_anneal_steps = sc["anneal_steps"]
        cfg.wandb_run_name = sc["wandb_name"]

        os.makedirs(sc["ckpt_dir"], exist_ok=True)
        run_training(
            model_config, cfg, vol, tokenizer_path,
            ckpt_dir=sc["ckpt_dir"],
        )
        print(f"\n>>> Run {sc['name']} complete.\n")


# =============================================================================
# OPTIMIZER × ROUTING ABLATION (cells A/B/C/D, 1200 steps each)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=21600,  # 6h for 4 sequential 1200-step runs (~80-90 min total + headroom)
)
def ablate():
    """Optimizer × routing ablation, 1200 Foundation-matched steps per cell.

    Reads each cell against:
      - the four-metric clean health gate (Phase 1 success criteria)
      - the three new prebias guards (router not collapsed under bias)

    Cells:
      | Cell | Optimizer   | Aux  | Routing      | Purpose              |
      |------|-------------|-----:|--------------|----------------------|
      | A    | Lion        | 0.10 | aux + bias   | old optimizer base   |
      | B    | Lion        | 0.0  | bias only    | aux-loss isolation   |
      | C    | Muon hybrid | 0.10 | aux + bias   | production default   |
      | D    | Muon hybrid | 0.0  | bias only    | aux-free failure     |

    Reading guide:
      - If A passes the clean gate but B fails marginal_entropy below 1.5 →
        the bias controller alone can't hold balance at this scale; keep aux.
      - If A and C both pass but C reaches lower task loss at step 1200 →
        Muon is paying off on the matrix updates; keep it for full pretrain.
      - If any cell trips a prebias guard (clean passes, raw collapses) →
        the bias controller is hiding raw-router collapse and the cell is
        misleading; do NOT promote that recipe to a full run.

    Each cell runs 1200 steps with Foundation-matched warmup (3000) so the
    first ~1000 steps are LR-warmup territory — exactly when v4 saw expert
    death. The 5k clean health gate is disabled because 1200 steps isn't
    enough to calibrate it.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_training
    from nano_osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config_kwargs = dict(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    cells = [
        {
            "name": "A",
            "label": "lion+aux (baseline)",
            "wandb_name": "osrt-ablate-A-lion-aux",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-A",
            "optimizer_name": "lion",
            "aux_coeff": 0.10,
        },
        {
            "name": "B",
            "label": "lion+bias-only",
            "wandb_name": "osrt-ablate-B-lion-biasonly",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-B",
            "optimizer_name": "lion",
            "aux_coeff": 0.0,
        },
        {
            "name": "C",
            "label": "muon+aux",
            "wandb_name": "osrt-ablate-C-muon-aux",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-C",
            "optimizer_name": "muon",
            "aux_coeff": 0.10,
        },
        {
            "name": "D",
            "label": "muon+bias-only",
            "wandb_name": "osrt-ablate-D-muon-biasonly",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-D",
            "optimizer_name": "muon",
            "aux_coeff": 0.0,
        },
    ]

    for cell in cells:
        # Skip cells that already produced a final checkpoint. Lets us
        # crash-recover the ablation without paying for cells that
        # already finished — important because cell A is ~$1 of compute.
        final_ckpt = f"{cell['ckpt_dir']}/osrt_v5_final.pt"
        if os.path.exists(final_ckpt):
            print("=" * 60)
            print(
                f"ABLATE CELL {cell['name']}: SKIP — final checkpoint "
                f"already exists at {final_ckpt}"
            )
            print("=" * 60)
            print(f"\n>>> Cell {cell['name']} ({cell['label']}) skipped.\n")
            continue

        print("=" * 60)
        print(
            f"ABLATE CELL {cell['name']}: {cell['label']} "
            f"(optimizer={cell['optimizer_name']}, aux={cell['aux_coeff']})"
        )
        print("=" * 60)

        # Each cell carries the new architectural defaults from today's
        # session: Z-loss on, seq-balance off, QK-Norm always-on,
        # softplus moe_gate, bias controller on. Only optimizer + aux
        # coefficient vary across cells.
        model_config = NanoOSRTConfig(
            router_aux_loss_coeff=cell["aux_coeff"],
            router_balance_bias_enabled=True,
            **model_config_kwargs,
        )

        class AblateConfig(PretrainConfig):
            # 1200 Foundation-matched steps — long enough to see expert
            # death during LR warmup but short enough that 4 cells fit
            # in one Modal run.
            total_steps = 1200
            warmup_steps = 3000
            log_interval = 50
            # Eval skipped — pays a 10-15 min FineWeb skip for telemetry
            # we already get from the four-metric health gate at every step.
            eval_interval = 10_000_000
            ckpt_interval = 600
            # Match the production Gumbel schedule so noise survives peak LR.
            router_gumbel_tau_init = 0.5
            router_gumbel_tau_final = 0.0
            router_gumbel_anneal_steps = 4000
            # 5k gate is calibrated for the full Foundation phase — at 1200
            # steps it would always trip, so disable it. Read the clean gate
            # plus the three prebias guards manually from W&B instead.
            early_stop_check_step = 10_000_000

        cfg = AblateConfig()
        cfg.optimizer_name = cell["optimizer_name"]
        cfg.wandb_run_name = cell["wandb_name"]

        os.makedirs(cell["ckpt_dir"], exist_ok=True)
        run_training(
            model_config, cfg, vol, tokenizer_path,
            ckpt_dir=cell["ckpt_dir"],
        )
        print(f"\n>>> Cell {cell['name']} ({cell['label']}) complete.\n")


# =============================================================================
# SFT
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft():
    """Run balanced SFT on top of the Foundation+Knowledge checkpoint.

    Loads /vol/checkpoints/v5/osrt_v5_final.pt (set by SFTConfig), injects
    HRA adapters for extra capacity, and trains on the math+code+STEM+general
    mixture with v4-style packing (inherited from v4_sft_data unchanged).
    """
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.sft_train import run_sft
    from nano_osrt.train_config import SFTConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_cfg = SFTConfig()
    run_sft(model_config, sft_cfg, vol, tok)


# =============================================================================
# SFT-LONG (long-context follow-up SFT, seq_len 4096, Nemotron-heavy mix)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_long():
    """Long-context SFT (seq_len 4096) resuming from osrt_v5_sft_final.pt.

    Configures a 1000-step run on a Nvidia-Nemotron-heavy data mix
    (math + stem + code + tool_calling = 75% Nemotron, 25% diversity)
    with HRA already loaded from the base SFT pass. Cooler LR
    (5e-6 peak) since we're fine-tuning a fine-tune.
    """
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.sft_train import run_sft
    from nano_osrt.train_config import SFTLongConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_long_cfg = SFTLongConfig()
    run_sft(model_config, sft_long_cfg, vol, tok)


# =============================================================================
# GRPO (REINFORCEMENT LEARNING)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def grpo():
    """Run GRPO with verifiable math rewards."""
    import copy
    import math
    import os
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        import wandb
    except ImportError:
        wandb = None

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.hra import get_param_groups, inject_hra
    from nano_osrt.model import NanoOSRTForCausalLM
    from nano_osrt.rewards import compute_group_advantages, compute_reward
    from nano_osrt.train import apply_router_balance_updates, load_model_state_or_raise
    from nano_osrt.train_config import GRPOConfig

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GRPOConfig()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    print("=" * 60)
    print("NanoOSRT — GRPO Training")
    print("=" * 60)

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    model = NanoOSRTForCausalLM(model_config).to(device)

    # Inject HRA before loading SFT checkpoint
    hra_params = []
    if cfg.hra_enabled:
        print(f"Injecting HRA (rank={cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=cfg.hra_rank)

    # Load SFT weights — GRPO MUST start from a real SFT checkpoint.
    ckpt_path = cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"GRPO refuses to start: SFT checkpoint not found at {ckpt_path}. "
            "Run SFT first (modal run app_v5.py --stage sft)."
        )

    print(f"Loading SFT weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    load_model_state_or_raise(
        model, state_dict, context=f"GRPO SFT load from {ckpt_path}",
    )
    print("  Clean load: all keys matched.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Group size: {cfg.group_size}")
    print(f"Total steps: {cfg.total_steps}")

    # Reference model
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Compiling policy model...")
    model = torch.compile(model)
    # Uncompiled handle for rollout — KV-cached generate() uses
    # eager-mode forward per decode step (shape changes each step
    # would trigger recompilation anyway).
    inner_for_gen = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Hold the policy in eval mode for the entire GRPO step so that the
    # rollout (generate) and the log-prob recompute (model(...)) see the
    # same routing distribution. With train(True) the MoE layer enforces
    # capacity drops (model.py:394-398), so dropped (token, expert) pairs
    # collapse to "shared expert + residual" only — different logits than
    # the no-drop rollout. That makes the assumed importance ratio ≈ 1
    # invalid and biases the policy gradient. The bias controller's
    # accumulators are gated on self.training so they simply don't update
    # during GRPO; the controller is already learned in pretrain.
    inner_for_gen.train(False)
    ref_model.train(False)

    # W&B
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={"stage": "grpo"},
        )

    # Optimizer
    if hra_params:
        param_groups = get_param_groups(
            model, hra_params, cfg.peak_lr, cfg.hra_lr, cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr,
                                       weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # Prompt dataset
    print("Loading prompt dataset...")
    load_kwargs = {"split": cfg.prompt_split, "streaming": True}
    if cfg.prompt_config:
        load_kwargs["name"] = cfg.prompt_config
    prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
    prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42)
    prompt_iter = iter(prompt_ds)

    # Resume. GRPO was previously write-only — it would reload the base
    # SFT weights on every launch and drop any partial progress. Now we
    # scan for existing grpo step and rescue checkpoints, prefer rescue
    # on ties (same logic as pretrain/sft), and start_step from there.
    ckpt_dir = "/vol/checkpoints/v5"
    os.makedirs(ckpt_dir, exist_ok=True)
    import glob as _glob
    best_grpo_step = -1
    best_grpo_ckpt: str | None = None
    for pattern in (
        f"{ckpt_dir}/osrt_v5_grpo_step_*.pt",
        f"{ckpt_dir}/osrt_v5_grpo_rescue_step_*.pt",
    ):
        for f in _glob.glob(pattern):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if s > best_grpo_step or (
                s == best_grpo_step and "rescue" in f
            ):
                best_grpo_step = s
                best_grpo_ckpt = f

    start_step = 0
    if best_grpo_step > 0 and best_grpo_ckpt is not None:
        print(
            f"Found grpo checkpoint at step {best_grpo_step}: "
            f"{best_grpo_ckpt}",
        )
        grpo_ckpt = torch.load(
            best_grpo_ckpt, map_location=device, weights_only=True,
        )
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        load_model_state_or_raise(
            inner,
            grpo_ckpt["model_state_dict"],
            context=f"GRPO resume from {best_grpo_ckpt}",
        )
        try:
            optimizer.load_state_dict(grpo_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
        start_step = grpo_ckpt["step"] + 1
        # Do NOT rebuild ref_model here. ref_model was frozen from the
        # SFT-loaded policy at line 470 and must remain the SFT anchor.
        # Rebuilding it from the resumed (already-drifted) policy would
        # make KL penalize drift from the drifted policy, not the SFT
        # baseline, so restarting would silently change the objective.
        print(f"  Resumed at step {start_step}")

    # Training loop
    start_time = time.time()

    for step in range(start_step, cfg.total_steps):
        # LR schedule
        if step < cfg.warmup_steps:
            lr = cfg.peak_lr * step / cfg.warmup_steps
        else:
            progress = (step - cfg.warmup_steps) / max(
                cfg.total_steps - cfg.warmup_steps, 1,
            )
            lr = cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in optimizer.param_groups:
            if pg.get("group_name") == "hra":
                pg["lr"] = lr * (cfg.hra_lr / cfg.peak_lr)
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_kl = 0.0
        step_rewards = []
        step_correct = 0
        step_total = 0

        for _accum in range(cfg.grad_accum_steps):
            try:
                example = next(prompt_iter)
            except StopIteration:
                prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
                prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42 + step)
                prompt_iter = iter(prompt_ds)
                example = next(prompt_iter)

            question = example["question"]
            ground_truth = example["answer"]

            prompt_text = f"{cfg.user_tag}{question}{cfg.assistant_tag}"
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)

            # Batched rollout using KV-cached generate(). The previous
            # implementation did group_size sequential loops, each
            # feeding the full prefix back into the compiled model
            # every step — O(N^2) per token and sequential across the
            # group. Replicating the prompt group_size times and calling
            # generate() once uses the per-effective-layer KV cache
            # built into NanoOSRTForCausalLM.generate(), decoding all
            # group_size samples in parallel at O(1) attention cost
            # per step.
            prompt_batch = prompt_tensor.expand(
                cfg.group_size, -1,
            ).contiguous()
            with torch.no_grad():
                generated_batch = inner_for_gen.generate(
                    prompt_batch,
                    max_new_tokens=cfg.max_gen_len,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tok.eos_token_id,
                )
            # generate() pads finished rows with EOS so the batch stays
            # rectangular. Truncate each row to its first EOS in the
            # completion region (inclusive) so downstream scoring and
            # policy log-prob computation don't see the EOS padding.
            completions = []
            for row in generated_batch:
                comp_region = row[prompt_len:]
                eos_hits = (
                    comp_region == tok.eos_token_id
                ).nonzero(as_tuple=False)
                if eos_hits.numel() > 0:
                    first_eos = int(eos_hits[0].item())
                    completions.append(row[: prompt_len + first_eos + 1])
                else:
                    completions.append(row)

            # Score — IMPORTANT: skip_special_tokens=False so native tags
            # like <|think|>, <|answer|> survive decoding for the reward
            # scorer. And explicitly pass the v4 native tag strings so
            # the reward function doesn't fall back to v3 defaults.
            rewards = []
            for comp_ids in completions:
                comp_text = tok.decode(
                    comp_ids[prompt_len:].tolist(),
                    skip_special_tokens=False,
                )
                comp_tokens = len(comp_ids) - prompt_len
                reward, breakdown = compute_reward(
                    comp_text, ground_truth,
                    correctness_weight=cfg.correctness_reward,
                    format_weight=cfg.format_reward,
                    length_penalty=cfg.length_penalty,
                    think_open=cfg.think_open,
                    think_close=cfg.think_close,
                    answer_open=cfg.answer_open,
                    answer_close=cfg.answer_close,
                    max_tokens=cfg.max_gen_len,
                    completion_tokens=comp_tokens,
                    reasoning_bonus=cfg.reasoning_bonus,
                    truncation_penalty=cfg.truncation_penalty,
                    empty_think_penalty=cfg.empty_think_penalty,
                )
                rewards.append(reward)
                if breakdown["correct"]:
                    step_correct += 1
                step_total += 1
            step_rewards.extend(rewards)

            advantages = compute_group_advantages(rewards)

            for comp_ids, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue
                comp_ids = comp_ids[:cfg.seq_len].to(device)
                comp_len = len(comp_ids) - prompt_len
                if comp_len <= 0:
                    continue

                # Policy log probs on the sampled completion
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(comp_ids.unsqueeze(0))
                    logits = out.logits[0, :, :model_config.real_vocab_size].float()
                shift_logits = logits[prompt_len - 1:-1]
                shift_labels = comp_ids[prompt_len:]
                policy_lp = F.log_softmax(shift_logits, dim=-1).gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                # Reference log probs (frozen, no grad)
                with torch.no_grad():
                    ref_out = ref_model(comp_ids.unsqueeze(0))
                    ref_logits = ref_out.logits[
                        0, :, :model_config.real_vocab_size
                    ].float()
                ref_shift = ref_logits[prompt_len - 1:-1]
                ref_lp = F.log_softmax(ref_shift, dim=-1).gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                # Direct policy gradient weighted by group-normalised advantage.
                # Since we perform only one gradient step per sampled batch,
                # importance-sampling ratio ~= 1, so PPO clipping is a no-op
                # here. We keep the formulation simple and correct.
                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -(policy_lp * adv_t).mean()

                # Schulman's unbiased non-negative KL approximation:
                #   approx_kl = exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
                # Always >= 0 (unlike the simple mean(policy_lp - ref_lp) which
                # can go negative and give a bogus "negative KL" penalty).
                log_ratio = ref_lp - policy_lp
                approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()

                loss = (policy_loss + cfg.kl_coeff * approx_kl) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()
                step_kl += approx_kl.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        apply_router_balance_updates(model)

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
            accuracy = step_correct / step_total if step_total > 0 else 0
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            mean_kl = step_kl / max(step_total, 1)
            print(f"step {step:>6d}/{cfg.total_steps} | loss {step_loss:.4f} | "
                  f"reward {mean_reward:.3f} | acc {accuracy:.1%} | "
                  f"kl {mean_kl:.4f} | lr {lr:.2e} | "
                  f"vram {vram:.1f}GB | elapsed {elapsed:.0f}s")
            if use_wandb:
                wandb.log({
                    "grpo/loss": step_loss,
                    "grpo/mean_reward": mean_reward,
                    "grpo/accuracy": accuracy,
                    "grpo/approx_kl": mean_kl,
                    "grpo/lr": lr,
                }, step=step)

        # Checkpoints
        if step > 0 and step % cfg.ckpt_interval == 0:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model_state_dict": inner.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       f"{ckpt_dir}/osrt_v5_grpo_step_{step}.pt")
            vol.commit()

        # 23h safety. Filename includes the step so the resume scanner
        # can rank it against numbered checkpoints (same convention as
        # pretrain/sft).
        if time.time() - start_time > 82_800:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            rescue_path = (
                f"{ckpt_dir}/osrt_v5_grpo_rescue_step_{step}.pt"
            )
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, rescue_path)
            vol.commit()
            print(f"\n23h boundary at step {step}. Rescue: {rescue_path}")
            if use_wandb:
                wandb.finish()
            return

    # Final
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model_state_dict": inner.state_dict(), "training_stage": "grpo"},
               f"{ckpt_dir}/osrt_v5_grpo_final.pt")
    vol.commit()
    elapsed_h = (time.time() - start_time) / 3600
    print(f"\nGRPO complete. {cfg.total_steps} steps in {elapsed_h:.1f}h")
    if use_wandb:
        wandb.finish()



# =============================================================================
# ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main(stage: str = "pretrain"):
    """Run v5 training stages.

    --stage sanity     200-step smoke test (config A)
    --stage sweep      Gumbel schedule sweep (configs B, C, D)
    --stage ablate     Optimizer × routing ablation (cells A/B/C/D, 1200 steps each)
    --stage pretrain   Full pre-training with progressive seq_len curriculum
    --stage sft        Balanced SFT on the final pretrained checkpoint
    --stage sft_long   Long-context SFT (seq 4096) resuming from sft_final.pt with Nemotron mix
    --stage grpo       GRPO RL on the SFT checkpoint (verifiable math rewards)
    """
    if stage == "sanity":
        sanity.remote()
    elif stage == "sweep":
        sweep.remote()
    elif stage == "ablate":
        ablate.remote()
    elif stage == "pretrain":
        pretrain.remote()
    elif stage == "sft":
        sft.remote()
    elif stage == "sft_long":
        sft_long.remote()
    elif stage == "grpo":
        grpo.remote()
    else:
        print(
            f"Unknown stage: {stage}. "
            f"Use sanity, sweep, ablate, pretrain, sft, sft_long, or grpo"
        )
