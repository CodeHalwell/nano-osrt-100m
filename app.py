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
        # Persistent HF datasets cache. Volume mounted by SFT/eval/GRPO
        # functions; pretrain doesn't mount it, but HF datasets handles
        # a non-existent path gracefully under streaming=True (the
        # iterable doesn't touch the cache; only metadata downloads do,
        # and those mkdir the path on demand within the container's
        # writable layer if no volume is mounted there).
        "HF_DATASETS_CACHE": "/vol/hf_cache",
    })
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers", "datasets", "lion-pytorch", "triton", "wandb",
        "tokenizers", "sentencepiece", "safetensors",
        # lm-eval baked into the base image so the `evaluate` function
        # doesn't need a derived image. Modal disallows .pip_install
        # after .add_local_dir; folding it in here keeps the build chain
        # linear (env → apt → pip → add_local) so any function can
        # evaluate without an image rebuild.
        #
        # [ifeval] extras: pulls in langdetect + immutabledict + nltk
        # which IFEval's instruction graders rely on. Without these,
        # `lm_eval.simple_evaluate(tasks=["ifeval", ...])` fails at
        # task-config load time with "ModuleNotFoundError: No module
        # named 'langdetect'" before the model even runs.
        "lm-eval[ifeval]",
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
# Persistent HF datasets cache. First run downloads dataset shards from
# the Hub into this volume; subsequent runs read from local volume
# storage, which removes Hub round-trips and the latency variance that
# caused 20→75 sec/step swings during SFT-long. Streaming mode bypasses
# the dataset cache for the iterable itself, but it still uses this
# directory for split metadata, dataset_info.json, and any non-streamed
# auxiliary downloads — small but noticeable wins.
#
# HF_DATASETS_CACHE=/vol/hf_cache is set in the base image's env block
# (above) so the datasets library auto-discovers it. SFT/eval/GRPO
# functions mount this volume; pretrain skips the mount and HF falls
# back gracefully under streaming=True.
hf_cache_vol = modal.Volume.from_name(
    "osrt-hf-cache", create_if_missing=True,
)

# MOPD rollout volume — holds Gemini teacher-rollout JSONL collected
# via scripts/collect_rollouts.py. Uploaded from local before launching
# the mopd stage. Per-workspace; create_if_missing so first launch on a
# fresh workspace works without manual setup.
rollouts_vol = modal.Volume.from_name(
    "osrt-rollouts", create_if_missing=True,
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
# PRETRAIN_EXTEND (continued pretraining / "mid-training" on top of SFT ckpt)
# =============================================================================
#
# Loads osrt_v5_sft_ultralong_final.pt, injects + freezes HRA, then runs
# 1,800 steps of continued pretraining at seq 4096 with a math/science/
# code-heavy mix plus 25 % SFT-formatted rehearsal data to combat
# chat-format forgetting. Output: osrt_v5_extend_step_N.pt and
# osrt_v5_extend_final.pt (distinct prefix so resume scans don't
# collide with base pretrain ckpts). See PretrainExtendConfig +
# train.py::run_pretrain_extend for the design rationale.
#
# Mounts the HF cache volume so the new datasets (Nemotron-CC-Math,
# RedPajama-arxiv, the-stack-smol, plus the existing FineWeb-Edu and
# Wikipedia) populate /vol/hf_cache on first run and reuse the cache
# on subsequent runs.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain_extend():
    """Continued pretraining on top of the SFT-ultralong checkpoint.

    See train_config.py::PretrainExtendConfig for the full design
    rationale (lineage decision, LR schedule, rehearsal mix, HRA
    freeze). Single phase, seq 4096, ~1,800 steps, ~$30 of H100
    time, ~485M new pretrain tokens concentrated in the
    math/science/code categories the original pretrain missed.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import PretrainExtendConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

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

    extend_cfg = PretrainExtendConfig()
    print(
        "pretrain_extend: 1,800 steps at seq 4096, peak LR 1.5e-5, "
        "HRA frozen, 25 % SFT-formatted rehearsal mix.",
    )
    print(
        f"Resume base: {extend_cfg.pretrained_checkpoint}",
    )

    run_pretrain_extend(model_config, extend_cfg, vol, tokenizer_name)


# =============================================================================
# PRETRAIN_EXTEND2 — broadened mid-training pass
# =============================================================================
# Reuses run_pretrain_extend (the training loop is config-driven). Resumes
# from osrt_v5_grpo_final.pt (canonical step-700 GRPO ckpt) with HRA frozen
# so the SFT+GRPO investment in chat/answer format stays put while the base
# weights absorb new reasoning/code/math knowledge. Output checkpoints use
# the `extend2` prefix to avoid colliding with extend1's scan.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain_extend2():
    """Second mid-training pass — broader reasoning + code + science mix.

    See train_config.py::PretrainExtend2Config for the design
    rationale (DeepSeek-R1 cold-start strategy, 30/40/15/15 mix,
    HRA freeze, tag-rewrite for R1 traces). Single phase, seq 2048,
    ~3,000 steps, ~$28 of H100 time. Resumes from GRPO step-700.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import PretrainExtend2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

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

    extend2_cfg = PretrainExtend2Config()
    print(
        "pretrain_extend2: 3,000 steps at seq 2048, peak LR 1e-5, "
        "HRA frozen, 30/40/15/15 code/math/reasoning/general mix.",
    )
    print(f"Resume base: {extend2_cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, extend2_cfg, vol, tokenizer_name)


# =============================================================================
# LOOP_FIX — architecture-fix continuation with per-loop aux LM-head losses
# =============================================================================
# Recursive-loop probe (probe_recursion.py, 2026-06-05) showed loop 5 doing
# ~6.0 of the CE loss reduction while loops 1-4 contributed ~0.75 combined.
# This stage attaches the weight-tied LM head to each non-final loop's hidden
# state (after norm_out for path consistency) and adds the resulting CE
# losses to the main loss with `aux_loop_loss_weight`. Forces gradient signal
# into the intermediate loops. ~1500 step continuation from extend2_final.
# See train_config.py::LoopFixConfig for the full design.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def loop_fix():
    """Per-loop aux-loss continuation from extend2_final.

    The aux_loop_loss_weight on NanoOSRTConfig (model config) is the
    real switch — without it the model forward is unchanged. We thread
    it through here from the training config (LoopFixConfig).
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import LoopFixConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    loopfix_cfg = LoopFixConfig()
    # Phase end is informational (the loop stops on total_steps), but
    # set it for accurate printed banner.
    loopfix_cfg.phases["extend"]["end"] = loopfix_cfg.total_steps
    # Aux losses materialise 5 extra logit tensors (B × T × V) per
    # forward pass at the same precision as the main logits — ~10 GB
    # extra at batch=8/seq=2048. Cut batch 8→4 and bump accum 8→16 to
    # keep effective batch=64 while halving activation memory.
    loopfix_cfg.phases["extend"]["batch_size"] = 4
    loopfix_cfg.phases["extend"]["grad_accum_steps"] = 16

    # Critical: pass aux_loop_loss_weight to the MODEL config so the
    # model's forward actually computes the aux losses. Without this,
    # the training config's aux_loop_loss_weight is a no-op.
    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=loopfix_cfg.aux_loop_loss_weight,
    )

    print(
        f"loop_fix: {loopfix_cfg.total_steps - loopfix_cfg.lr_anchor_step} "
        f"steps continuation, peak_lr={loopfix_cfg.peak_lr}, "
        f"aux_loop_loss_weight={loopfix_cfg.aux_loop_loss_weight}.",
    )
    print(f"Resume base: {loopfix_cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, loopfix_cfg, vol, tokenizer_name)


def loop_fix_sanity_inner():
    """Body of the 50-step sanity smoke test for loop_fix.
    Shared between the real sanity stage and any future ad-hoc test.
    """
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import LoopFixConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityLoopFix(LoopFixConfig):
        # Fresh step counter — no lr_anchor_step.
        total_steps = 50
        lr_anchor_step = 0
        warmup_steps = 10
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False      # fast first-step events

    sanity_cfg = SanityLoopFix()
    sanity_cfg.phases["extend"]["end"] = 50
    sanity_cfg.phases["extend"]["batch_size"] = 4
    sanity_cfg.phases["extend"]["grad_accum_steps"] = 16
    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=sanity_cfg.aux_loop_loss_weight,
    )
    print(f"loop_fix SANITY: 50 steps from extend2_final, "
          f"aux_loop_loss_weight={sanity_cfg.aux_loop_loss_weight}.")
    run_pretrain_extend(model_config, sanity_cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def loop_fix_sanity():
    loop_fix_sanity_inner()


# =============================================================================
# LOOP_FIX_V2 — stacked fixes (aux + dropout + curriculum + per-loop weights)
# =============================================================================
# Layered on top of loop_fix's aux LM-head loss with:
#   (1) loop dropout (stochastic depth, p=0.2)
#   (2) aux-weight curriculum (0.02 → 0.10 over 200 steps)
#   (3) per-loop aux weights biased toward earlier loops [2.0,1.5,1.0,0.7,0.5]
# See train_config.py::LoopFixV2Config for the full rationale.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def loop_fix_v2():
    """Stacked architecture fixes (aux + dropout + curriculum +
    per-loop weights) from loop_fix's final ckpt."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import LoopFixV2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = LoopFixV2Config()
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    # Resume base: loop_fix's step_400 (we plan to stop loop_fix early
    # at step 400 — fast gains were done by step 200, diminishing
    # returns thereafter; the v2 stacked-fix run is the better use of
    # the remaining compute). Update to loopfix_final.pt if we end up
    # running loop_fix to completion.
    cfg.pretrained_checkpoint = "/vol/checkpoints/v5/osrt_v5_loopfix_step_400.pt"

    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        # All three architecture-fix knobs go through the model config:
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"loop_fix_v2: {cfg.total_steps} steps, peak_lr={cfg.peak_lr}, "
        f"aux_loop_loss_weight={cfg.aux_loop_loss_weight} "
        f"(curriculum: {cfg.aux_loop_weight_start} → final over "
        f"{cfg.aux_loop_curriculum_steps} steps), "
        f"loop_dropout_prob={cfg.loop_dropout_prob}, "
        f"per_loop_weights={cfg.per_loop_aux_weights}."
    )
    print(f"Resume base: {cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def loop_fix_v2_sanity():
    """50-step sanity for loop_fix_v2 stacked-fix run."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import LoopFixV2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityCfg(LoopFixV2Config):
        total_steps = 50
        lr_anchor_step = 0
        warmup_steps = 10
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False
        # Shorter curriculum so we exercise the ramp within 50 steps.
        aux_loop_curriculum_steps = 30
        # Resume from extend2_final (loop_fix may not have a final.pt
        # yet during testing); the sanity is to validate the new code
        # paths run end-to-end, not to require a specific ckpt.
        pretrained_checkpoint = "/vol/checkpoints/v5/osrt_v5_extend2_final.pt"

    cfg = SanityCfg()
    cfg.phases["extend"]["end"] = 50
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"loop_fix_v2 SANITY: 50 steps, dropout={cfg.loop_dropout_prob}, "
        f"curriculum {cfg.aux_loop_weight_start}→{cfg.aux_loop_loss_weight} "
        f"over {cfg.aux_loop_curriculum_steps} steps, "
        f"per_loop_weights={cfg.per_loop_aux_weights}."
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


# =============================================================================
# PRETRAIN_EXTEND3 — first mid-training round with WORKING recursive depth
# =============================================================================
# All prior training (~30k+ steps) happened with the loop-collapsed
# architecture. extend2's 9-stream mix was absorbed at only ~6 effective
# layers of depth. With loop_fix + v2 done, the model can now actually use
# all 18 effective layers — so re-running on the same data mix lets it
# encode information it couldn't before. v2 already showed this happening
# (task CE dropped 1.80 → 1.54 in 300 steps with fix on, on data the model
# had seen 8100 steps of before).
# See train_config.py::PretrainExtend3Config for the full design.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain_extend3():
    """Third mid-training pass — first run with the architecture fix
    permanently in the loss path. Same 9-stream extend2 mix, softer
    fix knobs (aux=0.05, dropout=0.10), lower LR (peak 3e-6), 3000
    steps from loopfixv2_merged.pt."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import PretrainExtend3Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = PretrainExtend3Config()
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16

    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"pretrain_extend3: {cfg.total_steps} steps, peak_lr={cfg.peak_lr}, "
        f"aux_loop_loss_weight={cfg.aux_loop_loss_weight} "
        f"(curriculum {cfg.aux_loop_weight_start}→{cfg.aux_loop_loss_weight} "
        f"over {cfg.aux_loop_curriculum_steps} steps), "
        f"loop_dropout_prob={cfg.loop_dropout_prob}, "
        f"per_loop_weights=uniform."
    )
    print(f"Resume base: {cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def pretrain_extend3_sanity():
    """50-step sanity for pretrain_extend3."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import PretrainExtend3Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityCfg(PretrainExtend3Config):
        total_steps = 50
        lr_anchor_step = 0
        warmup_steps = 10
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False
        # Shorter curriculum to exercise the ramp within 50 steps.
        aux_loop_curriculum_steps = 30

    cfg = SanityCfg()
    cfg.phases["extend"]["end"] = 50
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"pretrain_extend3 SANITY: 50 steps, peak_lr={cfg.peak_lr}, "
        f"aux={cfg.aux_loop_loss_weight}, dropout={cfg.loop_dropout_prob}."
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


# =============================================================================
# MOPD — Multi-teacher On-Policy Distillation from Gemini rollouts
# =============================================================================
# Trains on a local JSONL of teacher rollouts (collected via
# scripts/collect_rollouts.py + uploaded to the osrt-rollouts volume).
# Reuses run_pretrain_extend with the rollout_dataset_path override so all
# the architecture-fix telemetry, LR schedule, MoE balance, and checkpoint
# infrastructure works unchanged. Resumes from extend3_final.pt.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def mopd():
    """MOPD distillation on Gemini-rollout JSONL from extend3_final.pt.

    Reads /vol/rollouts/mopd_v1.jsonl (upload from local with
    `modal volume put osrt-rollouts rollouts/mopd_v1.jsonl mopd_v1.jsonl`
    before launching). 1000 steps, peak_lr 1.5e-6, aux fix knobs at
    extend3 levels."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import MOPDConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = MOPDConfig()
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    # Shorter seq_len for rollouts — most are under 1024 tokens, so
    # 2048 is mostly wasted padding. Cuts compute ~50% per step.
    cfg.phases["extend"]["seq_len"] = 1024

    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"mopd: {cfg.total_steps} steps from {cfg.pretrained_checkpoint}, "
        f"peak_lr={cfg.peak_lr}, rollout_path={cfg.rollout_dataset_path}, "
        f"aux={cfg.aux_loop_loss_weight}, dropout={cfg.loop_dropout_prob}."
    )
    if not os.path.exists(cfg.rollout_dataset_path):
        raise FileNotFoundError(
            f"Rollout JSONL not found at {cfg.rollout_dataset_path}. "
            "Upload via: "
            "`modal volume put osrt-rollouts rollouts/mopd_v1.jsonl "
            "mopd_v1.jsonl`"
        )

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def mopd_sanity():
    """30-step MOPD sanity validating the rollout loader path."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import MOPDConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityCfg(MOPDConfig):
        total_steps = 30
        lr_anchor_step = 0
        warmup_steps = 5
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False
        aux_loop_curriculum_steps = 10
        # Resume from extend3 step ckpt (or loopfixv2_merged) — sanity
        # is to validate the rollout pipeline end-to-end, not to
        # depend on a specific final ckpt that may not exist yet.
        pretrained_checkpoint = "/vol/checkpoints/v5/osrt_v5_loopfixv2_merged.pt"

    cfg = SanityCfg()
    cfg.phases["extend"]["end"] = 30
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    cfg.phases["extend"]["seq_len"] = 1024
    model_config = NanoOSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"mopd SANITY: 30 steps, peak_lr={cfg.peak_lr}, "
        f"rollouts={cfg.rollout_dataset_path}."
    )
    if not os.path.exists(cfg.rollout_dataset_path):
        raise FileNotFoundError(
            f"Rollout JSONL not found at {cfg.rollout_dataset_path}. "
            "Upload via: "
            "`modal volume put osrt-rollouts rollouts/mopd_v1.jsonl "
            "mopd_v1.jsonl`"
        )

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def pretrain_extend2_sanity():
    """50-step smoke test of pretrain_extend2 — verify all 10 streams
    connect, format functions yield clean batches end-to-end, and the
    training loop completes a few cycles before committing $28 on the
    full 3,000-step run.

    Total cost ~$1 (~10 min including compile time). Disables
    checkpoint saving so the volume isn't polluted with throwaway
    sanity ckpts. Overrides `wandb_run_name` so the smoke run is
    visually separated from real extend2 runs in the dashboard.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.train import run_pretrain_extend
    from nano_osrt.train_config import PretrainExtend2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # Subclass so we don't mutate the real config. ckpt_interval set
    # past total_steps so save path is never triggered. warmup_steps
    # cut to 10 (still 20% of total) so we actually exit the warmup
    # and see a cosine-shaped LR at least once before the run ends.
    class SanityCfg(PretrainExtend2Config):
        total_steps = 50
        warmup_steps = 10
        ckpt_interval = 999_999
        log_interval = 5
        eval_interval = 999_999
        wandb_run_name = "osrt-pretrain-extend2-sanity"
        # Skip torch.compile — eager starts producing step events
        # immediately (compile takes ~10 min of silent GPU time).
        compile_enabled = False
        wandb_log = False
        # Differential diagnosis: extend1 (which worked) resumed from
        # sft_ultralong_final.pt; extend2 (which auto-cancels mid-
        # first-forward-pass) resumes from osrt_v5_grpo_final.pt.
        # Swapping to the pre-GRPO sft_math ckpt for sanity isolates
        # whether the GRPO checkpoint itself is the trigger.
        pretrained_checkpoint = (
            "/vol/checkpoints/v5/osrt_v5_sft_math_final.pt"
        )

    sanity_cfg = SanityCfg()
    sanity_cfg.phases["extend"]["end"] = 50
    # Datasets now live in PretrainExtend2Config; sanity inherits
    # the locked 9-stream working mix verified via v9-v25 bisection.

    print("pretrain_extend2 SANITY: 50 steps, no ckpts, no eval — "
          "validating all streams + format functions.")
    print(f"Resume base: {sanity_cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, sanity_cfg, vol, tokenizer_name)


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
        "/vol/hf_cache": hf_cache_vol,
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
        "/vol/hf_cache": hf_cache_vol,
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
# SFT-ULTRALONG (seq_len 8192, resumes from sft_long_final.pt)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_ultralong():
    """Ultra-long-context SFT (seq_len 8192) resuming from sft_long_final.pt.

    500 steps at the same Nemotron-heavy mix, batch 2 × accum 32 to
    keep effective batch at 64 sequences within H100 80GB at seq 8192.
    Cooler LR (3e-6 peak) for the third successive fine-tune.
    """
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.sft_train import run_sft
    from nano_osrt.train_config import SFTUltraLongConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_ultralong_cfg = SFTUltraLongConfig()
    run_sft(model_config, sft_ultralong_cfg, vol, tok)


# =============================================================================
# SFT_REFRESH (short SFT pass to re-anchor chat format after extend)
# =============================================================================
#
# Local probe on osrt_v5_extend_final.pt showed chat-format degradation
# (special tokens emitted in wrong positions, <|/answer|> never closes,
# some prompts produce immediate EOS) despite the 25 % rehearsal mix
# during pretrain_extend. This stage runs a short, low-LR SFT on top
# of the extended ckpt to re-anchor the format wrapping. The math /
# code learning the extend gave us is preserved; only the format
# placement is re-tuned.
#
# 500 steps at seq 2048 ≈ 50 min on H100 ≈ ~$3-4. Output:
# osrt_v5_sft_refresh_step_N.pt + osrt_v5_sft_refresh_final.pt.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_refresh():
    """Short SFT format-anchor pass on top of pretrain_extend ckpt.

    See train_config.py::SFTRefreshConfig for the full design
    rationale. 500 steps, peak LR 5e-6 (33 % of base SFT), HRA
    trainable, NO tool_calling in the data mix (preserves the
    anti-hallucination win from extend).
    """
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.sft_train import run_sft
    from nano_osrt.train_config import SFTRefreshConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_refresh_cfg = SFTRefreshConfig()
    print(
        "sft_refresh: 500 steps, peak LR 5e-6, HRA trainable, "
        "no tool_calling. Goal: re-anchor chat-format emission "
        "after pretrain_extend.",
    )
    print(f"Resume base: {sft_refresh_cfg.pretrained_checkpoint}")
    run_sft(model_config, sft_refresh_cfg, vol, tok)


# =============================================================================
# SFT_MATH (math-only SFT pass between sft_refresh and GRPO)
# =============================================================================
#
# Math probe of sft_refresh_final.pt revealed think→answer decoupling
# (think block had correct steps, answer block ignored them and
# emitted random wrong content). 200 steps of pure math SFT trains
# the answer block to commit to the think block's conclusion.
# Cheap (~$1.30) and gives GRPO a coherent base before RL kicks in.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_math():
    """Math-only SFT polish on top of sft_refresh_final.pt.

    See train_config.py::SFTMathConfig for design rationale. 200
    steps, pure math mix (GSM8K + Orca-Math + MathInstruct +
    NuminaMath-CoT, all warm-cached on gradio-winter-hack), peak
    LR 3e-6. Goal: tighten think→answer correlation before GRPO.
    """
    from transformers import AutoTokenizer

    from nano_osrt.config import NanoOSRTConfig
    from nano_osrt.sft_train import run_sft
    from nano_osrt.train_config import SFTMathConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_math_cfg = SFTMathConfig()
    print(
        "sft_math: 1,000 steps, peak LR 3e-6, math-only mix. "
        "Goal: tighten think→answer correlation before GRPO.",
    )
    print(f"Resume base: {sft_math_cfg.pretrained_checkpoint}")
    run_sft(model_config, sft_math_cfg, vol, tok)


# =============================================================================
# EVALUATE (lm-eval-harness pass: gsm8k + IFEval + MMLU-stem)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,  # lm-eval is in the base image's pip_install
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=14400,  # 4h: comfortable headroom for full lm-eval suite
)
def evaluate(
    ckpt_name: str = "osrt_v5_sft_ultralong_final.pt",
    tag: str = "pre-grpo",
    tasks: str = (
        # Original three: math reasoning + instruction following + STEM knowledge
        "gsm8k,ifeval,mmlu_stem,"
        # Small-model commonsense suite — gives a clean GPT-2-class
        # comparison anchor. GPT-2 medium (355M) reference numbers:
        #   hellaswag 33%, arc_easy 49%, arc_challenge 22%,
        #   piqa 63%, winogrande 52%.
        # All five are pure loglikelihood scoring (no generation),
        # so total added cost is ~$3-4 on top of the generate-heavy
        # gsm8k + ifeval. Adds ~30 min to the full pass.
        "hellaswag,arc_easy,arc_challenge,piqa,winogrande"
    ),
    limit: int | None = None,
):
    """Run lm-evaluation-harness on the latest SFT/GRPO checkpoint.

    Runs gsm8k + IFEval + MMLU-stem by default. Results are written to
    `/vol/checkpoints/v5/eval_<tag>.json` so pre-GRPO and post-GRPO
    runs can be diffed straightforwardly.

    Args:
        ckpt_name: filename under /vol/checkpoints/v5/. Default points
            at the SFT-ultralong final ckpt; pass
            "osrt_v5_grpo_final.pt" (or step-named variants) for the
            post-GRPO eval.
        tag: short label embedded in the output filename and W&B run.
            Use "pre-grpo" / "post-grpo" / "post-iter-grpo" for the
            comparison sequence.
        tasks: comma-separated lm-eval task names. Defaults match the
            three benchmarks we care about; override for ad-hoc runs.
        limit: cap problems per task for quick smoke tests. None =
            full benchmark. 50 is a reasonable smoke value.

    Cost: ~$5 for the default three-task pass on H100 (gsm8k 1319
    problems × 256 generated tokens dominates).
    """
    import json
    import os

    try:
        import wandb
    except ImportError:
        wandb = None

    from lm_eval import simple_evaluate

    from nano_osrt.lm_eval_wrapper import NanoOSRTLMEval

    ckpt_path = f"/vol/checkpoints/v5/{ckpt_name}"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Available files: "
            f"{sorted(os.listdir('/vol/checkpoints/v5'))[:10]}",
        )

    print("=" * 60)
    print(f"NanoOSRT — lm-eval-harness ({tag})")
    print("=" * 60)
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Tasks          : {tasks}")
    print(f"Per-task limit : {limit or 'full'}")
    print()

    wrapper = NanoOSRTLMEval(
        ckpt_path=ckpt_path,
        tokenizer_path="/vol/tokenizer",
        hra_enabled=True,
        hra_rank=256,
        batch_size=8,
        device="cuda",
    )

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    if wandb is not None:
        wandb.init(
            project="nano-osrt",
            name=f"osrt-eval-{tag}",
            config={
                "stage": "evaluate",
                "ckpt_name": ckpt_name,
                "tasks": task_list,
                "limit": limit,
            },
        )

    # Sample logging is gated on the smoke run — when limit is set
    # (i.e. iterating on the wrapper), we want every prompt+response
    # in the JSON to debug. Full eval (limit=None) skips it because
    # the transcript dump bloats the JSON ~100×.
    log_samples = limit is not None
    print(
        f"Running lm-eval on {task_list}... "
        f"(limit={limit}, log_samples={log_samples})",
        flush=True,
    )
    results = simple_evaluate(
        model=wrapper,
        tasks=task_list,
        limit=limit,
        log_samples=log_samples,
    )

    # Strip the bulky "model_dump" entry. Keep "samples" iff log_samples
    # was on (smoke runs) so we can read what the model actually emitted
    # — the whole point of running smokes.
    summary = {
        "tag": tag,
        "ckpt_name": ckpt_name,
        "tasks": task_list,
        "limit": limit,
        "results": results.get("results", {}),
        "configs": {k: v.get("task", k) for k, v in results.get("configs", {}).items()},
    }
    if log_samples and "samples" in results:
        # Cap to first 5 per task to keep JSON readable even at limit=50.
        # Five samples per task is enough to see if formatting / extraction
        # is working without burying us in transcripts.
        capped_samples = {}
        for task_name, sample_list in results["samples"].items():
            capped_samples[task_name] = sample_list[:5]
        summary["samples"] = capped_samples

    out_path = f"/vol/checkpoints/v5/eval_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {out_path}", flush=True)

    # Print headline numbers for stdout/Modal log readability.
    print("\n=== Headline metrics ===")
    for task_name, task_results in summary["results"].items():
        print(f"\n[{task_name}]")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    if wandb is not None:
        # Log flat metrics so they're queryable + plottable across runs.
        for task_name, task_results in summary["results"].items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"eval/{task_name}/{metric}": value})
        wandb.finish()

    vol.commit()


# =============================================================================
# GRPO (REINFORCEMENT LEARNING)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
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
        # Fall back to the filename-extracted step (best_grpo_step) when
        # the ckpt itself doesn't carry a "step" field — happens when a
        # final.pt is renamed to step_N.pt (the final-save path only
        # writes model_state_dict, not the step int, so naive resume
        # crashes with KeyError: 'step'). Caught during the
        # 500→700 extension restart.
        start_step = grpo_ckpt.get("step", best_grpo_step) + 1
        # Do NOT rebuild ref_model here. ref_model was frozen from the
        # SFT-loaded policy at line 470 and must remain the SFT anchor.
        # Rebuilding it from the resumed (already-drifted) policy would
        # make KL penalize drift from the drifted policy, not the SFT
        # baseline, so restarting would silently change the objective.
        print(f"  Resumed at step {start_step}")

    # Training loop
    start_time = time.time()

    for step in range(start_step, cfg.total_steps):
        # LR schedule. lr_anchor_step lets a resumed run re-warm:
        # the warmup/cosine treats `step - anchor` as the effective
        # step, so the new phase gets a real gradient instead of
        # the near-zero LR a continued cosine would yield.
        anchor = getattr(cfg, "lr_anchor_step", 0)
        eff_step = max(step - anchor, 0)
        eff_total = max(cfg.total_steps - anchor, 1)
        if eff_step < cfg.warmup_steps:
            lr = cfg.peak_lr * eff_step / cfg.warmup_steps
        else:
            progress = (eff_step - cfg.warmup_steps) / max(
                eff_total - cfg.warmup_steps, 1,
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
# GRPO_MULTI — multi-env GRPO from mopd_final.pt
# =============================================================================
# Same PPO-style loop as grpo() but with:
#   - per micro-batch env sampling (math 60% / ifeval 30% / mbpp 10%)
#   - env-aware prompt fetcher + ground-truth extractor
#   - env-aware reward dispatcher (compose_template_rewards + per-env scorer)
#   - per-env wandb keys (math_acc, ifeval_constraints_hit_rate, mbpp_pass_rate)
#   - stop_token_ids during rollout for clean <|/answer|> halt
#   - RewardEMA logging
# Resumes from mopd_final.pt. See train_config.py::MultiEnvGRPOConfig.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def grpo_multi():
    """Run multi-env GRPO from mopd_final.pt."""
    _run_grpo_multi(sanity=False)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def grpo_multi_sanity():
    """30-step multi-env GRPO sanity: validate env dispatch, rollout,
    reward computation, KV-cached generation with stop tokens."""
    _run_grpo_multi(sanity=True)


def _run_grpo_multi(sanity: bool = False) -> None:
    """Multi-env GRPO training loop.

    Shared by grpo_multi (full) and grpo_multi_sanity (30-step smoke).
    """
    import copy
    import math
    import os
    import random as _random
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
    from nano_osrt.rewards import (
        RewardEMA,
        compose_template_rewards,
        compute_group_advantages,
        ifeval_constraint_reward,
        mbpp_test_reward,
    )
    from nano_osrt.train import apply_router_balance_updates, load_model_state_or_raise
    from nano_osrt.train_config import MultiEnvGRPOConfig

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = MultiEnvGRPOConfig()
    if sanity:
        # Sanity overrides — small everything, no compile, no wandb
        cfg.total_steps = 30
        cfg.warmup_steps = 5
        cfg.log_interval = 2
        cfg.ckpt_interval = 999_999
        cfg.wandb_log = False
        cfg.grad_accum_steps = 2  # smaller for fast iteration
        cfg.aux_loop_curriculum_steps = 0

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    print("=" * 60)
    print(f"NanoOSRT — Multi-env GRPO {'(SANITY)' if sanity else ''}")
    print("=" * 60)
    print(f"  Envs: {dict(zip(cfg.env_names, cfg.env_weights))}")
    print(f"  Resume: {cfg.pretrained_checkpoint}")
    print(f"  Steps: {cfg.total_steps}, group_size: {cfg.group_size}, "
          f"max_gen_len: {cfg.max_gen_len}, kl_coeff: {cfg.kl_coeff}")
    print(f"  Stop token ids: {cfg.stop_token_ids}")

    # Model with architecture-fix knobs
    model_config = NanoOSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
    )
    model = NanoOSRTForCausalLM(model_config).to(device)

    hra_params = []
    if cfg.hra_enabled:
        print(f"Injecting HRA (rank={cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=cfg.hra_rank)

    ckpt_path = cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"grpo_multi refuses to start: ckpt not found at {ckpt_path}. "
            "Upload mopd_final.pt to the osrt-checkpoints volume first.",
        )
    print(f"Loading base weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    load_model_state_or_raise(
        model, state_dict, context=f"grpo_multi load from {ckpt_path}",
    )
    print("  Clean load: all keys matched.")

    # Frozen reference for KL anchor
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if not sanity:
        print("Compiling policy model...")
        model = torch.compile(model)
    inner_for_gen = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner_for_gen.train(False)
    ref_model.train(False)

    # W&B
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={"stage": "grpo_multi"},
        )

    # Optimizer
    if hra_params:
        param_groups = get_param_groups(
            model, hra_params, cfg.peak_lr, cfg.hra_lr, cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.peak_lr,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

    # ──────────────────────────────────────────────────────────────
    # Per-env prompt streams. Each env yields (prompt_text, gt_blob).
    # gt_blob is env-shaped: math = "#### N" string; ifeval = dict
    # with instruction_id_list + kwargs; mbpp_code = list of asserts.
    # We build one streaming iterator per env, then in the training
    # loop we sample which env to draw from per micro-batch.
    # ──────────────────────────────────────────────────────────────
    print("Loading per-env prompt datasets...")
    env_iters: dict[str, object] = {}
    env_ds_factories: dict[str, callable] = {}

    def _make_env_factory(env_name: str, ds_spec: dict):
        load_kwargs = {"split": ds_spec["split"], "streaming": True}
        if ds_spec.get("hf_config"):
            load_kwargs["name"] = ds_spec["hf_config"]

        def _build(seed: int):
            ds = load_dataset(ds_spec["hf_id"], **load_kwargs)
            try:
                ds = ds.shuffle(buffer_size=1_000, seed=seed)
            except Exception:
                pass
            return iter(ds)
        return _build

    for env_name in cfg.env_names:
        ds_spec = cfg.env_datasets[env_name]
        factory = _make_env_factory(env_name, ds_spec)
        env_ds_factories[env_name] = factory
        env_iters[env_name] = factory(seed=42)
        print(f"  [{env_name}] loaded from {ds_spec['hf_id']} "
              f"(split={ds_spec['split']})")

    def _next_example(env_name: str):
        """Get the next prompt + raw row from the env's iterator,
        re-creating the iterator if exhausted."""
        while True:
            try:
                return next(env_iters[env_name])
            except StopIteration:
                # Reshuffle with a different seed and continue
                env_iters[env_name] = env_ds_factories[env_name](
                    seed=42 + int(time.time()) % 100,
                )

    def _build_prompt_and_gt(env_name: str, ex: dict):
        """Env-aware prompt construction + ground-truth extraction.
        Returns (prompt_text, gt_blob) where gt_blob's shape depends
        on the env (see reward dispatcher below)."""
        ds_spec = cfg.env_datasets[env_name]
        prompt_field = ds_spec["prompt_field"]
        question = (ex.get(prompt_field) or "").strip()
        prompt_text = f"{cfg.user_tag}{question}{cfg.assistant_tag}"

        gt_format = ds_spec.get("ground_truth_format")
        if gt_format == "gsm8k_hash":
            gt = ex.get(ds_spec["gt_field"], "")
        elif gt_format == "ifeval_constraints":
            gt = {
                "instruction_id_list": ex.get("instruction_id_list") or [],
                "kwargs": ex.get("kwargs") or [],
            }
        elif gt_format == "mbpp_tests":
            gt = ex.get(ds_spec["gt_field"]) or []
        else:
            gt = None
        return prompt_text, gt

    def _score_completion(
        env_name: str, comp_text: str, gt: object,
    ) -> tuple[float, dict]:
        """Env-aware reward dispatcher. Returns (total_reward, breakdown).
        All envs get compose_template_rewards (shared format signal);
        ifeval/mbpp add their env-specific reward on top."""
        if env_name == "math":
            return compose_template_rewards(
                comp_text, ground_truth_answer=gt,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                exact_format_reward=cfg.reward_exact_format,
                approx_format_pos=cfg.reward_approx_format_pos,
                approx_format_neg=cfg.reward_approx_format_neg,
                answer_check=True,
                number_check_reward=cfg.reward_number_match,
                number_check_penalty=cfg.reward_number_miss,
                strict_template_weight=cfg.reward_strict_template_weight,
            )
        if env_name == "ifeval":
            total, bd = compose_template_rewards(
                comp_text, ground_truth_answer=None,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                exact_format_reward=cfg.reward_exact_format,
                approx_format_pos=cfg.reward_approx_format_pos,
                approx_format_neg=cfg.reward_approx_format_neg,
                answer_check=False,
                strict_template_weight=cfg.reward_strict_template_weight,
            )
            ifeval_s, ifeval_bd = ifeval_constraint_reward(
                comp_text,
                instruction_id_list=gt["instruction_id_list"] if gt else None,
                kwargs_list=gt["kwargs"] if gt else None,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
            )
            total += ifeval_s
            bd["r_ifeval"] = ifeval_s
            bd["ifeval_verdict"] = ifeval_bd.get("verdict", "")
            bd["ifeval_hits"] = ifeval_bd.get("constraints_hit", 0)
            bd["ifeval_misses"] = ifeval_bd.get("constraints_miss", 0)
            bd["total_reward"] = total
            return total, bd
        if env_name == "mbpp_code":
            total, bd = compose_template_rewards(
                comp_text, ground_truth_answer=None,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                exact_format_reward=cfg.reward_exact_format,
                approx_format_pos=cfg.reward_approx_format_pos,
                approx_format_neg=cfg.reward_approx_format_neg,
                answer_check=False,
                strict_template_weight=cfg.reward_strict_template_weight,
            )
            # Sandboxed exec: minimal env (no secrets), tempdir cwd,
            # process-group kill on timeout, absolute python path.
            # Modal containers ARE the outer isolation layer; this
            # in-process hardening is defence-in-depth. See
            # rewards.py::mbpp_test_reward for the full safety model.
            mbpp_s, mbpp_bd = mbpp_test_reward(
                comp_text,
                test_list=gt if isinstance(gt, list) else None,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                allow_unsafe_exec=True,  # explicit opt-in
            )
            total += mbpp_s
            bd["r_mbpp"] = mbpp_s
            bd["mbpp_verdict"] = mbpp_bd.get("verdict", "")
            bd["total_reward"] = total
            return total, bd
        raise ValueError(f"Unknown env: {env_name}")

    # Resume scan
    ckpt_dir = "/vol/checkpoints/v5"
    os.makedirs(ckpt_dir, exist_ok=True)
    import glob as _glob
    best_step = -1
    best_ckpt: str | None = None
    for pattern in (
        f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_step_*.pt",
        f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_rescue_step_*.pt",
    ):
        for f in _glob.glob(pattern):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if s > best_step or (s == best_step and "rescue" in f):
                best_step = s
                best_ckpt = f
    start_step = 0
    if best_step > 0 and best_ckpt is not None:
        print(f"Found {cfg.stage_prefix} checkpoint at step {best_step}: {best_ckpt}")
        resume_ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        load_model_state_or_raise(
            inner, resume_ckpt["model_state_dict"],
            context=f"grpo_multi resume from {best_ckpt}",
        )
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
        start_step = resume_ckpt.get("step", best_step) + 1
        print(f"  Resumed at step {start_step}")

    # Reward EMA per env (signal-quality monitor)
    ema_overall = RewardEMA(alpha=0.1, print_every_n_calls=cfg.log_interval)
    ema_per_env = {n: RewardEMA(alpha=0.1) for n in cfg.env_names}

    # Env sampler — weighted random, seeded so reruns are reproducible
    env_rng = _random.Random(42 + start_step)

    def _sample_env() -> str:
        return env_rng.choices(cfg.env_names, weights=cfg.env_weights, k=1)[0]

    start_time = time.time()
    print(f"\nStarting training at step {start_step}...")

    for step in range(start_step, cfg.total_steps):
        # LR schedule (cosine with re-warm anchor)
        anchor = getattr(cfg, "lr_anchor_step", 0)
        eff_step = max(step - anchor, 0)
        eff_total = max(cfg.total_steps - anchor, 1)
        if eff_step < cfg.warmup_steps:
            lr = cfg.peak_lr * eff_step / cfg.warmup_steps
        else:
            progress = (eff_step - cfg.warmup_steps) / max(
                eff_total - cfg.warmup_steps, 1,
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
        step_rewards: list[float] = []
        step_env_rewards: dict[str, list[float]] = {n: [] for n in cfg.env_names}
        step_env_counts: dict[str, int] = {n: 0 for n in cfg.env_names}

        for _accum in range(cfg.grad_accum_steps):
            env_name = _sample_env()
            step_env_counts[env_name] += 1
            ex = _next_example(env_name)
            prompt_text, gt = _build_prompt_and_gt(env_name, ex)
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor(
                [prompt_ids], dtype=torch.long, device=device,
            )
            prompt_len = len(prompt_ids)

            # Group rollout — KV-cached, batched, with stop tokens.
            prompt_batch = prompt_tensor.expand(cfg.group_size, -1).contiguous()
            with torch.no_grad():
                generated_batch = inner_for_gen.generate(
                    prompt_batch,
                    max_new_tokens=cfg.max_gen_len,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tok.eos_token_id,
                    stop_token_ids=list(cfg.stop_token_ids),
                )

            # Truncate at first EOS / stop token in completion region.
            stop_set = {tok.eos_token_id, *cfg.stop_token_ids}
            completions = []
            for row in generated_batch:
                comp_region = row[prompt_len:]
                stop_hits = torch.tensor(
                    [t.item() in stop_set for t in comp_region],
                    device=row.device,
                )
                hit_pos = stop_hits.nonzero(as_tuple=False)
                if hit_pos.numel() > 0:
                    first = int(hit_pos[0].item())
                    completions.append(row[: prompt_len + first + 1])
                else:
                    completions.append(row)

            # Score each completion using env-aware reward dispatcher.
            rewards: list[float] = []
            for comp_ids in completions:
                comp_text = tok.decode(
                    comp_ids[prompt_len:].tolist(),
                    skip_special_tokens=False,
                )
                r, bd = _score_completion(env_name, comp_text, gt)
                rewards.append(r)
                step_env_rewards[env_name].append(r)
            step_rewards.extend(rewards)

            advantages = compute_group_advantages(rewards)
            for comp_ids, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue
                comp_ids = comp_ids[:cfg.seq_len].to(device)
                comp_len = len(comp_ids) - prompt_len
                if comp_len <= 0:
                    continue

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(comp_ids.unsqueeze(0))
                    logits = out.logits[
                        0, :, :model_config.real_vocab_size,
                    ].float()
                shift_logits = logits[prompt_len - 1:-1]
                shift_labels = comp_ids[prompt_len:]
                policy_lp = F.log_softmax(shift_logits, dim=-1).gather(
                    1, shift_labels.unsqueeze(1),
                ).squeeze(1)

                with torch.no_grad():
                    ref_out = ref_model(comp_ids.unsqueeze(0))
                    ref_logits = ref_out.logits[
                        0, :, :model_config.real_vocab_size,
                    ].float()
                ref_shift = ref_logits[prompt_len - 1:-1]
                ref_lp = F.log_softmax(ref_shift, dim=-1).gather(
                    1, shift_labels.unsqueeze(1),
                ).squeeze(1)

                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -(policy_lp * adv_t).mean()
                log_ratio = ref_lp - policy_lp
                approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
                loss = (
                    policy_loss + cfg.kl_coeff * approx_kl
                ) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()
                step_kl += approx_kl.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        apply_router_balance_updates(model)

        # Per-env EMA updates
        for n, rs in step_env_rewards.items():
            if rs:
                ema_per_env[n].update(sum(rs) / len(rs))
        mean_reward_step = (
            sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        )
        ema_overall.update(
            mean_reward_step,
            **{
                f"env_{n}": step_env_counts[n] for n in cfg.env_names
            },
        )

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            n_rollouts = max(len(step_rewards), 1)
            env_breakdown = "  ".join(
                f"{n}={step_env_counts[n]}" for n in cfg.env_names
            )
            print(
                f"step {step:>5d}/{cfg.total_steps} | "
                f"loss {step_loss:.4f} | reward {mean_reward_step:+.3f} "
                f"(ema {ema_overall.value:+.3f}) | "
                f"kl {step_kl/n_rollouts:.4f} | lr {lr:.2e} | "
                f"vram {vram:.1f}GB | elapsed {elapsed:.0f}s",
                flush=True,
            )
            print(
                f"           envs: {env_breakdown}",
                flush=True,
            )
            per_env_str = "  ".join(
                f"{n}={ema_per_env[n].value:+.3f}"
                if ema_per_env[n].value is not None else f"{n}=—"
                for n in cfg.env_names
            )
            print(f"           ema_reward_per_env: {per_env_str}", flush=True)

            if use_wandb:
                log_dict = {
                    "grpo_multi/loss": step_loss,
                    "grpo_multi/mean_reward": mean_reward_step,
                    "grpo_multi/ema_reward": ema_overall.value or 0.0,
                    "grpo_multi/approx_kl": step_kl / n_rollouts,
                    "grpo_multi/lr": lr,
                    "grpo_multi/vram_gb": vram,
                }
                for n in cfg.env_names:
                    log_dict[f"grpo_multi/env_{n}_count"] = step_env_counts[n]
                    if ema_per_env[n].value is not None:
                        log_dict[f"grpo_multi/env_{n}_ema_reward"] = (
                            ema_per_env[n].value
                        )
                wandb.log(log_dict, step=step)

        # Checkpoints
        if step > 0 and step % cfg.ckpt_interval == 0:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_out = f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_out)
            vol.commit()
            print(f"  -> Checkpoint saved: {ckpt_out}", flush=True)

        # 23h safety
        if time.time() - start_time > 82_800:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            rescue_path = (
                f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_rescue_step_{step}.pt"
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

    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_out = f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_final.pt"
    torch.save({
        "model_state_dict": inner.state_dict(),
        "training_stage": cfg.stage_prefix,
    }, final_out)
    vol.commit()
    elapsed_h = (time.time() - start_time) / 3600
    print(f"\n{cfg.stage_prefix} complete. {cfg.total_steps} steps in "
          f"{elapsed_h:.1f}h. Final ckpt: {final_out}")
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
    --stage pretrain_extend  Continued pretraining on top of SFT-ultralong ckpt
                             (~1,800 steps, seq 4096, math/science/code mix +
                              SFT-formatted rehearsal, HRA frozen)
    --stage pretrain_extend2 Broader mid-training on top of GRPO step-700 ckpt
                             (~3,000 steps, seq 2048, 30/40/15/15
                              code/math/reasoning/general mix with R1
                              cold-start traces, HRA frozen)
    --stage pretrain_extend2_sanity  50-step smoke test of the extend2
                             pipeline; validates all 10 streams + format
                             functions before committing $28 on the full run
    --stage sft            Balanced SFT on the final pretrained checkpoint
    --stage sft_long       Long-context SFT (seq 4096) resuming from sft_final.pt with Nemotron mix
    --stage sft_ultralong  Ultra-long-context SFT (seq 8192) resuming from sft_long_final.pt
    --stage sft_refresh    Short format-anchor SFT on top of extend_final.pt
                             (500 steps, peak LR 5e-6, no tool_calling)
    --stage sft_math       Math-only SFT polish on top of sft_refresh_final.pt
                             (1,000 steps, peak LR 3e-6, GSM8K + Orca + MathInstruct + NuminaMath)
    --stage evaluate       lm-eval-harness pass (gsm8k + IFEval + MMLU-stem). Args:
                             --ckpt-name <filename in /vol/checkpoints/v5/>
                             --tag <pre-grpo|post-grpo|...>
                             --tasks <comma-separated; default 8-task suite:
                                gsm8k, ifeval, mmlu_stem, hellaswag,
                                arc_easy, arc_challenge, piqa, winogrande>
                             --limit <int or None for full benchmark>
    --stage grpo           GRPO RL on the SFT checkpoint (verifiable math rewards)
    """
    if stage == "sanity":
        sanity.remote()
    elif stage == "sweep":
        sweep.remote()
    elif stage == "ablate":
        ablate.remote()
    elif stage == "pretrain":
        pretrain.remote()
    elif stage == "pretrain_extend":
        pretrain_extend.remote()
    elif stage == "pretrain_extend2":
        # .spawn() instead of .remote() — the local CLI exits cleanly
        # in detached mode, and .remote() inside a local_entrypoint can
        # be cancelled when the caller disconnects (Modal warns about
        # this on launch). Observed: extend2 cancelled at ~2 min from
        # container start regardless of compile/wandb/ckpt/log config.
        # .spawn() is true fire-and-forget; the function continues even
        # if the local entrypoint exits immediately.
        call = pretrain_extend2.spawn()
        print(f"Spawned pretrain_extend2 as call: {call.object_id}")
    elif stage == "pretrain_extend2_sanity":
        call = pretrain_extend2_sanity.spawn()
        print(f"Spawned pretrain_extend2_sanity as call: {call.object_id}")
    elif stage == "loop_fix":
        call = loop_fix.spawn()
        print(f"Spawned loop_fix as call: {call.object_id}")
    elif stage == "loop_fix_sanity":
        call = loop_fix_sanity.spawn()
        print(f"Spawned loop_fix_sanity as call: {call.object_id}")
    elif stage == "loop_fix_v2":
        call = loop_fix_v2.spawn()
        print(f"Spawned loop_fix_v2 as call: {call.object_id}")
    elif stage == "loop_fix_v2_sanity":
        call = loop_fix_v2_sanity.spawn()
        print(f"Spawned loop_fix_v2_sanity as call: {call.object_id}")
    elif stage == "pretrain_extend3":
        call = pretrain_extend3.spawn()
        print(f"Spawned pretrain_extend3 as call: {call.object_id}")
    elif stage == "pretrain_extend3_sanity":
        call = pretrain_extend3_sanity.spawn()
        print(f"Spawned pretrain_extend3_sanity as call: {call.object_id}")
    elif stage == "mopd":
        call = mopd.spawn()
        print(f"Spawned mopd as call: {call.object_id}")
    elif stage == "mopd_sanity":
        call = mopd_sanity.spawn()
        print(f"Spawned mopd_sanity as call: {call.object_id}")
    elif stage == "grpo_multi":
        call = grpo_multi.spawn()
        print(f"Spawned grpo_multi as call: {call.object_id}")
    elif stage == "grpo_multi_sanity":
        call = grpo_multi_sanity.spawn()
        print(f"Spawned grpo_multi_sanity as call: {call.object_id}")
    elif stage == "sft":
        sft.remote()
    elif stage == "sft_long":
        sft_long.remote()
    elif stage == "sft_ultralong":
        sft_ultralong.remote()
    elif stage == "sft_refresh":
        sft_refresh.remote()
    elif stage == "sft_math":
        sft_math.remote()
    elif stage == "evaluate":
        evaluate.remote()
    elif stage == "grpo":
        grpo.remote()
    else:
        print(
            f"Unknown stage: {stage}. "
            f"Use sanity, sweep, ablate, pretrain, pretrain_extend, "
            f"sft, sft_long, sft_ultralong, sft_refresh, sft_math, "
            f"evaluate, or grpo"
        )
