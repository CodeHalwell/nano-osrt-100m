"""Training configurations for NanoOSRT.

v5 architecture: Mixtral-style MoE (8 routed × top-2, 1 shared, no dense FFN),
Switch balance loss, orthogonal expert init, eval-time drop-free capacity.

Progressive curriculum:
  Phase 1 (Foundation):  seq_len 2048, FineWeb-Edu + CodeParrot
  Phase 2 (Knowledge):   seq_len 4096, FineWeb-Edu + CodeParrot + Wikipedia
  Phase 3 (Instruction): seq_len 8192, SmolTalk + Evol-Code + OpenHermes

Post-training:
  SFT:  Balanced math + code + STEM + general (native tag format)
  GRPO: Verifiable math rewards


# Optimizer × routing ablation (Foundation-matched cells)
# ──────────────────────────────────────────────────────────────────────

Ran A/B/C to 1200 steps in `--stage ablate`; stopped D at step 600
once its task loss remained tied with C but the raw router had already
collapsed. Headline numbers:

| Cell | Optimizer | Aux  | Best / seen task | prebias emin | bal  |
|------|-----------|-----:|-----------------:|-------------:|-----:|
| A    | Lion      | 0.10 |             ~7.0 |       ~0.002 | ~1.2 |
| B    | Lion      | 0.0  |             ~7.6 |        0.000 | ~3.9 |
| C    | Muon      | 0.10 |         **3.43** |    **0.105** | 1.02 |
| D    | Muon      | 0.0  |   4.66 @ step 600|       ~0.001 | ~2.3 |

Three load-bearing conclusions that drive the v5 defaults:

1. Muon is a ~4-nat task-loss win at this scale, regardless of routing
   scheme. C and D both hit task < 5.0 by step 450; A and B were still
   at ~7.2 there.
2. Gradient aux loss is necessary for router health regardless of
   optimizer — bias controller alone collapses the raw router under
   both Lion and Muon. The DeepSeek-V3 "auxiliary-loss-free" claim
   does not hold at 363M params on this curriculum.
3. C (Muon + aux) is the production recipe: best loss, best balance,
   best emin, best margin. D gets C's loss but with B-style collapse
   on the raw router, which would degrade once task complexity grows
   beyond what 2-3 active experts can fit (Phases 2/3).

Defaults below reflect this. To rerun the ablation, use --stage ablate.


# Optional A/B configurations
# ──────────────────────────────────────────────────────────────────────

DeepSeek-style aux-loss-free routing (research only; failed here)
-----------------------------------------------------------------
v5 ships with both a gradient-driven Switch balance aux loss
(coefficient `router_aux_loss_coeff`, default 0.10) and a
non-gradient per-loop bias controller (`router_balance_bias_*`).
DeepSeek-V3 reports that the bias controller alone is sufficient on
their 671B model, and that removing the gradient aux loss eliminates
the well-known specialisation-vs-balance tradeoff (the gradient term
forces uniformity even when token-context says it shouldn't).

The v5 ablation rejected that recipe at 363M params: both bias-only
cells collapsed the raw pre-bias router. Cell D proved why clean
metrics are not enough: Muon kept task loss near Cell C through step
600, but prebias emin fell to ~0.001 and the model was effectively
using only a small expert subset. Treat aux-loss-free runs as research
or failure-reproduction only unless the routing algorithm itself
changes.

To reproduce aux-loss-free routing on this codebase, override the model
config when constructing it:

    model_config = NanoOSRTConfig(
        router_aux_loss_coeff=0.0,        # disable gradient aux
        router_balance_bias_enabled=True, # keep bias controller
        router_balance_bias_update_rate=0.10,
        ...,
    )

The clean four-metric Phase-1 health gate is not sufficient for this
experiment because the bias controller can hide collapse. Watch the
raw metrics instead:
`moe/prebias_marginal_entropy_mean`,
`moe/prebias_expert_min_mean`,
`moe/prebias_raw_max_prob_mean`, and
`moe/prebias_top_margin_mean`.
If `moe/prebias_expert_min_mean` falls near zero while task loss still
looks good, the recipe has failed even if clean balance appears healthy.

Sweep template (drop into app.py near the existing `sweep` stage)::

    sweep_configs = [
        {"name": "aux_only",  "aux": 0.10, "bias": True},   # current default
        {"name": "bias_only", "aux": 0.0,  "bias": True},   # DeepSeek-style
        {"name": "both_low",  "aux": 0.03, "bias": True},   # belt + braces
    ]
"""


class PretrainConfig:
    """Pre-training hyperparameters for v5."""

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    # Realistic step horizon for the available compute budget. The
    # original 300_000 was the "complete the full curriculum" target
    # but that requires ~$1700+ on Modal H100. With the realistic
    # multi-account budget (~$77 across 3 accounts) we're aiming for
    # roughly Chinchilla-optimal on active params (~3.8B tokens =
    # step ~21000 at Phase 2 sizes). 25_000 leaves headroom past that
    # so the cosine taper hits ~12% of peak LR by step 21000 — proper
    # cooked training rather than running at peak forever.
    # Bumped from 300_000 mid-run to fix the cosine being effectively
    # a constant LR at this horizon.
    total_steps: int = 25_000
    warmup_steps: int = 3_000
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.3
    grad_clip: float = 1.0
    log_interval: int = 50
    eval_interval: int = 1_000
    eval_steps: int = 20           # number of batches per eval
    # Frequent ckpts protect against budget-driven Modal kills: with a
    # capped credit pool, the function dies hard (no clean shutdown,
    # no rescue ckpt) when the wallet hits zero. 500-step intervals
    # bound progress loss to ~30 min on H100 at this throughput.
    ckpt_interval: int = 500
    # Default optimizer is Muon hybrid (Muon for 2D matrix weights,
    # AdamW for embeddings/norms/scalars/router/loop_embeddings). The
    # 1200-step ablation (A/B/C to completion; D stopped at step 600)
    # showed Muon+aux delivers a
    # ~4-nat task-loss improvement over Lion+aux at step 1200 (cell C
    # task ~3.4 vs cell A task ~7.4) AND keeps the learned pre-bias
    # routed-expert population balanced (prebias emin > 0.10 vs cell A's
    # late-warmup collapse to emin < 0.01). Lion is still available
    # via optimizer_name="lion" for comparison runs. AdamW is the
    # fallback when optimizer_name is anything else.
    optimizer_name: str = "muon"
    # Muon LR (used only when optimizer_name == "muon"). The Newton-Schulz
    # update is normalised, so Muon's effective step size is much smaller
    # per parameter than Lion/AdamW. The 1200-step Cell C run held
    # task-loss steady at lr=0.02 through 23 % of warmup with no fatal
    # divergence. If a full Phase 1 (10k steps, peak at step 3000)
    # destabilises, drop to 0.015 first — that's the next thing to
    # try before deeper changes. AdamW (the other half of the hybrid)
    # keeps using peak_lr / min_lr.
    muon_lr: float = 0.02
    muon_min_lr: float = 2e-3

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt"
    # Suffix the optimizer in the W&B name so dashboard runs from
    # different optimizer configs don't visually pile up on top of the
    # historical Lion runs. Override per-run if you want a custom label.
    wandb_run_name: str = "osrt-pretrain-muon"
    wandb_run_id: str = ""

    # --- Success criteria for Phase 1 (Foundation) ---
    # v4's router was never alive. v5 uses a four-metric check that would
    # have correctly diagnosed v4's failure (batch-marginal entropy stayed
    # high while per-token entropy also stayed high). All four must hold by
    # step `early_stop_check_step` or training is considered failed.
    early_stop_check_step: int = 5_000
    min_per_token_entropy_drop: float = 0.55   # init 2.08 → 1.53 (ln 8 = 2.08)
    min_raw_max_prob: float = 0.30             # well above uniform 1/8 = 0.125
    min_top_margin: float = 0.10               # clear gap between rank 0 and 1
    min_marginal_entropy: float = 1.80         # balanced across experts
    # The four checks above use clean deployed routing (router + balance bias).
    # These guardrails make sure the learned pre-bias router is not secretly
    # collapsed while the non-gradient bias controller hides it.
    min_prebias_marginal_entropy: float = 1.55
    min_prebias_expert_fraction: float = 0.01
    max_bias_saturation_fraction: float = 0.85

    # --- Router exploration ---
    # Sanity runs showed experts can die during the first 20 optimizer steps:
    # once an expert falls out of top-k it receives no task gradient. Add
    # noisy top-k exploration early. Anneal after LR warmup peaks so the
    # router does not face a rising task-loss gradient with no exploration,
    # but finish 1k steps before the 5k clean-router health gate.
    router_gumbel_tau_init: float = 0.5
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 4_000

    # Progressive seq_len curriculum
    # Tokens per step per phase:
    #   Phase 1 (foundation, 10K steps):  8 × 8  × 2048 = 131K tok/step → ~1.3B
    #   Phase 2 (knowledge, 240K steps):  4 × 16 × 4096 = 262K tok/step → ~63B
    #   Phase 3 (instruction, 50K steps): 2 × 32 × 8192 = 524K tok/step → ~26B
    # Total budget: ~90B tokens if the full 300K schedule completes.
    phases: dict = {  # noqa: RUF012
        "foundation": {
            "start": 0,
            "end": 9_500,
            "seq_len": 2048,
            "grad_accum_steps": 8,
            "datasets": [
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.6,
                },
                {
                    "name": "codeparrot-clean",
                    "hf_id": "codeparrot/codeparrot-clean",
                    "weight": 0.4,
                },
            ],
        },
        "knowledge": {
            "start": 9_500,
            "end": 250_000,
            "seq_len": 4096,
            # Bumped from batch_size=4, grad_accum_steps=16 once the
            # grad-checkpointing threshold was raised (see train.py
            # commit 57513a9). At seq_len 4096 with no checkpointing,
            # H100 80GB had ~31 GB unused at batch 4 (49 GB total).
            # Batch 8 OOMed at 76.7 GB (activations scale super-linearly
            # at this sequence length); batch 6 sits at ~59 GB, leaving
            # ~20 GB headroom for fragmentation and the optimizer step's
            # transient buffers. Effective batch 6*11=66 sequences,
            # close to the prior 4*16=64. If a future GPU has tighter
            # VRAM (3090, A100 40GB), drop batch_size back to 4 with
            # grad_accum_steps=16.
            "batch_size": 6,
            "grad_accum_steps": 11,
            "datasets": [
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.50,
                },
                {
                    "name": "codeparrot-clean",
                    "hf_id": "codeparrot/codeparrot-clean",
                    "weight": 0.30,
                },
                {
                    "name": "wikipedia",
                    "hf_id": "wikimedia/wikipedia",
                    "hf_config": "20231101.en",
                    "weight": 0.20,
                },
            ],
        },
        "instruction": {
            "start": 250_000,
            "end": 300_000,
            "seq_len": 8192,
            "batch_size": 2,
            "grad_accum_steps": 32,
            "datasets": [
                {
                    "name": "smoltalk",
                    "hf_id": "HuggingFaceTB/smoltalk",
                    "hf_config": "all",
                    "weight": 0.30,
                },
                {
                    "name": "evol-instruct-code",
                    "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
                    "weight": 0.20,
                },
                {
                    "name": "openhermes",
                    "hf_id": "teknium/OpenHermes-2.5",
                    "weight": 0.10,
                },
                {
                    "name": "nemotron-post-training-math",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "math",
                    "weight": 0.20,
                },
                {
                    "name": "nemotron-post-training-stem",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "stem",
                    "weight": 0.20,
                },
            ],
        },
    }

    # Budget note: the schedule is aspirational — the user runs in chunks as
    # Modal credits allow. Checkpoints every 1K steps keep stop/resume cheap.
    # Any early stopping still leaves a usable model for SFT.


class PretrainExtendConfig(PretrainConfig):
    """Continued pre-training ("mid-training") on top of an SFT checkpoint.

    Goal: fill the pretrain data-mix gaps identified after SFT —
    specifically math (Nemotron-CC-Math), better code (The Stack v2),
    and scientific text (RedPajama-arxiv). The original pretrain ran
    on FineWeb-Edu + CodeParrot + Wikipedia only, with effectively no
    math content; the result was a model that emits structurally
    correct chat output but can't do double-digit multiplication.

    Resume strategy
    ───────────────
    Loads from `osrt_v5_sft_ultralong_final.pt` rather than the pure
    pretrain ckpt (osrt_v5_step_17000.pt). This keeps the SFT
    investment intact so we don't have to redo SFT from scratch
    afterward. The risks (chat-format erosion, HRA drift, full-token
    loss vs prompt-masked) are mitigated by:

      1. Conservative LR: peak 1.5e-5 (2.5 % of original 6e-4).
      2. SFT-formatted rehearsal data (25 % of mix) — Nemotron rows
         wrapped in <|user|>...<|assistant|><|think|>...<|/think|>
         <|answer|>...<|/answer|> so the model keeps seeing chat
         tags during ostensibly "raw" pretraining.
      3. HRA frozen (`hra_frozen=True`) — the 86 M HRA delta layer
         stays as the SFT-trained delta; only base weights absorb new
         pretrain knowledge. Cleanly separates concerns and gives a
         small ~5–8 % throughput win from skipping HRA backward pass.

    Token budget
    ────────────
    $30 of H100 time (~7.6 hr) at Phase-2-ish throughput
    (~15 sec/step at seq 4096) ≈ 1,800 steps × 270 k tok/step ≈
    485 M new tokens. About 15 % of our prior 3.27 B pretrain budget,
    concentrated in the underrepresented categories.

    Lineage
    ───────
    Output checkpoint: osrt_v5_extend_step_N.pt and
    osrt_v5_extend_final.pt (distinct prefix so resume scans don't
    collide with base pretrain checkpoints). Subsequent SFT
    "refresh" pass (200 steps, ~$4) loads from the extend-final to
    re-anchor chat format if it has degraded.
    """

    # ── Schedule ─────────────────────────────────────────────────────
    # Extended from 1,800 → 2,800 mid-run (post-step 200) to use more
    # of the $30 workspace budget on actual training (Liquid AI / Phi
    # philosophy: small models benefit from heavy overtraining past
    # Chinchilla-optimal). At our 8.4 sec/step throughput, 2,800
    # steps fits ~$26 of compute, leaving ~$4 for a limit-100 eval
    # pass to measure the lift. The cosine schedule recalibrates
    # automatically — at step 200 with new total=2800, LR is at
    # ~99.7 % of peak (vs ~99 % under the old total=1800), so the
    # transition is a tiny upward LR bump (0.5 %), then cosine cools
    # more gently over the longer horizon.
    total_steps: int = 2_800
    warmup_steps: int = 50          # 3 % of original — kept fixed (re-warmup
                                    # already done in steps 0-50)
    peak_lr: float = 1.5e-5         # 2.5 % of original 6e-4
    min_lr: float = 1.5e-6          # cosine to 10 % of peak
    weight_decay: float = 0.1       # softer wd than pretrain (0.3)
    grad_clip: float = 1.0
    log_interval: int = 25
    eval_interval: int = 250
    eval_steps: int = 20
    ckpt_interval: int = 200        # ~14 ckpts over the 2,800-step run

    # Optimizer reuses the Muon hybrid from pretrain. The lower
    # peak_lr also propagates down to Muon via the same _peak_lr
    # tagging in train.py::_set_param_group_lrs. Override muon_lr
    # explicitly so we don't reuse the pretrain-tuned 0.02 (which
    # would shock SFT-flavored weights at this stage).
    optimizer_name: str = "muon"
    muon_lr: float = 5e-3           # 25 % of pretrain's 0.02
    muon_min_lr: float = 5e-4

    # ── Routing exploration ─────────────────────────────────────────
    # Disable Gumbel exploration entirely. The router has been trained
    # for 17k pretrain + 2.7k SFT steps and is well-formed; reintroducing
    # noise would hurt rather than help.
    router_gumbel_tau_init: float = 0.0
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 1   # avoid div-by-zero

    # ── Early-stop gate ────────────────────────────────────────────
    # Push past the budget so the gate never trips — the four-metric
    # gate was tuned for cold-start pretraining where the router needs
    # to specialise from scratch. At extend time the router is already
    # healthy (clean_per_token_H ~1.40, assn ~2.07 in last SFT) and the
    # gate's thresholds (designed for the "is this run salvageable?"
    # question) don't apply.
    early_stop_check_step: int = 9_999_999

    # ── HRA ────────────────────────────────────────────────────────
    # SFT-ultralong ckpt has HRA params (rank 256, +86.1M) in its
    # state_dict — must inject before load. `hra_frozen=True` is a
    # new flag (see train.py::run_pretrain_extend) that sets
    # requires_grad=False on adapters_a/adapters_b after load so the
    # frozen SFT-trained delta layer stays as-is while base absorbs
    # new pretrain content.
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_before_load: bool = True
    hra_frozen: bool = True
    pretrained_checkpoint: str = (
        "/vol/checkpoints/v5/osrt_v5_sft_ultralong_final.pt"
    )

    # Distinct ckpt prefix so this stage's checkpoints
    # (osrt_v5_extend_step_N.pt) don't collide with base pretrain
    # ckpts (osrt_v5_step_N.pt) under the resume scan.
    stage_prefix: str = "extend"

    # W&B labels
    wandb_run_name: str = "osrt-pretrain-extend"
    wandb_run_id: str = ""

    # ── Data mix (single phase, seq 4096) ──────────────────────────
    # Three new datasets + two existing for general-capability
    # maintenance + two SFT-formatted rehearsal streams.
    #
    # Token-weighted sampling (debt-based, see TokenStream._pick_stream)
    # produces actual token mixes matching these weights regardless of
    # per-stream example length. Weights need not sum to exactly 1; the
    # sampler normalises.
    phases: dict = {  # noqa: RUF012
        "extend": {
            "start": 0,
            "end": 1_800,
            "seq_len": 4096,
            # Phase 2 sizing — known to fit comfortably on H100 80GB
            # at ~60 GB VRAM with the bump from 4×16 to 6×11.
            "batch_size": 6,
            "grad_accum_steps": 11,
            "datasets": [
                # ── Math (35 %) — biggest gap, biggest expected lift
                {
                    "name": "nemotron-cc-math",
                    "hf_id": "nvidia/Nemotron-CC-Math-v1",
                    # `4plus` subset (FineMath-classifier ≥4) is the
                    # higher-quality 52B-token variant. Quality > quantity
                    # for our 243M-token math budget. Available subsets
                    # are: '3' (133B, broader), '4plus' (52B, higher
                    # quality), '4plus_MIND' (most curated). REQUIRES
                    # gated-access approval at
                    # https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1
                    "hf_config": "4plus",
                    "weight": 0.35,
                    "format": "nemotron_math",
                },
                # ── Math/scientific web text (12 %)
                # OpenWebMath replaces the original RedPajama-arxiv plan
                # because RedPajama-Data-1T uses a deprecated Python
                # loader script that modern HF datasets no longer
                # supports ("Dataset scripts are no longer supported").
                # OpenWebMath is the well-known 14.7B-token math web
                # corpus that Nemotron-CC-Math itself is positioned
                # against — provides math/science diversity beyond
                # Nemotron's curation pipeline.
                {
                    "name": "open-web-math",
                    "hf_id": "open-web-math/open-web-math",
                    "weight": 0.12,
                    "format": "arxiv",  # same `text` field shape
                },
                # ── Code (12 %) — CodeParrot
                # Originally planned bigcode/the-stack-smol but it is
                # gated. CodeParrot-Clean is the same dataset already
                # used in original pretrain so we know it streams
                # reliably; the goal here is to maintain code
                # capability under the new mix, not introduce a
                # different code distribution.
                {
                    "name": "codeparrot-clean",
                    "hf_id": "codeparrot/codeparrot-clean",
                    "weight": 0.12,
                    # Default extractor handles the `content` field
                    # natively (see TokenStream._extract_text).
                },
                # ── General-capability maintenance (16 %)
                {
                    "name": "fineweb-edu",
                    "hf_id": "HuggingFaceFW/fineweb-edu",
                    "weight": 0.08,
                },
                {
                    "name": "wikipedia",
                    "hf_id": "wikimedia/wikipedia",
                    "hf_config": "20231101.en",
                    "weight": 0.08,
                },
                # ── SFT-formatted rehearsal (25 %) — anti-forgetting
                # wraps Nemotron rows in <|user|>...<|/answer|> chat
                # schema before tokenisation. Pretrain loss is full-
                # token (no masking) so the model trains on every
                # token in the formatted string including chat tags.
                {
                    "name": "nemotron-math-rehearsal",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "math",
                    "weight": 0.15,
                    "format": "nemotron_sft",
                },
                {
                    "name": "nemotron-stem-rehearsal",
                    "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
                    "split": "stem",
                    "weight": 0.10,
                    "format": "nemotron_sft",
                },
            ],
        },
    }


class SFTConfig:
    """Balanced SFT config for v5 — math + code + STEM + general."""

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 8
    total_steps: int = 5_000
    warmup_steps: int = 250
    peak_lr: float = 1.5e-5
    min_lr: float = 1.5e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 25
    ckpt_interval: int = 500
    optimizer_name: str = "adamw"
    seq_len: int = 2048              # short seq_len + packing (see v4_sft_data)

    # HRA
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_lr: float = 7.5e-5
    hra_freeze_pretrained: bool = False
    hra_before_load: bool = False
    stage_prefix: str = "sft"

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt"
    wandb_run_name: str = "osrt-sft"
    wandb_run_id: str = ""

    # Pretrained checkpoint. Points to the explicit step file rather
    # than `osrt_v5_final.pt` because pretraining was stopped early at
    # step 17000 once the eval-loss curve flatlined (eval 3.48 / ppl
    # 32.4 — Chinchilla-knee on 192M active params). Path is set
    # explicitly so SFT loads the actual snapshot, not a stale
    # filename convention.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_step_17000.pt"

    # Chat format (native single-token tags)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"

    # Balanced dataset mixture (same as v4)
    datasets: list = [  # noqa: RUF012
        # Math (25%)
        {
            "name": "gsm8k",
            "hf_id": "openai/gsm8k",
            "hf_config": "main",
            "split": "train",
            "weight": 0.10,
            "format": "gsm8k",
        },
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.15,
            "format": "numina_math",
        },
        # Code (25%)
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.15,
            "format": "evol_code",
        },
        {
            "name": "code-instructions-122k",
            "hf_id": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca_code",
        },
        # STEM (20%)
        {
            "name": "orca-math",
            "hf_id": "microsoft/orca-math-word-problems-200k",
            "split": "train",
            "weight": 0.10,
            "format": "orca_math",
        },
        {
            "name": "math-instruct",
            "hf_id": "TIGER-Lab/MathInstruct",
            "split": "train",
            "weight": 0.10,
            "format": "math_instruct",
        },
        # General (20%)
        {
            "name": "alpaca-cleaned",
            "hf_id": "yahma/alpaca-cleaned",
            "split": "train",
            "weight": 0.10,
            "format": "alpaca",
        },
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.10,
            "format": "openhermes",
        },
        # Instruction following (10%)
        {
            "name": "ifeval-like",
            "hf_id": "argilla/ifeval-like-data",
            "hf_config": "filtered",
            "split": "train",
            "weight": 0.05,
            "format": "ifeval",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.05,
            "format": "longform",
        },
    ]


class SFTLongConfig(SFTConfig):
    """Long-context SFT — resumes from the seq-2048 SFT checkpoint and
    fine-tunes at seq_len 4096 with a Nemotron-heavy data mix.

    Why: NuminaMath CoT and longform responses cluster in the
    1500-3000 token range and got truncated under the seq 2048 base
    SFT. This phase teaches the model to maintain quality over longer
    completions. Includes Nvidia Nemotron splits (math, stem, code,
    tool_calling) which the base SFT didn't see — Nemotron has
    explicit `reasoning` field that maps directly to our
    `<|think|>{}<|/think|>` block.

    HRA contract:
      - `hra_before_load=True` because the saved SFT ckpt already has
        HRA params in its state_dict; injecting HRA structure first
        lets the load place them correctly.
      - `hra_enabled=True` (inherited) — keeps the existing rank-256
        adapters trained in the base SFT.

    LR contract:
      - Lower peak (5e-6 vs base SFT's 1.5e-5) because we're
        fine-tuning a fine-tune. Aggressive LR risks washing out the
        base SFT learning.
      - Cosine over total_steps=1000 cools to min_lr by the end.
    """

    total_steps: int = 1_000
    warmup_steps: int = 50
    seq_len: int = 4096
    batch_size: int = 4               # halved from 8 to fit longer ctx
    grad_accum_steps: int = 16        # doubled to keep effective batch 64
    peak_lr: float = 5e-6
    min_lr: float = 5e-7
    log_interval: int = 25
    ckpt_interval: int = 250

    # Resume from the base SFT checkpoint with HRA already applied.
    # Points at the explicit step file rather than osrt_v5_sft_final.pt
    # because base SFT was stopped early at step 2500 (Option B budget
    # plan — eval loss already at 1.02 train, no point in chasing the
    # remaining 2500 steps with diminishing returns and a loss curve
    # that's already in the "starting to memorise" zone).
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_step_2500.pt"
    hra_before_load: bool = True
    stage_prefix: str = "sft_long"

    wandb_run_name: str = "osrt-sft-long"

    # Nvidia Nemotron-heavy mix (60%) plus 40% diversity from existing
    # SFT data to prevent over-fitting to one teacher's style.
    datasets: list = [  # noqa: RUF012
        # Nvidia Nemotron Post-Training (60% total)
        {
            "name": "nemotron-math",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "math",
            "weight": 0.30,
            "format": "nemotron",
        },
        {
            "name": "nemotron-stem",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "stem",
            "weight": 0.20,
            "format": "nemotron",
        },
        {
            "name": "nemotron-code",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "code",
            "weight": 0.15,
            "format": "nemotron",
        },
        {
            "name": "nemotron-tool-calling",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "tool_calling",
            "weight": 0.10,
            "format": "nemotron_tool_calling",
        },
        # Diversity (25%) — long-form reasoning + code from non-Nvidia
        # sources to keep the teacher distribution mixed.
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.10,
            "format": "numina_math",
        },
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.10,
            "format": "evol_code",
        },
        {
            "name": "longform",
            "hf_id": "akoksal/LongForm",
            "split": "train",
            "weight": 0.05,
            "format": "longform",
        },
    ]


class SFTUltraLongConfig(SFTLongConfig):
    """Ultra-long-context SFT — resumes from the seq-4096 SFT-long
    checkpoint and pushes to seq_len 8192.

    Why: NuminaMath multi-page derivations and multi-file code
    generations exceed the 4096 context window. Pushing to 8192
    teaches the model to maintain coherence across genuinely long
    completions before GRPO (which runs at seq_len 8192).

    Compute cost: attention is O(N²) so seq 8192 vs 4096 is 4× the
    attention compute per token. With 4× more tokens per microbatch
    too (4096 → 8192 plus same effective batch shape), this is the
    most expensive SFT phase — kept short (200 steps) to fit budget.
    Trimmed from the original 500 once it became clear that SFT-long
    delivers most of the long-context adaptation; ultralong is a
    polish pass to anchor seq 8192 behaviour before GRPO, not a
    full curriculum stage.

    HRA contract:
      - hra_before_load=True (inherited) — sft_long_final.pt has HRA
        params already.
      - Same HRA rank 256 — keeps continuity with the previous SFT
        passes' learned adaptations.

    LR contract:
      - Even cooler peak (3e-6) — third successive fine-tune, the
        adapters are well-formed and need light polish, not heavy
        re-shaping.
      - Cosine over total_steps=200; warmup scaled down to 10 steps
        (5 % of total) to match the shorter horizon. Without this the
        old 25-step warmup would consume 12.5 % of the run before any
        cosine taper — too much.
    """

    total_steps: int = 200
    warmup_steps: int = 10
    seq_len: int = 8192
    # Halve batch_size again, double grad_accum_steps to keep effective
    # batch at 64 sequences per gradient step. Memory budget at 80 GB
    # H100: ~50-60 GB for activations at seq 8192 + batch 2 (estimated;
    # if it OOMs, drop batch to 1 and accum to 64).
    batch_size: int = 2
    grad_accum_steps: int = 32
    peak_lr: float = 3e-6
    min_lr: float = 3e-7
    log_interval: int = 10
    ckpt_interval: int = 50

    # Points at the explicit step file rather than osrt_v5_sft_long_final.pt
    # because SFT-long was stopped early at step 500 (budget-driven cut to
    # preserve compute for SFT-ultralong + GRPO; final loss 1.32 vs the
    # ~1.10 a full 1000-step run would have hit). The step-500 ckpt has the
    # same HRA contract as the final would have — no functional difference.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_long_step_500.pt"
    stage_prefix: str = "sft_ultralong"
    wandb_run_name: str = "osrt-sft-ultralong"

    # Same Nemotron-heavy + diversity mix as SFTLongConfig (inherited
    # from `datasets`). The mix doesn't change between SFT-long and
    # SFT-ultralong — only the context window does.


class SFTRefreshConfig(SFTConfig):
    """Short SFT pass to re-anchor chat format after pretrain_extend.

    Why this exists
    ───────────────
    Local probe on osrt_v5_extend_final.pt (post-pretrain-extend) showed
    chat-format degradation despite the 25 % rehearsal mix during
    extend. Symptoms:

      * Special tokens still emitted but in wrong positions —
        "<|think|>on<|/think|><|answer|><think>... reasoning ..."
        with <|/answer|> never closed. lm-eval's answer extractor
        sees broken/missing structure and returns [invalid].
      * Some prompts produce immediate <|end_of_text|> with no
        content at all.
      * Math content is genuinely improving (model decomposes
        17×23 as 17×20 + 17×3 — correct method, wrong arithmetic
        execution) — but the format wrapping is broken.

    Tool-call hallucination is FIXED by extend (no tool_calling data
    seen for 2,800 steps) — confirmed by probe. So this refresh
    deliberately keeps tool_calling OUT of the mix to preserve that
    win.

    Design
    ──────
    Very short (500 steps) at very low LR (5e-6, 33 % of SFTConfig's
    1.5e-5). Goal is to refine where the model places its existing
    chat tags, not to reshape what it knows about math/code. The
    extend gave us a genuinely better base; this just re-anchors the
    SFT format on top of it.

    HRA stays trainable here (hra_freeze_pretrained=False, inherited)
    so the adapters can re-tune toward the new pretrain-extended
    base. The frozen-HRA mode used in pretrain_extend is the opposite
    direction — preserve HRA's old SFT learning while base absorbed
    new content. Now that base has new content, HRA needs to adapt.

    Data mix: 70 % Nemotron post-training (math/stem/code, NO
    tool_calling) + 30 % SFTConfig diversity, all chat-formatted with
    response-only loss masking (standard SFT behaviour from
    sft_data.py).

    Expected outcome: clean <|think|>...<|/think|><|answer|>...
    <|/answer|> emission again, math reasoning improvements
    preserved. Should restore extraction validity for eval.
    """

    total_steps: int = 500
    warmup_steps: int = 25
    seq_len: int = 2048
    batch_size: int = 8
    grad_accum_steps: int = 8
    peak_lr: float = 5e-6
    min_lr: float = 5e-7
    log_interval: int = 25
    ckpt_interval: int = 100

    # HRA: keep trainable so adapters re-tune to the new base.
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_lr: float = 2.5e-5            # 33 % of base SFT hra_lr (7.5e-5)
    hra_freeze_pretrained: bool = False
    hra_before_load: bool = True      # extend ckpt has HRA params

    # Resume from the post-extend checkpoint.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_extend_final.pt"
    stage_prefix: str = "sft_refresh"
    wandb_run_name: str = "osrt-sft-refresh"

    # Nemotron-heavy SFT mix WITHOUT tool_calling. The pretrain_extend
    # eliminated tool-call hallucination; we deliberately omit
    # nemotron-tool-calling here to preserve that win. NuminaMath +
    # Evol-Code provide diversity beyond Nemotron's CoT style.
    # All formats are already wired in sft_data.py::FORMAT_FN.
    datasets: list = [  # noqa: RUF012
        # Math (40 %)
        {
            "name": "nemotron-math",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "math",
            "weight": 0.30,
            "format": "nemotron",
        },
        {
            "name": "numina-math-cot",
            "hf_id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "weight": 0.10,
            "format": "numina_math",
        },
        # STEM (20 %)
        {
            "name": "nemotron-stem",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "stem",
            "weight": 0.20,
            "format": "nemotron",
        },
        # Code (20 %)
        {
            "name": "nemotron-code",
            "hf_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": "code",
            "weight": 0.10,
            "format": "nemotron",
        },
        {
            "name": "evol-instruct-code",
            "hf_id": "nickrosh/Evol-Instruct-Code-80k-v1",
            "split": "train",
            "weight": 0.10,
            "format": "evol_code",
        },
        # General/IF (20 %) — keeps the chat-format anchor diverse
        # beyond Nemotron's CoT style so the model re-learns to handle
        # non-math prompts cleanly (the cats / planet-question probe
        # failures suggest this is needed).
        {
            "name": "openhermes",
            "hf_id": "teknium/OpenHermes-2.5",
            "split": "train",
            "weight": 0.10,
            "format": "openhermes",
        },
        {
            "name": "ifeval-like",
            "hf_id": "argilla/ifeval-like-data",
            "hf_config": "filtered",
            "split": "train",
            "weight": 0.10,
            "format": "ifeval",
        },
    ]


class GRPOConfig:
    """GRPO config for v5 — verifiable math rewards on top of SFT."""

    batch_size: int = 4
    grad_accum_steps: int = 4
    total_steps: int = 2_000
    warmup_steps: int = 100
    peak_lr: float = 3e-6
    min_lr: float = 3e-7
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    ckpt_interval: int = 250
    seq_len: int = 8192

    # GRPO-specific
    group_size: int = 16
    max_gen_len: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coeff: float = 0.05
    clip_range: float = 0.2

    # HRA (inherited on top of SFT-injected HRA weights)
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_lr: float = 1.5e-5
    hra_before_load: bool = True

    # Rewards
    correctness_reward: float = 1.0
    format_reward: float = 0.2
    reasoning_bonus: float = 0.3
    truncation_penalty: float = -0.5
    empty_think_penalty: float = -0.1
    length_penalty: float = 0.0

    # Weights & Biases
    wandb_log: bool = True
    wandb_project: str = "nano-osrt"
    wandb_run_name: str = "osrt-grpo"
    wandb_run_id: str = ""

    # Checkpoint
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_sft_final.pt"
    stage_prefix: str = "grpo"

    # Prompt source
    prompt_dataset: str = "openai/gsm8k"
    prompt_config: str = "main"
    prompt_split: str = "train"

    # Chat format (native single-token tags, same as v4/SFT)
    user_tag: str = "<|user|>"
    assistant_tag: str = "<|assistant|>"
    think_open: str = "<|think|>"
    think_close: str = "<|/think|>"
    answer_open: str = "<|answer|>"
    answer_close: str = "<|/answer|>"
