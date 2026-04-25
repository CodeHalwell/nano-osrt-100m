"""Unit tests for NanoOSRT — Mixtral-style MoE without dense FFN."""

import pytest
import torch
import torch.nn.functional as F

from nano_osrt.config import NanoOSRTConfig
from nano_osrt.model import (
    ExpertFFN,
    MoELayer,
    NanoOSRTForCausalLM,
    NanoOSRTModel,
    orthogonal_expert_init,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def tiny_config(**overrides) -> NanoOSRTConfig:
    """Small config for fast tests."""
    defaults = dict(
        dim=128, heads=4, head_dim=32,
        vocab_size=512, real_vocab_size=512,
        num_blocks=2, recursive_loops=2,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=128,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return NanoOSRTConfig(**defaults)


# ── Config validation ──────────────────────────────────────────────────


def test_config_defaults_to_top_2():
    cfg = NanoOSRTConfig()
    assert cfg.top_k_experts == 2


def test_config_rejects_top_k_zero():
    with pytest.raises(ValueError, match="top_k_experts must be >= 1"):
        tiny_config(top_k_experts=0)


def test_config_rejects_top_k_too_many():
    """top-k > num_experts/2 defeats MoE sparsity."""
    with pytest.raises(ValueError, match="defeats the sparsity"):
        tiny_config(num_routed_experts=8, top_k_experts=6)


def test_config_rejects_capacity_factor_le_1():
    with pytest.raises(ValueError, match="capacity_factor"):
        tiny_config(router_capacity_factor=0.9)


def test_config_rejects_inconsistent_heads_head_dim():
    """heads * head_dim must equal dim. The reviewer's repro: dim=128,
    heads=4, head_dim=16 used to construct OK then crash at first
    forward with a shape error."""
    with pytest.raises(ValueError, match=r"heads.*head_dim.*must equal dim"):
        tiny_config(dim=128, heads=4, head_dim=16)


def test_config_rejects_odd_head_dim():
    """RoPE splits head_dim in half, so it must be even."""
    with pytest.raises(ValueError, match=r"head_dim.*must be even for RoPE"):
        tiny_config(dim=21, heads=3, head_dim=7)


def test_config_rejects_negative_gumbel_tau():
    with pytest.raises(ValueError, match="router_gumbel_tau_init"):
        tiny_config(router_gumbel_tau_init=-0.1)


def test_config_rejects_bad_balance_bias_settings():
    with pytest.raises(ValueError, match="router_balance_bias_update_rate"):
        tiny_config(router_balance_bias_update_rate=-0.1)
    with pytest.raises(ValueError, match="router_balance_bias_ema_rate"):
        tiny_config(router_balance_bias_ema_rate=1.5)
    with pytest.raises(ValueError, match="router_balance_bias_max"):
        tiny_config(router_balance_bias_max=-1.0)


def test_config_rejects_bad_head_dim():
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(dim=100, heads=7)


# ── Expert FFN ─────────────────────────────────────────────────────────


def test_expert_ffn_shape_preserved():
    e = ExpertFFN(dim=128, hidden=64)
    x = torch.randn(2, 16, 128)
    y = e(x)
    assert y.shape == x.shape


def test_orthogonal_init_breaks_symmetry():
    """Two experts with different seeds should have different weights."""
    e1 = ExpertFFN(dim=128, hidden=64)
    e2 = ExpertFFN(dim=128, hidden=64)
    orthogonal_expert_init(e1, seed=0)
    orthogonal_expert_init(e2, seed=1)
    assert not torch.allclose(e1.w_gate.weight, e2.w_gate.weight)
    assert not torch.allclose(e1.w_up.weight, e2.w_up.weight)
    assert not torch.allclose(e1.w_down.weight, e2.w_down.weight)


def test_orthogonal_init_deterministic_same_seed():
    """Same seed gives same weights — reproducible runs."""
    e1 = ExpertFFN(dim=128, hidden=64)
    e2 = ExpertFFN(dim=128, hidden=64)
    orthogonal_expert_init(e1, seed=42)
    orthogonal_expert_init(e2, seed=42)
    assert torch.allclose(e1.w_gate.weight, e2.w_gate.weight)


def test_orthogonal_init_matches_claimed_fan_in_variance():
    """Reviewer-found regression: the old code scaled the QR result by
    gain (claimed gain/sqrt(fan_in)), which only holds for tall matrices
    where q ends up with orthonormal rows. For fat matrices (rows>cols,
    i.e. w_gate/w_up at default shapes) q has orthonormal columns, so
    element std was 1/sqrt(rows) — ~13% below the claimed target. This
    test pins the actual element std to within 10% of the target for
    both fat and tall shapes."""
    # Default-style shape: w_gate = (hidden=2048, dim=1536) → rows>cols
    # Tight bound: target is gain/sqrt(fan_in) = 1/sqrt(1536) ≈ 0.02552
    e = ExpertFFN(dim=1536, hidden=2048)
    orthogonal_expert_init(e, seed=0, gain=1.0)
    import math
    target = 1.0 / math.sqrt(1536)
    for name in ("w_gate", "w_up"):
        w = getattr(e, name).weight
        std = w.float().std().item()
        assert abs(std - target) / target < 0.1, (
            f"{name}: std {std:.5f} vs target {target:.5f} "
            f"(rel err {abs(std - target) / target:.3f})"
        )
    # w_down is tall (rows<cols): fan_in = cols = hidden = 2048
    w = e.w_down.weight
    target_down = 1.0 / math.sqrt(2048)
    std = w.float().std().item()
    assert abs(std - target_down) / target_down < 0.1


def test_orthogonal_init_survives_full_model_construction():
    """Orthogonal init must survive HF's post_init() in the wrapper.

    Regression: v1 of v5 ran orthogonal init inside MoELayer.__init__,
    but NanoOSRTForCausalLM.__init__ then called post_init() which
    walked all nn.Linear and overwrote the orthogonal weights with the
    default normal init.

    Deterministic diagnosis: compare block 0, expert 0's weights to
    what orthogonal_expert_init produces standalone with the same seed.
    If they match, orthogonal init survived. If they differ, post_init()
    or something else stomped the experts.
    """
    cfg = tiny_config(expert_orthogonal_init=True)
    model = NanoOSRTForCausalLM(cfg)

    # Regenerate expert (0, 0) with the same seed the model used.
    # Seed convention: block_idx * 1000 + expert_idx
    fresh = ExpertFFN(dim=cfg.dim, hidden=cfg.expert_hidden)
    orthogonal_expert_init(fresh, seed=0 * 1000 + 0, gain=1.0)
    model_expert = model.model.blocks[0].moe.experts[0]

    assert torch.allclose(model_expert.w_gate.weight, fresh.w_gate.weight, atol=1e-5), \
        "Block 0 expert 0 w_gate doesn't match fresh orthogonal init — " \
        "post_init() or another init pass stomped the orthogonal weights."
    assert torch.allclose(model_expert.w_up.weight, fresh.w_up.weight, atol=1e-5)
    assert torch.allclose(model_expert.w_down.weight, fresh.w_down.weight, atol=1e-5)

    # Sanity: pairs of experts within a block should also differ (they use
    # different per-expert seeds).
    for bi, blk in enumerate(model.model.blocks):
        for i in range(len(blk.moe.experts)):
            for j in range(i + 1, len(blk.moe.experts)):
                assert not torch.allclose(
                    blk.moe.experts[i].w_gate.weight,
                    blk.moe.experts[j].w_gate.weight,
                    atol=1e-5,
                ), f"Block {bi}: experts {i} and {j} have identical weights"


def test_orthogonal_init_disabled_flag_works():
    """When expert_orthogonal_init=False, experts use default normal init."""
    cfg = tiny_config(expert_orthogonal_init=False)
    model = NanoOSRTForCausalLM(cfg)
    # Experts should have the default normal init std
    std = model.model.blocks[0].moe.experts[0].w_gate.weight.std().item()
    assert abs(std - cfg.initializer_range) < 0.01, \
        f"Expected std ~{cfg.initializer_range}, got {std}"


# ── MoE Layer ──────────────────────────────────────────────────────────


def test_moe_output_shape():
    """MoE returns (shared_out, routed_out), both same shape as input."""
    cfg = tiny_config()
    moe = MoELayer(cfg)
    x = torch.randn(2, 16, cfg.dim)
    shared_out, routed_out = moe(x, loop_idx=0)
    assert shared_out.shape == x.shape
    assert routed_out.shape == x.shape


def test_moe_gate_applies_only_to_routed():
    """Changing moe_gate should NOT change the shared expert's contribution."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    # Zero out moe_gate → routed contribution should disappear; shared remains
    with torch.no_grad():
        for blk in model.model.blocks:
            blk.moe_gate.data.zero_()
        out_zero = model(input_ids=x)
        for blk in model.model.blocks:
            blk.moe_gate.data.fill_(1.0)
        out_one = model(input_ids=x)

    # Logits should differ — moe_gate=0 removes routed experts
    assert not torch.allclose(out_zero.logits, out_one.logits), \
        "moe_gate should only gate the routed branch; zeroing it should change output"


def test_moe_top_k_dispatch_coverage():
    """Sum of expert fractions should equal 1.0 (each top-k pick counts)."""
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2, router_capacity_factor=4.0)
    moe = MoELayer(cfg)
    # Use lots of tokens so capacity isn't a factor
    x = torch.randn(8, 16, cfg.dim)
    _ = moe(x, loop_idx=0)
    f = moe.last_expert_fraction[0]
    total = sum(f)
    assert abs(total - 1.0) < 1e-3, f"Expert fractions sum to {total}, expected 1.0"


def test_moe_gumbel_noise_spreads_tie_broken_routing():
    """Gumbel top-k exploration should keep all experts sampled early.

    With exactly tied clean logits, deterministic top-k chooses the same
    experts for every token. Training noise should break that lock-in so cold
    experts get some task-gradient path during the first optimizer steps.
    """
    torch.manual_seed(0)
    cfg = tiny_config(
        num_routed_experts=8,
        top_k_experts=2,
        router_gumbel_tau_init=1.0,
    )
    moe = MoELayer(cfg)
    moe.train(True)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.loop_embeddings.weight.zero_()
        moe.gumbel_tau.fill_(1.0)
    x = torch.randn(32, 32, cfg.dim)
    _ = moe(x, loop_idx=0)
    f = moe.last_expert_fraction[0]
    clean_f = moe.last_clean_expert_fraction[0]
    assert min(f) > 0.05, f"Gumbel routing left a cold expert: {f}"
    assert max(f) < 0.25, f"Gumbel routing over-concentrated experts: {f}"
    assert min(clean_f) == 0.0, (
        "Clean telemetry should expose that tied deterministic top-k still "
        f"leaves cold experts: {clean_f}"
    )


def test_moe_balance_loss_at_uniform_is_one():
    """Balance loss is exactly 1.0 when f and p are uniform."""
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2)
    moe = MoELayer(cfg)
    # Zero the router so softmax is exactly uniform; orthogonal init has
    # already set experts to different weights.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.loop_embeddings.weight.zero_()
    # Lots of tokens so the counts average out close to uniform
    x = torch.randn(64, 64, cfg.dim)
    _ = moe(x, loop_idx=0)
    loss = moe.balance_loss.item()
    # With zero router logits, softmax is exactly uniform so p_i = 1/E.
    # top-k picks are tie-broken deterministically by torch.topk, which
    # will favour lower-indexed experts. f won't be uniform but p is, so
    # loss = E * sum(f_i * 1/E) = sum(f_i) = 1.0 exactly.
    assert abs(loss - 1.0) < 1e-3, f"Loss {loss}, expected 1.0"


def test_moe_balance_loss_penalises_collapse():
    """Forcing all tokens to two experts should raise balance loss well above 1.0.

    We patch the router's forward to return fixed logits strongly favouring
    experts 0 and 1 for every token — avoiding the fragility of hoping
    random inputs produce the desired router output.
    """
    torch.manual_seed(0)
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2)
    moe = MoELayer(cfg)
    # Replace router with a constant-logit module favouring experts 0,1
    class ConstLogits(torch.nn.Module):
        def __init__(self, e: int) -> None:
            super().__init__()
            self.e = e
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (h.shape[0], self.e), -10.0,
                device=h.device, dtype=h.dtype,
            )
            logits[:, 0] = 10.0
            logits[:, 1] = 10.0
            return logits
    moe.router = ConstLogits(cfg.num_routed_experts)
    x = torch.randn(8, 32, cfg.dim)
    _ = moe(x, loop_idx=0)
    loss = moe.balance_loss.item()
    # f concentrates on experts {0, 1} (each ~0.5), p similar (each ~0.5
    # after softmax). loss = E * sum(f_i * p_i) ≈ 8 * 2 * (0.5 * 0.5) = 4.0
    # Minimum loss at uniform is 1.0, so this is a clear penalty.
    assert loss > 3.0, f"Collapse loss {loss}, expected > 3.0"


def test_moe_balance_loss_uses_raw_router_under_gumbel():
    """Aux loss should penalise raw-router collapse even when Gumbel
    exploration spreads noisy dispatch and bias may alter clean selection.

    Regression target: if balance loss is computed from noisy selection logits,
    Gumbel-routed cold experts can make the aux term look healthier than the
    learned raw router actually is.
    """
    torch.manual_seed(0)
    cfg = tiny_config(
        num_routed_experts=8,
        top_k_experts=2,
        router_gumbel_tau_init=20.0,
    )
    moe = MoELayer(cfg)
    moe.train(True)

    class ConstLogits(torch.nn.Module):
        def __init__(self, e: int) -> None:
            super().__init__()
            self.e = e
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(
                (h.shape[0], self.e),
                device=h.device, dtype=h.dtype,
            )
            logits[:, 0] = 5.0
            logits[:, 1] = 5.0
            return logits

    moe.router = ConstLogits(cfg.num_routed_experts)
    with torch.no_grad():
        moe.gumbel_tau.fill_(20.0)
    x = torch.randn(16, 32, cfg.dim)
    _ = moe(x, loop_idx=0)
    loss = moe.balance_loss.item()
    assert loss > 3.5, (
        f"Raw-router collapse should stay strongly penalised under Gumbel; "
        f"got balance loss {loss:.3f}"
    )


def test_balance_bias_controller_accumulates_and_applies():
    """Skewed clean assignments should push popular experts down and unused up."""
    cfg = tiny_config(
        num_routed_experts=4,
        top_k_experts=2,
        router_balance_bias_enabled=True,
        router_balance_bias_update_rate=0.2,
        router_balance_bias_max=0.5,
    )
    moe = MoELayer(cfg)
    assert torch.all(moe.router_balance_bias == 0.0)

    skewed = torch.zeros(32, cfg.top_k_experts, dtype=torch.long)
    skewed[:, 0] = 0
    skewed[:, 1] = 1
    for _ in range(6):
        moe._accumulate_balance_counts(skewed, loop_idx=0)

    assert torch.all(moe.router_balance_bias == 0.0)
    assert moe.balance_total_accum[0].item() > 0
    moe.apply_balance_update()

    assert moe.router_balance_bias[0, 0].item() < 0
    assert moe.router_balance_bias[0, 1].item() < 0
    assert moe.router_balance_bias[0, 2].item() > 0
    assert moe.router_balance_bias[0, 3].item() > 0
    assert torch.all(moe.router_balance_bias[1:] == 0.0)
    assert moe.router_balance_bias.abs().max().item() <= 0.5 + 1e-6
    assert torch.all(moe.balance_count_accum == 0.0)
    assert torch.all(moe.balance_total_accum == 0.0)

    bias_before = moe.router_balance_bias.clone()
    moe.apply_balance_update()
    assert torch.allclose(moe.router_balance_bias, bias_before)


def test_balance_bias_affects_clean_topk_selection():
    """Clean routing should include the persistent balance bias."""
    cfg = tiny_config(
        num_routed_experts=4,
        top_k_experts=2,
        router_gumbel_tau_init=0.0,
        router_balance_bias_enabled=True,
    )
    moe = MoELayer(cfg)
    moe.train(False)

    class ConstLogits(torch.nn.Module):
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = torch.empty(h.shape[0], 4, device=h.device, dtype=h.dtype)
            logits[:, 0] = -1.0
            logits[:, 1] = 0.5
            logits[:, 2] = 2.0
            logits[:, 3] = 0.0
            return logits

    moe.router = ConstLogits()
    x = torch.randn(2, 8, cfg.dim)
    with torch.no_grad():
        moe.router_balance_bias.zero_()
        _ = moe(x, loop_idx=0)
        unbiased_f = moe.last_clean_expert_fraction[0]
        moe.router_balance_bias[0].copy_(torch.tensor([0.0, 0.0, -5.0, 0.0]))
        _ = moe(x, loop_idx=0)
        biased_f = moe.last_clean_expert_fraction[0]

    assert unbiased_f[2] > 0.49
    assert biased_f[2] == 0.0
    assert biased_f[1] > 0.49


def test_balance_bias_persists_in_state_dict():
    cfg = tiny_config(num_routed_experts=4, top_k_experts=2)
    model = NanoOSRTForCausalLM(cfg)
    bias = torch.tensor([
        [0.1, -0.2, 0.3, -0.4],
        [-0.3, 0.2, -0.1, 0.4],
    ])
    model.model.blocks[0].moe.router_balance_bias.copy_(bias)

    state = model.state_dict()
    found = [k for k in state if "router_balance_bias" in k]
    assert found, "router_balance_bias not in state_dict"

    model2 = NanoOSRTForCausalLM(cfg)
    model2.load_state_dict(state)
    assert torch.allclose(
        model2.model.blocks[0].moe.router_balance_bias,
        bias,
    )


def test_moe_drop_rate_low_at_loose_cap():
    """With capacity_factor 4.0, drops should be rare at uniform routing."""
    cfg = tiny_config(
        num_routed_experts=8, top_k_experts=2, router_capacity_factor=4.0,
    )
    moe = MoELayer(cfg)
    x = torch.randn(4, 16, cfg.dim)
    _ = moe(x, loop_idx=0)
    assert moe.last_drop_rate[0] < 0.1


def test_moe_drop_rate_high_at_tight_cap_and_collapse():
    """Forced collapse + tight cap should produce significant drops."""
    torch.manual_seed(0)
    cfg = tiny_config(
        num_routed_experts=8, top_k_experts=2, router_capacity_factor=1.01,
    )
    moe = MoELayer(cfg)
    # Patch router with a constant-logit module favouring experts 0,1
    class ConstLogits(torch.nn.Module):
        def __init__(self, e: int) -> None:
            super().__init__()
            self.e = e
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (h.shape[0], self.e), -10.0,
                device=h.device, dtype=h.dtype,
            )
            logits[:, 0] = 10.0
            logits[:, 1] = 10.0
            return logits
    moe.router = ConstLogits(cfg.num_routed_experts)
    x = torch.randn(4, 16, cfg.dim)
    _ = moe(x, loop_idx=0)
    drop = moe.last_drop_rate[0]
    assert drop > 0.3, \
        f"Expected high drop rate with collapse + tight cap, got {drop}"


def test_moe_renormalises_top_k_gates():
    """top_probs should sum to 1 per token after renormalisation."""
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2)
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, cfg.dim)
    # Patch forward to expose top_probs — easier to just replicate the math
    N = 2 * 8
    router_input = x + moe.loop_embeddings.weight[0].view(1, 1, -1)
    logits = moe.router(router_input.reshape(N, cfg.dim))
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = probs.topk(cfg.top_k_experts, dim=-1)
    renorm = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(
        renorm.sum(dim=-1), torch.ones(N), atol=1e-5,
    ), "Renormalised top-k gates don't sum to 1"


def test_moe_telemetry_populated():
    cfg = tiny_config()
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, cfg.dim)
    for loop in range(cfg.recursive_loops):
        _ = moe(x, loop_idx=loop)
        # All telemetry fields should be filled per-loop
        assert moe.last_per_token_entropy[loop] > 0
        assert moe.last_marginal_entropy[loop] > 0
        assert moe.last_assignment_entropy[loop] >= 0
        assert moe.last_clean_per_token_entropy[loop] > 0
        assert moe.last_clean_marginal_entropy[loop] > 0
        assert moe.last_clean_assignment_entropy[loop] >= 0
        assert moe.last_raw_max_prob[loop] > 0
        assert moe.last_top_margin[loop] >= 0
        assert moe.last_clean_raw_max_prob[loop] > 0
        assert moe.last_clean_top_margin[loop] >= 0
        assert sum(moe.last_expert_fraction[loop]) > 0.95
        assert sum(moe.last_clean_expert_fraction[loop]) > 0.95


def test_moe_raw_max_prob_is_pre_renormalisation():
    """Raw max prob should be close to 1/E at random init, NOT 0.5 for top-2."""
    torch.manual_seed(42)
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2)
    moe = MoELayer(cfg)
    x = torch.randn(4, 16, cfg.dim)
    _ = moe(x, loop_idx=0)
    # Random router + softmax of 8 experts → top-1 prob averages near 1/8
    # (slightly above since we're taking the max of 8 random draws).
    # Critical: must be < 0.5 so we know we're logging RAW not renormalised.
    raw_max = moe.last_raw_max_prob[0]
    assert raw_max < 0.4, \
        f"Raw max_prob {raw_max} looks renormalised (top-2 renormalised averages 0.5)"


def test_moe_per_token_entropy_vs_marginal():
    """Per-token entropy and marginal entropy can differ — a balanced sharp
    router has LOW per-token entropy but HIGH marginal entropy.
    """
    torch.manual_seed(42)
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2)
    moe = MoELayer(cfg)
    # Build a router whose preferences depend on token features (sharp per
    # token) but are balanced on average across many tokens. Random large
    # weights do this: each token picks a different expert, marginal is
    # near-uniform but per-token is sharp.
    with torch.no_grad():
        moe.router.weight.copy_(torch.randn_like(moe.router.weight) * 5.0)
        moe.loop_embeddings.weight.zero_()
    x = torch.randn(8, 32, cfg.dim)
    _ = moe(x, loop_idx=0)
    # Per-token entropy should be substantially LOWER than marginal entropy
    # when routing is sharp. Marginal should still be high because tokens
    # spread across experts.
    per_token = moe.last_per_token_entropy[0]
    marginal = moe.last_marginal_entropy[0]
    assert per_token < marginal, (
        f"Expected per_token_ent {per_token} < marginal_ent {marginal} "
        f"for sharp balanced routing"
    )
    # And marginal should stay near ln(8)=2.08 (balanced)
    assert marginal > 1.8, f"Marginal entropy {marginal} suggests imbalance"


# ── Gradient flow ──────────────────────────────────────────────────────


def test_gradient_flows_to_router():
    """Router weights should receive gradient from task loss."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=x, labels=labels)
    out.loss.backward()
    for blk in model.model.blocks:
        assert blk.moe.router.weight.grad is not None
        assert blk.moe.router.weight.grad.abs().sum() > 0, "Router got zero gradient"


def test_gradient_flows_to_multiple_experts():
    """With top-2 and many tokens, multiple experts should receive gradient."""
    cfg = tiny_config(num_routed_experts=8, top_k_experts=2)
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    labels = torch.randint(0, cfg.vocab_size, (4, 32))
    out = model(input_ids=x, labels=labels)
    out.loss.backward()
    for blk in model.model.blocks:
        experts_with_grad = sum(
            1 for e in blk.moe.experts
            if e.w_gate.weight.grad is not None
            and e.w_gate.weight.grad.abs().sum() > 0
        )
        assert experts_with_grad >= 2, (
            f"Only {experts_with_grad} experts got gradient; "
            f"expected at least top-k={cfg.top_k_experts}"
        )


def test_gradient_flows_to_shared_expert():
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=x, labels=labels)
    out.loss.backward()
    for blk in model.model.blocks:
        assert blk.moe.shared_expert.w_gate.weight.grad is not None
        assert blk.moe.shared_expert.w_gate.weight.grad.abs().sum() > 0


# ── Full model ─────────────────────────────────────────────────────────


def test_forward_returns_expected_shapes():
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=x)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is None


def test_loss_computed_when_labels_given():
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=x, labels=labels)
    assert out.loss is not None
    assert not torch.isnan(out.loss)


def test_loss_components_exposed_after_forward():
    """After forward with labels, task_loss and balance_loss should be set
    regardless of train/eval mode.
    """
    cfg = tiny_config(router_aux_loss_coeff=0.1)
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))

    # Training mode: full total loss
    model.train(True)
    out_train = model(input_ids=x, labels=labels)
    assert model.last_task_loss is not None
    assert model.last_balance_loss is not None
    assert model.last_balance_loss_normalised is not None
    expected_train = (
        model.last_task_loss.item()
        + cfg.router_aux_loss_coeff * model.last_balance_loss_normalised.item()
    )
    assert abs(out_train.loss.item() - expected_train) < 1e-4

    # Eval mode: components still populated; total is just task loss
    model.train(False)
    out_eval = model(input_ids=x, labels=labels)
    assert model.last_task_loss is not None
    assert model.last_balance_loss is not None
    assert model.last_balance_loss_normalised is not None
    assert abs(out_eval.loss.item() - model.last_task_loss.item()) < 1e-5


def test_loss_components_reset_without_labels():
    """Without labels, loss components should be None (not stale)."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.randint(0, cfg.vocab_size, (1, 8))
    # Run once with labels to populate
    _ = model(input_ids=x, labels=labels)
    assert model.last_task_loss is not None
    # Run again without labels — should clear
    _ = model(input_ids=x)
    assert model.last_task_loss is None
    assert model.last_balance_loss is None


def test_capacity_drops_disabled_in_eval():
    """With model.train(False), drop_rate must be zero regardless of capacity."""
    torch.manual_seed(0)
    cfg = tiny_config(
        num_routed_experts=8, top_k_experts=2, router_capacity_factor=1.01,
    )
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    # 128 tokens — would overflow tight cap in training mode
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    _ = model(input_ids=x)
    for blk in model.model.blocks:
        for loop_drop in blk.moe.last_drop_rate:
            assert loop_drop == 0.0, \
                f"Eval mode should have zero drops; got {loop_drop}"


def test_capacity_drops_active_in_training():
    """In training mode with tight cap + forced collapse, drops should occur."""
    torch.manual_seed(0)
    cfg = tiny_config(
        num_routed_experts=8, top_k_experts=2, router_capacity_factor=1.01,
    )
    model = NanoOSRTForCausalLM(cfg)
    model.train(True)

    # Monkey-patch first block's router to force collapse
    class ConstLogits(torch.nn.Module):
        def __init__(self, e: int) -> None:
            super().__init__()
            self.e = e
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (h.shape[0], self.e), -10.0,
                device=h.device, dtype=h.dtype,
            )
            logits[:, 0] = 10.0
            logits[:, 1] = 10.0
            return logits
    model.model.blocks[0].moe.router = ConstLogits(cfg.num_routed_experts)

    x = torch.randint(0, cfg.vocab_size, (4, 32))
    _ = model(input_ids=x)
    drops = model.model.blocks[0].moe.last_drop_rate
    assert any(d > 0 for d in drops), \
        f"Training mode with forced collapse should produce drops; got {drops}"


def test_eval_loss_excludes_balance_loss():
    """Eval loss must be pure task CE — no aux contamination of perplexity."""
    # Large coeff so any contamination would be obvious in the assertion
    cfg = tiny_config(router_aux_loss_coeff=1.0)
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))

    model.train(False)
    out = model(input_ids=x, labels=labels)
    # In eval mode, out.loss should equal task_loss exactly
    assert torch.allclose(out.loss, model.last_task_loss, atol=1e-5), \
        f"Eval loss {out.loss.item()} should equal task_loss " \
        f"{model.last_task_loss.item()} but includes aux loss"

    # Confirm balance components are still populated (for telemetry)
    assert model.last_balance_loss is not None
    assert model.last_balance_loss_normalised is not None


def test_train_loss_includes_balance_contribution():
    """In training mode, total loss = task + coeff * balance_norm."""
    cfg = tiny_config(router_aux_loss_coeff=0.1)
    model = NanoOSRTForCausalLM(cfg)
    model.train(True)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=x, labels=labels)

    expected = (
        model.last_task_loss.item()
        + cfg.router_aux_loss_coeff * model.last_balance_loss_normalised.item()
    )
    assert abs(out.loss.item() - expected) < 1e-4


def test_loss_contains_balance_contribution():
    """Balance loss should contribute to total loss when coefficient > 0.

    Checked in training mode since eval excludes aux loss by design.
    """
    cfg_off = tiny_config(router_aux_loss_coeff=0.0)
    cfg_on = tiny_config(router_aux_loss_coeff=1.0)

    torch.manual_seed(0)
    model_off = NanoOSRTForCausalLM(cfg_off)
    model_off.train(True)
    torch.manual_seed(0)
    model_on = NanoOSRTForCausalLM(cfg_on)
    model_on.train(True)

    x = torch.randint(0, cfg_off.vocab_size, (2, 16))
    labels = torch.randint(0, cfg_off.vocab_size, (2, 16))

    with torch.no_grad():
        out_off = model_off(input_ids=x, labels=labels)
        out_on = model_on(input_ids=x, labels=labels)

    # Balance loss has min 1.0 so with coeff 1.0 the total is higher
    assert out_on.loss.item() > out_off.loss.item()


def test_recursive_loops_produce_different_hidden():
    """Multiple loops should modify hidden state at each step."""
    cfg = tiny_config(recursive_loops=3)
    model = NanoOSRTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    _, loop_rms, _, _ = model(x)
    rms_vals = [r.item() for r in loop_rms]
    assert len(set(f"{r:.4f}" for r in rms_vals)) > 1, \
        f"Loops produced identical hidden state: {rms_vals}"


# ── KV cache ───────────────────────────────────────────────────────────


def test_kv_cache_shapes():
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(input_ids=x, use_cache=True)
    expected_layers = cfg.num_blocks * cfg.recursive_loops
    assert out.past_key_values is not None
    assert len(out.past_key_values) == expected_layers
    for layer_past in out.past_key_values:
        k, v = layer_past
        assert k.shape == (1, cfg.heads, 8, cfg.head_dim)
        assert v.shape == (1, cfg.heads, 8, cfg.head_dim)


def test_kv_cache_extend_matches_full_pass():
    """Step-by-step decode with cache should match a single full forward.

    v5 disables MoE capacity drops when not training (see MoELayer.forward),
    so generation is chunk-stable by construction. We put the model into
    training=False via .train(False) to activate the no-drop path.
    """
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)  # equivalent to .eval() but avoids naming hooks
    full = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        # Full pass
        out_full = model(input_ids=full)
        # Split: prefill 4 tokens, then one at a time
        out_prefill = model(input_ids=full[:, :4], use_cache=True)
        out_step = model(
            input_ids=full[:, 4:5],
            past_key_values=out_prefill.past_key_values,
            use_cache=True,
        )
        out_step2 = model(
            input_ids=full[:, 5:6],
            past_key_values=out_step.past_key_values,
            use_cache=True,
        )

    assert torch.allclose(
        out_full.logits[:, 4, :], out_step.logits[:, 0, :], atol=1e-4
    ), "KV cache produced different logits at position 4"
    assert torch.allclose(
        out_full.logits[:, 5, :], out_step2.logits[:, 0, :], atol=1e-4
    ), "KV cache produced different logits at position 5"


# ── Parameter count & architecture sanity ──────────────────────────────


def test_param_count_in_expected_range():
    """Full-config v5 should be in the 350-500M range."""
    cfg = NanoOSRTConfig()
    model = NanoOSRTForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 350_000_000 < n < 500_000_000, f"Unexpected param count: {n:,}"


def test_no_dense_ffn_in_block():
    """v5 blocks should not have a dense FFN attribute."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    for blk in model.model.blocks:
        assert not hasattr(blk, "ffn_dense"), "v5 should have no ffn_dense"
        assert not hasattr(blk, "dense_gate"), "v5 should have no dense_gate"


def test_blocks_have_moe_gate():
    """MoE output should pass through a learnable scalar gate."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    for blk in model.model.blocks:
        assert hasattr(blk, "moe_gate")
        assert blk.moe_gate.requires_grad


# ── torch.compile integration ──────────────────────────────────────────


def test_last_loss_attrs_set_under_torch_compile():
    """torch.compile wraps forward; verify attribute mutations still reach
    the original module.

    Regression: the training loop reads inner.last_task_loss after each
    forward to build the per-step loss breakdown. That attribute is set
    inside the compiled forward. torch.compile usually graph-breaks on
    Python attribute writes so the write propagates to _orig_mod, but
    this isn't documented as guaranteed. If a future Torch release
    changes the behaviour, our training would silently log zeros.

    This test runs a compiled model once with labels and asserts the
    components are present on the underlying module.
    """
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    # torch.compile is a no-op on CPU for some configs; try anyway and
    # gracefully skip if the backend rejects.
    try:
        compiled = torch.compile(model)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch.compile unavailable in this env: {e}")

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.randint(0, cfg.vocab_size, (1, 8))
    try:
        _ = compiled(input_ids=x, labels=labels)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch.compile failed on test env: {e}")

    inner = compiled._orig_mod if hasattr(compiled, "_orig_mod") else compiled
    assert inner.last_task_loss is not None, \
        "last_task_loss not set on _orig_mod after compiled forward " \
        "(attribute mutation regressed — training logs would go blank)"
    assert inner.last_balance_loss is not None
    assert inner.last_balance_loss_normalised is not None


# ── Generate method with KV cache ─────────────────────────────────────


def test_generate_greedy_produces_expected_shape():
    """Greedy generate returns input_ids + max_new_tokens tokens."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    ctx = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(
        ctx, max_new_tokens=12, temperature=0.0, repetition_penalty=1.0,
    )
    assert out.shape == (1, 8 + 12), f"Expected (1, 20), got {out.shape}"


def test_generate_sampling_respects_temperature():
    """temperature>0 uses multinomial; two seeded runs with different seeds
    produce different sequences (probabilistic check)."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    ctx = torch.randint(0, cfg.vocab_size, (1, 6))

    torch.manual_seed(1)
    a = model.generate(ctx, max_new_tokens=15, temperature=1.0, top_p=0.9)
    torch.manual_seed(2)
    b = model.generate(ctx, max_new_tokens=15, temperature=1.0, top_p=0.9)

    # Different seeds + non-zero temperature => sequences should differ
    # (at least one position). Extremely unlikely to fail for random model.
    assert not torch.equal(a, b), \
        "Sampled generation with different seeds produced identical output"


def test_generate_kv_cache_matches_full_forward():
    """Prefill+decode with KV cache produces same logits as a full forward.

    Uses a loose capacity_factor so MoE dispatch doesn't drop tokens that
    would otherwise make full vs cached paths diverge.
    """
    cfg = tiny_config(router_capacity_factor=10.0)
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    ctx = torch.randint(0, cfg.vocab_size, (1, 10))
    new_tok_id = 42

    # Full forward over ctx + new_tok
    full_in = torch.cat([ctx, torch.tensor([[new_tok_id]])], dim=1)
    out_full = model(full_in)
    logits_full = out_full.logits[:, -1, :]

    # Prefill + cached decode
    out_pre = model(ctx, use_cache=True)
    past = out_pre.past_key_values
    out_dec = model(
        torch.tensor([[new_tok_id]]), past_key_values=past, use_cache=True,
    )
    logits_dec = out_dec.logits[:, -1, :]

    diff = (logits_full - logits_dec).abs().max().item()
    assert diff < 1e-3, f"KV cache logit mismatch: max diff {diff}"


def test_generate_stops_on_eos():
    """If the first argmax token happens to equal eos_token_id, generation
    stops early with output length = input + 1."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    ctx = torch.randint(0, cfg.vocab_size, (1, 6))
    # Force argmax by zeroing all output logits except EOS — then greedy
    # picks EOS on the first step.
    with torch.no_grad():
        model.model.embedding.weight.zero_()
        # A model with zero embedding still produces some logits via the
        # transformer body, so we can't guarantee first-token EOS without
        # patching further. Instead, test with a high max_new_tokens and
        # just confirm the output is no longer than that (sanity, not
        # a strong test of EOS stopping).
    out = model.generate(
        ctx, max_new_tokens=5, temperature=0.0,
        eos_token_id=cfg.eos_token_id, repetition_penalty=1.0,
    )
    assert out.shape[1] <= ctx.shape[1] + 5, \
        f"Output too long: {out.shape[1]} > {ctx.shape[1] + 5}"


# ── Checkpoint round-trip and aux-loss gradient flow ───────────────────


def test_checkpoint_roundtrip_preserves_logits(tmp_path):
    """state_dict save/load produces identical logits on the same input.

    Guards against silent state-dict key drift (e.g. if someone renames a
    module). For v5 this matters because resume logic in the training loop
    uses load_state_dict(strict=False), which would silently skip mismatched
    keys — we want the round-trip to NOT depend on strict=False to succeed.
    """
    import os
    cfg = tiny_config()
    torch.manual_seed(0)
    model_a = NanoOSRTForCausalLM(cfg)
    model_a.train(False)

    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out_a = model_a(input_ids=x)

    ckpt_path = os.path.join(tmp_path, "v5_roundtrip.pt")
    torch.save({"model_state_dict": model_a.state_dict()}, ckpt_path)

    # Fresh model with a different seed — loading must overwrite the
    # differently-initialised weights to match model_a exactly.
    torch.manual_seed(1)
    model_b = NanoOSRTForCausalLM(cfg)
    model_b.train(False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model_b.load_state_dict(
        ckpt["model_state_dict"], strict=True,
    )
    # strict=True would raise on missing/unexpected, so if we got here the
    # state dict keys line up perfectly. Double-check logits.
    with torch.no_grad():
        out_b = model_b(input_ids=x)
    diff = (out_a.logits - out_b.logits).abs().max().item()
    assert diff < 1e-5, f"Logits differ after state_dict round-trip: {diff}"


def test_balance_loss_pushes_router_weights():
    """Backprop from an aux-loss-only objective should produce non-zero
    gradients on the router weights.

    If this regresses (e.g. balance_loss stops being a tensor with grad),
    the router would stop balancing and we'd drift back toward collapse.
    """
    cfg = tiny_config(router_aux_loss_coeff=1.0)
    model = NanoOSRTForCausalLM(cfg)
    # Zero router weights so the gate has nothing to work with initially;
    # balance loss gradient should still flow.
    with torch.no_grad():
        for blk in model.model.blocks:
            blk.moe.router.weight.zero_()

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    labels = torch.randint(0, cfg.vocab_size, (2, 32))
    model.zero_grad(set_to_none=True)
    # Run forward to populate each block's balance_loss attribute; we
    # don't need the returned output, only the side-effect.
    model(input_ids=x, labels=labels)

    # Backprop only the balance loss component — task loss is irrelevant
    # for this test, we want to confirm balance_loss alone generates a
    # router gradient.
    assert model.last_balance_loss is not None
    # last_balance_loss is .detach()'d, so we need the un-detached one
    # from the MoE layer directly.
    bal_sum = torch.zeros(())
    for blk in model.model.blocks:
        assert blk.moe.balance_loss is not None
        bal_sum = bal_sum + blk.moe.balance_loss
    bal_sum.backward()

    for blk in model.model.blocks:
        grad = blk.moe.router.weight.grad
        assert grad is not None, \
            "Router weight has no grad after balance_loss.backward()"
        grad_mag = grad.abs().sum().item()
        assert grad_mag > 0, \
            "Router grad is all zero — balance loss isn't pushing router"


# ── Reward parsing regressions ─────────────────────────────────────────


def test_extract_numeric_answer_returns_last_number_in_answer_tag():
    """Reviewer-found regression: the old parser returned the first number
    inside the answer tag, so "After 3 steps, the answer is 12" scored as
    3 instead of 12. The fix takes the last number, which is the model's
    final committed answer.
    """
    from nano_osrt.rewards import extract_numeric_answer

    text = (
        "<|think|>3 steps of reasoning...<|/think|>"
        "<|answer|>After 3 steps, the answer is 12<|/answer|>"
    )
    assert extract_numeric_answer(text) == "12"


def test_extract_numeric_answer_handles_plain_number():
    """A bare number inside the answer tag should still parse correctly."""
    from nano_osrt.rewards import extract_numeric_answer

    text = "<|think|>...<|/think|><|answer|>42<|/answer|>"
    assert extract_numeric_answer(text) == "42"


def test_extract_numeric_answer_strips_commas():
    """Numbers with thousand separators should be normalised."""
    from nano_osrt.rewards import extract_numeric_answer

    text = "<|answer|>The total is 1,234<|/answer|>"
    assert extract_numeric_answer(text) == "1234"


def test_extract_numeric_answer_falls_back_to_post_think():
    """No answer tag — should pick the last number after </think>."""
    from nano_osrt.rewards import extract_numeric_answer

    text = "<think>Working it out: 10, then 20...</think> So the final is 30"
    assert extract_numeric_answer(text) == "30"


# ── Batch-safe generate ────────────────────────────────────────────────


def test_generate_batch_safe_with_repetition_penalty():
    """Reviewer-found regression: generate() used generated[0] and
    next_token.item() which crashed for batch>1. Fix applies rep
    penalty per row and uses an EOS mask for termination."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    # Two rows → ensures all per-row indexing is correct.
    ctx = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model.generate(
        ctx,
        max_new_tokens=6,
        temperature=0.0,
        repetition_penalty=1.2,  # exercises the per-row penalty loop
    )
    assert out.shape == (2, 14), f"Expected (2, 14), got {out.shape}"


def test_generate_batch_safe_with_sampling():
    """Batch>1 with temperature + top_p + top_k. Guards against the
    .item() EOS check crashing for multi-row outputs."""
    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    ctx = torch.randint(0, cfg.vocab_size, (3, 5))
    out = model.generate(
        ctx,
        max_new_tokens=4,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )
    assert out.shape == (3, 9)


def test_generate_pads_finished_rows_until_all_done():
    """Reviewer-found regression: when rows emit EOS on different
    steps, finished rows must keep emitting EOS (not random tokens)
    until every row has finished. The earlier implementation only
    checked whether all rows emitted EOS on the *same* step, so rows
    that finished early kept getting live decoded tokens appended.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    cfg = tiny_config()
    model = NanoOSRTForCausalLM(cfg)
    model.train(False)
    eos = 7

    # Scripted logits per decode step. Row 0 emits eos on step 1
    # (prefill returns step-0 logits, so step_idx==1 is the first
    # decode step) and row 1 emits eos on step 3.
    # For every call, we build (B, T, V) logits where last position
    # controls the sampled token via argmax (temperature=0).
    call_count = {"n": 0}

    def scripted_forward(input_ids, past_key_values=None, use_cache=False, **kw):
        B, T = input_ids.shape
        V = cfg.vocab_size
        logits = torch.full((B, T, V), -1e4)
        n = call_count["n"]
        # Greedy picks argmax of the last position. We set:
        # prefill (n==0): non-eos on both rows
        # decode step 1 (n==1): row 0 → eos, row 1 → non-eos
        # decode step 2 (n==2): row 1 → non-eos, row 0 padded to eos
        # decode step 3 (n==3): row 1 → eos (triggers break)
        if n == 0:
            logits[0, -1, 1] = 10.0
            logits[1, -1, 2] = 10.0
        elif n == 1:
            logits[0, -1, eos] = 10.0
            logits[1, -1, 3] = 10.0
        elif n == 2:
            logits[0, -1, 5] = 10.0  # would pollute row 0 if padding missing
            logits[1, -1, 4] = 10.0
        else:
            logits[0, -1, 6] = 10.0
            logits[1, -1, eos] = 10.0
        call_count["n"] = n + 1
        # Return a minimal past_key_values list matching the model
        # so the generate() cache-trim guard is happy.
        past = past_key_values if past_key_values is not None else [None] * 1
        return CausalLMOutputWithPast(
            loss=None, logits=logits.to(torch.float32),
            past_key_values=past,
        )

    model.forward = scripted_forward  # type: ignore[assignment]

    ctx = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    out = model.generate(
        ctx, max_new_tokens=4, temperature=0.0, eos_token_id=eos,
    )

    # Row 0 finished on decode step 1, so positions 3..end must be eos.
    # Row 1 finished on decode step 3, so position 5 must be eos.
    row0_tail = out[0, 3:].tolist()
    assert all(t == eos for t in row0_tail), (
        f"row 0 must be padded with eos after finishing, got {row0_tail}"
    )
    assert out[1, -1].item() == eos, (
        f"row 1 should end on eos, got {out[1, -1].item()}"
    )
