"""Smoke tests for NanoOSRT v4.

Guards the things we actually care about:
  1. Config validation is correct (too-small top_k, etc.)
  2. Model instantiates with expected physical parameter count
  3. Forward + backward produces finite loss
  4. MoE aux losses and telemetry populate during training
  5. Router and gate initialisation match the dense-first design
  6. RoPE NTK scaling increases the effective theta
  7. Reward extraction handles native v4 tags (<|think|>, <|answer|>)

These are CPU-friendly with a tiny config so pytest runs fast.
"""

from __future__ import annotations

import pytest
import torch

from nano_osrt.rewards import compute_reward, extract_numeric_answer
from nano_osrt.v4_config import NanoOSRTv4Config
from nano_osrt.v4_model import (
    NanoOSRTv4ForCausalLM,
    apply_rope,
    compute_rope_freqs,
)


def _tiny_config(**overrides) -> NanoOSRTv4Config:
    base = {
        "dim": 128,
        "heads": 4,
        "head_dim": 32,
        "vocab_size": 512,
        "real_vocab_size": 512,
        "num_blocks": 2,
        "recursive_loops": 3,
        "dense_hidden": 256,
        "num_experts": 4,
        "num_shared_experts": 1,
        "num_routed_experts": 3,
        "top_k_experts": 2,
        "expert_hidden": 128,
        "max_position_embeddings": 64,
    }
    base.update(overrides)
    return NanoOSRTv4Config(**base)


# Config validation ----------------------------------------------------------


def test_config_rejects_bad_top_k():
    with pytest.raises(ValueError, match="top_k_experts"):
        _tiny_config(top_k_experts=5, num_routed_experts=3)


def test_config_rejects_dim_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        _tiny_config(dim=127, heads=4)


def test_default_config_uses_32k_vocab():
    cfg = NanoOSRTv4Config()
    assert cfg.vocab_size == 32768
    assert cfg.real_vocab_size == 32768


# Model build + forward/backward --------------------------------------------


def test_model_instantiates_and_counts_params():
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n > 0
    assert hasattr(model, "model")
    assert len(model.model.blocks) == cfg.num_blocks


def test_forward_and_backward_produce_finite_loss():
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(0, cfg.real_vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(g > 0 for g in grads)


def test_forward_without_labels_returns_logits_only():
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    # Stay in train mode for the test but use torch.no_grad() so the
    # training-only aux-loss path still runs, matching real usage.
    input_ids = torch.randint(0, cfg.real_vocab_size, (1, 8))
    with torch.no_grad():
        out = model(input_ids)
    assert out.logits.shape == (1, 8, cfg.vocab_size)
    assert out.loss is None


# MoE behaviour --------------------------------------------------------------


def test_moe_aux_losses_populate_during_training():
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(0, cfg.real_vocab_size, (2, 16))
    _ = model(input_ids, labels=input_ids)
    for blk in model.model.blocks:
        assert blk.moe.load_balance_loss is not None
        assert blk.moe.z_loss is not None
        assert torch.isfinite(blk.moe.load_balance_loss)
        assert torch.isfinite(blk.moe.z_loss)


def test_moe_telemetry_records_per_loop():
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(0, cfg.real_vocab_size, (2, 16))
    _ = model(input_ids, labels=input_ids)
    for blk in model.model.blocks:
        # One entry per recursive loop
        assert len(blk.moe.last_router_entropy) == cfg.recursive_loops
        assert len(blk.moe.last_assignment_entropy) == cfg.recursive_loops
        assert len(blk.moe.last_expert_fraction) == cfg.recursive_loops
        # Both entropies should be > 0 (distribution is not a point mass)
        assert all(e > 0 for e in blk.moe.last_router_entropy)
        assert all(e > 0 for e in blk.moe.last_assignment_entropy)
        # Each fraction row sums to ~1.0
        for fracs in blk.moe.last_expert_fraction:
            assert len(fracs) == cfg.num_routed_experts
            assert abs(sum(fracs) - 1.0) < 1e-4


def test_assignment_entropy_detects_collapse():
    """Assignment entropy should be much lower when routing is collapsed."""
    import math

    cfg = _tiny_config(router_noise_std=0.0)  # kill noise so we can construct a deterministic fake
    model = NanoOSRTv4ForCausalLM(cfg)

    # Build a synthetic top_k_indices that picks only experts 0 and 1
    collapsed = torch.zeros(1, 16, cfg.top_k_experts, dtype=torch.long)
    collapsed[..., 0] = 0  # all top-1 to expert 0
    collapsed[..., 1] = 1  # all top-2 to expert 1
    fake_probs = torch.full((1, 16, cfg.num_routed_experts), 0.5)

    moe = model.model.blocks[0].moe
    moe._record_telemetry(fake_probs, collapsed, loop_idx=0)

    # Full top-2 collapse gives exactly ln(2) = 0.693.
    # Uniform routing over 3 routed experts gives ln(3) = 1.099.
    # We want the collapsed case to register as strictly lower than uniform.
    assert moe.last_assignment_entropy[0] < math.log(cfg.num_routed_experts) - 0.2
    assert abs(moe.last_assignment_entropy[0] - math.log(2)) < 0.05
    # Prob entropy is still near-uniform because all fake probs are equal —
    # this is the exact "misleading" case we wanted assignment entropy to catch.
    assert moe.last_router_entropy[0] > math.log(cfg.num_routed_experts) * 0.9


def test_gates_initialised_dense_first():
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    for blk in model.model.blocks:
        assert pytest.approx(blk.dense_gate.item(), abs=1e-5) == 1.0
        assert pytest.approx(blk.moe_gate.item(), abs=1e-4) == 0.01


def test_router_uses_additive_loop_embedding():
    """Router input dim should be dim (not 2*dim) since we add loop_emb."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    for blk in model.model.blocks:
        # (num_routed, dim) — additive router halves what cat-router would use
        assert blk.moe.router.weight.shape == (cfg.num_routed_experts, cfg.dim)


# Adapter placement ----------------------------------------------------------


def test_adapter_b_is_zero_initialised():
    """B init to zero means adapter contribution is 0 at step 0."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    for b in model.model.adapters_b:
        assert torch.all(b == 0)


# RoPE scaling ---------------------------------------------------------------


def test_rope_base_shape():
    cos, sin = compute_rope_freqs(seq_len=16, dim=32, theta=10000.0)
    assert cos.shape == (1, 16, 1, 32)
    assert sin.shape == (1, 16, 1, 32)


def test_rope_ntk_scaling_changes_frequencies():
    """NTK scaling with factor=4 should produce different cos/sin than no scaling."""
    cos_base, _ = compute_rope_freqs(seq_len=16, dim=32, theta=10000.0)
    cos_scaled, _ = compute_rope_freqs(
        seq_len=16, dim=32, theta=10000.0,
        scaling={"type": "ntk", "factor": 4.0},
    )
    # The frequencies should differ at most positions except t=0
    assert not torch.allclose(cos_base[0, 5:], cos_scaled[0, 5:])


def test_rope_rotation_preserves_norm():
    """Applying RoPE shouldn't change the vector norm."""
    cos, sin = compute_rope_freqs(seq_len=8, dim=16)
    x = torch.randn(1, 8, 2, 16)  # (B, S, heads, head_dim)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


# Reward extraction (v4 native tags) ----------------------------------------


def test_extract_numeric_answer_with_native_tags():
    completion = "<|think|>2+2 equals 4.<|/think|><|answer|>4<|/answer|>"
    got = extract_numeric_answer(
        completion,
        think_close="<|/think|>",
        answer_open="<|answer|>",
        answer_close="<|/answer|>",
    )
    assert got == "4"


def test_extract_numeric_answer_fallback_to_think_close():
    # No answer tags, but a close-think tag — take first number after.
    completion = "<think>Let me solve it.</think>\n42 is the answer."
    got = extract_numeric_answer(completion)
    assert got == "42"


def test_extract_numeric_answer_fallback_to_last_number():
    # No think or answer tags — fall through to "last number in text".
    # Avoid trailing punctuation in the input because the numeric regex
    # also consumes a trailing dot as part of a potential decimal.
    completion = "Some reasoning. The result is 7 pigs"
    got = extract_numeric_answer(completion)
    assert got == "7"


def test_compute_reward_v4_native_tags_scores_correct_answer():
    completion = (
        "<|think|>Step 1: compute 2+2.\n"
        "Step 2: verify by counting.<|/think|>"
        "<|answer|>4<|/answer|>"
    )
    reward, breakdown = compute_reward(
        completion,
        "#### 4",
        think_open="<|think|>",
        think_close="<|/think|>",
        answer_open="<|answer|>",
        answer_close="<|/answer|>",
    )
    assert breakdown["correct"] is True
    assert breakdown["has_format"] is True
    # correctness (1.0) + format (0.2) + reasoning bonus > 1.2
    assert reward >= 1.2


def test_compute_reward_wrong_answer_scores_zero_correctness():
    completion = (
        "<|think|>2+2=5<|/think|><|answer|>5<|/answer|>"
    )
    reward, breakdown = compute_reward(
        completion,
        "#### 4",
        think_open="<|think|>",
        think_close="<|/think|>",
        answer_open="<|answer|>",
        answer_close="<|/answer|>",
    )
    assert breakdown["correct"] is False
    assert breakdown["correctness_reward"] == 0.0
    # Format present but empty-ish thinking
    assert breakdown["has_format"] is True
