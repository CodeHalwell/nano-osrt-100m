"""Smoke tests for NanoOSRT v4.

Guards the things we actually care about:
  1. Config validation is correct (too-small top_k, etc.)
  2. Model instantiates with expected physical parameter count
  3. Forward + backward produces finite loss
  4. Differentiable MoE losses + telemetry populate during training
  5. Router row-norm init, gate init, loop-embedding init std
  6. Routing-mode dispatch (soft / blend / hard) produces different
     outputs and matches a reference soft weighted sum
  7. RoPE NTK scaling increases the effective theta
  8. Reward extraction handles native v4 tags (<|think|>, <|answer|>)

These are CPU-friendly with a tiny config so pytest runs fast.
"""

from __future__ import annotations

import math

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
        assert blk.moe.importance_loss is not None
        assert blk.moe.logit_bias_loss is not None
        assert blk.moe.z_loss is not None
        assert torch.isfinite(blk.moe.importance_loss)
        assert torch.isfinite(blk.moe.logit_bias_loss)
        assert torch.isfinite(blk.moe.z_loss)
        # importance loss is N * sum(p^2), minimum N * (1/N) = 1.0
        # when perfectly uniform. At init it should be close to 1.0
        # because the router (row-normalised) has minimal bias toward
        # any expert; allow slack for sampling noise.
        assert blk.moe.importance_loss.item() >= 0.99
        assert blk.moe.importance_loss.item() < 2.0


def test_importance_loss_minimum_at_uniform():
    """Hand-compute importance loss for a crafted uniform softmax vector."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe

    # Synthetic router logits that produce an exactly uniform softmax
    logits = torch.zeros(1, 4, cfg.num_routed_experts)
    probs = torch.softmax(logits, dim=-1)
    moe._compute_losses(logits, probs)
    # N * sum((1/N)^2) = 1.0
    assert abs(moe.importance_loss.item() - 1.0) < 1e-5
    # Concentrated distribution: 90% on expert 0, 10% evenly across rest
    other = 0.1 / (cfg.num_routed_experts - 1)
    conc = torch.full((1, 4, cfg.num_routed_experts), other)
    conc[..., 0] = 0.9
    # Fake logits aren't used for importance loss computation here
    moe._compute_losses(torch.zeros_like(conc), conc)
    # Should be >> 1.0: ~ N * (0.9^2 + 10 * other^2)
    expected = cfg.num_routed_experts * (
        0.9**2 + (cfg.num_routed_experts - 1) * other**2
    )
    assert abs(moe.importance_loss.item() - expected) < 1e-4
    # With only 3 experts the theoretical max is ~2.445 (N=3,
    # 0.9 on one expert). We just need "clearly above uniform (1.0)".
    assert moe.importance_loss.item() > 2.0


def test_router_rows_equal_norm_at_init():
    """Router.weight rows should all have the same norm after post_init
    thanks to the row-norm tag on the router linear layer.
    """
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    for blk in model.model.blocks:
        rows = blk.moe.router.weight  # (num_routed, dim)
        norms = rows.norm(dim=1)
        # All rows should have the same norm (equal init) — within
        # float32 precision the deviation should be ~0.
        assert (norms.max() - norms.min()).item() < 1e-5
        # And the mean norm should be > 0 (we didn't zero the router).
        assert norms.mean().item() > 0.0


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
    """Hard-phase telemetry assignment entropy should fall under collapse."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)

    # Synthetic top_k_indices picking only experts 0 and 1
    collapsed = torch.zeros(1, 16, cfg.top_k_experts, dtype=torch.long)
    collapsed[..., 0] = 0
    collapsed[..., 1] = 1
    # Uniform softmax probs — the "misleading" signal we want to catch
    fake_probs = torch.full(
        (1, 16, cfg.num_routed_experts), 1.0 / cfg.num_routed_experts
    )

    moe = model.model.blocks[0].moe
    moe._record_hard_telemetry(
        fake_probs,
        top_k_indices=collapsed,
        clean_biased_indices=collapsed,
        clean_raw_indices=collapsed,
        loop_idx=0,
    )

    # Full top-2 collapse gives exactly ln(2) = 0.693.
    # Uniform routing over 3 routed experts gives ln(3) = 1.099.
    assert moe.last_assignment_entropy[0] < math.log(cfg.num_routed_experts) - 0.2
    assert abs(moe.last_assignment_entropy[0] - math.log(2)) < 0.05
    # Softmax entropy is still near-uniform because all probs are equal.
    assert moe.last_router_entropy[0] > math.log(cfg.num_routed_experts) * 0.95


def test_soft_routing_mode_uses_all_experts():
    """In mode 0 (soft), every expert gets nonzero gradient from a single
    forward+backward — no expert should be "dead"."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    model.train()
    for blk in model.model.blocks:
        blk.moe.routing_mode = 0
        blk.moe.routing_alpha.fill_(0.0)

    input_ids = torch.randint(0, cfg.real_vocab_size, (1, 8))
    out = model(input_ids, labels=input_ids)
    out.loss.backward()

    # Every routed expert in every block should have nonzero gradient
    # on its first weight matrix, because every expert ran on every token.
    for blk in model.model.blocks:
        for eid, expert in enumerate(blk.moe.experts):
            g = expert.w_gate.weight.grad
            assert g is not None, f"expert {eid} has no grad"
            assert g.abs().sum().item() > 0.0, f"expert {eid} grad is all zeros"


def test_balance_bias_controller_accumulate_and_apply():
    """Accumulate skewed assignment counts, then apply a single update.
    The over-used experts' bias should go down, the unused up, and
    everything should be clamped. Applying without calling accumulate
    is a no-op (accumulator is zero).
    """
    cfg = _tiny_config(
        router_balance_bias_enabled=True,
        router_balance_bias_max=0.5,
        router_balance_bias_update_rate=0.2,
    )
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe

    # All zero at init
    assert torch.all(moe.router_balance_bias == 0.0)

    # Skewed indices: expert 0 always picked as top-1, expert 1 as top-2
    skewed = torch.zeros(1, 32, cfg.top_k_experts, dtype=torch.long)
    skewed[..., 0] = 0
    skewed[..., 1] = 1

    # Accumulate multiple forward's worth of assignments (simulating
    # the 6 recursive-loop calls in a single training step)
    for _ in range(6):
        moe._accumulate_balance_counts(skewed)

    # Before apply: bias still zero, accumulator non-zero
    assert torch.all(moe.router_balance_bias == 0.0)
    assert moe.balance_total_accum.item() > 0

    # Apply the update once
    moe.apply_balance_update()

    # Experts 0 and 1 should be pushed DOWN (negative bias)
    assert moe.router_balance_bias[0].item() < 0
    assert moe.router_balance_bias[1].item() < 0
    # Expert 2 (unused) should be pushed UP (positive bias)
    assert moe.router_balance_bias[2].item() > 0
    # All clamped to [-0.5, 0.5]
    assert moe.router_balance_bias.abs().max().item() <= 0.5 + 1e-6
    # Accumulator has been reset
    assert torch.all(moe.balance_count_accum == 0.0)
    assert moe.balance_total_accum.item() == 0.0

    # Multiple applies without new accumulation are safe no-ops
    bias_before = moe.router_balance_bias.clone()
    moe.apply_balance_update()
    assert torch.allclose(moe.router_balance_bias, bias_before)


def test_balance_bias_applied_in_hard_gate_selection():
    """The balance bias should change _hard_gate's top-k selection
    when the bias is large enough to reorder the logits.
    """
    cfg = _tiny_config(router_balance_bias_enabled=True)
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe
    moe.gumbel_tau.fill_(0.0)  # deterministic for the test

    # Logits: expert 2 is highest
    logits = torch.tensor([[[-1.0, 0.5, 2.0]]])

    # With zero bias, top-2 picks [2, 1]
    moe.router_balance_bias.zero_()
    _, indices = moe._hard_gate(logits)
    assert indices[0, 0].tolist() == [2, 1]

    # Push expert 2 down with a large bias — now top-2 should be [1, 0]
    moe.router_balance_bias.copy_(torch.tensor([0.0, 0.0, -5.0]))
    _, indices = moe._hard_gate(logits)
    assert indices[0, 0].tolist() == [1, 0]


def test_balance_bias_persists_in_state_dict():
    """router_balance_bias is a persistent buffer — it should appear
    in state_dict and survive save/load."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe
    moe.router_balance_bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    state = model.state_dict()
    found = [k for k in state.keys() if "router_balance_bias" in k]
    assert found, "router_balance_bias not in state_dict"

    # Round-trip via a fresh model
    model2 = NanoOSRTv4ForCausalLM(cfg)
    model2.load_state_dict(state)
    assert torch.allclose(
        model2.model.blocks[0].moe.router_balance_bias,
        torch.tensor([0.1, -0.2, 0.3]),
    )


def _reference_assign_with_caps(moe, cand_indices, cand_scores, N, capacity, device):
    """Python-loop reference oracle for capacity-capped assignment.

    Processes one expert at a time within each rank — guaranteed
    race-free, easy to verify by inspection. Used as ground truth
    for testing the vectorised production implementation.
    """
    assigned = torch.full((N, moe.top_k), -1, dtype=torch.long, device=device)
    assigned_w = torch.zeros(N, moe.top_k, device=device)
    expert_load = torch.zeros(moe.num_routed, dtype=torch.long, device=device)
    slots_filled = torch.zeros(N, dtype=torch.long, device=device)
    rank_sum = torch.zeros(N, dtype=torch.float32, device=device)

    for rank in range(moe.candidate_k):
        if (slots_filled >= moe.top_k).all():
            break
        eid_col = cand_indices[:, rank]
        score_col = cand_scores[:, rank] if cand_scores is not None else None
        for e in range(moe.num_routed):
            remaining = capacity - expert_load[e].item()
            if remaining <= 0:
                continue
            mask = (eid_col == e) & (slots_filled < moe.top_k)
            if not mask.any():
                continue
            candidates = mask.nonzero(as_tuple=True)[0]
            take = candidates[:remaining]
            sidx = slots_filled[take]
            assigned[take, sidx] = e
            if score_col is not None:
                assigned_w[take, sidx] = score_col[take].float()
            slots_filled[take] += 1
            rank_sum[take] += rank
            expert_load[e] += len(take)
    return assigned, assigned_w, slots_filled, rank_sum


class TestCapacityCappedAssignment:
    """Comprehensive tests for _assign_with_caps against a reference
    oracle. Covers: overflow, assigned/tok, max bound, no duplicate
    experts per token, rank_mean match, same selections as oracle,
    unfilled slot masking, and bf16 autocast compatibility.
    """

    @staticmethod
    def _make_moe(num_routed=11, top_k=2, candidate_k=11, capacity_factor=1.25):
        cfg = _tiny_config(
            num_routed_experts=num_routed,
            num_experts=num_routed + 1,
            top_k_experts=top_k,
            router_capacity_capped=True,
            router_candidate_k=candidate_k,
            router_capacity_factor=capacity_factor,
        )
        model = NanoOSRTv4ForCausalLM(cfg)
        return model.model.blocks[0].moe

    @staticmethod
    def _random_candidates(N, num_routed, candidate_k, device="cpu"):
        logits = torch.randn(N, num_routed, device=device)
        sig = torch.sigmoid(logits)
        cand_s, cand_i = torch.topk(sig, candidate_k, dim=-1)
        return cand_i, cand_s

    @staticmethod
    def _concentrated_candidates(N, num_routed, candidate_k, device="cpu"):
        """All tokens strongly prefer expert 0."""
        logits = torch.randn(N, num_routed, device=device) * 0.01
        logits[:, 0] = 10.0
        sig = torch.sigmoid(logits)
        cand_s, cand_i = torch.topk(sig, candidate_k, dim=-1)
        return cand_i, cand_s

    def test_oracle_match_random_logits(self):
        moe = self._make_moe(num_routed=3, candidate_k=3)
        N = 512
        cand_i, cand_s = self._random_candidates(N, 3, 3)
        cap = math.ceil(1.25 * N * 2 / 3)

        ref_a, ref_w, ref_sf, ref_rs = _reference_assign_with_caps(
            moe,
            cand_i,
            cand_s,
            N,
            cap,
            torch.device("cpu"),
        )
        vec_a, vec_w, vec_sf, vec_rs = moe._assign_with_caps(
            cand_i,
            cand_s,
            N,
            cap,
            torch.device("cpu"),
        )
        assert torch.equal(vec_a, ref_a), "assigned indices differ from oracle"
        assert torch.allclose(vec_w, ref_w, atol=1e-6), "weights differ"
        assert torch.equal(vec_sf, ref_sf), "slots_filled differ"
        assert torch.equal(vec_rs.long(), ref_rs.long()), "rank_sum differ"

    def test_oracle_match_concentrated_logits(self):
        moe = self._make_moe(num_routed=3, candidate_k=3)
        N = 512
        cand_i, cand_s = self._concentrated_candidates(N, 3, 3)
        cap = math.ceil(1.25 * N * 2 / 3)

        ref_a, ref_w, ref_sf, ref_rs = _reference_assign_with_caps(
            moe,
            cand_i,
            cand_s,
            N,
            cap,
            torch.device("cpu"),
        )
        vec_a, vec_w, vec_sf, vec_rs = moe._assign_with_caps(
            cand_i,
            cand_s,
            N,
            cap,
            torch.device("cpu"),
        )
        assert torch.equal(vec_a, ref_a)
        assert torch.allclose(vec_w, ref_w, atol=1e-6)
        assert torch.equal(vec_sf, ref_sf)

    def test_zero_overflow_when_candidate_k_equals_num_experts(self):
        moe = self._make_moe(num_routed=11, candidate_k=11)
        N = 2048
        cand_i, cand_s = self._random_candidates(N, 11, 11)
        cap = math.ceil(1.25 * N * 2 / 11)
        _, _, sf, _ = moe._assign_with_caps(cand_i, cand_s, N, cap, torch.device("cpu"))
        assert (sf == 2).all(), f"overflow: {(sf < 2).sum()} tokens unfilled"

    def test_assigned_per_token_equals_top_k(self):
        moe = self._make_moe(num_routed=11, candidate_k=11)
        N = 2048
        cand_i, cand_s = self._random_candidates(N, 11, 11)
        cap = math.ceil(1.25 * N * 2 / 11)
        _, _, sf, _ = moe._assign_with_caps(cand_i, cand_s, N, cap, torch.device("cpu"))
        assert sf.float().mean().item() == 2.0

    def test_expert_max_bounded_by_capacity(self):
        moe = self._make_moe(num_routed=11, candidate_k=11)
        N = 4096
        cand_i, cand_s = self._concentrated_candidates(N, 11, 11)
        cap = math.ceil(1.25 * N * 2 / 11)
        assigned, _, _, _ = moe._assign_with_caps(
            cand_i, cand_s, N, cap, torch.device("cpu")
        )
        valid = assigned[assigned >= 0]
        counts = torch.bincount(valid, minlength=11)
        assert counts.max().item() <= cap, f"expert got {counts.max()} > cap {cap}"

    def test_no_duplicate_expert_per_token(self):
        moe = self._make_moe(num_routed=11, candidate_k=11)
        N = 2048
        cand_i, cand_s = self._random_candidates(N, 11, 11)
        cap = math.ceil(1.25 * N * 2 / 11)
        assigned, _, sf, _ = moe._assign_with_caps(
            cand_i, cand_s, N, cap, torch.device("cpu")
        )
        for i in range(N):
            filled = assigned[i, : sf[i]]
            assert filled.unique().shape[0] == sf[i], (
                f"token {i} has duplicate expert: {filled.tolist()}"
            )

    def test_unfilled_slots_are_minus_one(self):
        """With capacity_factor=0.5, some tokens can't fill both slots.
        Unfilled slots should stay -1 with weight 0."""
        moe = self._make_moe(num_routed=3, candidate_k=3, capacity_factor=0.5)
        N = 512
        cand_i, cand_s = self._concentrated_candidates(N, 3, 3)
        cap = math.ceil(0.5 * N * 2 / 3)
        assigned, assigned_w, sf, _ = moe._assign_with_caps(
            cand_i,
            cand_s,
            N,
            cap,
            torch.device("cpu"),
        )
        unfilled_mask = assigned == -1
        assert (assigned_w[unfilled_mask] == 0).all()
        # Some tokens should have unfilled slots with factor=0.5
        assert (sf < 2).any(), "expected some overflow with factor=0.5"

    def test_scores_none_shadow_telemetry(self):
        """When cand_scores=None (shadow path), weights should be all 0
        and assignments should still be race-free."""
        moe = self._make_moe(num_routed=3, candidate_k=3)
        N = 256
        cand_i, _ = self._random_candidates(N, 3, 3)
        cap = math.ceil(1.25 * N * 2 / 3)
        assigned, assigned_w, sf, _ = moe._assign_with_caps(
            cand_i,
            None,
            N,
            cap,
            torch.device("cpu"),
        )
        assert (assigned_w == 0).all()
        assert (sf == 2).all()


def test_hard_gate_with_gumbel_noise_differs_from_clean_selection():
    """With gumbel_tau > 0 during training, _hard_gate should produce
    different selections across successive calls on the same logits.
    With gumbel_tau = 0 (or eval mode), selection should be
    deterministic.
    """
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe
    model.train()

    # Logits with a small margin so top-k isn't overwhelmingly decided
    torch.manual_seed(0)
    logits = torch.randn(1, 32, cfg.num_routed_experts) * 0.1

    moe.gumbel_tau.fill_(1.0)
    _, idx_a = moe._hard_gate(logits)
    _, idx_b = moe._hard_gate(logits)
    # Stochastic selection with tau=1.0 should diverge across calls
    assert not torch.equal(idx_a, idx_b)

    # With tau=0 selection must be deterministic
    moe.gumbel_tau.fill_(0.0)
    _, idx_c = moe._hard_gate(logits)
    _, idx_d = moe._hard_gate(logits)
    assert torch.equal(idx_c, idx_d)

    # In eval mode even with tau>0 the selection should be deterministic
    model.eval()
    moe.gumbel_tau.fill_(1.0)
    _, idx_e = moe._hard_gate(logits)
    _, idx_f = moe._hard_gate(logits)
    assert torch.equal(idx_e, idx_f)


def test_hard_gate_weights_come_from_clean_logits_not_noised():
    """Even when Gumbel selects different experts on different calls,
    the returned weights should always be the CLEAN sigmoid of the
    original logits (gathered at the chosen positions), not sigmoid of
    the noised logits."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe
    model.train()
    moe.gumbel_tau.fill_(0.0)  # deterministic for the comparison

    logits = torch.tensor([[[-1.0, 0.5, 2.0]]])
    weights, indices = moe._hard_gate(logits)
    # top-k=2 should pick experts 2 and 1 (sigmoid of 2.0 and 0.5)
    assert indices[0, 0].tolist() == [2, 1]
    # Weights should be sigmoid([2.0, 0.5]) renormalized to sum to 1
    expected_sig = torch.sigmoid(torch.tensor([2.0, 0.5]))
    expected = expected_sig / expected_sig.sum()
    assert torch.allclose(weights[0, 0], expected, atol=1e-5)


def test_hard_gate_returns_topk_indices_and_renormalised_weights():
    """Direct test of _hard_gate: top-k indices correspond to highest
    sigmoid scores and the returned weights sum to 1 per token.

    Must disable Gumbel noise (tau=0) for a deterministic comparison.
    """
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    moe = model.model.blocks[0].moe
    moe.gumbel_tau.fill_(0.0)  # disable Gumbel for deterministic top-k

    # Hand-crafted logits: per-token argmax is expert 2 then expert 0.
    logits = torch.tensor(
        [[[-1.0, 0.5, 2.0], [1.0, -0.5, 0.2]]],
    )
    top_k_weights, top_k_indices = moe._hard_gate(logits)

    # Shape checks
    assert top_k_weights.shape == (1, 2, cfg.top_k_experts)
    assert top_k_indices.shape == (1, 2, cfg.top_k_experts)
    # First position: top-2 from logits (-1, 0.5, 2.0) → experts 2, 1
    assert top_k_indices[0, 0].tolist() == [2, 1]
    # Second position: top-2 from logits (1, -0.5, 0.2) → experts 0, 2
    assert top_k_indices[0, 1].tolist() == [0, 2]
    # Weights must sum to 1 along the top-k axis (renormalisation)
    sums = top_k_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_blend_mode_interpolates_between_soft_and_hard():
    """With alpha=0.5 the routed output should fall between the pure
    soft and pure hard outputs (entry-wise)."""
    cfg = _tiny_config()
    model = NanoOSRTv4ForCausalLM(cfg)
    model.eval()  # deterministic, no loss computation
    moe = model.model.blocks[0].moe

    x = torch.randn(1, 4, cfg.dim)
    with torch.no_grad():
        moe.routing_mode = 0
        moe.routing_alpha.fill_(0.0)
        soft_out = moe(x, loop_idx=0)

        moe.routing_mode = 2
        moe.routing_alpha.fill_(1.0)
        hard_out = moe(x, loop_idx=0)

        moe.routing_mode = 1
        moe.routing_alpha.fill_(0.5)
        blend_out = moe(x, loop_idx=0)

    # forward() returns shared_out + routed_out. The shared_out term is
    # identical across all three modes, so blending the routed branches
    # by alpha=0.5 is equivalent to averaging the full outputs.
    expected_blend = 0.5 * soft_out + 0.5 * hard_out
    assert torch.allclose(blend_out, expected_blend, atol=1e-4)


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


def test_loop_embedding_init_std_survives_post_init():
    """Regression test: HF PreTrainedModel.post_init() walks the module
    tree and re-inits every nn.Embedding with initializer_range=0.02,
    which previously silently overwrote the larger loop-embedding init.
    Verify that the _osrt_init_std tag is respected and the actual
    stddev of the initialised weights matches config.loop_embedding_init_std.
    """
    cfg = _tiny_config(loop_embedding_init_std=0.15)
    model = NanoOSRTv4ForCausalLM(cfg)
    for blk in model.model.blocks:
        w = blk.moe.loop_embeddings.weight.detach()
        actual_std = w.std().item()
        # With small tensors (recursive_loops × dim = 3 × 128 = 384 values)
        # the empirical std has some sampling error, so allow 30% tolerance.
        assert (
            0.7 * cfg.loop_embedding_init_std
            < actual_std
            < 1.3 * cfg.loop_embedding_init_std
        ), (
            f"Expected loop_embeddings std ~ {cfg.loop_embedding_init_std}, "
            f"got {actual_std:.4f}. HF post_init probably stomped the custom init."
        )
        # Verify the tag is present so the custom init path is taken
        assert (
            getattr(blk.moe.loop_embeddings, "_osrt_init_std", None)
            == cfg.loop_embedding_init_std
        )


def test_token_embedding_still_uses_default_std():
    """Sanity: the main token embedding should still use initializer_range,
    not the loop-embedding std, since only loop_embeddings was tagged.
    """
    cfg = _tiny_config(loop_embedding_init_std=0.15, initializer_range=0.02)
    model = NanoOSRTv4ForCausalLM(cfg)
    w = model.model.embedding.weight.detach()
    actual_std = w.std().item()
    assert 0.7 * cfg.initializer_range < actual_std < 1.3 * cfg.initializer_range, (
        f"Token embedding std expected ~ {cfg.initializer_range}, got {actual_std:.4f}"
    )


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
        seq_len=16,
        dim=32,
        theta=10000.0,
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
    completion = "<|think|>2+2=5<|/think|><|answer|>5<|/answer|>"
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


# ── KV cache tests (merged from PR #10) ─────────────────────────────


@pytest.fixture()
def tiny_v4_config() -> NanoOSRTv4Config:
    """A tiny v4 config suitable for KV-cache CPU tests."""
    return NanoOSRTv4Config(
        dim=64,
        heads=4,
        head_dim=16,
        vocab_size=256,
        real_vocab_size=256,
        num_blocks=2,
        recursive_loops=2,
        adapter_rank=4,
        adapter_alpha=4.0,
        dense_hidden=128,
        num_experts=4,
        num_shared_experts=1,
        num_routed_experts=3,
        top_k_experts=2,
        expert_hidden=64,
        max_position_embeddings=128,
    )


class TestV4ModelBasic:
    def test_instantiation(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        assert isinstance(model, NanoOSRTv4ForCausalLM)

    def test_forward_without_cache(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 8, tiny_v4_config.vocab_size)
        assert outputs.past_key_values is None

    def test_forward_with_labels(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        labels = torch.randint(0, tiny_v4_config.real_vocab_size, (2, 8))
        outputs = model(input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.loss.ndim == 0


class TestKVCache:
    def test_forward_returns_cache_when_requested(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 8))
        outputs = model(input_ids, use_cache=True)

        expected_layers = tiny_v4_config.num_blocks * tiny_v4_config.recursive_loops
        assert outputs.past_key_values is not None
        assert len(outputs.past_key_values) == expected_layers

        for k, v in outputs.past_key_values:
            assert k.shape == (1, tiny_v4_config.heads, 8, tiny_v4_config.head_dim)
            assert v.shape == (1, tiny_v4_config.heads, 8, tiny_v4_config.head_dim)

    def test_incremental_matches_full(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 6))

        full_outputs = model(input_ids, use_cache=False)
        full_logits = full_outputs.logits

        prefill_ids = input_ids[:, :4]
        prefill_out = model(prefill_ids, use_cache=True)
        cache = prefill_out.past_key_values

        step1_ids = input_ids[:, 4:5]
        step1_out = model(step1_ids, past_key_values=cache, use_cache=True)
        cache = step1_out.past_key_values

        step2_ids = input_ids[:, 5:6]
        step2_out = model(step2_ids, past_key_values=cache, use_cache=True)

        torch.testing.assert_close(
            full_logits[:, 5, :],
            step2_out.logits[:, 0, :],
            atol=5e-4,
            rtol=5e-4,
        )
        torch.testing.assert_close(
            full_logits[:, 4, :],
            step1_out.logits[:, 0, :],
            atol=5e-4,
            rtol=5e-4,
        )

    def test_cache_grows_correctly(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))

        out = model(input_ids, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 4

        new_token = torch.randint(0, tiny_v4_config.vocab_size, (1, 1))
        out = model(new_token, past_key_values=out.past_key_values, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 5

        new_token = torch.randint(0, tiny_v4_config.vocab_size, (1, 1))
        out = model(new_token, past_key_values=out.past_key_values, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 6


class TestGenerate:
    def test_generate_with_cache(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(
            input_ids, max_new_tokens=5, temperature=1.0, use_cache=True
        )
        assert out.shape == (1, 9)

    def test_generate_without_cache(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(
            input_ids, max_new_tokens=5, temperature=1.0, use_cache=False
        )
        assert out.shape == (1, 9)

    def test_generate_greedy_deterministic(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))

        out_cached = model.generate(
            input_ids.clone(),
            max_new_tokens=8,
            temperature=0.0,
            use_cache=True,
        )
        out_nocache = model.generate(
            input_ids.clone(),
            max_new_tokens=8,
            temperature=0.0,
            use_cache=False,
        )
        assert torch.equal(out_cached, out_nocache)
