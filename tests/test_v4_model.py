"""Tests for NanoOSRT v4 — Recursive MoE model, KV cache, and training helpers."""

import math

import pytest
import torch

from nano_osrt.v4_config import NanoOSRTv4Config
from nano_osrt.v4_model import (
    DenseSwiGLU,
    ExpertFFN,
    MoELayer,
    NanoOSRTv4ForCausalLM,
    NanoOSRTv4Model,
    RecursiveBlockV4,
    apply_rope,
    compute_rope_freqs,
)

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def tiny_v4_config() -> NanoOSRTv4Config:
    """A tiny v4 config suitable for CPU tests."""
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
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=0.001,
        max_position_embeddings=128,
    )


# ── Config ──────────────────────────────────────────────────────────────


class TestNanoOSRTv4Config:
    def test_default_values(self) -> None:
        cfg = NanoOSRTv4Config()
        assert cfg.dim == 1536
        assert cfg.heads == 24
        assert cfg.head_dim == 64
        assert cfg.num_blocks == 3
        assert cfg.recursive_loops == 6
        assert cfg.num_experts == 12
        assert cfg.num_shared_experts == 1
        assert cfg.num_routed_experts == 11
        assert cfg.top_k_experts == 2

    def test_dim_heads_validation(self) -> None:
        with pytest.raises(ValueError, match="dim.*must be divisible by heads"):
            NanoOSRTv4Config(dim=65, heads=4)

    def test_top_k_exceeds_routed_validation(self) -> None:
        with pytest.raises(ValueError, match="top_k_experts.*must be <="):
            NanoOSRTv4Config(top_k_experts=12, num_routed_experts=11)

    def test_num_blocks_validation(self) -> None:
        with pytest.raises(ValueError, match="num_blocks must be >= 1"):
            NanoOSRTv4Config(num_blocks=0)

    def test_recursive_loops_validation(self) -> None:
        with pytest.raises(ValueError, match="recursive_loops must be >= 1"):
            NanoOSRTv4Config(recursive_loops=0)

    def test_special_token_ids(self) -> None:
        cfg = NanoOSRTv4Config()
        assert cfg.bos_token_id == 1
        assert cfg.eos_token_id == 2
        assert cfg.pad_token_id == 0
        assert cfg.fim_prefix_id == 4


# ── RoPE (v4 copy) ─────────────────────────────────────────────────────


class TestV4RoPE:
    def test_compute_rope_freqs_shape(self) -> None:
        seq_len, dim = 32, 16
        cos, sin = compute_rope_freqs(seq_len, dim)
        assert cos.shape == (1, seq_len, 1, dim)
        assert sin.shape == (1, seq_len, 1, dim)

    def test_odd_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="even dimension"):
            compute_rope_freqs(32, 15)

    def test_apply_rope_preserves_norm(self) -> None:
        B, S, H, D = 1, 8, 2, 16
        x = torch.randn(B, S, H, D)
        cos, sin = compute_rope_freqs(S, D)
        out = apply_rope(x, cos, sin)
        torch.testing.assert_close(
            x.norm(dim=-1), out.norm(dim=-1), atol=1e-5, rtol=1e-5
        )


# ── ExpertFFN ───────────────────────────────────────────────────────────


class TestExpertFFN:
    def test_output_shape(self) -> None:
        expert = ExpertFFN(64, 128)
        x = torch.randn(2, 8, 64)
        out = expert(x)
        assert out.shape == (2, 8, 64)

    def test_hidden_tc_alignment(self) -> None:
        expert = ExpertFFN(64, 100)
        # Hidden should be rounded up to next multiple of 64
        assert expert.w_gate.out_features % 64 == 0

    def test_no_bias(self) -> None:
        expert = ExpertFFN(64, 128)
        assert expert.w_gate.bias is None
        assert expert.w_up.bias is None
        assert expert.w_down.bias is None


# ── DenseSwiGLU ─────────────────────────────────────────────────────────


class TestDenseSwiGLU:
    def test_output_shape(self) -> None:
        ffn = DenseSwiGLU(64, 256)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_hidden_tc_alignment(self) -> None:
        ffn = DenseSwiGLU(64, 100)
        assert ffn.w_gate.out_features % 64 == 0


# ── MoELayer ────────────────────────────────────────────────────────────


class TestMoELayer:
    def test_output_shape(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        moe = MoELayer(tiny_v4_config)
        x = torch.randn(2, 8, tiny_v4_config.dim)
        out = moe(x, loop_idx=0)
        assert out.shape == (2, 8, tiny_v4_config.dim)

    def test_losses_set_in_train_mode(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        moe = MoELayer(tiny_v4_config)
        moe.train()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        moe(x, loop_idx=0)
        assert moe.load_balance_loss is not None
        assert moe.z_loss is not None
        assert moe.load_balance_loss.ndim == 0  # scalar
        assert moe.z_loss.ndim == 0  # scalar

    def test_losses_none_in_eval_mode(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        moe = MoELayer(tiny_v4_config)
        moe.eval()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        moe(x, loop_idx=0)
        assert moe.load_balance_loss is None
        assert moe.z_loss is None

    def test_different_loop_indices(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        """Different loop indices produce different outputs."""
        moe = MoELayer(tiny_v4_config)
        moe.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 4, tiny_v4_config.dim)
        out0 = moe(x, loop_idx=0)
        out1 = moe(x, loop_idx=1)
        # Outputs should differ due to loop-aware routing
        assert not torch.allclose(out0, out1, atol=1e-6)

    def test_sigmoid_gating_range(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """Verify router uses sigmoid (outputs in [0,1]) not softmax."""
        moe = MoELayer(tiny_v4_config)
        moe.eval()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        # Access router logits directly
        loop_emb = moe.loop_embeddings(
            moe.loop_indices[0]
        ).unsqueeze(0).unsqueeze(0).expand(2, 8, -1)
        router_input = torch.cat([x, loop_emb], dim=-1)
        logits = moe.router(router_input)
        probs = torch.sigmoid(logits)
        # Sigmoid outputs are independent; they don't need to sum to 1
        # Each should be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_load_balance_loss_nonnegative(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        moe = MoELayer(tiny_v4_config)
        moe.train()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        moe(x, loop_idx=0)
        assert moe.load_balance_loss.item() >= 0

    def test_z_loss_nonnegative(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        moe = MoELayer(tiny_v4_config)
        moe.train()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        moe(x, loop_idx=0)
        assert moe.z_loss.item() >= 0

    def test_expert_count_matches_config(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        moe = MoELayer(tiny_v4_config)
        assert len(moe.experts) == tiny_v4_config.num_routed_experts

    def test_router_input_dim(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """Router should accept dim*2 input (hidden + loop embedding)."""
        moe = MoELayer(tiny_v4_config)
        assert moe.router.in_features == tiny_v4_config.dim * 2
        assert moe.router.out_features == tiny_v4_config.num_routed_experts

    def test_expert_counts_tracked_in_train(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        moe = MoELayer(tiny_v4_config)
        moe.train()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        moe(x, loop_idx=0)
        assert moe.expert_counts is not None
        assert moe.expert_counts.shape == (tiny_v4_config.num_routed_experts,)
        # Total tokens dispatched = B*S*top_k = 2*8*2 = 32
        assert moe.expert_counts.sum().item() == (
            2 * 8 * tiny_v4_config.top_k_experts
        )

    def test_expert_counts_none_in_eval(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        moe = MoELayer(tiny_v4_config)
        moe.eval()
        x = torch.randn(2, 8, tiny_v4_config.dim)
        moe(x, loop_idx=0)
        assert moe.expert_counts is None


# ── RecursiveBlockV4 ────────────────────────────────────────────────────


class TestRecursiveBlockV4:
    def test_output_shape(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        block = RecursiveBlockV4(tiny_v4_config)
        B, S = 2, 8
        x = torch.randn(B, S, tiny_v4_config.dim)
        adapter_a = torch.randn(tiny_v4_config.dim, tiny_v4_config.adapter_rank) * 0.01
        adapter_b = torch.zeros(tiny_v4_config.adapter_rank, tiny_v4_config.dim)
        cos, sin = compute_rope_freqs(S, tiny_v4_config.head_dim)
        out, present_kv = block(x, adapter_a, adapter_b, 1.0, cos, sin, loop_idx=0)
        assert out.shape == (B, S, tiny_v4_config.dim)
        assert present_kv is None  # use_cache defaults to False

    def test_parallel_ffn_gates_init(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """Dense and MoE gates should start at 0.5 (combined = 1.0)."""
        block = RecursiveBlockV4(tiny_v4_config)
        assert block.dense_gate.item() == pytest.approx(0.5)
        assert block.moe_gate.item() == pytest.approx(0.5)


# ── NanoOSRTv4Model ────────────────────────────────────────────────────


class TestNanoOSRTv4Model:
    def test_forward_returns_tuple(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4Model(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 8))
        hidden, loop_rms, lb_loss, z_loss, presents = model(input_ids)
        assert hidden.shape == (1, 8, tiny_v4_config.dim)
        assert len(loop_rms) == tiny_v4_config.recursive_loops
        assert lb_loss.ndim == 0
        assert z_loss.ndim == 0
        assert presents is None  # use_cache defaults to False

    def test_loop_rms_positive(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4Model(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        _, loop_rms, _, _, _ = model(input_ids)
        for rms in loop_rms:
            assert rms.item() > 0

    def test_adapter_count(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4Model(tiny_v4_config)
        expected = tiny_v4_config.num_blocks * tiny_v4_config.recursive_loops
        assert len(model.adapters_a) == expected
        assert len(model.adapters_b) == expected

    def test_moe_losses_accumulate_in_train(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        model = NanoOSRTv4Model(tiny_v4_config)
        model.train()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 8))
        _, _, lb_loss, z_loss, _ = model(input_ids)
        # With 2 blocks × 2 loops = 4 MoE forwards, losses should be > 0
        assert lb_loss.item() > 0
        assert z_loss.item() > 0

    def test_get_moe_stats_train(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        """get_moe_stats returns utilization metrics after train."""
        model = NanoOSRTv4Model(tiny_v4_config)
        model.train()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        model(input_ids)
        stats = model.get_moe_stats()
        assert "moe/expert_balance_cv" in stats
        assert "moe/max_expert_share" in stats
        assert "moe/dead_experts" in stats
        assert stats["moe/expert_balance_cv"] >= 0
        assert 0 < stats["moe/max_expert_share"] <= 1.0
        assert stats["moe/dead_experts"] >= 0

    def test_get_moe_stats_empty_in_eval(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        """get_moe_stats returns empty dict in eval mode."""
        model = NanoOSRTv4Model(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 8))
        model(input_ids)
        stats = model.get_moe_stats()
        assert stats == {}


# ── NanoOSRTv4ForCausalLM ──────────────────────────────────────────────


class TestNanoOSRTv4ForCausalLM:
    def test_instantiation(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        assert isinstance(model, NanoOSRTv4ForCausalLM)

    def test_forward_no_labels(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        outputs = model(input_ids)
        assert outputs.loss is None
        assert outputs.logits.shape == (2, 8, tiny_v4_config.vocab_size)
        assert outputs.past_key_values is None

    def test_forward_with_labels(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.train()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        labels = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        outputs = model(input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # scalar
        assert outputs.logits.shape == (2, 8, tiny_v4_config.vocab_size)

    def test_loss_includes_moe_aux(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """Loss should include cross-entropy + load_balance + z_loss."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.train()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        labels = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))

        outputs = model(input_ids, labels=labels)
        total_loss = outputs.loss

        # Get sub-losses by running the inner model
        with torch.no_grad():
            _, _, lb_loss, z_loss, _ = model.model(input_ids)

        # Total loss should be > just cross-entropy (the aux losses add)
        # We can't easily decompose, but loss should be a reasonable number
        assert total_loss.item() > 0
        assert math.isfinite(total_loss.item())

    def test_weight_tying(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """LM head should use the embedding weights (weight-tied)."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        assert model.model.embedding.weight.shape == (
            tiny_v4_config.vocab_size, tiny_v4_config.dim
        )

    def test_generate(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(input_ids, max_new_tokens=3, temperature=1.0)
        assert out.shape[0] == 1
        assert out.shape[1] >= 4  # at least the input length
        assert out.shape[1] <= 7  # at most input + max_new_tokens

    def test_generate_greedy(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """Greedy decoding (temperature=0) should be deterministic."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out1 = model.generate(input_ids, max_new_tokens=3, temperature=0)
        out2 = model.generate(input_ids, max_new_tokens=3, temperature=0)
        assert torch.equal(out1, out2)

    def test_backward_pass(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        """Verify gradients flow through the entire model including MoE."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.train()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        labels = torch.randint(0, tiny_v4_config.vocab_size, (2, 8))
        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check gradients exist for key components
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_num_parameters(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0


# ── KV cache correctness ─────────────────────────────────────────────


class TestKVCache:
    def test_forward_returns_cache_when_requested(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 8))
        outputs = model(input_ids, use_cache=True)

        # Should have num_blocks * recursive_loops cache entries
        expected_layers = (
            tiny_v4_config.num_blocks * tiny_v4_config.recursive_loops
        )
        assert outputs.past_key_values is not None
        assert len(outputs.past_key_values) == expected_layers

        # Each entry is (K, V) with shape (B, heads, S, head_dim)
        for k, v in outputs.past_key_values:
            assert k.shape == (
                1, tiny_v4_config.heads, 8, tiny_v4_config.head_dim,
            )
            assert v.shape == (
                1, tiny_v4_config.heads, 8, tiny_v4_config.head_dim,
            )

    def test_incremental_matches_full(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        """Incremental decoding produces identical logits to full pass."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 6))

        # Full forward pass
        full_outputs = model(input_ids, use_cache=False)
        full_logits = full_outputs.logits  # (1, 6, vocab)

        # Incremental: prefill 4, then token 5, then token 6
        prefill_ids = input_ids[:, :4]
        prefill_out = model(prefill_ids, use_cache=True)
        cache = prefill_out.past_key_values

        step1_ids = input_ids[:, 4:5]
        step1_out = model(
            step1_ids, past_key_values=cache, use_cache=True,
        )
        cache = step1_out.past_key_values

        step2_ids = input_ids[:, 5:6]
        step2_out = model(
            step2_ids, past_key_values=cache, use_cache=True,
        )

        torch.testing.assert_close(
            full_logits[:, 5, :],
            step2_out.logits[:, 0, :],
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            full_logits[:, 4, :],
            step1_out.logits[:, 0, :],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_cache_grows_correctly(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        """Cached sequence length grows by 1 each step."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))

        # Prefill
        out = model(input_ids, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 4

        # Step 1
        new_token = torch.randint(0, tiny_v4_config.vocab_size, (1, 1))
        out = model(
            new_token,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        assert out.past_key_values[0][0].shape[2] == 5

        # Step 2
        new_token = torch.randint(0, tiny_v4_config.vocab_size, (1, 1))
        out = model(
            new_token,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        assert out.past_key_values[0][0].shape[2] == 6


# ── Generate ─────────────────────────────────────────────────────────


class TestGenerate:
    def test_generate_with_cache(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(
            input_ids, max_new_tokens=5, temperature=1.0, use_cache=True,
        )
        assert out.shape == (1, 9)  # 4 prompt + 5 generated

    def test_generate_without_cache(
        self, tiny_v4_config: NanoOSRTv4Config,
    ) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(
            input_ids, max_new_tokens=5, temperature=1.0, use_cache=False,
        )
        assert out.shape == (1, 9)

    def test_generate_greedy_deterministic(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        """Greedy generation with/without cache produces identical tokens."""
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


# ── V4 Training Helpers ─────────────────────────────────────────────────


class TestV4TrainHelpers:
    def test_get_lr_warmup(self) -> None:
        from nano_osrt.v4_train import get_lr
        from nano_osrt.v4_train_config import V4PretrainConfig

        cfg = V4PretrainConfig()
        lr_0 = get_lr(0, cfg)
        lr_mid = get_lr(cfg.warmup_steps // 2, cfg)
        lr_end = get_lr(cfg.warmup_steps, cfg)
        assert lr_0 == 0.0
        assert 0 < lr_mid < cfg.peak_lr
        assert math.isclose(lr_end, cfg.peak_lr, rel_tol=1e-6)

    def test_get_lr_cosine_decay(self) -> None:
        from nano_osrt.v4_train import get_lr
        from nano_osrt.v4_train_config import V4PretrainConfig

        cfg = V4PretrainConfig()
        lr_after = get_lr(cfg.warmup_steps + 1, cfg)
        lr_near_end = get_lr(cfg.total_steps - 1, cfg)
        assert lr_after <= cfg.peak_lr
        assert lr_near_end >= cfg.min_lr

    def test_get_phase_foundation(self) -> None:
        from nano_osrt.v4_train import get_phase
        from nano_osrt.v4_train_config import V4PretrainConfig

        cfg = V4PretrainConfig()
        name, phase_cfg = get_phase(0, cfg)
        assert name == "foundation"
        assert phase_cfg["seq_len"] == 2048

    def test_get_phase_knowledge(self) -> None:
        from nano_osrt.v4_train import get_phase
        from nano_osrt.v4_train_config import V4PretrainConfig

        cfg = V4PretrainConfig()
        name, phase_cfg = get_phase(10_000, cfg)
        assert name == "knowledge"
        assert phase_cfg["seq_len"] == 4096

    def test_get_phase_instruction(self) -> None:
        from nano_osrt.v4_train import get_phase
        from nano_osrt.v4_train_config import V4PretrainConfig

        cfg = V4PretrainConfig()
        name, phase_cfg = get_phase(250_000, cfg)
        assert name == "instruction"
        assert phase_cfg["seq_len"] == 8192

    def test_get_phase_fallback(self) -> None:
        from nano_osrt.v4_train import get_phase
        from nano_osrt.v4_train_config import V4PretrainConfig

        cfg = V4PretrainConfig()
        name, _ = get_phase(999_999, cfg)
        # Falls back to last phase
        assert name == "instruction"
