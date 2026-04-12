"""Tests for NanoOSRT v4 model with KV cache support."""

import pytest
import torch

from nano_osrt.v4_config import NanoOSRTv4Config
from nano_osrt.v4_model import NanoOSRTv4ForCausalLM


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
        max_position_embeddings=128,
    )


# ── Basic forward / backward compat ──────────────────────────────────


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
        expected_layers = tiny_v4_config.num_blocks * tiny_v4_config.recursive_loops
        assert outputs.past_key_values is not None
        assert len(outputs.past_key_values) == expected_layers

        # Each entry is (K, V) with shape (B, heads, S, head_dim)
        for k, v in outputs.past_key_values:
            assert k.shape == (1, tiny_v4_config.heads, 8, tiny_v4_config.head_dim)
            assert v.shape == (1, tiny_v4_config.heads, 8, tiny_v4_config.head_dim)

    def test_incremental_matches_full(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        """Verify that incremental decoding produces identical logits to a
        full forward pass.  This is the critical correctness test for KV
        cache: given tokens [A, B, C], the logit for position C should be
        the same whether we process [A, B, C] in one shot or process
        [A, B] then [C] with cache."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 6))

        # Full forward pass — logits at every position
        full_outputs = model(input_ids, use_cache=False)
        full_logits = full_outputs.logits  # (1, 6, vocab)

        # Incremental: first 4 tokens as prefill, then token 5, then token 6
        prefill_ids = input_ids[:, :4]
        prefill_out = model(prefill_ids, use_cache=True)
        cache = prefill_out.past_key_values

        step1_ids = input_ids[:, 4:5]
        step1_out = model(step1_ids, past_key_values=cache, use_cache=True)
        cache = step1_out.past_key_values

        step2_ids = input_ids[:, 5:6]
        step2_out = model(step2_ids, past_key_values=cache, use_cache=True)

        # The logit for the last position should match
        torch.testing.assert_close(
            full_logits[:, 5, :],
            step2_out.logits[:, 0, :],
            atol=1e-4,
            rtol=1e-4,
        )
        # The logit at position 4 should also match
        torch.testing.assert_close(
            full_logits[:, 4, :],
            step1_out.logits[:, 0, :],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_cache_grows_correctly(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        """Check that the cached sequence length grows by 1 each step."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))

        # Prefill
        out = model(input_ids, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 4

        # Step 1
        new_token = torch.randint(0, tiny_v4_config.vocab_size, (1, 1))
        out = model(new_token, past_key_values=out.past_key_values, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 5

        # Step 2
        new_token = torch.randint(0, tiny_v4_config.vocab_size, (1, 1))
        out = model(new_token, past_key_values=out.past_key_values, use_cache=True)
        assert out.past_key_values[0][0].shape[2] == 6


# ── Generate ─────────────────────────────────────────────────────────


class TestGenerate:
    def test_generate_with_cache(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(input_ids, max_new_tokens=5, temperature=1.0, use_cache=True)
        assert out.shape == (1, 9)  # 4 prompt + 5 generated

    def test_generate_without_cache(self, tiny_v4_config: NanoOSRTv4Config) -> None:
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))
        out = model.generate(input_ids, max_new_tokens=5, temperature=1.0, use_cache=False)
        assert out.shape == (1, 9)

    def test_generate_greedy_deterministic(
        self, tiny_v4_config: NanoOSRTv4Config
    ) -> None:
        """Greedy generation (temperature=0) with and without cache should
        produce identical token sequences."""
        model = NanoOSRTv4ForCausalLM(tiny_v4_config)
        model.eval()
        input_ids = torch.randint(0, tiny_v4_config.vocab_size, (1, 4))

        out_cached = model.generate(
            input_ids.clone(), max_new_tokens=8, temperature=0.0, use_cache=True,
        )
        out_nocache = model.generate(
            input_ids.clone(), max_new_tokens=8, temperature=0.0, use_cache=False,
        )
        assert torch.equal(out_cached, out_nocache)
