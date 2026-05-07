"""Tests for recursive-block NanoOSRT modules."""

import math

import pytest
import torch

from nano_osrt.modal_config import ModalConfig
from nano_osrt.recursive_model import RecursiveBlock, RecursiveNanoOSRT, SwiGLU
from nano_osrt.rope import apply_rope, compute_rope_freqs

# ---- ModalConfig -----------------------------------------------------------


class TestModalConfig:
    def test_default_values(self) -> None:
        cfg = ModalConfig()
        assert cfg.dim == 1280
        assert cfg.heads == 20
        assert cfg.head_dim == 64
        assert cfg.seq_len == 2048
        assert cfg.num_blocks == 2
        assert cfg.recursive_loops == 6
        assert cfg.optimizer_name == "lion"

    def test_phases_structure(self) -> None:
        cfg = ModalConfig()
        assert "tinystories" in cfg.phases
        assert "fineweb" in cfg.phases
        assert "smoltalk" in cfg.phases
        for phase in cfg.phases.values():
            assert "start" in phase
            assert "end" in phase
            assert "dataset" in phase

    def test_adapter_params(self) -> None:
        cfg = ModalConfig()
        assert cfg.adapter_rank == 16
        assert cfg.adapter_alpha == 16.0


# ---- RoPE ------------------------------------------------------------------


class TestRoPE:
    def test_compute_rope_freqs_shape(self) -> None:
        seq_len, dim = 128, 64
        cos, sin = compute_rope_freqs(seq_len, dim)
        assert cos.shape == (1, seq_len, 1, dim)
        assert sin.shape == (1, seq_len, 1, dim)

    def test_apply_rope_shape(self) -> None:
        B, S, H, D = 2, 16, 4, 32
        x = torch.randn(B, S, H, D)
        cos, sin = compute_rope_freqs(S, D)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_rope_preserves_norm(self) -> None:
        """RoPE is a rotation — it should (approximately) preserve L2 norm."""
        B, S, H, D = 1, 8, 2, 16
        x = torch.randn(B, S, H, D)
        cos, sin = compute_rope_freqs(S, D)
        out = apply_rope(x, cos, sin)
        torch.testing.assert_close(
            x.norm(dim=-1), out.norm(dim=-1), atol=1e-5, rtol=1e-5
        )


# ---- SwiGLU ----------------------------------------------------------------


class TestSwiGLU:
    def test_output_shape(self) -> None:
        dim = 64
        ffn = SwiGLU(dim)
        x = torch.randn(2, 8, dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_hidden_dim_tc_alignment(self) -> None:
        """Hidden dim should be aligned to multiples of 64."""
        ffn = SwiGLU(64)
        hidden = ffn.w_gate.out_features
        assert hidden % 64 == 0


# ---- RecursiveBlock ---------------------------------------------------------


class TestRecursiveBlock:
    def test_output_shape(self) -> None:
        dim, heads = 64, 4
        block = RecursiveBlock(dim, heads)
        B, S = 2, 16
        x = torch.randn(B, S, dim)
        adapter_a = torch.randn(dim, 8) * 0.01
        adapter_b = torch.zeros(8, dim)
        cos, sin = compute_rope_freqs(S, dim // heads)
        out = block(x, adapter_a, adapter_b, 1.0, cos, sin)
        assert out.shape == (B, S, dim)


# ---- RecursiveNanoOSRT ------------------------------------------------------


@pytest.fixture()
def tiny_modal_config() -> ModalConfig:
    """A tiny ModalConfig suitable for CPU tests."""
    cfg = ModalConfig()
    cfg.dim = 64
    cfg.heads = 4
    cfg.head_dim = 16
    cfg.seq_len = 32
    cfg.vocab_size = 256
    cfg.real_vocab_size = 256
    cfg.num_blocks = 2
    cfg.recursive_loops = 2
    cfg.adapter_rank = 4
    cfg.adapter_alpha = 4.0
    return cfg


class TestRecursiveNanoOSRT:
    def test_instantiation(self, tiny_modal_config: ModalConfig) -> None:
        model = RecursiveNanoOSRT(tiny_modal_config)
        assert isinstance(model, RecursiveNanoOSRT)

    def test_forward_shape(self, tiny_modal_config: ModalConfig) -> None:
        model = RecursiveNanoOSRT(tiny_modal_config)
        model.eval()
        input_ids = torch.randint(0, tiny_modal_config.vocab_size, (2, 8))
        logits, loop_rms = model(input_ids)
        assert logits.shape == (2, 8, tiny_modal_config.vocab_size)
        assert len(loop_rms) == tiny_modal_config.recursive_loops

    def test_loop_rms_values(self, tiny_modal_config: ModalConfig) -> None:
        model = RecursiveNanoOSRT(tiny_modal_config)
        model.eval()
        input_ids = torch.randint(0, tiny_modal_config.vocab_size, (1, 4))
        _, loop_rms = model(input_ids)
        for rms in loop_rms:
            assert rms.ndim == 0  # scalar
            assert rms.item() > 0

    def test_adapter_count(self, tiny_modal_config: ModalConfig) -> None:
        model = RecursiveNanoOSRT(tiny_modal_config)
        expected = (
            tiny_modal_config.num_blocks * tiny_modal_config.recursive_loops
        )
        assert len(model.adapters_a) == expected
        assert len(model.adapters_b) == expected

    def test_num_parameters(self, tiny_modal_config: ModalConfig) -> None:
        model = RecursiveNanoOSRT(tiny_modal_config)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0


# ---- modal_train helpers ----------------------------------------------------


class TestTrainHelpers:
    def test_get_lr_warmup(self) -> None:
        from nano_osrt.modal_train import get_lr

        cfg = ModalConfig()
        # During warmup, LR should increase linearly
        lr_0 = get_lr(0, cfg)
        lr_mid = get_lr(cfg.warmup_steps // 2, cfg)
        lr_end = get_lr(cfg.warmup_steps, cfg)
        assert lr_0 == 0.0
        assert 0 < lr_mid < cfg.peak_lr
        assert math.isclose(lr_end, cfg.peak_lr, rel_tol=1e-6)

    def test_get_lr_cosine_decay(self) -> None:
        from nano_osrt.modal_train import get_lr

        cfg = ModalConfig()
        lr_after_warmup = get_lr(cfg.warmup_steps + 1, cfg)
        lr_near_end = get_lr(cfg.total_steps - 1, cfg)
        assert lr_after_warmup <= cfg.peak_lr
        assert lr_near_end >= cfg.min_lr

    def test_get_phase_tinystories(self) -> None:
        from nano_osrt.modal_train import get_phase

        cfg = ModalConfig()
        name, dataset, dataset_config = get_phase(0, cfg)
        assert name == "tinystories"
        assert dataset == "roneneldan/TinyStories"
        assert dataset_config is None

    def test_get_phase_fineweb(self) -> None:
        from nano_osrt.modal_train import get_phase

        cfg = ModalConfig()
        name, dataset, dataset_config = get_phase(10_000, cfg)
        assert name == "fineweb"
        assert dataset == "HuggingFaceFW/fineweb-edu"
        assert dataset_config is None

    def test_get_phase_smoltalk(self) -> None:
        from nano_osrt.modal_train import get_phase

        cfg = ModalConfig()
        name, dataset, dataset_config = get_phase(145_000, cfg)
        assert name == "smoltalk"
        assert dataset == "HuggingFaceTB/smoltalk"
        assert dataset_config == "all"

    def test_get_phase_fallback(self) -> None:
        from nano_osrt.modal_train import get_phase

        cfg = ModalConfig()
        name, dataset, dataset_config = get_phase(999_999, cfg)
        assert name == "fineweb"
        assert dataset == "HuggingFaceFW/fineweb-edu"
        assert dataset_config is None
