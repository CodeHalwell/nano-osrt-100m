"""Smoke tests for nano-osrt-100m model and config."""

import pytest
import torch

from nano_osrt.config import ModelConfig, TrainConfig
from nano_osrt.model import NanoOSRT


@pytest.fixture()
def tiny_config() -> ModelConfig:
    """A tiny model config suitable for CPU tests."""
    return ModelConfig(
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        ffn_hidden_mult=4,
        dropout=0.0,
        bias=True,
        block_size=32,
    )


def test_model_config_head_dim(tiny_config: ModelConfig) -> None:
    assert tiny_config.head_dim == 32  # 64 / 2


def test_model_instantiation(tiny_config: ModelConfig) -> None:
    model = NanoOSRT(tiny_config)
    assert isinstance(model, NanoOSRT)


def test_model_num_parameters(tiny_config: ModelConfig) -> None:
    model = NanoOSRT(tiny_config)
    n = model.num_parameters()
    assert n > 0


def test_forward_no_targets(tiny_config: ModelConfig) -> None:
    model = NanoOSRT(tiny_config)
    model.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert loss is None
    assert logits.shape == (2, 1, tiny_config.vocab_size)


def test_forward_with_targets(tiny_config: ModelConfig) -> None:
    model = NanoOSRT(tiny_config)
    model.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    assert logits.shape == (2, 8, tiny_config.vocab_size)


def test_generate(tiny_config: ModelConfig) -> None:
    model = NanoOSRT(tiny_config)
    model.eval()
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 9)


def test_weight_tying(tiny_config: ModelConfig) -> None:
    """Token embedding and lm_head should share weights."""
    model = NanoOSRT(tiny_config)
    assert model.transformer["wte"].weight is model.lm_head.weight


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.learning_rate > 0
    assert cfg.min_lr < cfg.learning_rate
    assert isinstance(cfg.model, ModelConfig)


def test_block_size_exceeded_raises(tiny_config: ModelConfig) -> None:
    model = NanoOSRT(tiny_config)
    idx = torch.zeros((1, tiny_config.block_size + 1), dtype=torch.long)
    with pytest.raises(AssertionError):
        model(idx)
