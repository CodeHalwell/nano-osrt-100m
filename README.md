# nano-osrt-100m

A nano-scale (~100M parameter) Open-Set Reasoning Transformer.

## Project structure

```
src/nano_osrt/
├── __init__.py            # Package exports
├── config.py              # ModelConfig & TrainConfig dataclasses
├── model.py               # GPT-2 style NanoOSRT model
├── data.py                # Memory-mapped token datasets & batch helpers
├── train.py               # Local training loop
├── modal_config.py        # ModalConfig for recursive-block deployment
├── rope.py                # Rotary Position Embedding utilities
├── recursive_model.py     # RecursiveNanoOSRT (weight-shared blocks + adapters)
├── modal_data.py          # Streaming HuggingFace data pipeline
└── modal_train.py         # Modal training loop & helpers

scripts/
└── train.py               # CLI entry-point for local training

app.py                     # Modal cloud deployment entry-point

tests/
├── test_model.py          # Tests for GPT-2 style model
└── test_recursive_model.py # Tests for recursive-block model & utilities
```

## Quick start

```bash
# Install dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Lint
uv run ruff check src/ scripts/ tests/

# Local training (requires pre-tokenised data)
uv run train

# Modal cloud training
modal run app.py
```
