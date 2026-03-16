#!/usr/bin/env python3
"""Train a custom 64K SuperBPE tokenizer for NanoOSRT v4.

Uses the SuperBPE two-stage process:
  Stage 1: Standard BPE with whitespace pretokenization (learns subwords)
  Stage 2: Continue training without pretokenization (learns superwords)

SuperBPE produces 33% fewer tokens and +4% on benchmarks vs standard BPE (COLM 2025).

Training data: sampled from the pre-training mix (FineWeb-Edu + StarCoder + Wikipedia)
to ensure the tokenizer is optimized for our exact distribution.

Usage:
    # Full training (~2-4 hours on CPU, ~30 min on GPU)
    uv run python scripts/train_tokenizer.py

    # Quick test (1M chars)
    uv run python scripts/train_tokenizer.py --sample-size 1000000

    # Custom output path
    uv run python scripts/train_tokenizer.py --output ./my-tokenizer
"""

import argparse
import os
import tempfile
import time

# Try SuperBPE first, fall back to HuggingFace tokenizers
USE_SUPERBPE = False
try:
    import superbpe
    USE_SUPERBPE = True
except ImportError:
    pass


def sample_training_data(sample_size: int = 50_000_000, seed: int = 42) -> str:
    """Sample training data from pre-training mix and save to temp file.

    Samples proportionally:
      - 55% FineWeb-Edu (general text)
      - 30% StarCoder Python (code)
      - 15% Wikipedia (factual/STEM)

    Args:
        sample_size: Target number of characters to sample.
        seed: Random seed.

    Returns:
        Path to temporary file with training text.
    """
    from datasets import load_dataset
    import random

    rng = random.Random(seed)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")

    sources = [
        {
            "name": "FineWeb-Edu",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "fraction": 0.55,
            "text_key": "text",
        },
        {
            "name": "StarCoder Python",
            "hf_id": "bigcode/starcoderdata",
            "hf_config": "python",
            "fraction": 0.30,
            "text_key": "content",
        },
        {
            "name": "Wikipedia",
            "hf_id": "wikimedia/wikipedia",
            "hf_config": "20231101.en",
            "fraction": 0.15,
            "text_key": "text",
        },
    ]

    total_chars = 0
    for src in sources:
        target_chars = int(sample_size * src["fraction"])
        chars = 0

        print(f"  Sampling {src['name']} ({target_chars:,} chars target)...")
        load_kwargs = {"split": "train", "streaming": True}
        if "hf_config" in src:
            load_kwargs["name"] = src["hf_config"]

        ds = load_dataset(src["hf_id"], **load_kwargs)
        ds = ds.shuffle(buffer_size=10_000, seed=seed)

        for example in ds:
            text = example.get(src["text_key"], "")
            if not text or len(text) < 100:
                continue
            tmp.write(text + "\n")
            chars += len(text)
            if chars >= target_chars:
                break

        total_chars += chars
        print(f"    Collected {chars:,} chars")

    tmp.close()
    print(f"  Total: {total_chars:,} chars -> {tmp.name}")
    return tmp.name


def train_with_hf_tokenizers(data_path: str, vocab_size: int, output_dir: str) -> None:
    """Train a BPE tokenizer using HuggingFace tokenizers library.

    Falls back to this when SuperBPE is not installed.
    Still produces a high-quality BPE tokenizer, just without superword merges.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    print("\nTraining BPE tokenizer with HuggingFace tokenizers...")
    print(f"  Vocab size: {vocab_size:,}")

    # BPE model with byte fallback
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization: byte-level (handles any Unicode)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Special tokens
    special_tokens = [
        "<|padding|>",        # 0: pad
        "<|begin_of_text|>",  # 1: bos
        "<|end_of_text|>",    # 2: eos
        "<|unknown|>",        # 3: unk
        "<|fim_prefix|>",     # 4: fill-in-middle (code)
        "<|fim_middle|>",     # 5: fill-in-middle
        "<|fim_suffix|>",     # 6: fill-in-middle
        "<|think|>",          # 7: reasoning open
        "<|/think|>",         # 8: reasoning close
        "<|answer|>",         # 9: answer open
        "<|/answer|>",        # 10: answer close
        "<|user|>",           # 11: user turn
        "<|assistant|>",      # 12: assistant turn
        "<|system|>",         # 13: system prompt
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print("  Training (this may take a while)...")
    t0 = time.time()
    tokenizer.train([data_path], trainer)
    print(f"  Training done in {time.time() - t0:.0f}s")

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A <|end_of_text|>",
        pair="<|begin_of_text|> $A <|end_of_text|> <|begin_of_text|> $B:1 <|end_of_text|>:1",
        special_tokens=[
            ("<|begin_of_text|>", 1),
            ("<|end_of_text|>", 2),
        ],
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    # Create HF tokenizer config files
    _create_hf_tokenizer_config(output_dir, special_tokens)

    print(f"  Saved to {output_dir}/")

    # Verify
    _verify_tokenizer(output_dir)


def train_with_superbpe(data_path: str, vocab_size: int, output_dir: str) -> None:
    """Train a SuperBPE tokenizer (two-stage: subwords then superwords).

    SuperBPE produces 33% fewer tokens vs standard BPE.
    """
    print("\nTraining SuperBPE tokenizer (two-stage)...")
    print(f"  Vocab size: {vocab_size:,}")

    # Stage 1: Standard BPE with pretokenization (subwords)
    subword_vocab = int(vocab_size * 0.75)  # 75% subwords, 25% superwords
    print(f"  Stage 1: BPE subwords (vocab={subword_vocab:,})...")
    t0 = time.time()

    os.makedirs(output_dir, exist_ok=True)
    import subprocess
    subprocess.run(
        ["python", "-m", "superbpe.train",
         "--input", data_path,
         "--vocab_size", str(subword_vocab),
         "--output", f"{output_dir}/stage1",
         "--pretokenize"],
        check=True,
    )
    print(f"  Stage 1 done in {time.time() - t0:.0f}s")

    # Stage 2: Continue without pretokenization (superwords)
    print(f"  Stage 2: SuperBPE extension to {vocab_size:,}...")
    t0 = time.time()
    subprocess.run(
        ["python", "-m", "superbpe.extend",
         "--input", data_path,
         "--base_tokenizer", f"{output_dir}/stage1",
         "--vocab_size", str(vocab_size),
         "--output", f"{output_dir}/final"],
        check=True,
    )
    print(f"  Stage 2 done in {time.time() - t0:.0f}s")

    # Convert to HF format
    subprocess.run(
        ["python", "-m", "superbpe.construct_hf_tokenizer",
         "--tokenizer_path", f"{output_dir}/final",
         "--output_path", output_dir],
        check=True,
    )

    _verify_tokenizer(output_dir)


def _create_hf_tokenizer_config(output_dir: str, special_tokens: list[str]) -> None:
    """Create HuggingFace tokenizer config files."""
    import json

    # tokenizer_config.json
    config = {
        "model_type": "nano-osrt-v4",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|padding|>",
        "unk_token": "<|unknown|>",
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # special_tokens_map.json
    special_map = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|padding|>",
        "unk_token": "<|unknown|>",
        "additional_special_tokens": special_tokens[4:],  # FIM + think tokens
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_map, f, indent=2)


def _verify_tokenizer(output_dir: str) -> None:
    """Verify the tokenizer works correctly."""
    from transformers import AutoTokenizer

    print("\n  Verifying tokenizer...")
    tok = AutoTokenizer.from_pretrained(output_dir)

    test_cases = [
        "Hello, world!",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "The derivative of x^2 is 2x.",
        "<|user|>What is 2+2?<|assistant|><|think|>2+2 equals 4<|/think|><|answer|>4<|/answer|>",
        "import torch\nimport torch.nn as nn\n\nclass MyModel(nn.Module):\n    pass",
    ]

    print(f"  Vocab size: {tok.vocab_size:,}")
    print(f"  BOS: {tok.bos_token} (id={tok.bos_token_id})")
    print(f"  EOS: {tok.eos_token} (id={tok.eos_token_id})")
    print(f"  PAD: {tok.pad_token} (id={tok.pad_token_id})")

    for text in test_cases:
        tokens = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(tokens)
        ratio = len(text) / len(tokens) if tokens else 0
        print(f"  [{len(tokens):>4d} tokens, {ratio:.1f} chars/tok] {text[:60]}...")

    # Roundtrip test
    for text in test_cases:
        tokens = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(tokens)
        assert decoded == text, f"Roundtrip failed: {repr(text)} != {repr(decoded)}"
    print("  Roundtrip tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Train custom 64K tokenizer for NanoOSRT v4")
    parser.add_argument("--vocab-size", type=int, default=65536, help="Target vocabulary size (64K default, TC-aligned)")
    parser.add_argument("--sample-size", type=int, default=50_000_000, help="Training text size in chars (~50MB default)")
    parser.add_argument("--output", type=str, default="./tokenizer-v4", help="Output directory")
    parser.add_argument("--data-path", type=str, default=None, help="Pre-existing training text file (skip download)")
    args = parser.parse_args()

    print("NanoOSRT v4 — Custom Tokenizer Training")
    print("=" * 50)

    # Get training data
    if args.data_path and os.path.exists(args.data_path):
        data_path = args.data_path
        print(f"Using existing data: {data_path}")
    else:
        print(f"Sampling {args.sample_size:,} chars from pre-training mix...")
        data_path = sample_training_data(args.sample_size)

    # Train tokenizer
    if USE_SUPERBPE:
        print("\nUsing SuperBPE (COLM 2025 — 33% fewer tokens)")
        train_with_superbpe(data_path, args.vocab_size, args.output)
    else:
        print("\nSuperBPE not installed, using HuggingFace BPE")
        print("  (install superbpe for better tokenization: pip install superbpe)")
        train_with_hf_tokenizers(data_path, args.vocab_size, args.output)

    print(f"\nTokenizer saved to: {args.output}/")
    print(f"Use with: AutoTokenizer.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
