#!/usr/bin/env python3
"""Inference script for NanoOSRT 100M.

Usage:
    # From a training checkpoint:
    python inference.py --checkpoint /path/to/osrt100m_code_final.pt --prompt "Write a Python function to sort a list"

    # From a saved HF-format model:
    python inference.py --model ./nano-osrt-model --prompt "What is 2+2?"

    # Interactive mode:
    python inference.py --checkpoint /path/to/checkpoint.pt --interactive

    # Convert checkpoint to HF format (for sharing):
    python inference.py --checkpoint /path/to/checkpoint.pt --export ./nano-osrt-model
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer

from src.nano_osrt.hf_model import NanoOSRTConfig, NanoOSRTForCausalLM


def main():
    parser = argparse.ArgumentParser(description="NanoOSRT 100M Inference")
    parser.add_argument("--checkpoint", type=str, help="Path to training checkpoint (.pt)")
    parser.add_argument("--model", type=str, help="Path to HF-format model directory")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--export", type=str, help="Export checkpoint to HF format at this path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty (1.0=none, 1.2=default)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    if not args.checkpoint and not args.model:
        parser.error("Provide either --checkpoint or --model")

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    # Load model
    if args.model:
        print(f"Loading model from {args.model}...")
        model = NanoOSRTForCausalLM.from_pretrained(args.model, device=device)
    else:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model = NanoOSRTForCausalLM.from_checkpoint(args.checkpoint, device=device)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Export if requested
    if args.export:
        print(f"Exporting to {args.export}...")
        model.save_pretrained(args.export)
        print("Done. You can now load with NanoOSRTForCausalLM.from_pretrained()")
        if not args.prompt and not args.interactive:
            return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    def generate_response(prompt_text: str) -> str:
        # Format as chat
        if not prompt_text.startswith("user:"):
            formatted = f"user: {prompt_text}\nassistant:"
        else:
            formatted = prompt_text
            if not formatted.endswith("assistant:"):
                formatted += "\nassistant:"

        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Single prompt mode
    if args.prompt:
        response = generate_response(args.prompt)
        print(f"\n--- Response ---\n{response}")
        return

    # Interactive mode
    if args.interactive:
        print("\nNanoOSRT 100M — Interactive Mode")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            response = generate_response(user_input)
            print(f"Assistant: {response}\n")

    # Default: show usage
    if not args.prompt and not args.interactive and not args.export:
        parser.print_help()


if __name__ == "__main__":
    main()
