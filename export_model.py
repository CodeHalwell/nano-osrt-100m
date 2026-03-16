#!/usr/bin/env python3
"""Download checkpoint from Modal volume and export to HF format.

Usage:
    # Download and export in one step:
    python export_model.py

    # Or specify paths:
    python export_model.py --checkpoint /path/to/local/checkpoint.pt --output ./nano-osrt-model
"""

import argparse
import os


def download_from_modal(remote_path: str, local_path: str) -> str:
    """Download a file from the Modal volume."""
    import modal

    vol = modal.Volume.from_name("osrt-checkpoints")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    print(f"Downloading {remote_path} from Modal volume...")
    with open(local_path, "wb") as f:
        for chunk in vol.read_file(remote_path):
            f.write(chunk)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"Downloaded: {local_path} ({size_mb:.1f} MB)")
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Export NanoOSRT to HF format")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Local path to checkpoint. If not provided, downloads from Modal.",
    )
    parser.add_argument(
        "--remote-path", type=str, default="checkpoints/osrt100m_code_final.pt",
        help="Remote path in Modal volume (default: checkpoints/osrt100m_code_final.pt)",
    )
    parser.add_argument(
        "--output", type=str, default="./nano-osrt-model",
        help="Output directory for HF-format model",
    )
    args = parser.parse_args()

    # Get checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = "./checkpoints/osrt100m_code_final.pt"
        if not os.path.exists(ckpt_path):
            download_from_modal(args.remote_path, ckpt_path)

    # Export
    from src.nano_osrt.hf_model import NanoOSRTConfig, NanoOSRTForCausalLM

    print(f"\nLoading checkpoint: {ckpt_path}")
    model = NanoOSRTForCausalLM.from_checkpoint(ckpt_path, device="cpu")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    print(f"\nExporting to: {args.output}")
    model.save_pretrained(args.output)

    # Verify reload
    print("\nVerifying reload...")
    model2 = NanoOSRTForCausalLM.from_pretrained(args.output, device="cpu")
    total2 = sum(p.numel() for p in model2.parameters())
    assert total == total2, f"Parameter count mismatch: {total} vs {total2}"
    print(f"Reload OK: {total2:,} parameters")

    print(f"\nModel exported to {args.output}/")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        size = os.path.getsize(os.path.join(args.output, f))
        print(f"  {f} ({size / 1e6:.1f} MB)")

    print(f"\nUsage:")
    print(f'  python inference.py --model {args.output} --prompt "Write a Python function to sort a list"')
    print(f'  python inference.py --model {args.output} --interactive')


if __name__ == "__main__":
    main()
