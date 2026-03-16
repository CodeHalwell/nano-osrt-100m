#!/usr/bin/env python3
"""Transfer checkpoints and tokenizer between Modal accounts.

When switching Modal accounts mid-training, use this to:
1. Download from the old account's volume
2. Switch accounts (modal token new)
3. Upload to the new account's volume

Usage:
    # Download everything from current account
    uv run python scripts/transfer_checkpoints.py download

    # Switch Modal account
    modal token new

    # Upload to new account
    uv run python scripts/transfer_checkpoints.py upload

    # Download only tokenizer
    uv run python scripts/transfer_checkpoints.py download --tokenizer-only

    # Download only latest checkpoint
    uv run python scripts/transfer_checkpoints.py download --latest-only
"""

import argparse
import os
import sys


LOCAL_DIR = "./transfer"
CKPT_VOL = "osrt-v4-checkpoints"
TOK_VOL = "osrt-v4-tokenizer"


def download(args):
    """Download from Modal volumes to local disk."""
    import modal

    os.makedirs(f"{LOCAL_DIR}/checkpoints/v4", exist_ok=True)
    os.makedirs(f"{LOCAL_DIR}/tokenizer", exist_ok=True)

    # Tokenizer
    if not args.latest_only:
        print("Downloading tokenizer...")
        try:
            tok_vol = modal.Volume.from_name(TOK_VOL)
            for entry in tok_vol.listdir("/"):
                path = f"{LOCAL_DIR}/tokenizer/{entry.path}"
                print(f"  {entry.path}")
                with open(path, "wb") as f:
                    for chunk in tok_vol.read_file(entry.path):
                        f.write(chunk)
            print(f"  Tokenizer saved to {LOCAL_DIR}/tokenizer/")
        except Exception as e:
            print(f"  Tokenizer volume not found or empty: {e}")

    if args.tokenizer_only:
        return

    # Checkpoints
    print("\nDownloading checkpoints...")
    try:
        ckpt_vol = modal.Volume.from_name(CKPT_VOL)
        entries = list(ckpt_vol.listdir("/"))

        # Check for v4 subdirectory
        v4_entries = []
        try:
            v4_entries = list(ckpt_vol.listdir("/v4"))
        except Exception:
            pass

        all_entries = [(e.path, "") for e in entries] + [(e.path, "v4/") for e in v4_entries]

        if args.latest_only:
            # Find latest numbered checkpoint
            best_step = -1
            best_path = None
            best_prefix = ""
            for path, prefix in all_entries:
                try:
                    if "step_" in path:
                        s = int(path.rsplit("_", 1)[1].split(".")[0])
                        if s > best_step:
                            best_step = s
                            best_path = path
                            best_prefix = prefix
                except (ValueError, IndexError):
                    continue

            # Also grab rescue and final if they exist
            targets = []
            if best_path:
                targets.append((best_path, best_prefix))
            for path, prefix in all_entries:
                if "rescue" in path or "final" in path:
                    targets.append((path, prefix))

            for path, prefix in targets:
                full_remote = f"{prefix}{path}" if prefix else path
                local_path = f"{LOCAL_DIR}/checkpoints/{prefix}{path}"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                size_before = 0
                print(f"  Downloading {full_remote}...")
                with open(local_path, "wb") as f:
                    for chunk in ckpt_vol.read_file(full_remote):
                        f.write(chunk)
                size_mb = os.path.getsize(local_path) / 1e6
                print(f"    {size_mb:.1f} MB")
        else:
            # Download everything
            for path, prefix in all_entries:
                full_remote = f"{prefix}{path}" if prefix else path
                local_path = f"{LOCAL_DIR}/checkpoints/{prefix}{path}"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"  Downloading {full_remote}...")
                with open(local_path, "wb") as f:
                    for chunk in ckpt_vol.read_file(full_remote):
                        f.write(chunk)
                size_mb = os.path.getsize(local_path) / 1e6
                print(f"    {size_mb:.1f} MB")

    except Exception as e:
        print(f"  Checkpoint volume not found or empty: {e}")

    print(f"\nAll files saved to {LOCAL_DIR}/")
    print("Now switch Modal accounts with: modal token new")
    print(f"Then run: uv run python scripts/transfer_checkpoints.py upload")


def upload(args):
    """Upload from local disk to Modal volumes."""
    import modal

    # Tokenizer
    tok_dir = f"{LOCAL_DIR}/tokenizer"
    if os.path.isdir(tok_dir) and os.listdir(tok_dir):
        print("Uploading tokenizer...")
        tok_vol = modal.Volume.from_name(TOK_VOL, create_if_missing=True)
        for fname in os.listdir(tok_dir):
            fpath = os.path.join(tok_dir, fname)
            if os.path.isfile(fpath):
                print(f"  {fname}")
                with open(fpath, "rb") as f:
                    tok_vol.write_file(fname, f)
        tok_vol.commit()
        print("  Tokenizer uploaded.")

    if args.tokenizer_only:
        return

    # Checkpoints
    ckpt_dir = f"{LOCAL_DIR}/checkpoints"
    if os.path.isdir(ckpt_dir):
        print("\nUploading checkpoints...")
        ckpt_vol = modal.Volume.from_name(CKPT_VOL, create_if_missing=True)

        for root, dirs, files in os.walk(ckpt_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                # Preserve relative path structure
                rel_path = os.path.relpath(local_path, ckpt_dir)
                size_mb = os.path.getsize(local_path) / 1e6
                print(f"  {rel_path} ({size_mb:.1f} MB)")
                with open(local_path, "rb") as f:
                    ckpt_vol.write_file(rel_path, f)

        ckpt_vol.commit()
        print("  Checkpoints uploaded.")

    print(f"\nTransfer complete. You can now run:")
    print(f"  uv run modal run --detach app_v4.py --stage pretrain")


def status(args):
    """Show what's in both local transfer dir and Modal volumes."""
    print("=== Local transfer directory ===")
    if os.path.isdir(LOCAL_DIR):
        for root, dirs, files in os.walk(LOCAL_DIR):
            for f in sorted(files):
                path = os.path.join(root, f)
                size = os.path.getsize(path) / 1e6
                rel = os.path.relpath(path, LOCAL_DIR)
                print(f"  {rel} ({size:.1f} MB)")
    else:
        print("  (empty)")

    print("\n=== Modal volumes ===")
    import modal

    for vol_name in [TOK_VOL, CKPT_VOL]:
        print(f"\n  {vol_name}:")
        try:
            vol = modal.Volume.from_name(vol_name)
            for entry in vol.listdir("/"):
                print(f"    {entry.path}")
            try:
                for entry in vol.listdir("/v4"):
                    print(f"    v4/{entry.path}")
            except Exception:
                pass
        except Exception:
            print("    (not found)")


def main():
    parser = argparse.ArgumentParser(description="Transfer checkpoints between Modal accounts")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download from Modal to local")
    dl.add_argument("--tokenizer-only", action="store_true")
    dl.add_argument("--latest-only", action="store_true", help="Only latest checkpoint + rescue + final")

    ul = sub.add_parser("upload", help="Upload from local to Modal")
    ul.add_argument("--tokenizer-only", action="store_true")

    sub.add_parser("status", help="Show local and remote files")

    args = parser.parse_args()

    if args.command == "download":
        download(args)
    elif args.command == "upload":
        upload(args)
    elif args.command == "status":
        status(args)


if __name__ == "__main__":
    main()
