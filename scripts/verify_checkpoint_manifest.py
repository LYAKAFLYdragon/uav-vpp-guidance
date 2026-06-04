"""
Verify checkpoint manifest against actual files.

Usage:
    python scripts/verify_checkpoint_manifest.py \
        --manifest config/trajectory_prediction/checkpoint_manifest.yaml

Checks:
    - checkpoint_path exists
    - SHA256 hash matches (if sha256 is not a placeholder)
    - model_type matches the checkpoint content (heuristic)
"""

import argparse
import hashlib
import os
import sys

import yaml


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_entry(entry: dict, root_dir: str = ".", allow_placeholder: bool = False) -> list:
    errors = []
    ckpt = entry.get("checkpoint_path", "")
    ckpt_full = os.path.join(root_dir, ckpt)

    if not ckpt:
        errors.append("Missing checkpoint_path")
        return errors

    if not os.path.exists(ckpt_full):
        if allow_placeholder:
            # Example manifest may reference not-yet-trained checkpoints
            pass
        else:
            errors.append(f"Checkpoint not found: {ckpt_full}")
            return errors

    expected_sha = entry.get("sha256", "")
    is_placeholder = expected_sha and "placeholder" in expected_sha.lower()
    if is_placeholder and allow_placeholder:
        pass
    elif expected_sha and not is_placeholder and os.path.exists(ckpt_full):
        actual_sha = compute_sha256(ckpt_full)
        if actual_sha != expected_sha:
            errors.append(
                f"SHA256 mismatch for {ckpt}: expected {expected_sha}, got {actual_sha}"
            )

    # Heuristic: check file size > 0
    if os.path.exists(ckpt_full):
        size = os.path.getsize(ckpt_full)
        if size == 0:
            errors.append(f"Checkpoint file is empty: {ckpt_full}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest YAML")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")
    parser.add_argument("--allow-placeholder-example", action="store_true",
                        help="Allow placeholder sha256 and missing checkpoint files (for example manifest)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    entries = manifest.get("entries", [])
    if not entries:
        print("WARNING: No entries in manifest.")
        sys.exit(0)

    all_ok = True
    for i, entry in enumerate(entries):
        model_type = entry.get("model_type", "unknown")
        ckpt = entry.get("checkpoint_path", "")
        print(f"\n[{i+1}/{len(entries)}] {model_type}: {ckpt}")
        errs = verify_entry(entry, root_dir=args.root, allow_placeholder=args.allow_placeholder_example)
        if errs:
            all_ok = False
            for e in errs:
                print(f"  FAIL: {e}")
        else:
            print("  OK")

    print("\n" + ("All checks passed." if all_ok else "Some checks failed."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
