#!/usr/bin/env python3
"""Compute SHA256 preservation baseline for control-method-rationality-audit.

This script hashes every source file in the preserved set and writes the
manifest to .kiro/specs/control-method-rationality-audit/
preservation_baseline.sha256.json. It does NOT modify any source or config
file.

Preserved set:
  - src/uav_vpp_guidance/guidance/**
  - src/uav_vpp_guidance/flight_control/**
  - src/uav_vpp_guidance/virtual_point/**
  - src/uav_vpp_guidance/gain_optimizer/**
  - src/uav_vpp_guidance/envs/reward.py
  - src/uav_vpp_guidance/envs/observation.py
  - src/uav_vpp_guidance/envs/tracking_env.py
  - config/**
"""

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def collect_files(base: Path) -> list[Path]:
    """Recursively collect all files under base, excluding __pycache__."""
    results = []
    for root, dirs, files in os.walk(base):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in sorted(files):
            results.append(Path(root) / fname)
    return results


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Directories to hash recursively
    directories = [
        repo_root / "src" / "uav_vpp_guidance" / "guidance",
        repo_root / "src" / "uav_vpp_guidance" / "flight_control",
        repo_root / "src" / "uav_vpp_guidance" / "virtual_point",
        repo_root / "src" / "uav_vpp_guidance" / "gain_optimizer",
        repo_root / "config",
    ]

    # Individual files to hash
    individual_files = [
        repo_root / "src" / "uav_vpp_guidance" / "envs" / "reward.py",
        repo_root / "src" / "uav_vpp_guidance" / "envs" / "observation.py",
        repo_root / "src" / "uav_vpp_guidance" / "envs" / "tracking_env.py",
    ]

    manifest: dict[str, str] = {}

    # Hash directory contents
    for directory in directories:
        if not directory.exists():
            raise FileNotFoundError(f"Expected directory not found: {directory}")
        for fpath in collect_files(directory):
            rel = fpath.relative_to(repo_root).as_posix()
            manifest[rel] = sha256_file(fpath)

    # Hash individual files
    for fpath in individual_files:
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file not found: {fpath}")
        rel = fpath.relative_to(repo_root).as_posix()
        manifest[rel] = sha256_file(fpath)

    # Sort keys for deterministic output
    manifest = dict(sorted(manifest.items()))

    # Write manifest only within the audit spec allowlist.
    output_path = (
        repo_root
        / ".kiro"
        / "specs"
        / "control-method-rationality-audit"
        / "preservation_baseline.sha256.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Preservation baseline written to {output_path}")
    print(f"Total files hashed: {len(manifest)}")


if __name__ == "__main__":
    main()
