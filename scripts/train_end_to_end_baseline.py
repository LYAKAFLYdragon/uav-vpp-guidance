#!/usr/bin/env python3
"""Train end-to-end DRL baseline.

Wrapper around uav_vpp_guidance.training.train_end_to_end_ppo that:
  1. Resolves output paths from checkpoint registry
  2. Supports --dry-run to validate config without training
  3. Writes run_manifest.json with provenance

Usage:
    # Full training (default)
    python scripts/train_end_to_end_baseline.py

    # Dry-run (validate config, no training)
    python scripts/train_end_to_end_baseline.py --dry-run

    # Custom output directory
    python scripts/train_end_to_end_baseline.py --output-dir custom/path
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve output path from registry
def _get_registry_path() -> str:
    registry_path = Path("config/checkpoint_registry.yaml")
    if not registry_path.exists():
        return "outputs/experiments/end_to_end_ppo"
    try:
        import yaml
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        entry = registry.get("training", {}).get("end_to_end", {})
        out = entry.get("output_dir", "outputs/experiments/end_to_end_ppo")
        if "{seed}" in out:
            out = out.format(seed=0)
        return out
    except Exception:
        return "outputs/experiments/end_to_end_ppo"


def _get_git_info() -> dict:
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
        info["dirty"] = (
            len(subprocess.check_output(["git", "status", "--short"], text=True).strip()) > 0
        )
        info["branch"] = (
            subprocess.check_output(["git", "branch", "--show-current"], text=True)
            .strip()
        )
    except Exception:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description="Train end-to-end DRL baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/train_end_to_end_ppo.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device override",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without training",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (fast, minimal timesteps)",
    )
    args = parser.parse_args()

    # Resolve default output dir from registry if not overridden
    output_dir = args.output_dir or _get_registry_path()

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Config: {args.config}")
        print(f"Output dir: {output_dir}")
        print(f"Mode: END-TO-END baseline")
        # Validate config exists
        if not Path(args.config).exists():
            print(f"ERROR: Config not found: {args.config}")
            sys.exit(1)
        print("Config file exists: OK")
        # Try import
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from uav_vpp_guidance.training.train_end_to_end_ppo import load_experiment_config
            config = load_experiment_config(args.config)
            obs_dim = config.get("policy", {}).get("action_dim", 3)
            print(f"Config loaded: action_dim={obs_dim}")
            print("DRY RUN PASSED")
        except Exception as e:
            print(f"ERROR loading config: {e}")
            sys.exit(1)
        return

    # Build CLI for underlying training script
    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.training.train_end_to_end_ppo",
        "--config", args.config,
        "--output-dir", output_dir,
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.smoke:
        cmd.append("--smoke")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        # Write run_manifest.json
        manifest = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "command_line": sys.argv,
            "config_path": args.config,
            "output_dir": output_dir,
            "git_info": _get_git_info(),
            "method": "end_to_end_ppo",
            "status": "completed",
        }
        manifest_path = Path(output_dir) / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Manifest saved: {manifest_path}")
    else:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
