#!/usr/bin/env python3
"""
Run the 10-training-seed JSBSim fair-comparison matrix.

Methods:
  - vpp    : hierarchical VPP + LOS-rate guidance
  - no_vpp : hierarchical with zero-offset VPP
  - e2e    : end-to-end PPO direct control

Outputs:
  outputs/jsbsim_10seed_matrix/<method>_s<seed>/
  outputs/jsbsim_10seed_matrix/summary.json

Usage:
    # Full 10-seed matrix (default)
    python scripts/run_jsbsim_10seed_matrix.py --seeds 10 --device cuda

    # Quick smoke test
    python scripts/run_jsbsim_10seed_matrix.py --seeds 1 --smoke

    # Resume / skip existing checkpoints
    python scripts/run_jsbsim_10seed_matrix.py --seeds 10 --skip-existing
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

METHODS = {
    "vpp": {
        "module": "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_prediction_vpp_ppo_jsbsim.yaml",
        "exp_name": "no_prediction_vpp_ppo_jsbsim",
    },
    "no_vpp": {
        "module": "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_vpp_ppo_jsbsim.yaml",
        "exp_name": "no_vpp_ppo_jsbsim",
    },
    "e2e": {
        "module": "uav_vpp_guidance.training.train_end_to_end_ppo",
        "config": "config/experiment/train_end_to_end_ppo_jsbsim.yaml",
        "exp_name": "end_to_end_ppo_jsbsim",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VPP/No-VPP/E2E on JSBSim across multiple seeds."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of independent training seeds (default: 10).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/jsbsim_10seed_matrix",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode (minimal training) for all methods/seeds.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose output directories already contain a checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override compute device (default: from config).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()),
        default=list(METHODS.keys()),
        help="Methods to train (default: all).",
    )
    return parser.parse_args()


def has_checkpoint(output_dir: Path) -> bool:
    """Check whether a checkpoint already exists in output_dir."""
    ckpt = output_dir / "checkpoints" / "best.pt"
    return ckpt.exists()


def run_single(method: str, seed: int, output_dir: Path, smoke: bool, device: str = None):
    meta = METHODS[method]
    cmd = [
        sys.executable,
        "-m",
        meta["module"],
        "--config",
        str(ROOT / meta["config"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if smoke:
        cmd.append("--smoke")
    if device:
        cmd.extend(["--device", device])

    print(f"\n[{method}] seed {seed}: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    elapsed = time.time() - start
    success = result.returncode == 0
    print(f"[{method}] seed {seed}: {'OK' if success else 'FAILED'} in {elapsed:.1f}s")
    return success, elapsed


def read_eval_metrics(output_dir: Path) -> dict:
    """Read the last row of eval_log.csv as a dict."""
    path = output_dir / "logs" / "eval_log.csv"
    if not path.exists():
        return {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}


def aggregate(method: str, seed_dirs: list, seeds: list) -> dict:
    results = []
    for seed, seed_dir in zip(seeds, seed_dirs):
        metrics = read_eval_metrics(seed_dir)
        if metrics:
            results.append({"seed": seed, **metrics})

    if not results:
        return {"num_seeds": 0}

    numeric_keys = [
        "mean_return",
        "std_return",
        "success_rate",
        "crash_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "mean_final_range_m",
        "mean_final_ata_deg",
    ]

    summary = {"num_seeds": len(results)}
    for key in numeric_keys:
        vals = []
        for r in results:
            v = r.get(key)
            if v is not None and v != "":
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        vals.append(fv)
                except (ValueError, TypeError):
                    pass
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
            summary[f"{key}_min"] = float(np.min(vals))
            summary[f"{key}_max"] = float(np.max(vals))
    return summary


def main():
    args = parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))
    manifest = {
        "seeds": seeds,
        "smoke": args.smoke,
        "device": args.device,
        "methods": args.methods,
        "output_root": str(output_root),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_summaries = {}
    for method in args.methods:
        meta = METHODS[method]
        print(f"\n{'='*60}")
        print(f"Method: {method} ({meta['exp_name']})")
        print(f"{'='*60}")

        seed_dirs = []
        for seed in seeds:
            seed_dir = output_root / f"{method}_s{seed}"
            seed_dirs.append(seed_dir)

            if args.skip_existing and has_checkpoint(seed_dir):
                print(f"[{method}] seed {seed}: checkpoint exists, skipping.")
                continue

            success, elapsed = run_single(
                method, seed, seed_dir, args.smoke, args.device
            )
            if not success:
                print(f"[{method}] seed {seed}: training failed, continuing...")

        all_summaries[method] = aggregate(method, seed_dirs, seeds)

    manifest["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["summaries"] = all_summaries

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    # Print a concise table
    print("\n" + "=" * 80)
    print("Aggregate results (eval_log.csv metrics):")
    print("=" * 80)
    for method, summary in all_summaries.items():
        n = summary.get("num_seeds", 0)
        sr_mean = summary.get("success_rate_mean", float("nan"))
        sr_std = summary.get("success_rate_std", float("nan"))
        print(f"  {method:8s}: n={n:2d}  success_rate={sr_mean:.3f} ± {sr_std:.3f}")


if __name__ == "__main__":
    main()
