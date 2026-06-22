#!/usr/bin/env python3
"""
10-seed training robustness review for the flight-control comparison controllers.

Controllers:
  - ppo     : no-prediction VPP PPO (3D action)
  - ppo_pid : PPO+PID hybrid (4D action: Δx, Δy, Δz, aggressiveness)

Outputs:
  outputs/ppo_ppo_pid_10seed_robustness/<controller>_s<seed>/
  outputs/ppo_ppo_pid_10seed_robustness/summary.json

Usage:
    # Full 10-seed robustness review (CPU, serial)
    python scripts/run_ppo_ppo_pid_10seed_robustness.py --seeds 10

    # Quick smoke test (1 seed each, minimal training)
    python scripts/run_ppo_ppo_pid_10seed_robustness.py --seeds 1 --smoke

    # Resume / skip existing checkpoints
    python scripts/run_ppo_ppo_pid_10seed_robustness.py --seeds 10 --skip-existing

    # CUDA device override
    python scripts/run_ppo_ppo_pid_10seed_robustness.py --seeds 10 --device cuda
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

CONTROLLERS = {
    "ppo": {
        "module": "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_prediction_vpp_ppo_jsbsim_compare.yaml",
        "exp_name": "no_prediction_vpp_ppo_jsbsim_compare",
    },
    "ppo_pid": {
        "module": "uav_vpp_guidance.training.train_ppo_pid_jsbsim",
        "config": "config/experiment/train_ppo_pid_jsbsim.yaml",
        "exp_name": "ppo_pid_jsbsim",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO and PPO+PID on JSBSim across multiple seeds for robustness review."
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
        default="outputs/ppo_ppo_pid_10seed_robustness",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode (minimal training) for all controllers/seeds.",
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
        help="Override compute device (default: from config, currently cpu).",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=list(CONTROLLERS.keys()),
        default=list(CONTROLLERS.keys()),
        help="Controllers to train (default: all).",
    )
    return parser.parse_args()


def has_checkpoint(output_dir: Path) -> bool:
    """Check whether a checkpoint already exists in output_dir."""
    return (output_dir / "checkpoints" / "best.pt").exists() or \
           (output_dir / "checkpoints" / "last.pt").exists()


def run_single(controller: str, seed: int, output_dir: Path, smoke: bool, device: str = None):
    meta = CONTROLLERS[controller]
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

    print(f"\n[{controller}] seed {seed}: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    elapsed = time.time() - start
    success = result.returncode == 0
    print(f"[{controller}] seed {seed}: {'OK' if success else 'FAILED'} in {elapsed:.1f}s")
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


def aggregate(controller: str, seed_dirs: list, seeds: list) -> dict:
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
        "controllers": args.controllers,
        "output_root": str(output_root),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_summaries = {}
    for controller in args.controllers:
        meta = CONTROLLERS[controller]
        print(f"\n{'='*60}")
        print(f"Controller: {controller} ({meta['exp_name']})")
        print(f"{'='*60}")

        seed_dirs = []
        for seed in seeds:
            seed_dir = output_root / f"{controller}_s{seed}"
            seed_dirs.append(seed_dir)

            if args.skip_existing and has_checkpoint(seed_dir):
                print(f"[{controller}] seed {seed}: checkpoint exists, skipping.")
                continue

            success, elapsed = run_single(
                controller, seed, seed_dir, args.smoke, args.device
            )
            if not success:
                print(f"[{controller}] seed {seed}: training failed, continuing...")

        all_summaries[controller] = aggregate(controller, seed_dirs, seeds)

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
    for controller, summary in all_summaries.items():
        n = summary.get("num_seeds", 0)
        sr_mean = summary.get("success_rate_mean", float("nan"))
        sr_std = summary.get("success_rate_std", float("nan"))
        ret_mean = summary.get("mean_return_mean", float("nan"))
        ret_std = summary.get("mean_return_std", float("nan"))
        print(
            f"  {controller:10s}: n={n:2d}  "
            f"success_rate={sr_mean:.3f} ± {sr_std:.3f}  "
            f"mean_return={ret_mean:.1f} ± {ret_std:.1f}"
        )

    # M-2: Auto-copy a representative checkpoint to the default evaluation path.
    # This bridges the 10-seed robustness output tree with the single-seed
    # evaluation runner expected by run_flight_control_comparison.py.
    print("\n" + "=" * 80)
    print("Auto-copying representative checkpoint to default evaluation path")
    print("=" * 80)
    for controller, meta in CONTROLLERS.items():
        if controller not in args.controllers:
            continue
        # Try best.pt first, then last.pt
        for ckpt_name in ("best.pt", "last.pt"):
            for seed in seeds:
                src = output_root / f"{controller}_s{seed}" / "checkpoints" / ckpt_name
                if src.exists():
                    dst = ROOT / "outputs" / "experiments" / meta["exp_name"] / "checkpoints" / "last.pt"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(str(src), str(dst))
                    print(f"  Copied {src} -> {dst}")
                    break
            else:
                continue
            break
        else:
            print(f"  WARNING: No checkpoint found for {controller}")

    # Also emit a CLI snippet for the comparison runner.
    print("\n" + "=" * 80)
    print("Next step: run the flight-control comparison")
    print("=" * 80)
    for controller, meta in CONTROLLERS.items():
        if controller not in args.controllers:
            continue
        default_ckpt = ROOT / "outputs" / "experiments" / meta["exp_name"] / "checkpoints" / "last.pt"
        print(f"  --checkpoint-{controller} {default_ckpt}")


if __name__ == "__main__":
    main()
