#!/usr/bin/env python3
"""
Multi-seed launcher for the No-VPP baseline.

Runs train_no_prediction_vpp_ppo with virtual_point.enabled=false,
so the VPP offset is forced to zero and the policy tracks the target's
current position directly through the LOS-rate guidance law.

This serves as an ablation baseline to quantify the tactical value
added by the VPP offset layer.

Usage:
    # Standard 3-seed run (200K steps each)
    python scripts/run_no_vpp_baseline.py \
        --config config/experiment/train_no_vpp_ppo.yaml \
        --seeds 3

    # Quick smoke test across seeds
    python scripts/run_no_vpp_baseline.py \
        --config config/experiment/train_no_vpp_ppo.yaml \
        --seeds 3 --smoke

    # Resume / skip existing
    python scripts/run_no_vpp_baseline.py ... --skip-existing
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed No-VPP baseline training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML (e.g. train_no_vpp_ppo.yaml).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of independent training seeds (default: 3).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/experiments",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name override (default: derived from config).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode (minimal steps) for all seeds.",
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
        help="Override PPO device (default: from config).",
    )
    return parser.parse_args()


def _load_yaml_name(config_path: str) -> str:
    """Extract a short experiment name from the config file stem."""
    name = Path(config_path).stem
    if name.startswith("train_"):
        name = name[len("train_"):]
    return name


def run_single_seed(
    config_path: str, seed: int, output_dir: str, smoke: bool, device: str = None
):
    """Launch one training run for a given seed."""
    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
    ]
    if smoke:
        cmd.append("--smoke")
    if device:
        cmd.extend(["--device", device])

    print(f"\n[SEED {seed}] {'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start
    success = result.returncode == 0
    print(f"[SEED {seed}] {'OK' if success else 'FAILED'} in {elapsed:.1f}s")
    return success, elapsed


def _read_eval_metrics(eval_log_path: str):
    """Read the last row of eval_log.csv as a dict."""
    if not os.path.exists(eval_log_path):
        return {}
    try:
        with open(eval_log_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        return rows[-1]
    except Exception:
        return {}


def _read_smoke_summary(log_dir: str):
    """Read smoke_summary.json if present."""
    path = os.path.join(log_dir, "smoke_summary.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def aggregate_results(seed_dirs: list, seeds: list):
    """Aggregate evaluation metrics across seeds."""
    results = []
    for seed, seed_dir in zip(seeds, seed_dirs):
        eval_log = os.path.join(seed_dir, "logs", "eval_log.csv")
        metrics = _read_eval_metrics(eval_log)
        if metrics:
            results.append({"seed": seed, **metrics})

    if not results:
        print("\n[WARN] No evaluation logs found across seeds.")
        return {}

    # Numeric aggregation
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

    exp_name = args.exp_name or "no_vpp_baseline"
    output_root = Path(args.output_root) / exp_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Config:     {args.config}")
    print(f"Seeds:      {args.seeds}")
    print(f"Output:     {output_root}")
    print("-" * 60)

    seed_dirs = []
    run_results = []

    for seed in range(args.seeds):
        seed_dir = str(output_root / f"seed_{seed}")
        seed_dirs.append(seed_dir)

        # Skip-existing check
        if args.skip_existing:
            best_ckpt = Path(seed_dir) / "checkpoints" / "best.pt"
            last_ckpt = Path(seed_dir) / "checkpoints" / "last.pt"
            if best_ckpt.exists() or last_ckpt.exists():
                print(f"\n[SEED {seed}] Skipping (checkpoint exists).")
                run_results.append(
                    {"seed": seed, "status": "skipped", "elapsed_s": 0.0}
                )
                continue

        success, elapsed = run_single_seed(
            args.config, seed, seed_dir, args.smoke, args.device
        )
        run_results.append(
            {
                "seed": seed,
                "status": "success" if success else "failed",
                "elapsed_s": elapsed,
            }
        )

    # ------------------------------------------------------------------
    # Aggregate & report
    # ------------------------------------------------------------------
    successful_seeds = [r["seed"] for r in run_results if r["status"] == "success"]
    failed_seeds = [r["seed"] for r in run_results if r["status"] == "failed"]
    skipped_seeds = [r["seed"] for r in run_results if r["status"] == "skipped"]

    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful_seeds)} {successful_seeds}")
    print(f"Failed:     {len(failed_seeds)} {failed_seeds}")
    print(f"Skipped:    {len(skipped_seeds)} {skipped_seeds}")

    summary = aggregate_results(seed_dirs, list(range(args.seeds)))
    if summary:
        print(f"\nPerformance (across seeds):")
        print(
            f"  mean_return:  {summary.get('mean_return_mean', 'N/A')} +/- {summary.get('mean_return_std', 'N/A')}"
        )
        print(
            f"  success_rate: {summary.get('success_rate_mean', 'N/A')} +/- {summary.get('success_rate_std', 'N/A')}"
        )

    # Save manifest
    manifest = {
        "experiment_name": exp_name,
        "config": args.config,
        "seeds": args.seeds,
        "smoke": args.smoke,
        "seed_results": run_results,
        "aggregate": summary,
    }
    manifest_path = output_root / "multiseed_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved: {manifest_path}")

    if failed_seeds:
        print(f"\n[ERROR] {len(failed_seeds)} seed(s) failed. Check individual logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
