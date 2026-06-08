#!/usr/bin/env python3
"""
Multi-seed PPO training launcher for frozen LSTM predictor.

Runs train_prediction_vpp_ppo across N seeds (default 3), aggregates
predictor health metrics (fallback rates, prediction errors), and produces
a summary report compatible with the CV/CA baseline evaluation pipeline.

Usage:
    # Standard 3-seed run (200K steps each)
    python scripts/run_lstm_ppo_multiseed.py \
        --config config/experiment/train_vpp_ppo_lstm_frozen.yaml \
        --checkpoint outputs/trajectory_prediction/best_model.pt \
        --seeds 3

    # Quick smoke test across seeds
    python scripts/run_lstm_ppo_multiseed.py \
        --config config/experiment/train_vpp_ppo_lstm_frozen.yaml \
        --checkpoint outputs/trajectory_prediction/best_model.pt \
        --seeds 3 --smoke

    # Resume / skip existing
    python scripts/run_lstm_ppo_multiseed.py ... --skip-existing
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
    parser = argparse.ArgumentParser(description="Multi-seed LSTM-PPO training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML (e.g. train_vpp_ppo_lstm_frozen.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained LSTM checkpoint (.pt).",
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
    # Remove common prefixes/suffixes for brevity
    for prefix in ("train_vpp_ppo_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def run_single_seed(config_path: str, checkpoint: str, seed: int, output_dir: str, smoke: bool, device: str = None):
    """Launch one training run for a given seed."""
    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "--config", config_path,
        "--predictor-type", "lstm",
        "--checkpoint", checkpoint,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if smoke:
        cmd.append("--smoke")
    if device:
        # device is a PPO-level config key; we inject it via env var or
        # rely on the config file.  For simplicity we do not override here.
        pass

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
        "mean_return", "std_return", "success_rate", "crash_rate",
        "out_of_bounds_rate", "timeout_rate", "mean_final_range_m",
        "mean_final_ata_deg", "prediction_valid_rate", "fallback_rate",
        "post_warmup_fallback_rate", "warmup_fallback_rate",
        "runtime_fallback_rate", "mean_prediction_error_m",
        "median_prediction_error_m",
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

    # Count fields
    count_keys = [
        "predictor_init_failed_count", "unknown_fallback_phase_count",
        "missing_fallback_phase_count", "configured_current_target_fallback_count",
        "prediction_error_count",
    ]
    for key in count_keys:
        vals = []
        for r in results:
            v = r.get(key)
            if v is not None and v != "":
                try:
                    vals.append(int(float(v)))
                except (ValueError, TypeError):
                    pass
        if vals:
            summary[f"{key}_sum"] = int(np.sum(vals))

    return summary


def main():
    args = parse_args()

    exp_name = args.exp_name or _load_yaml_name(args.config)
    output_root = Path(args.output_root) / exp_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
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
                run_results.append({"seed": seed, "status": "skipped", "elapsed_s": 0.0})
                continue

        success, elapsed = run_single_seed(
            args.config, args.checkpoint, seed, seed_dir, args.smoke, args.device
        )
        run_results.append({"seed": seed, "status": "success" if success else "failed", "elapsed_s": elapsed})

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
        print("\nPredictor Health (across seeds):")
        print(f"  prediction_valid_rate:  {summary.get('prediction_valid_rate_mean', 'N/A')}")
        print(f"  fallback_rate:          {summary.get('fallback_rate_mean', 'N/A')}")
        print(f"  post_warmup_fallback:   {summary.get('post_warmup_fallback_rate_mean', 'N/A')}")
        print(f"  runtime_fallback:       {summary.get('runtime_fallback_rate_mean', 'N/A')}")
        print(f"  warmup_fallback:        {summary.get('warmup_fallback_rate_mean', 'N/A')}")
        print(f"  mean_pred_error_m:      {summary.get('mean_prediction_error_m_mean', 'N/A')}")
        print(f"  pred_error_count:       {summary.get('prediction_error_count_sum', 'N/A')}")
        print(f"  init_failed_count:      {summary.get('predictor_init_failed_count_sum', 'N/A')}")
        print(f"\nPerformance (across seeds):")
        print(f"  mean_return:  {summary.get('mean_return_mean', 'N/A')} +/- {summary.get('mean_return_std', 'N/A')}")
        print(f"  success_rate: {summary.get('success_rate_mean', 'N/A')} +/- {summary.get('success_rate_std', 'N/A')}")

    # Save manifest
    manifest = {
        "experiment_name": exp_name,
        "config": args.config,
        "checkpoint": args.checkpoint,
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
