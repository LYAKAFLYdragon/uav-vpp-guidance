#!/usr/bin/env python3
"""
Stage 6F Full Ablation Pipeline.

Runs full PPO training for all 5 ablation methods, then evaluates
them with the unified comparison script.

Methods:
  1. no_prediction
  2. cv_prediction
  3. ca_prediction
  4. lstm_frozen
  5. gru_frozen

Usage:
    python scripts/run_stage6f_full_ablation.py
"""

import os
import shutil
import subprocess
import sys
import time


METHODS = [
    {
        "name": "no_prediction",
        "train_config": "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "output_dir": "outputs/experiments/no_prediction_vpp_ppo",
    },
    {
        "name": "cv_prediction",
        "train_config": "config/experiment/train_vpp_ppo_cv.yaml",
        "output_dir": "outputs/experiments/vpp_ppo_cv_prediction",
    },
    {
        "name": "ca_prediction",
        "train_config": "config/experiment/train_vpp_ppo_ca.yaml",
        "output_dir": "outputs/experiments/vpp_ppo_ca_prediction",
    },
    {
        "name": "lstm_frozen",
        "train_config": "config/experiment/train_vpp_ppo_lstm_frozen.yaml",
        "output_dir": "outputs/experiments/vpp_ppo_lstm_frozen",
    },
    {
        "name": "gru_frozen",
        "train_config": "config/experiment/train_vpp_ppo_gru_frozen.yaml",
        "output_dir": "outputs/experiments/vpp_ppo_gru_frozen",
    },
]


def backup_existing(output_dir: str):
    """Move existing output dir to a backup location."""
    if os.path.exists(output_dir):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{output_dir}_backup_{timestamp}"
        print(f"  Backing up existing output to {backup_dir}")
        shutil.move(output_dir, backup_dir)


def run_training(method: dict) -> bool:
    """Run full training for a single method."""
    name = method["name"]
    config_path = method["train_config"]
    output_dir = method["output_dir"]

    print(f"\n{'='*60}")
    print(f"Training method: {name}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    backup_existing(output_dir)

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "--config",
        config_path,
    ]

    start = time.time()
    result = subprocess.run(cmd, cwd=os.getcwd())
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Training failed for {name} (exit {result.returncode})")
        return False

    print(f"Training completed for {name} in {elapsed/60:.1f} minutes")
    return True


def run_comparison_eval() -> bool:
    """Run unified comparison evaluation across all trained methods."""
    print(f"\n{'='*60}")
    print("Running Stage 6F comparison evaluation")
    print(f"{'='*60}")

    comparison_config = "config/experiment/evaluate_vpp_prediction_comparison.yaml"
    output_dir = "outputs/tables/stage6f_full_ablation"

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config",
        comparison_config,
        "--backend",
        "simple",
        "--episodes",
        "50",
        "--seeds",
        "0", "1", "2",
        "--scenarios",
        "favorable", "neutral", "disadvantage", "challenging",
        "--save-trajectories",
        "--output-dir",
        output_dir,
    ]

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"ERROR: Comparison evaluation failed (exit {result.returncode})")
        return False

    print(f"Comparison results saved to {output_dir}")
    return True


def main():
    print("Stage 6F Full Ablation Pipeline")
    print(f"Methods: {[m['name'] for m in METHODS]}")
    print(f"Total timesteps per method: 200,000")
    print("")

    overall_start = time.time()
    successes = []

    for method in METHODS:
        ok = run_training(method)
        successes.append(ok)
        if not ok:
            print(f"Aborting pipeline due to training failure for {method['name']}")
            break

    if all(successes):
        print("\nAll trainings completed successfully!")
        run_comparison_eval()
    else:
        print("\nSome trainings failed. Skipping comparison evaluation.")

    overall_elapsed = time.time() - overall_start
    print(f"\nTotal pipeline time: {overall_elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()
