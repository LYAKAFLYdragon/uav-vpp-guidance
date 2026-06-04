#!/usr/bin/env python3
"""
Stage 6F Full Ablation Pipeline.

Runs full PPO training for all 5 ablation methods across multiple training seeds,
then evaluates each training seed's policies with the unified comparison script.

Methods:
  1. no_prediction
  2. cv_prediction
  3. ca_prediction
  4. lstm_frozen
  5. gru_frozen

Usage:
    # Dry-run: print commands but do not execute
    python scripts/run_stage6f_full_ablation.py --dry-run --training-seeds 0 1 2

    # Smoke test: quick training + evaluation for one training seed
    python scripts/run_stage6f_full_ablation.py --smoke --training-seeds 0

    # Full formal ablation
    python scripts/run_stage6f_full_ablation.py --training-seeds 0 1 2 --evaluation-seeds 0 1 2

    # Resume: skip training if checkpoint already exists
    python scripts/run_stage6f_full_ablation.py --resume --training-seeds 0 1 2
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


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

METRICS_SCHEMA_VERSION = "6f.2"
SCENARIOS = ["favorable", "neutral", "disadvantage", "challenging"]


def get_git_info():
    """Return current git commit and branch, or placeholders if not available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=os.getcwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        branch = "unknown"
    return commit, branch


def compute_file_hash(path: str) -> str:
    """Compute MD5 hash of a file."""
    if not os.path.exists(path):
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_existing(output_dir: str):
    """Move existing output dir to a backup location."""
    if os.path.exists(output_dir):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{output_dir}_backup_{timestamp}"
        print(f"  Backing up existing output to {backup_dir}")
        shutil.move(output_dir, backup_dir)


def write_manifest(
    output_dir: str,
    method: str,
    seed: int,
    config_path: str,
    policy_checkpoint_path: str,
    predictor_checkpoint_path: str,
    backend: str,
    validation_mode: str,
    allow_random_policy: bool,
):
    """Write per-run experiment manifest."""
    os.makedirs(output_dir, exist_ok=True)
    commit, branch = get_git_info()
    manifest = {
        "git_commit": commit,
        "branch": branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": method,
        "seed": seed,
        "config_path": config_path,
        "config_hash": compute_file_hash(config_path),
        "output_dir": output_dir,
        "policy_checkpoint_path": policy_checkpoint_path,
        "predictor_checkpoint_path": predictor_checkpoint_path,
        "backend": backend,
        "validation_mode": validation_mode,
        "allow_random_policy": allow_random_policy,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest saved to {manifest_path}")


def write_experiment_plan(
    output_dir: str,
    training_seeds,
    evaluation_seeds,
    episodes_per_scenario: int,
    formal: bool,
):
    """Write top-level experiment plan for the whole ablation."""
    os.makedirs(output_dir, exist_ok=True)
    commit, branch = get_git_info()
    plan = {
        "git_commit": commit,
        "branch": branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methods": [m["name"] for m in METHODS],
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seeds,
        "scenarios": SCENARIOS,
        "episodes_per_scenario": episodes_per_scenario,
        "backend": "simple",
        "formal": formal,
        "allow_random_policy": False,
        "comparison_config": "config/experiment/evaluate_vpp_prediction_comparison.yaml",
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
    }
    plan_path = os.path.join(output_dir, "experiment_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"Experiment plan saved to {plan_path}")


def run_training(method: dict, seed: int, smoke: bool, dry_run: bool, resume: bool) -> bool:
    """Run full training for a single method and training seed."""
    name = method["name"]
    config_path = method["train_config"]
    output_dir = f"{method['output_dir']}_seed{seed}"
    checkpoint_path = os.path.join(output_dir, "checkpoints", "best.pt")

    print(f"\n{'='*60}")
    print(f"Training method: {name} | training_seed: {seed}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    if resume and os.path.exists(checkpoint_path):
        print(f"  Checkpoint already exists: {checkpoint_path}")
        print(f"  Skipping training (--resume).")
        return True

    if not dry_run and os.path.exists(output_dir) and not resume:
        backup_existing(output_dir)

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "--config", config_path,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if smoke:
        cmd.append("--smoke")

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=os.getcwd())
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: Training failed for {name} training_seed {seed} (exit {result.returncode})")
        return False

    # Write manifest
    predictor_ckpt = None
    if name in ("lstm_frozen", "gru_frozen"):
        from uav_vpp_guidance.utils.config import load_yaml_config
        cfg = load_yaml_config(config_path)
        predictor_ckpt = cfg.get("trajectory_prediction", {}).get("checkpoint_path")
    write_manifest(
        output_dir=output_dir,
        method=name,
        seed=seed,
        config_path=config_path,
        policy_checkpoint_path=checkpoint_path,
        predictor_checkpoint_path=predictor_ckpt,
        backend="simple",
        validation_mode="raise",
        allow_random_policy=False,
    )

    print(f"Training completed for {name} training_seed {seed} in {elapsed/60:.1f} minutes")
    return True


def build_method_checkpoint_overrides(training_seed: int) -> list:
    """Build --method-checkpoint overrides for a given training seed."""
    overrides = []
    for method in METHODS:
        ckpt = os.path.join(f"{method['output_dir']}_seed{training_seed}", "checkpoints", "best.pt")
        overrides.append(f"{method['name']}={ckpt}")
    return overrides


def run_comparison_for_training_seed(
    training_seed: int,
    evaluation_seeds,
    episodes_per_scenario: int,
    smoke: bool,
    dry_run: bool,
    comparison_root: str,
) -> bool:
    """Run comparison evaluation for all methods at a specific training seed."""
    print(f"\n{'='*60}")
    print(f"Comparison evaluation for training_seed: {training_seed}")
    print(f"{'='*60}")

    comparison_config = "config/experiment/evaluate_vpp_prediction_comparison.yaml"
    output_dir = os.path.join(comparison_root, f"train_seed{training_seed}")

    method_overrides = build_method_checkpoint_overrides(training_seed)

    episodes = "1" if smoke else str(episodes_per_scenario)
    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", comparison_config,
        "--backend", "simple",
        "--training-seed", str(training_seed),
        "--episodes-per-scenario", episodes,
        "--seeds", *map(str, evaluation_seeds),
        "--scenarios", *SCENARIOS,
        "--save-trajectories",
        "--output-dir", output_dir,
        "--validation-mode", "raise",
    ]
    for override in method_overrides:
        cmd.extend(["--method-checkpoint", override])

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"ERROR: Comparison evaluation failed for training_seed {training_seed} (exit {result.returncode})")
        return False

    print(f"Comparison results saved to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6F Full Ablation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_stage6f_full_ablation.py --dry-run --training-seeds 0 1 2\n"
            "  python scripts/run_stage6f_full_ablation.py --smoke --training-seeds 0\n"
            "  python scripts/run_stage6f_full_ablation.py --resume --training-seeds 0 1 2\n"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip training if checkpoint already exists")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode with reduced training/eval")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Training random seeds (default: 0 1 2)")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Evaluation random seeds (default: 0 1 2)")
    parser.add_argument("--episodes-per-scenario", type=int, default=25,
                        help="Episodes per scenario for formal evaluation (default: 25)")
    parser.add_argument("--comparison-output-dir", type=str,
                        default="outputs/tables/stage6f_full_ablation",
                        help="Root output directory for comparison evaluations")
    args = parser.parse_args()

    # Backward compatibility: --seeds is alias for --training-seeds
    if "--seeds" in sys.argv:
        print("WARNING: --seeds is deprecated; use --training-seeds for training seeds and --evaluation-seeds for evaluation seeds.")

    print("Stage 6F Full Ablation Pipeline")
    print(f"Methods: {[m['name'] for m in METHODS]}")
    print(f"Training seeds: {args.training_seeds}")
    print(f"Evaluation seeds: {args.evaluation_seeds}")
    print(f"Episodes per scenario: {args.episodes_per_scenario}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Resume: {args.resume}")
    print(f"Smoke: {args.smoke}")
    print("")

    if args.smoke:
        print("[SMOKE] Running reduced training and evaluation.")

    formal = not args.smoke
    write_experiment_plan(
        output_dir=args.comparison_output_dir,
        training_seeds=args.training_seeds,
        evaluation_seeds=args.evaluation_seeds,
        episodes_per_scenario=args.episodes_per_scenario,
        formal=formal,
    )

    overall_start = time.time()
    training_successes = []

    # Phase 1: Training
    for method in METHODS:
        for seed in args.training_seeds:
            ok = run_training(
                method, seed,
                smoke=args.smoke,
                dry_run=args.dry_run,
                resume=args.resume,
            )
            training_successes.append((method["name"], seed, ok))
            if not ok and not args.dry_run:
                print(f"Aborting pipeline due to training failure for {method['name']} training_seed {seed}")
                sys.exit(1)

    # Phase 2: Comparison evaluation per training seed
    all_ok = all(ok for _, _, ok in training_successes)
    if not all_ok and not args.dry_run:
        print("\nSome trainings failed. Skipping comparison evaluation.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY-RUN] All training commands prepared.")

    comparison_successes = []
    for training_seed in args.training_seeds:
        ok = run_comparison_for_training_seed(
            training_seed=training_seed,
            evaluation_seeds=args.evaluation_seeds,
            episodes_per_scenario=args.episodes_per_scenario,
            smoke=args.smoke,
            dry_run=args.dry_run,
            comparison_root=args.comparison_output_dir,
        )
        comparison_successes.append((training_seed, ok))

    if args.dry_run:
        print("\n[DRY-RUN] Pipeline complete.")
    elif all(ok for _, ok in comparison_successes):
        print("\nAll comparison evaluations completed successfully!")
    else:
        print("\nSome comparison evaluations failed.")

    overall_elapsed = time.time() - overall_start
    print(f"\nTotal pipeline time: {overall_elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()
