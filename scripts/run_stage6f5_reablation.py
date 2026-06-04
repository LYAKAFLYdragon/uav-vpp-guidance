#!/usr/bin/env python3
"""
Stage 6F.5 Re-Ablation Pipeline.

Supports two experimental suites:
  A. feasible_geometry: fixed favorable/disadvantage scenarios, constant-velocity target
  B. maneuvering_target: sinusoidal target motion to distinguish CV vs CA

Uses existing trained checkpoints (or trains if missing) and evaluates with
suite-specific comparison configurations.

Usage:
    # Dry-run feasible geometry suite
    python scripts/run_stage6f5_reablation.py \
        --suite feasible_geometry --dry-run \
        --training-seeds 0 1 2 --evaluation-seeds 0 1 2

    # Dry-run maneuvering target suite
    python scripts/run_stage6f5_reablation.py \
        --suite maneuvering_target --dry-run \
        --training-seeds 0 1 2 --evaluation-seeds 0 1 2

    # Full formal re-ablation
    python scripts/run_stage6f5_reablation.py \
        --suite feasible_geometry \
        --training-seeds 0 1 2 --evaluation-seeds 0 1 2
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

SUITES = {
    "feasible_geometry": {
        "comparison_config": "config/experiment/stage6f5_feasible_geometry.yaml",
        "output_root": "outputs/tables/stage6f5_feasible_geometry",
        "description": "Fixed favorable/disadvantage geometry with max_range_m=12000m",
    },
    "maneuvering_target": {
        "comparison_config": "config/experiment/stage6f5_maneuvering_target.yaml",
        "output_root": "outputs/tables/stage6f5_maneuvering_target",
        "description": "Sinusoidal target motion to distinguish CV vs CA",
    },
}

METRICS_SCHEMA_VERSION = "6f.2"


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
    suite_name: str,
    comparison_config: str,
    training_seeds,
    evaluation_seeds,
    episodes_per_scenario: int,
    scenarios: list,
    formal: bool,
):
    """Write top-level experiment plan for the re-ablation suite."""
    os.makedirs(output_dir, exist_ok=True)
    commit, branch = get_git_info()
    plan = {
        "git_commit": commit,
        "branch": branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite": suite_name,
        "methods": [m["name"] for m in METHODS],
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seeds,
        "scenarios": scenarios,
        "episodes_per_scenario": episodes_per_scenario,
        "backend": "simple",
        "formal": formal,
        "allow_random_policy": False,
        "comparison_config": comparison_config,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
    }
    plan_path = os.path.join(output_dir, "experiment_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"Experiment plan saved to {plan_path}")


def run_training(method: dict, seed: int, smoke: bool, dry_run: bool, resume: bool, force_resume: bool = False) -> bool:
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
        manifest_path = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            mismatches = []
            if manifest.get("method") != name:
                mismatches.append(f"method: manifest={manifest.get('method')} expected={name}")
            if manifest.get("seed") != seed:
                mismatches.append(f"seed: manifest={manifest.get('seed')} expected={seed}")
            current_config_hash = compute_file_hash(config_path)
            if manifest.get("config_hash") != current_config_hash:
                mismatches.append(f"config_hash: manifest={manifest.get('config_hash')} expected={current_config_hash}")
            if manifest.get("metrics_schema_version") != METRICS_SCHEMA_VERSION:
                mismatches.append(f"schema_version: manifest={manifest.get('metrics_schema_version')} expected={METRICS_SCHEMA_VERSION}")
            if mismatches:
                print(f"  WARNING: Manifest mismatch detected:")
                for mm in mismatches:
                    print(f"    - {mm}")
                if not force_resume:
                    print(f"  ERROR: Use --force-resume to override, or re-run without --resume.")
                    return False
                print(f"  --force-resume set; continuing despite mismatches.")
        else:
            print(f"  WARNING: No manifest found at {manifest_path}; cannot validate provenance.")
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


def extract_scenarios_from_config(config_path: str) -> list:
    """Extract scenario names from a comparison config."""
    if not os.path.exists(config_path):
        return []
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("scenarios", {}).keys())


def run_comparison_for_training_seed(
    training_seed: int,
    evaluation_seeds,
    episodes_per_scenario: int,
    smoke: bool,
    dry_run: bool,
    comparison_root: str,
    comparison_config: str,
    scenarios: list,
) -> bool:
    """Run comparison evaluation for all methods at a specific training seed."""
    print(f"\n{'='*60}")
    print(f"Comparison evaluation for training_seed: {training_seed}")
    print(f"Config: {comparison_config}")
    print(f"{'='*60}")

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
        "--scenarios", *scenarios,
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
        description="Stage 6F.5 Re-Ablation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_stage6f5_reablation.py --suite feasible_geometry --dry-run --training-seeds 0 1 2\n"
            "  python scripts/run_stage6f5_reablation.py --suite maneuvering_target --dry-run --training-seeds 0\n"
            "  python scripts/run_stage6f5_reablation.py --suite feasible_geometry --resume --training-seeds 0 1 2\n"
        ),
    )
    parser.add_argument("--suite", type=str, required=True, choices=list(SUITES.keys()),
                        help="Re-ablation suite to run")
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
    parser.add_argument("--force-resume", action="store_true",
                        help="Override manifest mismatch when using --resume")
    args = parser.parse_args()

    suite = SUITES[args.suite]
    comparison_config = suite["comparison_config"]
    comparison_output_dir = suite["output_root"]
    scenarios = extract_scenarios_from_config(comparison_config)

    if not scenarios:
        print(f"ERROR: Could not extract scenarios from {comparison_config}")
        sys.exit(1)

    print("Stage 6F.5 Re-Ablation Pipeline")
    print(f"Suite: {args.suite}")
    print(f"Description: {suite['description']}")
    print(f"Comparison config: {comparison_config}")
    print(f"Scenarios: {scenarios}")
    print(f"Training seeds: {args.training_seeds}")
    print(f"Evaluation seeds: {args.evaluation_seeds}")
    print(f"Episodes per scenario: {args.episodes_per_scenario}")
    print(f"Output directory: {comparison_output_dir}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Resume: {args.resume}")
    print(f"Smoke: {args.smoke}")
    print("")

    if args.smoke:
        print("[SMOKE] Running reduced training and evaluation.")

    formal = not args.smoke
    write_experiment_plan(
        output_dir=comparison_output_dir,
        suite_name=args.suite,
        comparison_config=comparison_config,
        training_seeds=args.training_seeds,
        evaluation_seeds=args.evaluation_seeds,
        episodes_per_scenario=args.episodes_per_scenario,
        scenarios=scenarios,
        formal=formal,
    )

    overall_start = time.time()
    training_successes = []

    # Phase 1: Training (reuse existing training configs)
    for method in METHODS:
        for seed in args.training_seeds:
            ok = run_training(
                method, seed,
                smoke=args.smoke,
                dry_run=args.dry_run,
                resume=args.resume,
                force_resume=args.force_resume,
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
            comparison_root=comparison_output_dir,
            comparison_config=comparison_config,
            scenarios=scenarios,
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
