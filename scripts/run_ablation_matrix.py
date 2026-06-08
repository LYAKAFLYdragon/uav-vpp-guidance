#!/usr/bin/env python3
"""
Run the complete ablation experiment matrix (with parallel worker support).

Usage:
    python scripts/run_ablation_matrix.py \
        --gpu 0 \
        --methods "no_vpp,vpp_single,vpp_bilevel,vpp_cv,vpp_ca,vpp_lstm,vpp_gru" \
        --target-modes "constant,sinusoidal" \
        --seeds 5 \
        --steps 500000 \
        --eval-seeds 10 \
        --eval-eps 50 \
        --workers 4 \
        --output-root outputs/ablation_matrix
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Experiment matrix definition
# NOTE: vpp_fixed excluded because baseline_fixed_gain.yaml is incompatible
# with train_bilevel module (requires checkpoint which vpp_fixed doesn't have)
METHODS = {
    "no_vpp": {
        "train_config": "config/experiment/train_no_vpp_direct_command.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "Direct Command (No VPP)",
    },
    # "vpp_fixed": {  # excluded - baseline_fixed_gain.yaml incompatible with train_bilevel
    #     "train_config": "config/experiment/baseline_fixed_gain.yaml",
    #     "train_module": "uav_vpp_guidance.training.train_bilevel",
    #     "train_extra_args": ["--allow-random-init", "--n-episodes", "100"],
    #     "name": "VPP + Fixed Gain",
    # },
    "vpp_single": {
        "train_config": "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "VPP + Single-Layer PPO",
    },
    "vpp_bilevel": {
        "train_config": "config/experiment/proposed_bilevel.yaml",
        "train_module": "uav_vpp_guidance.training.train_bilevel",
        "train_extra_args": ["--n-episodes", "200", "--outer-every", "10", "--inner-iter", "20"],
        "name": "VPP + Bilevel PPO",
    },
    "vpp_cv": {
        "train_config": "config/experiment/train_vpp_ppo_cv.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "VPP + Single PPO + CV",
    },
    "vpp_ca": {
        "train_config": "config/experiment/train_vpp_ppo_ca.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "VPP + Single PPO + CA",
    },
    "vpp_lstm": {
        "train_config": "config/experiment/train_vpp_ppo_lstm_frozen.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "VPP + Single PPO + LSTM",
    },
    "vpp_gru": {
        "train_config": "config/experiment/train_vpp_ppo_gru_frozen.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "VPP + Single PPO + GRU",
    },
}

TARGET_MODES = {
    "constant": {"target_mode": "constant_velocity"},
    "sinusoidal": {"target_mode": "sinusoidal"},
}

EVAL_METHOD_NAMES = {
    "no_vpp": "no_vpp",
    # "vpp_fixed": "no_prediction",
    "vpp_single": "no_prediction",
    "vpp_bilevel": "no_prediction",
    "vpp_cv": "cv_prediction",
    "vpp_ca": "ca_prediction",
    "vpp_lstm": "lstm_frozen",
    "vpp_gru": "gru_frozen",
}


def _modify_config(config_path: Path, target_mode: str, total_timesteps: int = None):
    """Temporarily replace target_mode and total_timesteps in config. Returns backup path."""
    # Use PID-specific backup to avoid collisions between parallel workers
    backup_path = config_path.with_suffix(f".yaml.bak.{os.getpid()}")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    content = config_path.read_text(encoding="utf-8")
    content = content.replace("target_mode: constant_velocity", f"target_mode: {target_mode}")
    content = content.replace("target_mode: sinusoidal", f"target_mode: {target_mode}")

    if total_timesteps is not None:
        import re
        content = re.sub(
            r"total_timesteps:\s*\d+",
            f"total_timesteps: {total_timesteps}",
            content,
        )

    config_path.write_text(content, encoding="utf-8")
    return backup_path


def _restore_config(config_path: Path, backup_path: Path):
    """Restore config from backup."""
    if backup_path.exists():
        shutil.copy2(backup_path, config_path)


def _create_single_method_eval_config(original_config_path: Path, method_key: str, checkpoint_path: Path, output_dir: Path):
    """Create a temporary eval config with only one method and the given checkpoint."""
    import yaml

    with open(original_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_method_name = EVAL_METHOD_NAMES[method_key]
    methods = config.get("methods", {})

    if eval_method_name in methods:
        single_method = dict(methods[eval_method_name])
        single_method["checkpoint"] = str(checkpoint_path)
    elif method_key == "no_vpp":
        single_method = {
            "name": "no_vpp",
            "checkpoint": str(checkpoint_path),
            "trajectory_prediction": {"enabled": False},
            "virtual_point": {"enabled": False},
        }
    else:
        single_method = dict(methods.get("no_prediction", {}))
        single_method["name"] = eval_method_name
        single_method["checkpoint"] = str(checkpoint_path)

    config["methods"] = {eval_method_name: single_method}

    # PID-specific temp config to avoid collisions
    temp_path = output_dir / f"eval_config_{method_key}_pid{os.getpid()}.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return temp_path


def run_training(method_key, target_mode_key, seed, steps, output_dir, gpu_id):
    """Run a single training run. Returns checkpoint path or None."""
    method = METHODS[method_key]
    target = TARGET_MODES[target_mode_key]

    exp_name = f"{method_key}_{target_mode_key}_s{seed}"
    output_path = output_dir / "training" / exp_name

    config_path = Path(method["train_config"])
    backup_path = _modify_config(config_path, target["target_mode"], steps)

    try:
        cmd = [
            sys.executable, "-m",
            method["train_module"],
            "--config", str(config_path),
            "--output-dir", str(output_path),
        ]

        if "train_prediction_vpp_ppo" in method["train_module"]:
            cmd.extend(["--seed", str(seed)])
        elif "train_bilevel" in method["train_module"]:
            cmd.extend(["--seed", str(seed)])
            if method_key == "vpp_bilevel":
                single_ckpt = output_dir / "training" / f"vpp_single_{target_mode_key}_s{seed}" / "checkpoints" / "best.pt"
                if not single_ckpt.exists():
                    return None
                cmd.extend(["--checkpoint", str(single_ckpt)])
            elif method_key == "vpp_fixed":
                cmd.append("--allow-random-init")
            cmd.extend(method.get("train_extra_args", []))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            return None

        ckpt = output_path / "checkpoints" / "best.pt"
        if not ckpt.exists():
            ckpt = output_path / "checkpoints" / "last.pt"
        if not ckpt.exists() and "train_bilevel" in method["train_module"]:
            # train_bilevel saves checkpoints as policy_ep{episode}.pt
            import glob
            ckpt_files = sorted(
                glob.glob(str(output_path / "checkpoints" / "policy_ep*.pt")),
                key=lambda x: int(x.split("_ep")[-1].split(".")[0])
            )
            if ckpt_files:
                ckpt = Path(ckpt_files[-1])

        return str(ckpt) if ckpt.exists() else None

    finally:
        _restore_config(config_path, backup_path)


def run_evaluation(method_key, target_mode_key, seed, checkpoint, eval_seeds, eval_eps, output_dir, gpu_id):
    """Run evaluation for a trained checkpoint. Returns bool."""
    target = TARGET_MODES[target_mode_key]
    exp_name = f"{method_key}_{target_mode_key}_s{seed}"
    eval_output = output_dir / "evaluation" / exp_name
    eval_output.mkdir(parents=True, exist_ok=True)

    if target["target_mode"] == "constant_velocity":
        eval_config = Path("config/experiment/stage6f5_feasible_geometry.yaml")
    else:
        eval_config = Path("config/experiment/stage6f5_maneuvering_target.yaml")

    temp_config = _create_single_method_eval_config(eval_config, method_key, Path(checkpoint), eval_output)

    cmd = [
        sys.executable, "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", str(temp_config),
        "--backend", "simple",
        "--episodes", str(eval_eps),
        "--seeds", *[str(s) for s in range(eval_seeds)],
        "--output-dir", str(eval_output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return result.returncode == 0


def run_single_experiment(task):
    """Run one complete experiment (train + eval). Called by worker process."""
    method_key, target_mode_key, seed, steps, output_dir, gpu_id, eval_seeds, eval_eps = task
    exp_name = f"{method_key}_{target_mode_key}_s{seed}"

    train_start = datetime.now()
    checkpoint = run_training(method_key, target_mode_key, seed, steps, output_dir, gpu_id)
    train_time = (datetime.now() - train_start).total_seconds()

    if checkpoint is None:
        return {
            "method": method_key,
            "target_mode": target_mode_key,
            "seed": seed,
            "checkpoint": None,
            "status": "training_failed",
            "train_time_s": train_time,
        }

    eval_start = datetime.now()
    eval_ok = run_evaluation(method_key, target_mode_key, seed, checkpoint, eval_seeds, eval_eps, output_dir, gpu_id)
    eval_time = (datetime.now() - eval_start).total_seconds()

    status = "success" if eval_ok else "eval_failed"
    return {
        "method": method_key,
        "target_mode": target_mode_key,
        "seed": seed,
        "checkpoint": checkpoint,
        "status": status,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment matrix")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method keys or 'all'")
    parser.add_argument("--target-modes", type=str, default="all", help="Comma-separated target mode keys or 'all'")
    parser.add_argument("--seeds", type=int, default=5, help="Number of training seeds")
    parser.add_argument("--steps", type=int, default=500000, help="Training steps per run")
    parser.add_argument("--eval-seeds", type=int, default=10, help="Number of evaluation seeds")
    parser.add_argument("--eval-eps", type=int, default=50, help="Episodes per eval seed")
    parser.add_argument("--output-root", type=str, default="outputs/ablation_matrix")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with existing checkpoints")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    methods = list(METHODS.keys()) if args.methods == "all" else args.methods.split(",")
    target_modes = list(TARGET_MODES.keys()) if args.target_modes == "all" else args.target_modes.split(",")

    for m in methods:
        assert m in METHODS, f"Unknown method: {m}"
    for t in target_modes:
        assert t in TARGET_MODES, f"Unknown target mode: {t}"

    # Load existing manifest if any
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = []

    # Build task list
    tasks = []
    skipped_existing = 0
    for method_key in methods:
        for target_mode_key in target_modes:
            for seed in range(args.seeds):
                exp_name = f"{method_key}_{target_mode_key}_s{seed}"
                ckpt_path = output_root / "training" / exp_name / "checkpoints" / "best.pt"

                # Check if already in manifest as success/skipped
                already_done = any(
                    r["method"] == method_key and r["target_mode"] == target_mode_key and r["seed"] == seed
                    and r["status"] in ("success", "skipped")
                    for r in manifest
                )

                if args.skip_existing and (ckpt_path.exists() or already_done):
                    skipped_existing += 1
                    if not already_done:
                        manifest.append({
                            "method": method_key,
                            "target_mode": target_mode_key,
                            "seed": seed,
                            "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
                            "status": "skipped",
                        })
                    continue

                tasks.append((method_key, target_mode_key, seed, args.steps, output_root, args.gpu, args.eval_seeds, args.eval_eps))

    total_runs = len(methods) * len(target_modes) * args.seeds
    remaining = len(tasks)
    print(f"Experiment matrix: {len(methods)} methods x {len(target_modes)} target modes x {args.seeds} seeds = {total_runs} runs")
    print(f"Already completed/skipped: {total_runs - remaining}")
    print(f"Remaining to run: {remaining}")
    print(f"Workers: {args.workers}")
    print(f"Estimated GPU time: {remaining * args.steps / 200000 * 8 / 60 / args.workers:.1f} hours (approximate, parallel)")
    print(f"Output directory: {output_root}")
    print()

    if not tasks:
        print("All runs already complete!")
        success = sum(1 for m in manifest if m["status"] == "success")
        print(f"Complete: {success}/{total_runs} runs successful")
        return

    # Run tasks in parallel
    run_idx = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_experiment, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            run_idx += 1
            method_key, target_mode_key, seed = task[0], task[1], task[2]
            exp_name = f"{method_key}_{target_mode_key}_s{seed}"

            try:
                result = future.result()
            except Exception as exc:
                print(f"[{run_idx}/{remaining}] {exp_name} ... EXCEPTION: {exc}")
                result = {
                    "method": method_key,
                    "target_mode": target_mode_key,
                    "seed": seed,
                    "checkpoint": None,
                    "status": f"exception: {exc}",
                }

            # Update manifest
            # Remove any existing entry for this run
            manifest = [r for r in manifest if not (
                r["method"] == method_key and r["target_mode"] == target_mode_key and r["seed"] == seed
            )]
            manifest.append(result)

            status = result["status"]
            train_time = result.get("train_time_s", 0)
            eval_time = result.get("eval_time_s", 0)
            print(f"[{run_idx}/{remaining}] {exp_name} ... {status} (train: {train_time:.0f}s, eval: {eval_time:.0f}s)")

            # Save manifest after each run
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Summary
    success = sum(1 for m in manifest if m["status"] == "success")
    print(f"\n{'='*60}")
    print(f"Complete: {success}/{total_runs} runs successful")
    print(f"Manifest saved to: {output_root}/manifest.json")


if __name__ == "__main__":
    main()
