#!/usr/bin/env python3
"""
Run the complete ablation experiment matrix.

Usage:
    python scripts/run_ablation_matrix.py \
        --gpu 0 \
        --methods "no_vpp,vpp_single,vpp_bilevel,vpp_cv,vpp_ca,vpp_lstm,vpp_gru" \
        --target-modes "constant,sinusoidal_1g,sinusoidal_2g,sinusoidal_3g" \
        --seeds 5 \
        --steps 500000 \
        --eval-seeds 10 \
        --eval-eps 50 \
        --output-root outputs/ablation_matrix
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Experiment matrix definition
METHODS = {
    "no_vpp": {
        "train_config": "config/experiment/train_no_vpp_direct_command.yaml",
        "train_module": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "name": "Direct Command (No VPP)",
    },
    "vpp_fixed": {
        "train_config": "config/experiment/baseline_fixed_gain.yaml",
        "train_module": "uav_vpp_guidance.training.train_bilevel",
        "train_extra_args": ["--allow-random-init", "--n-episodes", "100"],
        "name": "VPP + Fixed Gain",
    },
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


def _modify_config(config_path: Path, target_mode: str, total_timesteps: int = None):
    """Temporarily replace target_mode and total_timesteps in config. Returns backup path."""
    backup_path = config_path.with_suffix(".yaml.bak")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    content = config_path.read_text(encoding="utf-8")
    # Replace both constant_velocity and sinusoidal to ensure idempotency
    content = content.replace("target_mode: constant_velocity", f"target_mode: {target_mode}")
    content = content.replace("target_mode: sinusoidal", f"target_mode: {target_mode}")

    # Modify total_timesteps if specified
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


def run_training(method_key, target_mode_key, seed, steps, output_dir, gpu_id):
    """Run a single training run."""
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

        # Module-specific args
        if "train_prediction_vpp_ppo" in method["train_module"]:
            cmd.extend(["--seed", str(seed)])
            # total_timesteps is modified in config file by _modify_config
        elif "train_bilevel" in method["train_module"]:
            cmd.extend(["--seed", str(seed)])
            # bilevel needs a checkpoint for vpp_bilevel; skip if none
            if method_key == "vpp_bilevel":
                single_ckpt = output_dir / "training" / f"vpp_single_{target_mode_key}_s{seed}" / "checkpoints" / "best.pt"
                if not single_ckpt.exists():
                    print(f"  SKIP: vpp_single_{target_mode_key}_s{seed} missing, cannot init bilevel")
                    return None
                cmd.extend(["--checkpoint", str(single_ckpt)])
            elif method_key == "vpp_fixed":
                # fixed gain uses random init with bilevel trainer
                cmd.append("--allow-random-init")

            cmd.extend(method.get("train_extra_args", []))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            print(f"ERROR training {exp_name}: {result.stderr[-500:]}")
            return None

        # Find checkpoint
        ckpt = output_path / "checkpoints" / "best.pt"
        if not ckpt.exists():
            ckpt = output_path / "checkpoints" / "last.pt"

        return ckpt if ckpt.exists() else None

    finally:
        _restore_config(config_path, backup_path)


# Method name mapping for evaluation configs
EVAL_METHOD_NAMES = {
    "no_vpp": "no_vpp",
    "vpp_fixed": "no_prediction",  # Same policy structure
    "vpp_single": "no_prediction",
    "vpp_bilevel": "no_prediction",  # Same policy structure
    "vpp_cv": "cv_prediction",
    "vpp_ca": "ca_prediction",
    "vpp_lstm": "lstm_frozen",
    "vpp_gru": "gru_frozen",
}


def _create_single_method_eval_config(original_config_path: Path, method_key: str, checkpoint_path: Path, output_dir: Path):
    """Create a temporary eval config with only one method and the given checkpoint."""
    import yaml

    with open(original_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_method_name = EVAL_METHOD_NAMES[method_key]

    # Build single-method config
    methods = config.get("methods", {})

    if eval_method_name in methods:
        # Use existing method config, override checkpoint
        single_method = dict(methods[eval_method_name])
        single_method["checkpoint"] = str(checkpoint_path)
    elif method_key == "no_vpp":
        # Build minimal no_vpp config
        single_method = {
            "name": "no_vpp",
            "checkpoint": str(checkpoint_path),
            "trajectory_prediction": {"enabled": False},
            "virtual_point": {"enabled": False},
        }
    else:
        # Fallback: use no_prediction config as base
        single_method = dict(methods.get("no_prediction", {}))
        single_method["name"] = eval_method_name
        single_method["checkpoint"] = str(checkpoint_path)

    config["methods"] = {eval_method_name: single_method}

    # Save temp config
    temp_path = output_dir / f"eval_config_{method_key}.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return temp_path


def run_evaluation(method_key, target_mode_key, seed, checkpoint, eval_seeds, eval_eps, output_dir, gpu_id):
    """Run evaluation for a trained checkpoint."""
    target = TARGET_MODES[target_mode_key]

    exp_name = f"{method_key}_{target_mode_key}_s{seed}"
    eval_output = output_dir / "evaluation" / exp_name
    eval_output.mkdir(parents=True, exist_ok=True)

    # Select eval config based on target mode
    if target["target_mode"] == "constant_velocity":
        eval_config = Path("config/experiment/stage6f5_feasible_geometry.yaml")
    else:
        eval_config = Path("config/experiment/stage6f5_maneuvering_target.yaml")

    # Create single-method temp config
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

    if result.returncode != 0:
        print(f"ERROR evaluating {exp_name}: {result.stderr[-500:]}")
        return False

    return True


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
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Select methods
    methods = list(METHODS.keys()) if args.methods == "all" else args.methods.split(",")
    target_modes = list(TARGET_MODES.keys()) if args.target_modes == "all" else args.target_modes.split(",")

    # Validate
    for m in methods:
        assert m in METHODS, f"Unknown method: {m}"
    for t in target_modes:
        assert t in TARGET_MODES, f"Unknown target mode: {t}"

    total_runs = len(methods) * len(target_modes) * args.seeds
    print(f"Experiment matrix: {len(methods)} methods × {len(target_modes)} target modes × {args.seeds} seeds = {total_runs} runs")
    print(f"Estimated GPU time: {total_runs * args.steps / 200000 * 8 / 60:.1f} hours (approximate)")
    print(f"Output directory: {output_root}")
    print()

    # Run matrix
    manifest = []
    run_idx = 0

    for method_key in methods:
        for target_mode_key in target_modes:
            for seed in range(args.seeds):
                run_idx += 1
                exp_name = f"{method_key}_{target_mode_key}_s{seed}"

                print(f"[{run_idx}/{total_runs}] {exp_name} ...", end=" ", flush=True)

                # Check if exists
                ckpt_path = output_root / "training" / exp_name / "checkpoints" / "best.pt"
                if args.skip_existing and ckpt_path.exists():
                    print("SKIP (exists)")
                    manifest.append({
                        "method": method_key,
                        "target_mode": target_mode_key,
                        "seed": seed,
                        "checkpoint": str(ckpt_path),
                        "status": "skipped",
                    })
                    continue

                # Training
                start = datetime.now()
                checkpoint = run_training(
                    method_key, target_mode_key, seed, args.steps,
                    output_root, args.gpu
                )
                train_time = (datetime.now() - start).total_seconds()

                if checkpoint is None:
                    print(f"FAIL (training)")
                    manifest.append({
                        "method": method_key,
                        "target_mode": target_mode_key,
                        "seed": seed,
                        "checkpoint": None,
                        "status": "training_failed",
                        "train_time_s": train_time,
                    })
                    continue

                print(f"OK (train: {train_time:.0f}s)", end=" ")

                # Evaluation
                start = datetime.now()
                eval_ok = run_evaluation(
                    method_key, target_mode_key, seed, checkpoint,
                    args.eval_seeds, args.eval_eps, output_root, args.gpu
                )
                eval_time = (datetime.now() - start).total_seconds()

                status = "success" if eval_ok else "eval_failed"
                print(f"{status} (eval: {eval_time:.0f}s)")

                manifest.append({
                    "method": method_key,
                    "target_mode": target_mode_key,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "status": status,
                    "train_time_s": train_time,
                    "eval_time_s": eval_time,
                })

                # Save manifest after each run
                manifest_path = output_root / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Summary
    success = sum(1 for m in manifest if m["status"] == "success")
    print(f"\n{'='*60}")
    print(f"Complete: {success}/{total_runs} runs successful")
    print(f"Manifest saved to: {output_root}/manifest.json")


if __name__ == "__main__":
    main()
