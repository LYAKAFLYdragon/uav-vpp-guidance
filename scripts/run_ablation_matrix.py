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
    "sinusoidal_1g": {"target_mode": "sinusoidal", "weaving_amplitude_g": 1.0},
    "sinusoidal_2g": {"target_mode": "sinusoidal", "weaving_amplitude_g": 2.0},
    "sinusoidal_3g": {"target_mode": "sinusoidal", "weaving_amplitude_g": 3.0},
}


def _modify_config(config_path: Path, target_mode: str, weaving_amplitude_g: float = None):
    """Temporarily replace target_mode and weaving_amplitude_g in config. Returns backup path."""
    backup_path = config_path.with_suffix(".yaml.bak")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    content = config_path.read_text(encoding="utf-8")
    # Replace both constant_velocity and sinusoidal to ensure idempotency
    content = content.replace("target_mode: constant_velocity", f"target_mode: {target_mode}")
    content = content.replace("target_mode: sinusoidal", f"target_mode: {target_mode}")

    # Modify weaving_amplitude_g if specified
    if weaving_amplitude_g is not None:
        import re
        if "weaving_amplitude_g:" in content:
            content = re.sub(
                r"weaving_amplitude_g:\s*[\d.]+",
                f"weaving_amplitude_g: {weaving_amplitude_g}",
                content,
            )
        else:
            # Insert after target_mode line (4-space indent for env block)
            content = content.replace(
                f"target_mode: {target_mode}",
                f"target_mode: {target_mode}\n    weaving_amplitude_g: {weaving_amplitude_g}",
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
    weaving_amp = target.get("weaving_amplitude_g")
    backup_path = _modify_config(config_path, target["target_mode"], weaving_amp)

    try:
        cmd = [
            sys.executable, "-m",
            method["train_module"],
            "--config", str(config_path),
            "--output-dir", str(output_path),
        ]

        # Module-specific args
        if "train_prediction_vpp_ppo" in method["train_module"]:
            cmd.extend(["--seed", str(seed), "--total-timesteps", str(steps)])
            if gpu_id >= 0:
                cmd.extend(["--device", f"cuda:{gpu_id}"])
            else:
                cmd.extend(["--device", "cpu"])
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


def run_evaluation(method_key, target_mode_key, seed, checkpoint, eval_seeds, eval_eps, output_dir, gpu_id):
    """Run evaluation for a trained checkpoint."""
    method = METHODS[method_key]
    target = TARGET_MODES[target_mode_key]

    exp_name = f"{method_key}_{target_mode_key}_s{seed}"
    eval_output = output_dir / "evaluation" / exp_name
    eval_output.mkdir(parents=True, exist_ok=True)

    # Select eval config based on target mode
    if target["target_mode"] == "constant_velocity":
        eval_config = "config/experiment/stage6f5_feasible_geometry.yaml"
    else:
        eval_config = "config/experiment/stage6f5_maneuvering_target.yaml"

    cmd = [
        sys.executable, "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", eval_config,
        "--backend", "simple",
        "--checkpoint", str(checkpoint),
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
