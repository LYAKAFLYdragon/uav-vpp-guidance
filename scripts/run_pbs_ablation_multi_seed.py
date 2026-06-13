#!/usr/bin/env python3
"""
Run a multi-seed potential-based reward shaping (PBS) ablation.

Conditions:
  - with_pbs:    config with potential-based shaping enabled
  - without_pbs: config with potential-based shaping disabled

Usage:
    python scripts/run_pbs_ablation_multi_seed.py [--seeds 3] [--parallel 1]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


BASE_CONFIG = Path("config/method_innovation_comparison.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/pbs_ablation_multi")
CONDITIONS = {
    "with_pbs": "true",
    "without_pbs": "false",
}


def run_single(condition: str, seed: int, config: Path, output_root: Path, device: str, smoke: bool = False) -> tuple[str, int, int]:
    """Launch one training run. Returns (condition, seed, returncode)."""
    output_dir = output_root / condition / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config", str(config),
        "--seed", str(seed),
        "--reward-shaping", CONDITIONS[condition],
        "--output-dir", str(output_dir),
        "--device", device,
    ]
    if smoke:
        cmd.append("--smoke")

    log_path = output_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    return condition, seed, proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Multi-seed PBS ablation")
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke test")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    tasks = [
        (cond, seed)
        for cond in CONDITIONS
        for seed in range(args.seeds)
    ]

    print(f"Launching PBS ablation: {len(tasks)} runs ({args.seeds} seeds x {len(CONDITIONS)} conditions)")
    print(f"Output root: {args.output_root.resolve()}")

    if args.parallel <= 1:
        results = [
            run_single(cond, seed, args.config, args.output_root, args.device, smoke=args.smoke)
            for cond, seed in tasks
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single, cond, seed, args.config, args.output_root, args.device, args.smoke): (cond, seed)
                for cond, seed in tasks
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())

    failed = [(c, s, rc) for c, s, rc in results if rc != 0]
    if failed:
        print("FAILED runs:")
        for c, s, rc in failed:
            print(f"  {c}/seed{s}: exit {rc}")
        sys.exit(1)

    print("All PBS ablation runs completed successfully.")


if __name__ == "__main__":
    main()
