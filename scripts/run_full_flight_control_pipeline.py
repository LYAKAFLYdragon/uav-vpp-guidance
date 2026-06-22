#!/usr/bin/env python3
"""
End-to-end pipeline for the flight-control comparison benchmark.

Runs the comparison benchmark, aggregates results, generates figures, and exports
Table 3 (CSV/Excel/Markdown + summary report).

Usage:
    python scripts/run_full_flight_control_pipeline.py \
        --run-id fc_compare_20260621_103000 \
        --checkpoint-ppo-pid outputs/experiments/ppo_pid_jsbsim/checkpoints/last.pt \
        --checkpoint-ppo outputs/experiments/no_prediction_vpp_ppo_jsbsim_compare/checkpoints/last.pt \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --jobs 4

Optional flags mirror run_flight_control_comparison.py where applicable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print("\n[PIPELINE] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=_repo_root())
    if result.returncode != 0:
        print(f"[PIPELINE] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flight-control comparison pipeline")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--checkpoint-ppo-pid",
        default="outputs/experiments/ppo_pid_jsbsim/checkpoints/last.pt",
        help="PPO+PID-Hybrid checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-ppo",
        default="outputs/experiments/no_prediction_vpp_ppo_jsbsim_compare/checkpoints/last.pt",
        help="PPO-FixedPID checkpoint path",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["ppo_pid", "ppo", "enhanced_pid", "baseline_pid"],
        help="Controllers to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["multi_waypoint", "sustained_turn"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20)),
        help="Random seeds (paired across controllers)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--status",
        type=str,
        default="formal",
        choices=["smoke", "formal"],
        help="Run status for manifest and report (smoke = exploratory, formal = final)",
    )
    parser.add_argument(
        "--config-base",
        default="config/experiment/compare_flight_control_base.yaml",
        help="Base evaluation config",
    )
    parser.add_argument(
        "--config-mw",
        default="config/experiment/task_multi_waypoint.yaml",
        help="Multi-waypoint task config",
    )
    parser.add_argument(
        "--config-st",
        default="config/experiment/task_sustained_turn.yaml",
        help="Sustained-turn task config",
    )
    args = parser.parse_args()

    run_dir = _repo_root() / "outputs" / "flight_control_compare" / args.run_id

    t0 = time.time()

    # 1. Run comparison benchmark
    _run(
        [
            sys.executable,
            "scripts/run_flight_control_comparison.py",
            "--run-id",
            args.run_id,
            "--controllers",
            *args.controllers,
            "--tasks",
            *args.tasks,
            "--seeds",
            *[str(s) for s in args.seeds],
            "--jobs",
            str(args.jobs),
            "--checkpoint-ppo-pid",
            args.checkpoint_ppo_pid,
            "--checkpoint-ppo",
            args.checkpoint_ppo,
            "--config-base",
            args.config_base,
            "--config-mw",
            args.config_mw,
            "--config-st",
            args.config_st,
            "--status",
            args.status,
        ]
    )

    # 2. Aggregate results
    _run(
        [
            sys.executable,
            "scripts/aggregate_flight_control_results.py",
            "--run-dir",
            str(run_dir),
        ]
    )

    # 3. Generate figures
    _run(
        [
            sys.executable,
            "scripts/plot_flight_control_figures.py",
            "--run-dir",
            str(run_dir),
        ]
    )

    # 4. Export Table 3
    _run(
        [
            sys.executable,
            "scripts/export_table3.py",
            "--run-dir",
            str(run_dir),
        ]
    )

    elapsed = time.time() - t0
    print(f"\n[PIPELINE] Completed in {elapsed:.1f}s")
    print(f"[PIPELINE] Outputs: {run_dir}")


if __name__ == "__main__":
    main()
