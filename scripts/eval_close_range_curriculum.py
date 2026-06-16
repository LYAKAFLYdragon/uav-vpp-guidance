#!/usr/bin/env python3
"""Evaluate the three close-range curriculum stages side-by-side.

This script runs monte_carlo_dogfight.py for each stage config and prints a
unified comparison table. It is meant to validate the difficulty ordering
before launching full RL training.

Usage:
    python scripts/eval_close_range_curriculum.py --runs-per-cell 50 --max-steps 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STAGES = [
    {
        "name": "stage1_easy",
        "config": "config/experiment/close_range_curriculum_stage1.yaml",
        "scenarios": "head_on",
        "description": "CAC default + head-on + loose",
    },
    {
        "name": "stage2_medium",
        "config": "config/experiment/close_range_curriculum_stage2.yaml",
        "scenarios": "head_on,favorable",
        "description": "CAC tuned v2 + head-on/favorable + medium",
    },
    {
        "name": "stage3_hard",
        "config": "config/experiment/close_range_curriculum_stage3.yaml",
        "scenarios": "favorable,head_on,challenging,neutral",
        "description": "Maneuver library + all scenarios + strict",
    },
]


def run_stage(stage: dict, runs_per_cell: int, max_steps: int, seed: int) -> dict:
    output = f"outputs/mc_curriculum_{stage['name']}.json"
    cmd = [
        sys.executable,
        "scripts/monte_carlo_dogfight.py",
        "--config", stage["config"],
        "--runs-per-cell", str(runs_per_cell),
        "--max-steps", str(max_steps),
        "--seed", str(seed),
        "--scenario-types", stage["scenarios"],
        "--output", output,
    ]
    print(f"\n>>> Running {stage['name']}: {' '.join(cmd[-6:])}")
    subprocess.run(cmd, check=True)
    return json.load(open(output, encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-per-cell", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {}
    for stage in STAGES:
        results[stage["name"]] = run_stage(stage, args.runs_per_cell, args.max_steps, args.seed)

    print("\n" + "=" * 100)
    print(f"{'Stage':<18} {'Description':<35} {'N':>5} {'Success%':>10} {'Timeouts':>10} {'AvgZone':>9} {'MeanMinRng':>12} {'MeanFinRng':>12}")
    print("-" * 100)
    for stage in STAGES:
        ov = results[stage["name"]]["summary"]["overall"]
        timeouts = ov.get("reason_counts", {}).get("timeout", 0)
        print(
            f"{stage['name']:<18} {stage['description']:<35} {ov['n']:>5} "
            f"{ov['success_rate']*100:>9.1f}% {timeouts:>10} {ov['mean_steps_in_zone']:>9.2f} "
            f"{ov['mean_min_range_m']:>11.0f} {ov['mean_final_range_m']:>11.0f}"
        )

    print("\nCurriculum ordering check:")
    srs = [results[s["name"]]["summary"]["overall"]["success_rate"] for s in STAGES]
    if srs[0] >= srs[1] >= srs[2]:
        print("  PASS: success rate decreases monotonically across stages.")
    else:
        print("  WARNING: success rate does NOT decrease monotonically.")
        print(f"  Stage success rates: {[f'{s*100:.1f}%' for s in srs]}")


if __name__ == "__main__":
    main()
