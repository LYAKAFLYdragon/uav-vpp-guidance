#!/usr/bin/env python3
"""Smoke-test each disadvantage bridge scenario with a short solo training run.

Usage:
    python scripts/smoke_test_disadvantage_bridges.py --timesteps 10000

For each scenario a temporary config is generated that trains only on that
scenario.  This lets us verify the difficulty ordering of the feasibility
gradient without running the full curriculum.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Scenarios to smoke-test, from easy to hard.
DEFAULT_SCENARIOS = [
    "favorable",
    "neutral",
    "disadvantage_bridge_1",
    "disadvantage_bridge_2",
    "disadvantage_bridge_3",
    "disadvantage_bridge_4",
    "disadvantage_bridge_5",
    "disadvantage",
]


def make_temp_config(scenario_name: str, timesteps: int) -> Path:
    """Create a temporary config that trains on a single scenario."""
    # Use absolute paths for includes so the temp config works regardless of
    # where it is written (the loader resolves includes relative to the config
    # file's directory).
    base_abs = Path.cwd() / "config" / "experiment" / "maneuver_target_pilot_base.yaml"
    bridge_abs = Path.cwd() / "config" / "experiment" / "disadvantage_bridge_scenarios.yaml"
    cfg = f"""includes:
  - {base_abs.as_posix()}
  - {bridge_abs.as_posix()}

experiment:
  name: smoke_{scenario_name}

virtual_point:
  enabled: true
  mode: normal
  anchor_mode: current_target
  action_frame: world
  return_info: true
  action_dim: 3
  d_long_range: [-1000.0, 1000.0]
  d_lat_range: [-600.0, 600.0]
  d_vert_range: [-150.0, 150.0]
  smoothing_alpha: 0.5
  dynamics_aware: false
  max_heading_rate: 0.2
  lookahead_steps: 5

guidance:
  params:
    altitude_hold:
      enabled: true
      k_alt: 0.8
      altitude_reference_m: 5000.0
      min_altitude_m: 1000.0
    roll_angle_protection:
      enabled: true
      max_roll_rad: 1.0472
      attenuation_power: 2.0

reward:
  w_range: 0.3
  w_angle: 0.5
  w_safety: 8.0
  w_boundary: 4.0
  w_overshoot: 1.0
  w_alive: 0.2
  w_closing: 0.5
  w_position_advantage: 0.2
  position_advantage_tail_deg: 30.0
  terminal_crash: -3000.0
  terminal_failure: -200.0
  min_altitude_m: 700.0

ppo:
  total_timesteps: {timesteps}
  rollout_steps: 1024
  minibatch_size: 256
  update_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  learning_rate: 3.0e-4
  max_grad_norm: 0.5
  normalize_advantage: true
  device: cpu

evaluation:
  eval_interval: 5000
  eval_episodes: 10
  eval_episodes_per_scenario: 20
  seeds: [0, 1, 2]
  save_trajectories: false

checkpoint:
  save_interval: 5000
  save_best: true
  save_last: true

curriculum:
  stage_gate_sr: 0.50
  stages:
    - progress_end: 1.0
      scenario_names:
        - {scenario_name}
"""
    out_dir = Path("outputs/_temp_smoke/configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"smoke_{scenario_name}.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return cfg_path


def run_training(scenario_name: str, cfg_path: Path, timesteps: int) -> dict:
    """Run a short training and return final eval metrics."""
    output_dir = f"outputs/_temp_smoke/smoke_{scenario_name}"
    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config", str(cfg_path),
        "--output-dir", output_dir,
        "--backend", "jsbsim",
        "--device", "cpu",
        "--total-timesteps", str(timesteps),
        "--eval-scenario", scenario_name,
    ]
    print(f"\n{'='*60}")
    print(f"Smoke testing {scenario_name} -> {output_dir}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)

    # Read final eval log row.
    eval_csv = Path(output_dir) / "logs" / "eval_log.csv"
    if not eval_csv.exists():
        return {"error": "eval_log.csv not found"}

    with open(eval_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"error": "eval_log.csv empty"}

    final = rows[-1]
    return {
        "success_rate": float(final.get("success_rate", 0.0)),
        "crash_rate": float(final.get("crash_rate", 0.0)),
        "timeout_rate": float(final.get("timeout_rate", 0.0)),
        "out_of_bounds_rate": float(final.get("out_of_bounds_rate", 0.0)),
        "mean_return": float(final.get("mean_return", 0.0)),
        "mean_final_range_m": float(final.get("mean_final_range_m", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    args = parser.parse_args()

    results = {}
    for scenario in args.scenarios:
        cfg_path = make_temp_config(scenario, args.timesteps)
        try:
            metrics = run_training(scenario, cfg_path, args.timesteps)
        except subprocess.CalledProcessError as e:
            metrics = {"error": str(e)}
        results[scenario] = metrics

    # Summary table.
    print("\n" + "=" * 100)
    print(f"{'Scenario':<28} {'Success%':>10} {'Crash%':>8} {'Timeout%':>10} {'OOB%':>8} {'MeanReturn':>12} {'MeanFinalRng':>14}")
    print("-" * 100)
    for scenario in args.scenarios:
        m = results.get(scenario, {})
        if "error" in m:
            print(f"{scenario:<28} ERROR: {m['error']}")
            continue
        print(
            f"{scenario:<28} "
            f"{m['success_rate']*100:>9.1f}% "
            f"{m['crash_rate']*100:>7.1f}% "
            f"{m['timeout_rate']*100:>9.1f}% "
            f"{m['out_of_bounds_rate']*100:>7.1f}% "
            f"{m['mean_return']:>11.1f} "
            f"{m['mean_final_range_m']:>13.0f}"
        )
    print("=" * 100)

    # Save JSON.
    out_file = Path("outputs/_temp_smoke/smoke_test_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved summary to: {out_file}")


if __name__ == "__main__":
    main()
