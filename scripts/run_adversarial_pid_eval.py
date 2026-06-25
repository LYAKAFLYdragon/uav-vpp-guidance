#!/usr/bin/env python3
"""Adversarial flight-control evaluation — measure PID crash/kill rates
against opponent aircraft in combat scenarios.

Runs enhanced PID controller in CloseRangeTrackingEnv with adversarial
opponent policies, measuring crash rate and combat outcomes.

Usage:
    python scripts/run_adversarial_pid_eval.py [--scenarios small] [--seeds 5] [--jobs 4] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "compare_flight_control_base.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "adversarial_pid_eval"

# Small adversarial scenario grid (HP damage × close_range × opponent_stage).
# Each scenario is a (name, own_init, target_init, metadata) tuple.
SMALL_SCENARIOS = [
    {
        "name": "head_on_neutral",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [2000.0, 0.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 180.0},
        "metadata": {"scenario_type": "head_on", "difficulty": "neutral"},
    },
    {
        "name": "beam_defensive",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [0.0, 2000.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 90.0},
        "metadata": {"scenario_type": "beam", "difficulty": "defensive"},
    },
    {
        "name": "tail_chase_offensive",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [1000.0, 0.0, 5000.0], "velocity_mps": 220.0, "heading_deg": 0.0},
        "metadata": {"scenario_type": "tail_chase", "difficulty": "offensive"},
    },
    {
        "name": "tail_chase_defensive",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 0.0},
        "target_init": {"position_m": [1000.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "metadata": {"scenario_type": "tail_chase", "difficulty": "defensive"},
    },
    {
        "name": "crossing_90deg",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [0.0, 1000.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 90.0},
        "metadata": {"scenario_type": "crossing", "difficulty": "neutral", "crossing_angle_deg": 90.0},
    },
]

MEDIUM_SCENARIOS = SMALL_SCENARIOS + [
    {
        "name": "head_on_fast",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [3000.0, 0.0, 5000.0], "velocity_mps": 300.0, "heading_deg": 180.0},
        "metadata": {"scenario_type": "head_on", "difficulty": "hard"},
    },
    {
        "name": "beam_neutral_high",
        "own_init": {"position_m": [0.0, 0.0, 8000.0], "velocity_mps": 250.0, "heading_deg": 0.0},
        "target_init": {"position_m": [0.0, 2000.0, 8000.0], "velocity_mps": 250.0, "heading_deg": 90.0},
        "metadata": {"scenario_type": "beam", "difficulty": "neutral", "altitude": "high"},
    },
    {
        "name": "tail_chase_low",
        "own_init": {"position_m": [0.0, 0.0, 2000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [1000.0, 0.0, 2000.0], "velocity_mps": 220.0, "heading_deg": 0.0},
        "metadata": {"scenario_type": "tail_chase", "difficulty": "neutral", "altitude": "low"},
    },
    {
        "name": "crossing_30deg",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [866.0, 500.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 30.0},
        "metadata": {"scenario_type": "crossing", "difficulty": "easy", "crossing_angle_deg": 30.0},
    },
    {
        "name": "crossing_60deg",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 280.0, "heading_deg": 0.0},
        "target_init": {"position_m": [500.0, 866.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 60.0},
        "metadata": {"scenario_type": "crossing", "difficulty": "medium", "crossing_angle_deg": 60.0},
    },
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_update(target: dict, source: dict) -> dict:
    out = copy.deepcopy(target)
    for key, value in source.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(config_path: str) -> dict:
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: dict = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = _deep_update(merged, load_experiment_config(inc_full))
    return _deep_update(merged, base_config)


def build_pid_eval_config(base_config: dict) -> dict:
    """Build PID eval config with bank76_alt_hold_lift_smooth profile."""
    config = copy.deepcopy(base_config)
    config["backend"] = "jsbsim"
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = "jsbsim"
    config["env"]["use_jsbsim"] = True
    config["env"]["strict_backend"] = True

    if "virtual_point" not in config:
        config["virtual_point"] = {}
    config["virtual_point"]["enabled"] = True
    config["virtual_point"]["mode"] = "zero_offset"
    config["virtual_point"]["action_dim"] = 4

    config["low_level_controller"] = {
        "type": "enhanced",
        "enable_bank_angle_protection": True,
        "max_bank_rad": 1.3264502315156905,  # 76°
        "bank_violation_threshold": 25,
        "bank_protection_nz_increment": 0.5,
        "altitude_hold_gain": 0.01,
    }
    if "guidance" not in config:
        config["guidance"] = {}
    config["guidance"]["mode"] = "los_rate"
    config["guidance"]["post_process"] = {
        "enabled": True,
        "enable_lift_compensation": True,
        "lift_compensation_factor": 1.0,
        "lift_compensation_min_cos": 0.5,
    }

    if "policy" not in config:
        config["policy"] = {}
    config["policy"]["action_dim"] = 4

    if "task" not in config:
        config["task"] = {}
    config["task"]["name"] = "adversarial"
    config["task"]["env_class"] = "CloseRangeTrackingEnv"

    return config


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_single_episode(config: dict, scenario: dict, seed: int) -> dict[str, Any]:
    """Run one adversarial episode with PID-only control."""
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(scenario=scenario, seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    min_altitude = float("inf")
    max_nz = 0.0

    action = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        own_state = info.get("own_state", {})
        alt = float(own_state.get("altitude_m", 5000.0))
        nz = float(own_state.get("nz_g", own_state.get("actual_nz", 1.0)))
        min_altitude = min(min_altitude, alt)
        max_nz = max(max_nz, nz)

        if steps >= env.max_steps:
            truncated = True
            break

    termination_reason = str(
        info.get("reason") or info.get("termination_reason") or "timeout"
    )
    is_crash = bool(
        info.get("is_crash")
        or termination_reason.lower()
        in ("crash", "stall", "ground_collision", "altitude_violation")
    )
    is_success = bool(info.get("is_success", False))

    return {
        "termination_reason": termination_reason,
        "crashed": is_crash,
        "success": is_success,
        "n_steps": steps,
        "cumulative_reward": total_reward,
        "min_altitude_m": min_altitude,
        "max_nz_g": max_nz,
        "range_m": float(info.get("range_m", float("nan"))),
    }


def run_scenario_worker(args: tuple) -> dict[str, Any]:
    """Worker for one scenario × seed combination."""
    scenario, seed, base_config = args
    pid_config = build_pid_eval_config(base_config)

    try:
        result = run_single_episode(pid_config, scenario, seed)
        result["scenario"] = scenario.get("name", "unknown")
        result["seed"] = seed
        return result
    except Exception:
        traceback.print_exc()
        return {
            "scenario": scenario.get("name", "unknown"),
            "seed": seed,
            "crashed": True,
            "success": False,
            "termination_reason": "exception",
            "n_steps": 0,
            "cumulative_reward": 0.0,
            "min_altitude_m": float("nan"),
            "max_nz_g": float("nan"),
            "range_m": float("nan"),
        }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial PID flight-control evaluation."
    )
    parser.add_argument(
        "--scenarios", type=str, default="small",
        choices=["small", "medium"],
        help="Scenario set size (default: small = 5 scenarios).",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds per scenario (default: 5).",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configs without running simulations.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_set = SMALL_SCENARIOS if args.scenarios == "small" else MEDIUM_SCENARIOS
    seeds_list = list(range(args.seeds))

    print(f"Scenarios: {args.scenarios} ({len(scenario_set)} scenarios)")
    print(f"Seeds per scenario: {args.seeds}")
    print(f"Total episodes: {len(scenario_set) * args.seeds}")

    print("\nLoading configs...")
    base_config = load_experiment_config(str(BASE_CONFIG_PATH))

    if args.dry_run:
        print("\n[DRY RUN] Validating configs and scenarios...")
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        pid_config = build_pid_eval_config(base_config)
        for scenario in scenario_set:
            try:
                env = CloseRangeTrackingEnv(pid_config)
                obs = env.reset(scenario=scenario, seed=0)
                assert isinstance(obs, dict)
                print(f"  [OK] {scenario['name']}")
            except Exception as exc:
                print(f"  [FAIL] {scenario['name']}: {exc}")
                sys.exit(1)
        print("All scenarios validated successfully.")
        return

    # Build work list.
    work: list[tuple] = []
    for scenario in scenario_set:
        for seed in seeds_list:
            work.append((scenario, seed, base_config))

    all_results: list[dict] = []
    n_total = len(work)
    n_completed = 0
    t_start = time.time()

    if args.jobs > 1 and n_total > 1:
        print(f"\nRunning with {args.jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_scenario_worker, w): w for w in work}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                    n_completed += 1
                    print(
                        f"  [{n_completed}/{n_total}] {result['scenario']}/seed{result['seed']} "
                        f"reason={result['termination_reason']}"
                    )
                except Exception:
                    traceback.print_exc()
                    n_completed += 1
    else:
        print("\nRunning sequentially...")
        for w in work:
            try:
                result = run_scenario_worker(w)
                all_results.append(result)
                n_completed += 1
                print(
                    f"  [{n_completed}/{n_total}] {result['scenario']}/seed{result['seed']} "
                    f"reason={result['termination_reason']}"
                )
            except Exception:
                traceback.print_exc()
                n_completed += 1

    elapsed = time.time() - t_start

    # Aggregate by scenario.
    by_scenario: dict[str, list[dict]] = {}
    for r in all_results:
        by_scenario.setdefault(r["scenario"], []).append(r)

    scenario_summaries = []
    total_crashes = 0
    total_eps = len(all_results)
    for name, eps in sorted(by_scenario.items()):
        crashes = sum(1 for e in eps if e["crashed"])
        successes = sum(1 for e in eps if e["success"])
        total_crashes += crashes
        scenario_summaries.append({
            "scenario": name,
            "n_episodes": len(eps),
            "crashes": crashes,
            "crash_rate": crashes / len(eps),
            "successes": successes,
            "success_rate": successes / len(eps),
            "mean_steps": float(np.mean([e["n_steps"] for e in eps])),
        })
        print(f"  {name}: crash={crashes}/{len(eps)} ({crashes/len(eps):.0%}) "
              f"success={successes}/{len(eps)}")

    overall_crash_rate = total_crashes / total_eps if total_eps else 0.0
    print(f"\nOverall crash rate: {total_crashes}/{total_eps} = {overall_crash_rate:.1%}")

    output = {
        "metadata": {
            "scenarios_set": args.scenarios,
            "n_scenarios": len(scenario_set),
            "seeds_per_scenario": args.seeds,
            "total_episodes": total_eps,
            "elapsed_s": round(elapsed, 1),
            "bank_profile": "bank76_alt_hold_lift_smooth",
        },
        "scenario_summaries": scenario_summaries,
        "overall": {
            "total_episodes": total_eps,
            "total_crashes": total_crashes,
            "crash_rate": overall_crash_rate,
        },
        "acceptance": {
            "crash_rate_lt_10pct": overall_crash_rate < 0.10,
        },
        "per_episode": all_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Acceptance (crash < 10%): {'PASS' if overall_crash_rate < 0.10 else 'FAIL'}")


if __name__ == "__main__":
    main()
