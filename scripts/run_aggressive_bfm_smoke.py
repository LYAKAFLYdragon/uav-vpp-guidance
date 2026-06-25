#!/usr/bin/env python3
"""Aggressive BFM smoke test — validate flight controller under high-g yo-yo maneuvers.

Runs the Enhanced PID controller against an aggressively maneuvering yo-yo target
and checks for crashes, nz limits, and spiral dive events.

Usage:
    python scripts/run_aggressive_bfm_smoke.py [--seeds 5] [--dry-run]
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
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "compare_flight_control_base.yaml"
TASK_CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "task_aggressive_bfm.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "aggressive_bfm_smoke"


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


def build_pid_eval_config(base_config: dict, task_config: dict) -> dict:
    """Build PID eval config for aggressive BFM yo-yo task."""
    config = copy.deepcopy(base_config)
    config = _deep_update(config, task_config)

    backend = task_config.get("backend") or base_config.get("backend") or "jsbsim"
    config["backend"] = str(backend).lower()
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = config["backend"]
    config["env"]["use_jsbsim"] = config["backend"] == "jsbsim"

    if "virtual_point" not in config:
        config["virtual_point"] = {}
    config["virtual_point"]["enabled"] = True
    config["virtual_point"]["mode"] = "zero_offset"
    config["virtual_point"]["action_dim"] = 4

    config["low_level_controller"] = {
        "type": "enhanced",
        "enable_bank_angle_protection": True,
        "max_bank_rad": 1.3264502315156905,
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

    return config


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_single_episode(config: dict, seed: int) -> dict[str, Any]:
    """Run one aggressive BFM episode with PID-only control."""
    from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv

    env = BreakTurnEnv(config)
    obs = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    altitude_hist: list[float] = []
    nz_hist: list[float] = []
    speed_hist: list[float] = []

    action = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        own_state = info.get("own_state", {})
        alt = float(own_state.get("altitude_m", 5000.0))
        nz = float(own_state.get("nz_g", own_state.get("actual_nz", 1.0)))
        speed = float(
            own_state.get("speed_mps", 250.0)
            or np.linalg.norm(own_state.get("velocity_vector_mps", [0.0, 0.0, 0.0]))
        )
        altitude_hist.append(alt)
        nz_hist.append(nz)
        speed_hist.append(speed)

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

    # Spiral dive detection: sustained altitude loss > 2000m.
    max_alt = max(altitude_hist) if altitude_hist else 5000.0
    min_alt = min(altitude_hist) if altitude_hist else 5000.0
    alt_loss = max_alt - min_alt
    spiral_dive = alt_loss > 2000.0

    return {
        "termination_reason": termination_reason,
        "crashed": is_crash,
        "n_steps": steps,
        "cumulative_reward": total_reward,
        "nz_max": float(max(nz_hist)) if nz_hist else 1.0,
        "nz_min": float(min(nz_hist)) if nz_hist else 1.0,
        "altitude_loss_m": float(alt_loss),
        "spiral_dive": spiral_dive,
        "altitude_start_m": float(altitude_hist[0]) if altitude_hist else 5000.0,
        "altitude_end_m": float(altitude_hist[-1]) if altitude_hist else 5000.0,
        "speed_start_mps": float(speed_hist[0]) if speed_hist else 250.0,
        "speed_end_mps": float(speed_hist[-1]) if speed_hist else 250.0,
    }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggressive BFM smoke test for flight envelope limits."
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds (default: 5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configs without running simulations.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds_list = list(range(args.seeds))

    print(f"Aggressive BFM Smoke Test")
    print(f"Seeds: {args.seeds}")
    print(f"Task config: {TASK_CONFIG_PATH}")

    print("\nLoading configs...")
    base_config = load_experiment_config(str(BASE_CONFIG_PATH))
    task_config = load_experiment_config(str(TASK_CONFIG_PATH))

    if args.dry_run:
        print("\n[DRY RUN] Validating config...")
        from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv
        pid_config = build_pid_eval_config(base_config, task_config)
        try:
            env = BreakTurnEnv(pid_config)
            obs = env.reset(seed=0)
            assert isinstance(obs, dict)
            maneuver_type = task_config.get("task", {}).get("break_turn", {}).get("maneuver_type", "yo_yo")
            print(f"  [OK] Aggressive BFM ({maneuver_type}) config validated")
        except Exception as exc:
            print(f"  [FAIL] Config validation failed: {exc}")
            sys.exit(1)
        return

    pid_config = build_pid_eval_config(base_config, task_config)

    results: list[dict] = []
    crashes = 0
    spiral_dives = 0
    nz_violations = 0
    t_start = time.time()

    print(f"\nRunning {args.seeds} episodes...")
    for seed in seeds_list:
        try:
            result = run_single_episode(pid_config, seed)
            results.append(result)
            if result["crashed"]:
                crashes += 1
            if result["spiral_dive"]:
                spiral_dives += 1
            if result["nz_max"] >= 9.0:
                nz_violations += 1
            print(
                f"  [seed {seed}] "
                f"reason={result['termination_reason']} "
                f"nz_max={result['nz_max']:.1f}g "
                f"alt_loss={result['altitude_loss_m']:.0f}m "
                f"spiral={'YES' if result['spiral_dive'] else 'no'}"
            )
        except Exception:
            traceback.print_exc()
            results.append({
                "termination_reason": "exception",
                "crashed": True,
                "n_steps": 0,
                "nz_max": 0.0,
                "nz_min": 0.0,
                "altitude_loss_m": 0.0,
                "spiral_dive": False,
                "altitude_start_m": float("nan"),
                "altitude_end_m": float("nan"),
                "speed_start_mps": 0.0,
                "speed_end_mps": 0.0,
            })
            crashes += 1

    elapsed = time.time() - t_start
    n_total = len(seeds_list)

    nz_max_all = max((r["nz_max"] for r in results), default=0.0)
    alt_loss_max = max((r["altitude_loss_m"] for r in results), default=0.0)

    passed = crashes == 0 and nz_violations == 0 and spiral_dives == 0

    print(f"\nSmoke test {'PASSED' if passed else 'FAILED'}")
    print(f"  Crashes:       {crashes}/{n_total}")
    print(f"  Spiral dives:  {spiral_dives}/{n_total}")
    print(f"  nz violations: {nz_violations}/{n_total}")
    print(f"  nz_max overall: {nz_max_all:.1f}g")
    print(f"  Max alt loss:   {alt_loss_max:.0f}m")

    output = {
        "metadata": {
            "task": "aggressive_bfm_yo_yo",
            "seeds": args.seeds,
            "elapsed_s": round(elapsed, 1),
            "acceptance_criteria": {
                "crashes_eq_0": "crash / 5 = 0",
                "nz_max_lt_9g": "nz_max < 9g",
                "no_spiral_dive": "altitude loss <= 2000m",
            },
        },
        "summary": {
            "total_episodes": n_total,
            "crashes": crashes,
            "spiral_dives": spiral_dives,
            "nz_violations": nz_violations,
            "nz_max_overall": round(nz_max_all, 2),
            "max_altitude_loss_m": round(alt_loss_max, 2),
            "passed": passed,
        },
        "per_episode": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "smoke_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
