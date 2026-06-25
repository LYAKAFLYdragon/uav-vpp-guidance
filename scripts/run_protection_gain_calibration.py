#!/usr/bin/env python3
"""Protection gain calibration for flight-control supervisory settings.

Scans a grid of:
  - max_bank_rad
  - bank_protection_nz_increment
  - bank_protection_nz_increment_max
  - altitude_hold_gain

The task is configurable via ``--task-config``. Multi-waypoint uses
``completed_waypoints`` as the primary score, while sustained-turn uses
``completed_orbits``.
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.common.provenance import record_config_override_if_changed

BASE_CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "compare_flight_control_base.yaml"
DEFAULT_TASK_CONFIG_PATH = (
    REPO_ROOT / "config" / "experiment" / "task_multi_waypoint.yaml"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "protection_gain_calibration"

MAX_BANK_RAD_VALUES: list[tuple[float, float]] = [
    (1.3265, 76.0),
    (1.3614, 78.0),
    (1.3963, 80.0),
    (1.4312, 82.0),
]
BANK_PROTECTION_NZ_INCREMENT_VALUES = [0.3, 0.5, 0.7, 1.0]
BANK_PROTECTION_NZ_INCREMENT_MAX_VALUES = [0.5, 1.0, 1.5]
ALTITUDE_HOLD_GAIN_VALUES = [0.005, 0.01, 0.02, 0.03]

BASE_PROTECTION = {
    "enable_bank_angle_protection": True,
    "bank_violation_threshold": 25,
    "enable_altitude_hold": True,
}

ENV_CLASS_MAP = {
    "MultiWaypointTrackingEnv": (
        "uav_vpp_guidance.envs.multi_waypoint_tracking_env",
        "MultiWaypointTrackingEnv",
    ),
    "SustainedTurnEnv": (
        "uav_vpp_guidance.envs.sustained_turn_env",
        "SustainedTurnEnv",
    ),
    "BreakTurnEnv": (
        "uav_vpp_guidance.envs.break_turn_env",
        "BreakTurnEnv",
    ),
}

PRIMARY_METRIC_MAP = {
    "multi_waypoint": "completed_waypoints",
    "sustained_turn": "completed_orbits",
    "break_turn": "completed_orbits",
}


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


def merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


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
            merged = merge_config(merged, load_experiment_config(inc_full))
    return merge_config(merged, base_config)


def _get_nested_value(config: dict, key: str):
    current = config
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if not isinstance(current, dict):
        return None
    return current.get(parts[-1])


def _set_nested_value(config: dict, key: str, value: Any) -> None:
    current = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _set_config_override(config: dict, key: str, value: Any, source: str) -> None:
    old_value = _get_nested_value(config, key)
    _set_nested_value(config, key, value)
    record_config_override_if_changed(
        config,
        key=key,
        new_value=value,
        old_value=old_value,
        source=source,
    )


def resolve_env_class(config: dict):
    env_class_name = config.get("task", {}).get("env_class", "MultiWaypointTrackingEnv")
    try:
        module_name, attr_name = ENV_CLASS_MAP[env_class_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported env_class for calibration: {env_class_name}") from exc
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def extract_primary_metric(task_name: str, info: dict[str, Any]) -> tuple[str, float]:
    metric_name = PRIMARY_METRIC_MAP.get(task_name, "completed_waypoints")
    return metric_name, float(info.get(metric_name, 0.0))


def stability_threshold_for_task(task_name: str) -> float:
    if task_name == "sustained_turn":
        return 1.0
    if task_name == "break_turn":
        return 1.0
    return 3.0


def build_protection_config(
    base_config: dict,
    task_config: dict,
    max_bank_rad: float,
    bank_protection_nz_increment: float,
    bank_protection_nz_increment_max: float,
    altitude_hold_gain: float,
) -> dict:
    """Build evaluation config for a specific protection parameter set."""
    config = copy.deepcopy(base_config)
    config = _deep_update(config, task_config)
    source = "run_protection_gain_calibration.py:build_protection_config"

    backend = str(
        task_config.get("backend") or base_config.get("backend") or "jsbsim"
    ).lower()
    _set_config_override(config, "backend", backend, source)
    _set_config_override(config, "env.backend", backend, source)
    _set_config_override(config, "env.use_jsbsim", backend == "jsbsim", source)

    _set_config_override(config, "virtual_point.enabled", True, source)
    _set_config_override(config, "virtual_point.mode", "zero_offset", source)
    _set_config_override(config, "virtual_point.action_dim", 4, source)

    ll_cfg = copy.deepcopy(config.get("low_level_controller", {}))
    if isinstance(ll_cfg, str):
        ll_cfg = {}
    ll_cfg["type"] = "enhanced"
    ll_cfg.update(BASE_PROTECTION)
    ll_cfg["max_bank_rad"] = max_bank_rad
    ll_cfg["bank_protection_nz_increment"] = bank_protection_nz_increment
    ll_cfg["bank_protection_nz_increment_max"] = bank_protection_nz_increment_max
    ll_cfg["altitude_hold_gain"] = altitude_hold_gain
    for key, value in ll_cfg.items():
        _set_config_override(config, f"low_level_controller.{key}", value, source)

    _set_config_override(config, "policy.action_dim", 4, source)
    return config


def run_single_episode(config: dict, seed: int) -> dict[str, Any]:
    """Run one task episode with PID-only control."""
    env_class = resolve_env_class(config)
    task_name = str(config.get("task", {}).get("name", "multi_waypoint"))
    env = env_class(config)
    env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    action = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if steps >= env.max_steps:
            truncated = True
            break

    metric_name, metric_value = extract_primary_metric(task_name, info)
    termination_reason = str(
        info.get("reason") or info.get("termination_reason") or "timeout"
    )
    is_crash = bool(
        info.get("is_crash")
        or termination_reason.lower()
        in ("crash", "stall", "ground_collision", "altitude_violation")
    )
    final_alt = float((info.get("own_state") or {}).get("altitude_m", 5000.0))

    if hasattr(env, "close"):
        env.close()

    return {
        "task": task_name,
        "primary_metric_name": metric_name,
        "primary_metric": metric_value,
        "completed_waypoints": int(info.get("completed_waypoints", 0)),
        "completed_orbits": float(info.get("completed_orbits", 0.0)),
        "n_steps": steps,
        "crashed": is_crash,
        "termination_reason": termination_reason,
        "cumulative_reward": total_reward,
        "final_altitude_m": final_alt,
    }


def run_combo_worker(args: tuple) -> dict[str, Any]:
    """Worker for one (bank, nz_inc, nz_inc_max, alt_gain) combination."""
    (
        max_bank_rad,
        bank_deg,
        nz_inc,
        nz_inc_max,
        alt_gain,
        seeds_list,
        base_config,
        task_config,
    ) = args

    pid_config = build_protection_config(
        base_config,
        task_config,
        max_bank_rad,
        nz_inc,
        nz_inc_max,
        alt_gain,
    )
    task_name = str(pid_config.get("task", {}).get("name", "multi_waypoint"))
    metric_name = PRIMARY_METRIC_MAP.get(task_name, "completed_waypoints")

    episode_results: list[dict[str, Any]] = []
    crashes = 0
    for seed in seeds_list:
        try:
            result = run_single_episode(pid_config, seed)
            episode_results.append(result)
            if result["crashed"]:
                crashes += 1
        except Exception:
            traceback.print_exc()
            episode_results.append(
                {
                    "task": task_name,
                    "primary_metric_name": metric_name,
                    "primary_metric": 0.0,
                    "completed_waypoints": 0,
                    "completed_orbits": 0.0,
                    "n_steps": 0,
                    "crashed": True,
                    "termination_reason": "exception",
                    "cumulative_reward": 0.0,
                    "final_altitude_m": float("nan"),
                }
            )
            crashes += 1

    metric_vals = [float(r.get("primary_metric", 0.0)) for r in episode_results]
    mean_task_metric = float(np.mean(metric_vals)) if metric_vals else float("nan")
    std_task_metric = float(np.std(metric_vals)) if metric_vals else float("nan")

    return {
        "task": task_name,
        "task_metric_name": metric_name,
        "max_bank_deg": bank_deg,
        "max_bank_rad": round(max_bank_rad, 4),
        "bank_protection_nz_increment": nz_inc,
        "bank_protection_nz_increment_max": nz_inc_max,
        "altitude_hold_gain": alt_gain,
        "crashes": crashes,
        "total_episodes": len(seeds_list),
        "mean_task_metric": mean_task_metric,
        "std_task_metric": std_task_metric,
        "mean_wp": (
            mean_task_metric if metric_name == "completed_waypoints" else float("nan")
        ),
        "mean_completed_orbits": (
            mean_task_metric if metric_name == "completed_orbits" else float("nan")
        ),
        "per_episode": episode_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protection gain calibration for max stable bank angle."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per combination (default: 5).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs without running simulations.",
    )
    parser.add_argument(
        "--task-config",
        default=str(DEFAULT_TASK_CONFIG_PATH),
        help="Task config to merge on top of the base comparison config.",
    )
    parser.add_argument(
        "--bank-deg",
        default=None,
        help="Comma-separated bank angles to test (default: 76,78,80,82).",
    )
    parser.add_argument(
        "--nz-inc",
        default=None,
        help="Comma-separated bank_protection_nz_increment values.",
    )
    parser.add_argument(
        "--nz-inc-max",
        default=None,
        help="Comma-separated bank_protection_nz_increment_max values.",
    )
    parser.add_argument(
        "--alt-gain",
        default=None,
        help="Comma-separated altitude_hold_gain values.",
    )
    return parser.parse_args()


def _parse_float_list(raw: Optional[str], default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _validate_configs(
    bank_values: list[tuple[float, float]],
    nz_inc_values: list[float],
    nz_inc_max_values: list[float],
    alt_gain_values: list[float],
    base_config: dict,
    task_config: dict,
) -> bool:
    env_class = resolve_env_class(_deep_update(base_config, task_config))
    task_name = str(task_config.get("task", {}).get("name", "multi_waypoint"))

    for max_bank_rad, bank_deg in bank_values:
        for nz_inc in nz_inc_values:
            for nz_inc_max in nz_inc_max_values:
                for alt_gain in alt_gain_values:
                    cfg = build_protection_config(
                        base_config,
                        task_config,
                        max_bank_rad,
                        nz_inc,
                        nz_inc_max,
                        alt_gain,
                    )
                    try:
                        env = env_class(cfg)
                        obs = env.reset(seed=0)
                        assert isinstance(obs, dict)
                        if hasattr(env, "close"):
                            env.close()
                        print(
                            f"  [OK] task={task_name} bank={bank_deg}° nz_inc={nz_inc:.1f} "
                            f"nz_inc_max={nz_inc_max:.1f} alt_gain={alt_gain:.3f}"
                        )
                    except Exception as exc:
                        print(
                            f"  [FAIL] task={task_name} bank={bank_deg}° nz_inc={nz_inc:.1f} "
                            f"nz_inc_max={nz_inc_max:.1f} alt_gain={alt_gain:.3f}: {exc}"
                        )
                        return False
    return True


def main() -> None:
    args = _parse_args()

    if args.bank_deg is not None:
        bank_deg_list = [float(x.strip()) for x in args.bank_deg.split(",")]
        bank_values = [(math.radians(d), d) for d in bank_deg_list]
    else:
        bank_values = list(MAX_BANK_RAD_VALUES)

    nz_inc_values = _parse_float_list(args.nz_inc, BANK_PROTECTION_NZ_INCREMENT_VALUES)
    nz_inc_max_values = _parse_float_list(
        args.nz_inc_max, BANK_PROTECTION_NZ_INCREMENT_MAX_VALUES
    )
    alt_gain_values = _parse_float_list(args.alt_gain, ALTITUDE_HOLD_GAIN_VALUES)
    seeds_per_combo = args.seeds
    seeds_list = list(range(seeds_per_combo))

    n_combos = (
        len(bank_values)
        * len(nz_inc_values)
        * len(nz_inc_max_values)
        * len(alt_gain_values)
    )
    print(f"Bank angles: {[f'{d}°' for _, d in bank_values]}")
    print(f"bank_protection_nz_increment: {nz_inc_values}")
    print(f"bank_protection_nz_increment_max: {nz_inc_max_values}")
    print(f"altitude_hold_gain: {alt_gain_values}")
    print(f"Seeds per combo: {seeds_per_combo}")
    print(f"Total combinations: {n_combos}")
    print(f"Total episodes: {n_combos * seeds_per_combo}")

    print("\nLoading configs...")
    base_config = load_experiment_config(str(BASE_CONFIG_PATH))
    task_config = load_experiment_config(str(args.task_config))
    task_name = str(task_config.get("task", {}).get("name", "multi_waypoint"))
    task_metric_name = PRIMARY_METRIC_MAP.get(task_name, "completed_waypoints")
    stability_threshold = stability_threshold_for_task(task_name)
    print(f"Task: {task_name} ({task_metric_name})")

    if args.dry_run:
        print("\n[DRY RUN] Validating configs...")
        ok = _validate_configs(
            bank_values,
            nz_inc_values,
            nz_inc_max_values,
            alt_gain_values,
            base_config,
            task_config,
        )
        if ok:
            print("All configs validated successfully.")
        else:
            print("Validation failed.")
            sys.exit(1)
        return

    work: list[tuple] = []
    for max_bank_rad, bank_deg in sorted(bank_values, key=lambda x: -x[1]):
        for nz_inc in nz_inc_values:
            for nz_inc_max in nz_inc_max_values:
                for alt_gain in alt_gain_values:
                    work.append(
                        (
                            max_bank_rad,
                            bank_deg,
                            nz_inc,
                            nz_inc_max,
                            alt_gain,
                            seeds_list,
                            base_config,
                            task_config,
                        )
                    )

    results: list[dict[str, Any]] = []
    n_total = len(work)
    n_completed = 0
    t_start = time.time()

    if args.jobs > 1 and n_total > 1:
        print(f"\nRunning with {args.jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_combo = {executor.submit(run_combo_worker, w): w for w in work}
            for future in as_completed(future_to_combo):
                try:
                    result = future.result()
                    results.append(result)
                    n_completed += 1
                    print(
                        f"  [{n_completed}/{n_total}] "
                        f"task={result['task']} bank={result['max_bank_deg']}° "
                        f"nz_inc={result['bank_protection_nz_increment']:.1f} "
                        f"nz_inc_max={result['bank_protection_nz_increment_max']:.1f} "
                        f"alt_gain={result['altitude_hold_gain']:.3f} "
                        f"{result['task_metric_name']}={result['mean_task_metric']:.2f} "
                        f"crashes={result['crashes']}/{result['total_episodes']}"
                    )
                except Exception:
                    traceback.print_exc()
                    n_completed += 1
    else:
        print("\nRunning sequentially...")
        for w in work:
            try:
                result = run_combo_worker(w)
                results.append(result)
                n_completed += 1
                print(
                    f"  [{n_completed}/{n_total}] "
                    f"task={result['task']} bank={result['max_bank_deg']}° "
                    f"nz_inc={result['bank_protection_nz_increment']:.1f} "
                    f"nz_inc_max={result['bank_protection_nz_increment_max']:.1f} "
                    f"alt_gain={result['altitude_hold_gain']:.3f} "
                    f"{result['task_metric_name']}={result['mean_task_metric']:.2f} "
                    f"crashes={result['crashes']}/{result['total_episodes']}"
                )
            except Exception:
                traceback.print_exc()
                n_completed += 1

    elapsed = time.time() - t_start
    print(f"\nCalibration complete in {elapsed:.1f}s.")

    stable = [
        r
        for r in results
        if (
            r["crashes"] <= 1
            and not math.isnan(r["mean_task_metric"])
            and r["mean_task_metric"] >= stability_threshold
        )
    ]
    best_stable = max(stable, key=lambda r: r["max_bank_deg"]) if stable else None

    output = {
        "metadata": {
            "bank_deg_values": [d for _, d in bank_values],
            "nz_inc_values": nz_inc_values,
            "nz_inc_max_values": nz_inc_max_values,
            "alt_gain_values": alt_gain_values,
            "seeds_per_combo": seeds_per_combo,
            "task": task_name,
            "task_metric_name": task_metric_name,
            "stability_threshold": stability_threshold,
            "elapsed_s": round(elapsed, 1),
        },
        "results": sorted(
            results,
            key=lambda r: (r["max_bank_deg"], r["crashes"], -r["mean_task_metric"]),
        ),
        "best_stable": (
            {
                "task": best_stable["task"],
                "task_metric_name": best_stable["task_metric_name"],
                "max_bank_deg": best_stable["max_bank_deg"],
                "bank_protection_nz_increment": best_stable["bank_protection_nz_increment"],
                "bank_protection_nz_increment_max": best_stable["bank_protection_nz_increment_max"],
                "altitude_hold_gain": best_stable["altitude_hold_gain"],
                "mean_task_metric": best_stable["mean_task_metric"],
                "crashes": best_stable["crashes"],
            }
            if best_stable
            else None
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"calibration_results_{task_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if best_stable is not None:
        print(
            f"Best stable config: task={best_stable['task']} bank={best_stable['max_bank_deg']}° "
            f"nz_inc={best_stable['bank_protection_nz_increment']:.1f} "
            f"nz_inc_max={best_stable['bank_protection_nz_increment_max']:.1f} "
            f"alt_gain={best_stable['altitude_hold_gain']:.3f} "
            f"{best_stable['task_metric_name']}={best_stable['mean_task_metric']:.2f}"
        )
    else:
        print("WARNING: No stable configuration found above crash threshold.")


if __name__ == "__main__":
    main()
