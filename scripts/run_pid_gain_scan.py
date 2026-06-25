#!/usr/bin/env python3
"""PID gain grid scan for break-turn overshoot reduction.

Scans a grid of (Kp_nz, Kd_nz) combinations with fixed Ki_nz=0.05,
measuring nz tracking RMSE, overshoot percentage, crash rate, and
recovery delay for each combination.

Usage:
    python scripts/run_pid_gain_scan.py [--seeds 5] [--jobs 4] [--dry-run]
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
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "compare_flight_control_base.yaml"
TASK_CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "task_break_turn.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pid_gain_scan"

# Gain grid
KP_NZ_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
KD_NZ_RANGE = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
KI_NZ_FIXED = 0.05


# ---------------------------------------------------------------------------
# Config helpers (mirrors run_flight_control_comparison.py pattern)
# ---------------------------------------------------------------------------

def _deep_update(target: dict, source: dict) -> dict:
    """Recursively update *target* with *source* without mutating source."""
    out = copy.deepcopy(target)
    for key, value in source.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(config_path: str) -> dict:
    """Load YAML config recursively resolving ``includes`` relative to the file."""
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


def build_pid_eval_config(
    base_config: dict,
    task_config: dict,
    kp_nz: float,
    kd_nz: float,
    ki_nz: float,
) -> dict:
    """Build an evaluation config for a fixed-gain PID controller on the break-turn task.

    Merges base + task configs, forces zero-offset VPP mode so guidance tracks
    the reference target directly, and sets the low-level controller gains.
    """
    config = copy.deepcopy(base_config)
    config = _deep_update(config, task_config)

    # Backend: inherit from base/task, default to jsbsim.
    backend = (
        task_config.get("backend")
        or base_config.get("backend")
        or "jsbsim"
    )
    config["backend"] = str(backend).lower()
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = config["backend"]
    config["env"]["use_jsbsim"] = config["backend"] == "jsbsim"

    # Zero-offset VPP -- PID-only operation.
    if "virtual_point" not in config:
        config["virtual_point"] = {}
    config["virtual_point"]["enabled"] = True
    config["virtual_point"]["mode"] = "zero_offset"
    config["virtual_point"]["action_dim"] = 4

    # Low-level controller with bank76 protection baseline.
    ll_cfg = copy.deepcopy(config.get("low_level_controller", {}))
    if isinstance(ll_cfg, str):
        ll_cfg = {}
    ll_cfg["type"] = "enhanced"
    ll_cfg["Kp_nz"] = kp_nz
    ll_cfg["Kd_nz"] = kd_nz
    ll_cfg["Ki_nz"] = ki_nz
    ll_cfg["enable_bank_angle_protection"] = True
    ll_cfg["max_bank_rad"] = 1.3264502315156905  # 76 deg
    ll_cfg["bank_violation_threshold"] = 25
    ll_cfg["bank_protection_nz_increment"] = 0.5
    ll_cfg.setdefault("altitude_hold_gain", 0.01)
    config["low_level_controller"] = ll_cfg

    # Guidance with lift compensation (bank76 profile).
    if "guidance" not in config:
        config["guidance"] = {}
    config["guidance"].setdefault("mode", "los_rate")
    config["guidance"]["post_process"] = {
        "enabled": True,
        "enable_lift_compensation": True,
        "lift_compensation_factor": 1.0,
        "lift_compensation_min_cos": 0.5,
    }

    # Policy block for config completeness.
    if "policy" not in config:
        config["policy"] = {}
    config["policy"]["action_dim"] = 4

    return config


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(values: list[float], targets: list[float]) -> float:
    """Root-mean-square error between values and targets."""
    if not values:
        return float("nan")
    sq_errors = [(v - t) ** 2 for v, t in zip(values, targets)]
    return float(math.sqrt(sum(sq_errors) / len(sq_errors)))


def compute_overshoot_pct(
    actual: list[float],
    cmd: list[float],
    settle_band: float = 0.05,
) -> float:
    """Compute overshoot as percentage of steps where |actual - cmd| > settle_band."""
    if not actual:
        return float("nan")
    count = sum(1 for a, c in zip(actual, cmd) if abs(a - c) > settle_band)
    return 100.0 * count / len(actual)


def compute_recovery_delay(
    actual: list[float],
    cmd: list[float],
    threshold: float = 0.1,
    min_gap: int = 3,
) -> float:
    """Compute mean number of steps to recover after a threshold crossing.

    A "crossing" is when |actual - cmd| exceeds *threshold*.
    Recovery delay is how many steps it takes to get back within threshold.
    """
    if not actual:
        return float("nan")
    delays: list[int] = []
    in_violation = False
    violation_start = 0
    for i, (a, c) in enumerate(zip(actual, cmd)):
        if abs(a - c) > threshold:
            if not in_violation:
                in_violation = True
                violation_start = i
        else:
            if in_violation:
                delay = i - violation_start
                if delay >= min_gap:
                    delays.append(delay)
                in_violation = False
    if in_violation:
        delay = len(actual) - violation_start
        if delay >= min_gap:
            delays.append(delay)
    return float(np.mean(delays)) if delays else 0.0


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_single_episode(
    config: dict,
    seed: int,
) -> dict[str, Any]:
    """Run a single evaluation episode and return per-step metrics.

    Returns a dict with keys:
        nz_rmse, overshoot_pct, recovery_delay, n_steps, crashed,
        termination_reason, cumulative_reward.
    """
    from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv

    env = BreakTurnEnv(config)
    obs = env.reset(seed=seed)

    nz_cmd_list: list[float] = []
    actual_nz_list: list[float] = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    action = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        nz_cmd_list.append(float(info.get("nz_cmd", 1.0)))
        # actual_nz from own_state if available, else from info
        own_state = info.get("own_state", {})
        actual_nz = float(own_state.get("nz_g", own_state.get("actual_nz", 1.0)))
        actual_nz_list.append(actual_nz)

        if steps >= env.max_steps:
            truncated = True
            break

    # Final state if loop never ran.
    if not info:
        own_state, _target_state = env._get_current_states()
        info = {"own_state": own_state}

    termination_reason = str(info.get("reason") or info.get("termination_reason") or "timeout")
    is_crash = bool(
        info.get("is_crash")
        or termination_reason.lower() in ("crash", "stall", "ground_collision", "altitude_violation")
    )

    nz_rmse = compute_rmse(actual_nz_list, nz_cmd_list)
    overshoot_pct = compute_overshoot_pct(actual_nz_list, nz_cmd_list)
    recovery_delay = compute_recovery_delay(actual_nz_list, nz_cmd_list)

    return {
        "nz_rmse": nz_rmse,
        "overshoot_pct": overshoot_pct,
        "recovery_delay": recovery_delay,
        "n_steps": steps,
        "crashed": is_crash,
        "termination_reason": termination_reason,
        "cumulative_reward": total_reward,
    }


def run_combo_worker(args: tuple) -> dict[str, Any]:
    """Worker function for ProcessPoolExecutor.

    Args:
        args: (kp_nz, kd_nz, ki_nz, seeds_list, base_config, task_config)

    Returns:
        Aggregated metrics dict for this gain combination.
    """
    kp_nz, kd_nz, ki_nz, seeds_list, base_config, task_config = args
    pid_config = build_pid_eval_config(base_config, task_config, kp_nz, kd_nz, ki_nz)

    episode_results: list[dict] = []
    crashes = 0
    for seed in seeds_list:
        try:
            result = run_single_episode(pid_config, seed)
            episode_results.append(result)
            if result["crashed"]:
                crashes += 1
        except Exception:
            traceback.print_exc()
            episode_results.append({
                "nz_rmse": float("nan"),
                "overshoot_pct": float("nan"),
                "recovery_delay": float("nan"),
                "n_steps": 0,
                "crashed": True,
                "termination_reason": "exception",
                "cumulative_reward": 0.0,
            })
            crashes += 1

    nz_rmse_vals = [r["nz_rmse"] for r in episode_results if not math.isnan(r["nz_rmse"])]
    overshoot_vals = [r["overshoot_pct"] for r in episode_results if not math.isnan(r["overshoot_pct"])]
    delay_vals = [r["recovery_delay"] for r in episode_results if not math.isnan(r["recovery_delay"])]

    return {
        "Kp_nz": kp_nz,
        "Kd_nz": kd_nz,
        "Ki_nz": ki_nz,
        "nz_rmse": float(np.mean(nz_rmse_vals)) if nz_rmse_vals else float("nan"),
        "nz_rmse_std": float(np.std(nz_rmse_vals)) if nz_rmse_vals else float("nan"),
        "overshoot_pct": float(np.mean(overshoot_vals)) if overshoot_vals else float("nan"),
        "overshoot_pct_std": float(np.std(overshoot_vals)) if overshoot_vals else float("nan"),
        "crashes": crashes,
        "total_episodes": len(seeds_list),
        "mean_recovery_delay": float(np.mean(delay_vals)) if delay_vals else float("nan"),
        "per_episode": episode_results,
    }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PID gain grid scan for break-turn overshoot reduction."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per (Kp_nz, Kd_nz) combination (default: 5).",
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
        "--kp-range",
        default=None,
        help="Comma-separated override for Kp_nz values.",
    )
    parser.add_argument(
        "--kd-range",
        default=None,
        help="Comma-separated override for Kd_nz values.",
    )
    return parser.parse_args()


def _validate_configs(
    kp_values: list[float],
    kd_values: list[float],
    base_config: dict,
    task_config: dict,
) -> bool:
    """Dry-run: validate that configs can be built and env can be constructed."""
    from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv

    for kp in kp_values:
        for kd in kd_values:
            pid_config = build_pid_eval_config(base_config, task_config, kp, kd, KI_NZ_FIXED)
            try:
                env = BreakTurnEnv(pid_config)
                obs = env.reset(seed=0)
                assert isinstance(obs, dict), f"Expected dict obs, got {type(obs)}"
                print(f"  [OK] Kp_nz={kp:.2f}, Kd_nz={kd:.2f}")
            except Exception as exc:
                print(f"  [FAIL] Kp_nz={kp:.2f}, Kd_nz={kd:.2f}: {exc}")
                return False
    return True


def _parse_float_list(raw: Optional[str], default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()

    kp_values = _parse_float_list(args.kp_range, KP_NZ_RANGE)
    kd_values = _parse_float_list(args.kd_range, KD_NZ_RANGE)
    seeds_per_combo = args.seeds
    seeds_list = list(range(seeds_per_combo))

    print(f"Kp_nz range: {kp_values}")
    print(f"Kd_nz range: {kd_values}")
    print(f"Ki_nz fixed: {KI_NZ_FIXED}")
    print(f"Seeds per combo: {seeds_per_combo}")
    print(f"Total combinations: {len(kp_values) * len(kd_values)}")
    print(f"Total episodes: {len(kp_values) * len(kd_values) * seeds_per_combo}")

    # Load configs once.
    print("\nLoading configs...")
    base_config = load_experiment_config(str(BASE_CONFIG_PATH))
    task_config = load_experiment_config(str(TASK_CONFIG_PATH))

    if args.dry_run:
        print("\n[DRY RUN] Validating configs...")
        ok = _validate_configs(kp_values, kd_values, base_config, task_config)
        if ok:
            print("All configs validated successfully.")
        else:
            print("Validation failed.")
            sys.exit(1)
        return

    # Generate work list using tuples for ProcessPoolExecutor.
    work: list[tuple] = []
    for kp in kp_values:
        for kd in kd_values:
            work.append((kp, kd, KI_NZ_FIXED, seeds_list, base_config, task_config))

    results: list[dict] = []
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
                        f"Kp={result['Kp_nz']:.2f} Kd={result['Kd_nz']:.2f} "
                        f"RMSE={result['nz_rmse']:.3f} "
                        f"Overshoot={result['overshoot_pct']:.1f}% "
                        f"Crashes={result['crashes']}/{result['total_episodes']}"
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
                    f"Kp={result['Kp_nz']:.2f} Kd={result['Kd_nz']:.2f} "
                    f"RMSE={result['nz_rmse']:.3f} "
                    f"Overshoot={result['overshoot_pct']:.1f}% "
                    f"Crashes={result['crashes']}/{result['total_episodes']}"
                )
            except Exception:
                traceback.print_exc()
                n_completed += 1

    elapsed = time.time() - t_start
    print(f"\nScan complete in {elapsed:.1f}s ({elapsed / max(n_total, 1):.1f}s per combo).")

    # Find best combination by min nz_rmse.
    valid_results = [r for r in results if not math.isnan(r["nz_rmse"])]
    best = min(valid_results, key=lambda r: r["nz_rmse"]) if valid_results else None

    output = {
        "metadata": {
            "Kp_range": kp_values,
            "Kd_range": kd_values,
            "Ki_fixed": KI_NZ_FIXED,
            "seeds_per_combo": seeds_per_combo,
            "task": "break_turn",
            "elapsed_s": round(elapsed, 1),
        },
        "results": sorted(results, key=lambda r: (r["Kp_nz"], r["Kd_nz"])),
        "best": (
            {
                "Kp_nz": best["Kp_nz"],
                "Kd_nz": best["Kd_nz"],
                "nz_rmse": best["nz_rmse"],
                "overshoot_pct": best["overshoot_pct"],
                "criterion": "min nz_rmse",
            }
            if best
            else None
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "scan_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if best is not None:
        print(
            f"Best: Kp_nz={best['Kp_nz']:.2f} Kd_nz={best['Kd_nz']:.2f} "
            f"RMSE={best['nz_rmse']:.4f} Overshoot={best['overshoot_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
