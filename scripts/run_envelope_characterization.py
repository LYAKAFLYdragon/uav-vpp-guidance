#!/usr/bin/env python3
"""Flight envelope characterization via boundary tracking.

Characterizes the safe operating envelope of the Enhanced PID controller
across three dimensions: bank angle, nz, and speed × altitude.

Method: boundary tracking — start from a known-safe point and step outward
until crashes are detected, then refine at higher resolution near the boundary.

Usage:
    python scripts/run_envelope_characterization.py [--phase 1|2|3|all] [--dry-run]
    python scripts/run_envelope_characterization.py --phase 1 --bank-range 60,82
    python scripts/run_envelope_characterization.py --phase 3 --speeds 120,170,220,270,320 --alts 2000,4000,6000,8000,10000
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
MW_TASK_PATH = REPO_ROOT / "config" / "experiment" / "task_multi_waypoint.yaml"
ST_TASK_PATH = REPO_ROOT / "config" / "experiment" / "task_sustained_turn.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "envelope_characterization"

# Default scan parameters
DEFAULT_BANK_RANGE = list(range(60, 84, 2))  # 60, 62, ..., 82
DEFAULT_NZ_RANGE = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
DEFAULT_SPEEDS = [120.0, 170.0, 220.0, 270.0, 320.0]
DEFAULT_ALTITUDES = [2000.0, 4000.0, 6000.0, 8000.0, 10000.0]


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


def build_envelope_config(
    base_config: dict,
    task_config: dict,
    max_bank_rad: Optional[float] = None,
    nz_cmd_override: Optional[float] = None,
    own_speed: Optional[float] = None,
    own_altitude: Optional[float] = None,
) -> dict:
    """Build a config for envelope characterization at a specific operating point."""
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

    ll_cfg = {
        "type": "enhanced",
        "enable_bank_angle_protection": True,
        "bank_violation_threshold": 25,
        "bank_protection_nz_increment": 0.5,
        "altitude_hold_gain": 0.01,
    }
    if max_bank_rad is not None:
        ll_cfg["max_bank_rad"] = max_bank_rad
        ll_cfg["enable_bank_angle_protection"] = True
    if nz_cmd_override is not None:
        ll_cfg["nz_max"] = max(7.0, nz_cmd_override + 2.0)
    if own_speed is not None:
        ll_cfg["target_speed_mps"] = own_speed
    if own_altitude is not None:
        ll_cfg["altitude_reference_m"] = own_altitude
    config["low_level_controller"] = ll_cfg

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
# Episode runners
# ---------------------------------------------------------------------------

def run_mw_episode(config: dict, seed: int) -> dict[str, Any]:
    """Run one multi-waypoint episode."""
    from uav_vpp_guidance.envs.multi_waypoint_tracking_env import MultiWaypointTrackingEnv

    env = MultiWaypointTrackingEnv(config)
    obs = env.reset(seed=seed)

    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    action = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
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
    wp = int(info.get("completed_waypoints", 0))

    return {
        "termination_reason": termination_reason,
        "crashed": is_crash,
        "completed_waypoints": wp,
        "n_steps": steps,
    }


def run_st_episode(config: dict, seed: int) -> dict[str, Any]:
    """Run one sustained-turn episode."""
    from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv

    env = SustainedTurnEnv(config)
    obs = env.reset(seed=seed)

    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    speed_hist: list[float] = []
    action = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        speed = float(
            (info.get("own_state") or {}).get("speed_mps", 250.0)
        )
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
    orbits = float(info.get("completed_orbits", 0.0))

    # Energy loss rate (m/s²)
    energy_loss = 0.0
    if len(speed_hist) >= 2:
        dt = float(env.env_config.get("high_level_dt", 0.2))
        energy_loss = (speed_hist[0] - speed_hist[-1]) / (len(speed_hist) * dt)

    return {
        "termination_reason": termination_reason,
        "crashed": is_crash,
        "completed_orbits": orbits,
        "n_steps": steps,
        "speed_start": speed_hist[0] if speed_hist else 250.0,
        "speed_end": speed_hist[-1] if speed_hist else 250.0,
        "energy_loss_rate_mps2": energy_loss,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_phase_worker(args: tuple) -> dict[str, Any]:
    """Generic worker for ProcessPoolExecutor."""
    (phase, point, seed, base_config, task_config) = args

    if phase == "bank":
        bank_deg, bank_rad = point
        cfg = build_envelope_config(
            base_config, task_config,
            max_bank_rad=bank_rad,
        )
        result = run_mw_episode(cfg, seed)
        result["param"] = {"bank_deg": bank_deg}
    elif phase == "nz":
        nz_cmd, = point
        cfg = build_envelope_config(
            base_config, task_config,
            nz_cmd_override=nz_cmd,
        )
        result = run_mw_episode(cfg, seed)
        result["param"] = {"nz_cmd": nz_cmd}
    elif phase == "speed_alt":
        speed, alt = point
        cfg = build_envelope_config(
            base_config, task_config,
            own_speed=speed,
            own_altitude=alt,
        )
        result = run_st_episode(cfg, seed)
        result["param"] = {"speed_mps": speed, "altitude_m": alt}
    else:
        result = {"crashed": True, "termination_reason": "unknown_phase"}
    result["phase"] = phase
    result["seed"] = seed
    return result


def run_phase_bank(
    base_config: dict,
    task_config: dict,
    bank_deg_values: list[int],
    seeds: list[int],
    jobs: int,
) -> list[dict]:
    """Phase 1: Bank boundary sweep."""
    print(f"\n{'='*60}")
    print("Phase 1: Bank boundary sweep")
    print(f"Bank range: {bank_deg_values[0]}° – {bank_deg_values[-1]}°")
    print(f"Seeds: {len(seeds)}")
    print(f"{'='*60}")

    work = []
    for bank_deg in bank_deg_values:
        bank_rad = math.radians(bank_deg)
        for seed in seeds:
            work.append(("bank", (bank_deg, bank_rad), seed, base_config, task_config))

    results = _execute_work(work, jobs)

    # Aggregate by bank angle.
    by_bank: dict[int, list[dict]] = {}
    for r in results:
        bd = int(r["param"]["bank_deg"])
        by_bank.setdefault(bd, []).append(r)

    summary = []
    for bd in sorted(by_bank):
        eps = by_bank[bd]
        crashes = sum(1 for e in eps if e["crashed"])
        wp_vals = [e["completed_waypoints"] for e in eps if not e["crashed"]]
        mean_wp = float(np.mean(wp_vals)) if wp_vals else 0.0
        stability = (
            "stable" if crashes <= 1 and mean_wp >= 3.0
            else "marginal" if crashes <= 2
            else "unstable"
        )
        summary.append({
            "bank_deg": bd,
            "crashes": crashes,
            "n_episodes": len(eps),
            "mean_wp": round(mean_wp, 2),
            "stability": stability,
        })
        print(f"  bank={bd}°: crash={crashes}/{len(eps)} mean_wp={mean_wp:.2f} [{stability}]")

    stable_banks = [s["bank_deg"] for s in summary if s["stability"] == "stable"]
    max_stable = max(stable_banks) if stable_banks else None
    if max_stable:
        print(f"  Max stable bank: {max_stable}°")
    else:
        print("  WARNING: No stable bank configuration found!")

    return summary


def run_phase_nz(
    base_config: dict,
    task_config: dict,
    nz_values: list[float],
    seeds: list[int],
    jobs: int,
) -> list[dict]:
    """Phase 2: NZ boundary sweep."""
    print(f"\n{'='*60}")
    print("Phase 2: NZ boundary sweep")
    print(f"nz range: [{nz_values[0]:.0f}, {nz_values[-1]:.0f}] g")
    print(f"Seeds: {len(seeds)}")
    print(f"{'='*60}")

    work = []
    for nz in nz_values:
        for seed in seeds:
            work.append(("nz", (nz,), seed, base_config, task_config))

    results = _execute_work(work, jobs)

    by_nz: dict[float, list[dict]] = {}
    for r in results:
        nz = float(r["param"]["nz_cmd"])
        by_nz.setdefault(nz, []).append(r)

    summary = []
    for nz in sorted(by_nz):
        eps = by_nz[nz]
        crashes = sum(1 for e in eps if e["crashed"])
        wp_vals = [e["completed_waypoints"] for e in eps if not e["crashed"]]
        mean_wp = float(np.mean(wp_vals)) if wp_vals else 0.0
        stability = (
            "stable" if crashes <= 1 and mean_wp >= 3.0
            else "marginal" if crashes <= 2
            else "unstable"
        )
        summary.append({
            "nz_cmd_g": nz,
            "crashes": crashes,
            "n_episodes": len(eps),
            "mean_wp": round(mean_wp, 2),
            "stability": stability,
        })
        print(f"  nz={nz:.0f}g: crash={crashes}/{len(eps)} mean_wp={mean_wp:.2f} [{stability}]")

    return summary


def run_phase_speed_alt(
    base_config: dict,
    task_config: dict,
    speeds: list[float],
    altitudes: list[float],
    seeds: list[int],
    jobs: int,
) -> list[dict]:
    """Phase 3: Speed × Altitude grid."""
    print(f"\n{'='*60}")
    print("Phase 3: Speed × Altitude grid")
    print(f"Speeds:  {speeds}")
    print(f"Alts:    {altitudes}")
    print(f"Seeds: {len(seeds)}")
    print(f"{'='*60}")

    work = []
    for speed in speeds:
        for alt in altitudes:
            for seed in seeds:
                work.append(("speed_alt", (speed, alt), seed, base_config, task_config))

    results = _execute_work(work, jobs)

    by_point: dict[tuple, list[dict]] = {}
    for r in results:
        key = (float(r["param"]["speed_mps"]), float(r["param"]["altitude_m"]))
        by_point.setdefault(key, []).append(r)

    summary = []
    for (speed, alt) in sorted(by_point):
        eps = by_point[(speed, alt)]
        crashes = sum(1 for e in eps if e["crashed"])
        orbit_vals = [e["completed_orbits"] for e in eps if not e["crashed"]]
        mean_orbits = float(np.mean(orbit_vals)) if orbit_vals else 0.0
        el_vals = [e["energy_loss_rate_mps2"] for e in eps if not e["crashed"]]
        mean_el = float(np.mean(el_vals)) if el_vals else float("nan")
        stability = "stable" if crashes == 0 else "marginal" if crashes <= 1 else "unstable"
        summary.append({
            "speed_mps": speed,
            "altitude_m": alt,
            "crashes": crashes,
            "n_episodes": len(eps),
            "mean_orbits": round(mean_orbits, 2),
            "energy_loss_rate_mps2": round(mean_el, 4),
            "stability": stability,
        })
        print(
            f"  v={speed:.0f}m/s h={alt:.0f}m: "
            f"crash={crashes}/{len(eps)} orbits={mean_orbits:.2f} [{stability}]"
        )

    return summary


def _execute_work(work: list, jobs: int) -> list[dict]:
    """Execute work list with optional parallelism."""
    results: list[dict] = []
    n_total = len(work)
    n_completed = 0

    if jobs > 1 and n_total > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_run_phase_worker, w): w for w in work}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    n_completed += 1
                except Exception:
                    traceback.print_exc()
                    n_completed += 1
    else:
        for w in work:
            try:
                result = _run_phase_worker(w)
                results.append(result)
                n_completed += 1
            except Exception:
                traceback.print_exc()
                n_completed += 1

    return results


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flight envelope characterization via boundary tracking."
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="Phase to run (1=bank, 2=nz, 3=speed×alt, all=1+2+3).",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds per operating point (default: 5).",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configs without running simulations.",
    )
    parser.add_argument(
        "--bank-range", default=None,
        help="Comma-separated bank angles, e.g. 60,64,68,72,76 (default: 60-82 step 2).",
    )
    parser.add_argument(
        "--nz-range", default=None,
        help="Comma-separated nz values, e.g. 2,3,4,5,6,7.",
    )
    parser.add_argument(
        "--speeds", default=None,
        help="Comma-separated speeds for Phase 3, e.g. 120,170,220,270,320.",
    )
    parser.add_argument(
        "--alts", default=None,
        help="Comma-separated altitudes for Phase 3, e.g. 2000,4000,6000,8000,10000.",
    )
    return parser.parse_args()


def _parse_float_list(raw: Optional[str], default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()
    run_phase = args.phase

    bank_values = (
        [int(x.strip()) for x in args.bank_range.split(",")]
        if args.bank_range
        else DEFAULT_BANK_RANGE
    )
    nz_values = _parse_float_list(args.nz_range, DEFAULT_NZ_RANGE)
    speeds = _parse_float_list(args.speeds, DEFAULT_SPEEDS)
    altitudes = _parse_float_list(args.alts, DEFAULT_ALTITUDES)
    seeds = list(range(args.seeds))

    print("Flight Envelope Characterization")
    print(f"Phase(s): {run_phase}")
    print(f"Seeds per point: {args.seeds}")
    print(f"Parallel workers: {args.jobs}")

    print("\nLoading configs...")
    base_config = load_experiment_config(str(BASE_CONFIG_PATH))
    mw_task_config = load_experiment_config(str(MW_TASK_PATH))
    st_task_config = load_experiment_config(str(ST_TASK_PATH))

    if args.dry_run:
        print("\n[DRY RUN] Validating configs...")
        from uav_vpp_guidance.envs.multi_waypoint_tracking_env import MultiWaypointTrackingEnv
        from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv

        if run_phase in ("1", "all"):
            for bd in bank_values[:3]:  # Check a few.
                cfg = build_envelope_config(base_config, mw_task_config, max_bank_rad=math.radians(bd))
                env = MultiWaypointTrackingEnv(cfg)
                obs = env.reset(seed=0)
                print(f"  [OK] bank={bd}° (MW)")
        if run_phase in ("2", "all"):
            for nz in nz_values[:2]:
                cfg = build_envelope_config(base_config, mw_task_config, nz_cmd_override=nz)
                env = MultiWaypointTrackingEnv(cfg)
                obs = env.reset(seed=0)
                print(f"  [OK] nz={nz:.0f}g (MW)")
        if run_phase in ("3", "all"):
            for speed in speeds[:2]:
                for alt in altitudes[:2]:
                    cfg = build_envelope_config(base_config, st_task_config, own_speed=speed, own_altitude=alt)
                    env = SustainedTurnEnv(cfg)
                    obs = env.reset(seed=0)
                    print(f"  [OK] v={speed:.0f}m/s h={alt:.0f}m (ST)")
        print("All validated successfully.")
        return

    t_start = time.time()
    output: dict[str, Any] = {
        "metadata": {
            "phases_run": run_phase,
            "seeds_per_point": args.seeds,
            "bank_range": bank_values,
            "nz_range": nz_values,
            "speeds": speeds,
            "altitudes": altitudes,
        },
    }

    if run_phase in ("1", "all"):
        output["phase1_bank"] = run_phase_bank(
            base_config, mw_task_config, bank_values, seeds, args.jobs,
        )

    if run_phase in ("2", "all"):
        output["phase2_nz"] = run_phase_nz(
            base_config, mw_task_config, nz_values, seeds, args.jobs,
        )

    if run_phase in ("3", "all"):
        output["phase3_speed_alt"] = run_phase_speed_alt(
            base_config, st_task_config, speeds, altitudes, seeds, args.jobs,
        )

    elapsed = time.time() - t_start
    output["metadata"]["elapsed_s"] = round(elapsed, 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "envelope.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nEnvelope characterization complete in {elapsed:.0f}s")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
