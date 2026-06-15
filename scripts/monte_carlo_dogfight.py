#!/usr/bin/env python3
"""Monte-Carlo close-range air combat evaluation.

Red (target/bandit) is the maneuver-library expert system.
Blue (ownship) flies one of four simple maneuvers:
    straight, turn, dive, climb.

Initial geometry follows four canonical configurations:
    favorable, head-on, challenging, neutral.

Usage:
    source activate jsbenv
    python scripts/monte_carlo_dogfight.py \
        --config config/env_bandit.yaml \
        --runs-per-cell 50 \
        --max-steps 100 \
        --range 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


GRAVITY = 9.80665
SCENARIO_TYPES = ["favorable", "head_on", "challenging", "neutral"]
BLUE_POLICIES = ["straight", "turn", "dive", "climb"]


def _resolve_config(config_path: str) -> dict:
    path = Path(config_path)
    root_dir = path.parent
    base = load_yaml_config(str(path))
    includes = base.pop("includes", []) or []
    merged = {}
    for inc in includes:
        inc_path = root_dir / inc
        if inc_path.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_path)))
    return merge_config(merged, base)


def _sample_from_union(rng: np.random.Generator, lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    """Sample uniformly from [lo1, hi1] U [lo2, hi2] (degrees)."""
    if rng.random() < 0.5:
        return float(rng.uniform(lo1, hi1))
    return float(rng.uniform(lo2, hi2))


def _build_scenario(
    scenario_type: str,
    rng: np.random.Generator,
    range_m: float,
    alt_m: float,
    speed_mps: float,
) -> dict:
    """Build an initial-geometry scenario dict."""
    if scenario_type == "favorable":
        q_T = _sample_from_union(rng, -180.0, -90.0, 90.0, 180.0)
        phi_A = float(rng.uniform(-90.0, 90.0))
    elif scenario_type == "head_on":
        q_T = float(rng.uniform(-90.0, 90.0))
        phi_A = float(rng.uniform(-90.0, 90.0))
    elif scenario_type == "challenging":
        q_T = float(rng.uniform(-90.0, 90.0))
        phi_A = _sample_from_union(rng, -180.0, -90.0, 90.0, 180.0)
    elif scenario_type == "neutral":
        q_T = _sample_from_union(rng, -180.0, -90.0, 90.0, 180.0)
        phi_A = _sample_from_union(rng, -180.0, -90.0, 90.0, 180.0)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    # Own (blue) at origin, target (red) at +x (north in NEU frame).
    own_pos = np.array([0.0, 0.0, alt_m], dtype=np.float64)
    tgt_pos = np.array([range_m, 0.0, alt_m], dtype=np.float64)

    # phi_A = 0  => blue pointing toward red (+x).
    # q_T   = 0  => red pointing toward blue, i.e. heading 180 deg.
    own_heading = phi_A
    tgt_heading = 180.0 - q_T

    return {
        "own_init": {
            "position_m": own_pos,
            "velocity_mps": speed_mps,
            "heading_deg": own_heading,
        },
        "target_init": {
            "position_m": tgt_pos,
            "velocity_mps": speed_mps,
            "heading_deg": tgt_heading,
        },
    }


class BlueSimpleController:
    """Simple closed-loop controller for blue-side baseline maneuvers."""

    def __init__(self, own_state: dict, policy: str, rng: np.random.Generator):
        self.policy = policy
        self.rng = rng
        self.v_ref = float(own_state.get("speed_mps", 250.0))
        self.alt_ref = float(own_state.get("altitude_m", 5000.0))

        self.phi_ref = 0.0
        self.theta_ref = 0.0
        if policy == "turn":
            self.phi_ref = float(np.radians(30.0 * (1.0 if rng.random() < 0.5 else -1.0)))
        elif policy == "dive":
            self.theta_ref = float(np.radians(-15.0))
        elif policy == "climb":
            self.theta_ref = float(np.radians(15.0))

    def command(self, own_state: dict, dt: float) -> dict:
        alt = float(own_state.get("altitude_m", self.alt_ref))
        v = float(own_state.get("speed_mps", self.v_ref))

        vel_ned = own_state.get("velocity_ned", np.zeros(3))
        vz = float(vel_ned[2])  # positive down

        attitude = own_state.get("attitude_rpy", np.zeros(3))
        phi = float(own_state.get("roll_rad", attitude[0]))
        theta = float(own_state.get("pitch_rad", attitude[1]))

        rates = own_state.get("body_rates_rps", np.zeros(3))
        p = float(rates[0])
        q = float(rates[1])

        # Roll channel: track bank angle for turn, wings-level otherwise.
        if self.policy == "turn":
            roll_rate_cmd = 2.0 * (self.phi_ref - phi) - 0.5 * p
        else:
            roll_rate_cmd = -0.5 * p

        # Pitch / normal load factor channel.
        if self.policy in ("dive", "climb"):
            # Track a fixed flight-path angle.
            nz_cmd = 1.0 + 2.0 * (self.theta_ref - theta) - 0.5 * q
        else:
            # Straight: hold initial altitude.
            az_cmd = 0.5 * (self.alt_ref - alt) - 0.5 * vz
            nz_cmd = 1.0 + az_cmd / GRAVITY

        # Throttle: hold reference speed.
        throttle_cmd = 0.5 + 0.02 * (self.v_ref - v)

        return {
            "nz_cmd": float(np.clip(nz_cmd, -2.0, 7.0)),
            "roll_rate_cmd": float(np.clip(roll_rate_cmd, -1.5, 1.5)),
            "throttle_cmd": float(np.clip(throttle_cmd, 0.0, 1.0)),
        }


def _run_episode(
    env: CloseRangeTrackingEnv,
    scenario: dict,
    policy: str,
    max_steps: int,
    rng: np.random.Generator,
) -> dict:
    """Run a single Monte-Carlo episode and return summary metrics."""
    obs = env.reset(scenario=scenario, seed=int(rng.integers(0, 1_000_000)))
    # If an ownship controller is configured, the simple blue policy is not used.
    use_simple_blue = env._own_controller is None
    if use_simple_blue:
        ctrl = BlueSimpleController(obs["own_state"], policy, rng)

    ranges: List[float] = []
    steps = 0
    steps_in_zone = 0
    reason = "max_steps"
    terminated = truncated = False
    info = {}

    # Success zone criteria from the env's termination checker.
    tc = env.termination_checker
    success_range_m = float(getattr(tc, "success_range_m", 900.0))
    success_ata_deg = float(getattr(tc, "success_ata_deg", 25.0))

    try:
        for step in range(max_steps):
            if use_simple_blue:
                own_state, _ = env._get_current_states()
                cmd = ctrl.command(own_state, 0.2)
                obs, reward, terminated, truncated, info = env.step(command_override=cmd)
            else:
                obs, reward, terminated, truncated, info = env.step()
            steps = step + 1
            range_m = float(info.get("range_m", np.nan))
            ata_deg = float(info.get("ata_deg", np.nan))
            ranges.append(range_m)
            if range_m <= success_range_m and ata_deg <= success_ata_deg:
                steps_in_zone += 1
            if terminated or truncated:
                reason = info.get("termination_info", {}).get("reason", "unknown")
                break
    except Exception as exc:
        return {
            "steps": steps,
            "reason": f"exception: {exc}",
            "min_range_m": np.nan,
            "final_range_m": np.nan,
            "steps_in_zone": 0,
            "success": False,
            "own_crash": False,
            "bandit_crash": False,
        }

    own_state = info.get("own_state", {})
    target_state = info.get("target_state", {})
    own_alt = float(own_state.get("altitude_m", 5000.0))
    tgt_alt = float(target_state.get("altitude_m", 5000.0))
    min_alt_crash = 500.0

    final_range = ranges[-1] if ranges else np.nan
    min_range = float(np.nanmin(ranges)) if ranges else np.nan

    return {
        "steps": steps,
        "reason": reason,
        "min_range_m": min_range,
        "final_range_m": final_range,
        "steps_in_zone": steps_in_zone,
        "success": reason == "success",
        "own_crash": reason == "crash" or own_alt < min_alt_crash,
        "bandit_crash": tgt_alt < min_alt_crash,
    }


def _summarize(records: List[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    successes = sum(1 for r in records if r["success"])
    own_crashes = sum(1 for r in records if r["own_crash"])
    bandit_crashes = sum(1 for r in records if r["bandit_crash"])
    min_ranges = [r["min_range_m"] for r in records if np.isfinite(r["min_range_m"])]
    final_ranges = [r["final_range_m"] for r in records if np.isfinite(r["final_range_m"])]
    steps = [r["steps"] for r in records]
    steps_in_zone = [r.get("steps_in_zone", 0) for r in records]
    reasons = Counter(r.get("reason", "unknown") for r in records)
    return {
        "n": n,
        "success_rate": successes / n,
        "own_crash_rate": own_crashes / n,
        "bandit_crash_rate": bandit_crashes / n,
        "mean_min_range_m": float(np.mean(min_ranges)) if min_ranges else np.nan,
        "mean_final_range_m": float(np.mean(final_ranges)) if final_ranges else np.nan,
        "mean_steps": float(np.mean(steps)) if steps else np.nan,
        "mean_steps_in_zone": float(np.mean(steps_in_zone)) if steps_in_zone else np.nan,
        "reason_counts": dict(reasons),
    }


def main():
    parser = argparse.ArgumentParser(description="Monte-Carlo dogfight evaluation")
    parser.add_argument("--config", type=str, default="config/env_bandit.yaml")
    parser.add_argument("--runs-per-cell", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--range", type=float, default=2000.0)
    parser.add_argument("--altitude", type=float, default=5000.0)
    parser.add_argument("--speed", type=float, default=250.0)
    parser.add_argument("--bandit-controller-type", type=str, default=None,
                        choices=["maneuver_library", "close_air_combat"],
                        help="Override bandit (red) controller type in config.")
    parser.add_argument("--own-controller-type", type=str, default=None,
                        choices=["maneuver_library", "close_air_combat"],
                        help="Override ownship (blue) controller type in config.")
    parser.add_argument("--controller-type", type=str, default=None,
                        choices=["maneuver_library", "close_air_combat"],
                        help="Convenience alias: set both own and bandit controller type.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-types",
        type=str,
        default=None,
        help="Comma-separated list of scenario types to run (default: all).",
    )
    args = parser.parse_args()

    config = _resolve_config(args.config)
    config["env"]["max_high_level_steps"] = args.max_steps
    controller_type = args.controller_type
    bandit_type = args.bandit_controller_type or controller_type
    own_type = args.own_controller_type or controller_type
    if bandit_type is not None:
        config["env"].setdefault("bandit", {})["controller_type"] = bandit_type
    if own_type is not None:
        config["env"].setdefault("own", {})["controller_type"] = own_type

    scenario_types = SCENARIO_TYPES
    if args.scenario_types is not None:
        scenario_types = [s.strip() for s in args.scenario_types.split(",") if s.strip()]
        invalid = set(scenario_types) - set(SCENARIO_TYPES)
        if invalid:
            raise ValueError(f"Unknown scenario types: {invalid}. Valid: {SCENARIO_TYPES}")

    env = CloseRangeTrackingEnv(config)
    rng = np.random.default_rng(args.seed)

    # If an ownship controller is configured, the simple blue policy is not used;
    # collapse the policy loop to a single "controller" entry.
    blue_policies = (
        ["controller"]
        if config["env"].get("own", {}).get("controller_type") is not None
        else BLUE_POLICIES
    )

    results: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    total_runs = len(scenario_types) * len(blue_policies) * args.runs_per_cell
    completed = 0

    print(f"Running {total_runs} Monte-Carlo episodes ({args.runs_per_cell} per cell)")
    print(f"Scenario range: {args.range:.0f} m, altitude: {args.altitude:.0f} m, speed: {args.speed:.0f} m/s")
    print("-" * 80)

    for scenario_type in scenario_types:
        for policy in blue_policies:
            for run in range(args.runs_per_cell):
                scenario = _build_scenario(
                    scenario_type, rng, args.range, args.altitude, args.speed
                )
                record = _run_episode(env, scenario, policy, args.max_steps, rng)
                results[scenario_type][policy].append(record)
                completed += 1
                if completed % 50 == 0 or completed == total_runs:
                    print(f"  Progress: {completed}/{total_runs}")

    env.close()

    # Print summary table.
    print("\n" + "=" * 80)
    print(f"{'Scenario':<14} {'Policy':<10} {'N':>4} {'Success%':>9} {'OwnCrash%':>10} {'BanditCrash%':>13} {'AvgZone':>8} {'Timeouts':>9} {'MeanMinRng':>11} {'MeanFinRng':>11}")
    print("-" * 95)
    overall_records = []
    for scenario_type in scenario_types:
        for policy in blue_policies:
            stats = _summarize(results[scenario_type][policy])
            overall_records.extend(results[scenario_type][policy])
            timeouts = stats.get("reason_counts", {}).get("timeout", 0)
            print(
                f"{scenario_type:<14} {policy:<10} {stats['n']:>4} "
                f"{stats['success_rate']*100:>8.1f}% {stats['own_crash_rate']*100:>9.1f}% "
                f"{stats['bandit_crash_rate']*100:>12.1f}% "
                f"{stats['mean_steps_in_zone']:>8.2f} {timeouts:>9} "
                f"{stats['mean_min_range_m']:>10.0f} {stats['mean_final_range_m']:>10.0f}"
            )

    overall = _summarize(overall_records)
    print("-" * 80)
    overall_timeouts = overall.get("reason_counts", {}).get("timeout", 0)
    print(
        f"{'OVERALL':<14} {'ALL':<10} {overall['n']:>4} "
        f"{overall['success_rate']*100:>8.1f}% {overall['own_crash_rate']*100:>9.1f}% "
        f"{overall['bandit_crash_rate']*100:>12.1f}% "
        f"{overall['mean_steps_in_zone']:>8.2f} {overall_timeouts:>9} "
        f"{overall['mean_min_range_m']:>10.0f} {overall['mean_final_range_m']:>10.0f}"
    )

    if args.output:
        out = {
            "args": vars(args),
            "summary": {
                "overall": overall,
                "cells": {
                    f"{s}_{p}": _summarize(results[s][p])
                    for s in scenario_types
                    for p in blue_policies
                },
            },
            "raw": {
                s: {p: results[s][p] for p in blue_policies}
                for s in scenario_types
            },
        }
        Path(args.output).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
