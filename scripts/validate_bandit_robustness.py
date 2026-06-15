#!/usr/bin/env python3
"""Robustness validation for the JSBSim bandit maneuver baseline (no missiles).

Runs multiple episodes across difficulties and seeds, records why episodes end,
and checks that the bandit stays inside a safe flight envelope.  Designed for
close-range dogfight baseline verification where missiles are disabled.

The blue aircraft can be controlled by either mild random actions or a simple
3D proportional-navigation (PN) pursuit law to verify that the bandit provides
a challenging but stable adversary.

Usage:
    source activate jsbenv
    python scripts/validate_bandit_robustness.py \
        --config config/env_bandit.yaml \
        --policy pn \
        --episodes 30 \
        --max-steps 300
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.guidance.proportional_navigation import ProportionalNavigationGuidance
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def _resolve_config(config_path: str, overrides: dict | None = None) -> dict:
    """Load config and inline includes the same way training scripts do."""
    path = Path(config_path)
    root_dir = path.parent
    base = load_yaml_config(str(path))
    includes = base.pop("includes", []) or []
    merged = {}
    for inc in includes:
        inc_path = root_dir / inc
        if inc_path.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_path)))
    config = merge_config(merged, base)
    if overrides:
        config = merge_config(config, overrides)
    return config


def _build_scenario(scenario: str, rng: np.random.Generator) -> dict | None:
    """Build an initial-geometry scenario for close-range dogfight validation."""
    if scenario == "default":
        return None
    if scenario == "tail_chase":
        return {
            "own_init": {
                "position_m": np.array([0.0, 0.0, 5000.0], dtype=np.float64),
                "velocity_mps": 260.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": np.array([2000.0, 0.0, 5000.0], dtype=np.float64),
                "velocity_mps": 240.0,
                "heading_deg": 0.0,
            },
        }
    if scenario == "head_on":
        return {
            "own_init": {
                "position_m": np.array([0.0, 0.0, 5000.0], dtype=np.float64),
                "velocity_mps": 250.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": np.array([2000.0, 0.0, 5000.0], dtype=np.float64),
                "velocity_mps": 250.0,
                "heading_deg": 180.0,
            },
        }
    raise ValueError(f"Unknown scenario: {scenario}")


def _run_episode(
    env: CloseRangeTrackingEnv,
    seed: int,
    max_steps: int,
    policy: str,
    action_scale: float,
    scenario: dict | None = None,
) -> Dict[str, object]:
    """Run a single episode and return telemetry summary."""
    rng = np.random.default_rng(seed)
    obs = env.reset(scenario=scenario, seed=seed)

    pn_guidance: ProportionalNavigationGuidance | None = None
    if policy == "pn":
        pn_guidance = ProportionalNavigationGuidance(env.config.get("guidance", {}))
        pn_guidance.reset()

    min_bandit_alt = float("inf")
    max_bandit_alt = float("-inf")
    min_bandit_speed = float("inf")
    max_bandit_speed = float("-inf")
    maneuvers_seen: List[str] = []
    last_maneuver: str | None = None
    ranges: List[float] = []
    steps = 0
    reason = "max_steps"
    exception: str | None = None

    try:
        for step in range(max_steps):
            if policy == "random":
                action = rng.uniform(-action_scale, action_scale, size=3).astype(np.float64)
                obs, reward, terminated, truncated, info = env.step(action)
            elif policy == "pn":
                own_state, target_state = env._get_current_states()
                target_pos = target_state.get("position_m")
                if target_pos is None:
                    target_pos = target_state.get("position_neu")
                virtual_point = {"position_neu": np.asarray(target_pos, dtype=np.float64)}
                cmd = pn_guidance.compute_command(own_state, target_state, virtual_point, None)
                obs, reward, terminated, truncated, info = env.step(command_override=cmd)
            elif policy == "los_rate":
                own_state, target_state = env._get_current_states()
                target_pos = target_state.get("position_m")
                if target_pos is None:
                    target_pos = target_state.get("position_neu")
                virtual_point = {"position_neu": np.asarray(target_pos, dtype=np.float64)}
                cmd = env.guidance.compute_command(
                    own_state, target_state, virtual_point, env.current_gains
                )
                obs, reward, terminated, truncated, info = env.step(command_override=cmd)
            else:
                raise ValueError(f"Unknown policy: {policy}")

            steps = step + 1

            target_state = info.get("target_state", {})
            alt = float(target_state.get("altitude_m", np.nan))
            speed = float(target_state.get("speed_mps", np.nan))
            if np.isfinite(alt):
                min_bandit_alt = min(min_bandit_alt, alt)
                max_bandit_alt = max(max_bandit_alt, alt)
            if np.isfinite(speed):
                min_bandit_speed = min(min_bandit_speed, speed)
                max_bandit_speed = max(max_bandit_speed, speed)

            range_m = float(info.get("range_m", np.nan))
            if np.isfinite(range_m):
                ranges.append(range_m)

            maneuver = info.get("bandit_maneuver")
            if maneuver is not None and maneuver != last_maneuver:
                maneuvers_seen.append(maneuver)
                last_maneuver = maneuver

            if terminated or truncated:
                reason = info.get("termination_info", {}).get("reason", "unknown")
                break
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return {
        "steps": steps,
        "reason": reason,
        "exception": exception,
        "min_bandit_alt_m": min_bandit_alt if np.isfinite(min_bandit_alt) else np.nan,
        "max_bandit_alt_m": max_bandit_alt if np.isfinite(max_bandit_alt) else np.nan,
        "min_bandit_speed_mps": min_bandit_speed if np.isfinite(min_bandit_speed) else np.nan,
        "max_bandit_speed_mps": max_bandit_speed if np.isfinite(max_bandit_speed) else np.nan,
        "maneuvers": maneuvers_seen,
        "ranges": np.asarray(ranges, dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser(description="Bandit baseline robustness validation")
    parser.add_argument("--config", type=str, default="config/env_bandit.yaml")
    parser.add_argument(
        "--policy",
        type=str,
        default="pn",
        choices=["random", "pn", "los_rate"],
        help=(
            "Blue-side policy: mild random actions, simple proportional-navigation "
            "pursuit, or the environment's built-in LOS-rate guidance with direct tracking."
        ),
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument(
        "--scenario",
        type=str,
        default="tail_chase",
        choices=["default", "tail_chase", "head_on"],
        help=(
            "Initial geometry. 'default' uses the env's built-in JSBSim reset; "
            "'tail_chase' places the bandit 2000 m ahead with the ownship closing from behind."
        ),
    )
    args = parser.parse_args()

    # Ensure missiles stay disabled for this dogfight baseline.
    overrides = {
        "env": {
            "bandit": {"missile": {"enabled": False}},
            "max_high_level_steps": args.max_steps,
        }
    }
    config = _resolve_config(args.config, overrides)

    print(f"Config: {args.config}")
    print(f"Policy: {args.policy}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes per difficulty: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    if args.policy == "random":
        print(f"Action scale: {args.action_scale}")
    print("-" * 70)

    overall_exceptions = 0
    overall_crashes = 0
    overall_steps = []
    overall_ranges: List[np.ndarray] = []

    for difficulty in args.difficulties:
        config["env"]["bandit"]["difficulty"] = difficulty
        env = CloseRangeTrackingEnv(config)

        reasons = Counter()
        steps_list: List[int] = []
        alt_mins: List[float] = []
        alt_maxs: List[float] = []
        speed_mins: List[float] = []
        speed_maxs: List[float] = []
        maneuver_counts: Counter = Counter()
        ranges_list: List[np.ndarray] = []
        exceptions = 0
        successes = 0

        for seed in range(args.episodes):
            scenario = _build_scenario(args.scenario, np.random.default_rng(seed))
            result = _run_episode(
                env, seed, args.max_steps, args.policy, args.action_scale, scenario
            )
            reasons[result["reason"]] += 1
            steps_list.append(result["steps"])
            ranges_list.append(result["ranges"])
            if result["reason"] == "success":
                successes += 1
            if result["exception"] is not None:
                exceptions += 1
                print(f"  [{difficulty}] seed {seed} EXCEPTION: {result['exception']}")
            if not np.isnan(result["min_bandit_alt_m"]):
                alt_mins.append(result["min_bandit_alt_m"])
                alt_maxs.append(result["max_bandit_alt_m"])
                speed_mins.append(result["min_bandit_speed_mps"])
                speed_maxs.append(result["max_bandit_speed_mps"])
            maneuver_counts.update(result["maneuvers"])

        env.close()

        mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
        std_steps = float(np.std(steps_list)) if steps_list else 0.0
        crash_count = reasons.get("crash", 0) + reasons.get("bandit_out_of_bounds", 0)
        overall_exceptions += exceptions
        overall_crashes += crash_count
        overall_steps.extend(steps_list)
        overall_ranges.extend(ranges_list)

        if ranges_list:
            all_ranges = np.concatenate(ranges_list)
            min_range = float(np.min(all_ranges))
            mean_range = float(np.mean(all_ranges))
            final_ranges = np.array([r[-1] for r in ranges_list if r.size], dtype=np.float64)
            mean_final_range = float(np.mean(final_ranges)) if final_ranges.size else np.nan
        else:
            min_range = mean_range = mean_final_range = np.nan

        print(f"\nDifficulty: {difficulty}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Exceptions: {exceptions}")
        print(f"  Crashes/bandit-OOB: {crash_count}")
        print(f"  Successes: {successes}")
        print(f"  Steps: {mean_steps:.1f} ± {std_steps:.1f}")
        print(f"  Range over all steps: min={min_range:.0f}m, mean={mean_range:.0f}m")
        print(f"  Final range (mean): {mean_final_range:.0f}m")
        print(f"  Termination reasons: {dict(reasons)}")
        if alt_mins:
            print(
                f"  Bandit altitude: [{min(alt_mins):.0f}, {max(alt_maxs):.0f}] m "
                f"(min-of-mins {min(alt_mins):.0f}, max-of-maxs {max(alt_maxs):.0f})"
            )
            print(
                f"  Bandit speed:    [{min(speed_mins):.1f}, {max(speed_maxs):.1f}] m/s "
                f"(min-of-mins {min(speed_mins):.1f}, max-of-maxs {max(speed_maxs):.1f})"
            )
        print(f"  Maneuver switches: {dict(maneuver_counts)}")

    print("\n" + "=" * 70)
    print("OVERALL")
    print(f"  Total episodes: {len(overall_steps)}")
    print(f"  Total exceptions: {overall_exceptions}")
    print(f"  Total crashes/bandit-OOB: {overall_crashes}")
    if overall_steps:
        print(f"  Steps: {float(np.mean(overall_steps)):.1f} ± {float(np.std(overall_steps)):.1f}")
    if overall_ranges:
        all_ranges = np.concatenate(overall_ranges)
        final_ranges = np.array([r[-1] for r in overall_ranges if r.size], dtype=np.float64)
        print(f"  Range over all steps: min={float(np.min(all_ranges)):.0f}m, mean={float(np.mean(all_ranges)):.0f}m")
        print(f"  Final range (mean): {float(np.mean(final_ranges)):.0f}m")

    if overall_exceptions > 0 or overall_crashes > 0:
        print("\nRESULT: FAIL (exceptions or crashes detected)")
        sys.exit(1)
    print("\nRESULT: PASS (no exceptions or crashes)")


if __name__ == "__main__":
    main()
