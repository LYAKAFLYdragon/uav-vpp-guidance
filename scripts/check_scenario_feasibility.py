#!/usr/bin/env python3
"""
Scenario Feasibility Checker.

Reads a Stage 6F.5 scenario configuration and computes geometric feasibility metrics:
  - initial_range_m
  - closure_rate_mps
  - time_to_capture_lower_bound_s
  - time_to_oob_upper_bound_s
  - required_turn_angle_deg
  - feasible flag

Usage:
    python scripts/check_scenario_feasibility.py \
        --config config/experiment/stage6f5_feasible_geometry.yaml \
        --output outputs/tables/stage6f5_feasible_geometry/scenario_feasibility_report.md

    python scripts/check_scenario_feasibility.py \
        --config config/experiment/stage6f5_maneuvering_target.yaml \
        --output outputs/tables/stage6f5_maneuvering_target/scenario_feasibility_report.md
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uav_vpp_guidance.utils.config import load_yaml_config


def parse_state(state: dict) -> tuple:
    """Parse position, velocity, heading from state dict."""
    pos = np.array(state.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    vel_scalar = state.get("velocity_mps", 0.0)
    heading_deg = state.get("heading_deg", 0.0)
    heading_rad = math.radians(heading_deg)
    vel = vel_scalar * np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    return pos, vel, heading_deg


def compute_scenario_feasibility(scenario_name: str, scenario: dict, max_range_m: float) -> dict:
    """Compute feasibility metrics for a single scenario."""
    own = scenario.get("own_init", {})
    target = scenario.get("target_init", {})

    own_pos, own_vel, own_heading = parse_state(own)
    tgt_pos, tgt_vel, tgt_heading = parse_state(target)

    rel_pos = tgt_pos - own_pos
    initial_range_m = float(np.linalg.norm(rel_pos))

    rel_vel = tgt_vel - own_vel
    # Closure rate = negative of relative velocity projected onto line of sight
    los_unit = rel_pos / (initial_range_m + 1e-9)
    closure_rate_mps = float(-np.dot(rel_vel, los_unit))

    # Time to capture lower bound (straight-line pursuit at success_range)
    success_range_m = 900.0  # default
    if closure_rate_mps > 0:
        time_to_capture_lower_bound_s = (initial_range_m - success_range_m) / closure_rate_mps
    else:
        time_to_capture_lower_bound_s = float("inf")

    # Time to OOB upper bound (range expanding at closure_rate)
    if closure_rate_mps < 0:
        time_to_oob_upper_bound_s = (max_range_m - initial_range_m) / abs(closure_rate_mps)
    else:
        # If closure_rate > 0, range is decreasing, so OOB due to expansion is not a concern
        time_to_oob_upper_bound_s = float("inf")

    # Required turn angle (difference between own heading and LOS to target)
    los_heading_rad = math.atan2(rel_pos[1], rel_pos[0])
    required_turn_angle_deg = math.degrees(abs(((los_heading_rad - math.radians(own_heading) + math.pi) % (2 * math.pi)) - math.pi))

    # Feasibility checks
    warnings = []
    if closure_rate_mps <= 0:
        warnings.append("closure_rate <= 0: range will not decrease without maneuver")
    if math.isfinite(time_to_capture_lower_bound_s) and math.isfinite(time_to_oob_upper_bound_s):
        if time_to_capture_lower_bound_s > time_to_oob_upper_bound_s:
            warnings.append("time_to_capture > time_to_oob: scenario likely infeasible")
    if required_turn_angle_deg > 90.0:
        warnings.append("required_turn_angle > 90deg: large initial heading correction needed")

    feasible = len(warnings) == 0

    metadata = scenario.get("metadata", {})

    return {
        "scenario_name": scenario_name,
        "initial_range_m": round(initial_range_m, 1),
        "closure_rate_mps": round(closure_rate_mps, 1),
        "time_to_capture_lower_bound_s": round(time_to_capture_lower_bound_s, 1) if math.isfinite(time_to_capture_lower_bound_s) else "inf",
        "time_to_oob_upper_bound_s": round(time_to_oob_upper_bound_s, 1) if math.isfinite(time_to_oob_upper_bound_s) else "inf",
        "required_turn_angle_deg": round(required_turn_angle_deg, 1),
        "max_range_m": max_range_m,
        "feasible": feasible,
        "warnings": "; ".join(warnings) if warnings else "none",
        "expected_feasible": metadata.get("expected_feasible", "unknown"),
        "rationale": metadata.get("rationale", ""),
    }


def render_report(results: list, config_path: str) -> str:
    lines = []
    lines.append("# Scenario Feasibility Report")
    lines.append("")
    lines.append(f"**Config**: `{config_path}`")
    lines.append(f"**Total scenarios**: {len(results)}")
    feasible_count = sum(1 for r in results if r["feasible"])
    lines.append(f"**Feasible**: {feasible_count}/{len(results)}")
    lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Scenario | Initial Range (m) | Closure Rate (m/s) | TTC LB (s) | TTOOB UB (s) | Turn (deg) | Feasible | Warnings |")
    lines.append("|----------|------------------:|-------------------:|-----------:|-------------:|-----------:|:--------:|----------|")
    for r in results:
        ttc = str(r["time_to_capture_lower_bound_s"])
        ttoob = str(r["time_to_oob_upper_bound_s"])
        lines.append(
            f"| {r['scenario_name']} | {r['initial_range_m']} | {r['closure_rate_mps']} | "
            f"{ttc} | {ttoob} | {r['required_turn_angle_deg']} | "
            f"{'Yes' if r['feasible'] else '**NO**'} | {r['warnings']} |"
        )
    lines.append("")

    lines.append("## Per-Scenario Details")
    lines.append("")
    for r in results:
        lines.append(f"### {r['scenario_name']}")
        lines.append(f"- **Initial range**: {r['initial_range_m']} m")
        lines.append(f"- **Closure rate**: {r['closure_rate_mps']} m/s")
        lines.append(f"- **Time-to-capture lower bound**: {r['time_to_capture_lower_bound_s']} s")
        lines.append(f"- **Time-to-OOB upper bound**: {r['time_to_oob_upper_bound_s']} s")
        lines.append(f"- **Required turn angle**: {r['required_turn_angle_deg']} deg")
        lines.append(f"- **Feasible**: {'Yes' if r['feasible'] else '**NO**'}")
        lines.append(f"- **Warnings**: {r['warnings']}")
        if r['rationale']:
            lines.append(f"- **Rationale**: {r['rationale']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check scenario feasibility")
    parser.add_argument("--config", type=str, required=True, help="Path to scenario config YAML")
    parser.add_argument("--output", type=str, required=True, help="Output markdown report path")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    scenarios = config.get("scenarios", {})
    max_range_m = config.get("env", {}).get("max_range_m", 8000.0)

    results = []
    for name, sc in scenarios.items():
        results.append(compute_scenario_feasibility(name, sc, max_range_m))

    # Sort by feasibility then by name
    results.sort(key=lambda r: (not r["feasible"], r["scenario_name"]))

    # Save Markdown report
    report = render_report(results, args.config)
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved feasibility report: {output_path}")

    # Save CSV
    csv_path = output_path.with_suffix(".csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved feasibility CSV: {csv_path}")

    # Exit code: 1 if any scenario is infeasible
    if any(not r["feasible"] for r in results):
        print("WARNING: Some scenarios are flagged as potentially infeasible.")
        # Do not exit with error; just warn. The user may intend marginal scenarios.


if __name__ == "__main__":
    main()
