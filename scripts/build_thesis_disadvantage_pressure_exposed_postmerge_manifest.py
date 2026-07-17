#!/usr/bin/env python3
"""Build the frozen, disjoint manifest for the pressure-exposure collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


SOURCE_ID = "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-V1-MANIFEST16"

# These are geometry packages, not result-selected scenarios.  They perturb
# range, LOS bearing, speed, and height around a rear-quarter pressure setup.
PACKAGES = (
    ("short_tight", 2800.0, 12.0, 245.0, 290.0),
    ("short_wide", 3100.0, 26.0, 245.0, 290.0),
    ("medium_tight", 3600.0, 16.0, 235.0, 300.0),
    ("medium_wide", 4100.0, 30.0, 235.0, 300.0),
)
HEIGHTS = (("own_below", 650.0), ("co_altitude", 0.0))


def _scenario(
    *, package: tuple[str, float, float, float, float], height: tuple[str, float], sign: int, seed: int
) -> dict[str, Any]:
    package_name, range_m, bearing_deg, own_speed, target_speed = package
    height_name, altitude_diff = height
    horizontal = math.sqrt(range_m * range_m - altitude_diff * altitude_diff)
    bearing_rad = math.radians(sign * bearing_deg)
    # The target begins in the ego aircraft's rear hemisphere and points
    # approximately at it.  Both mirror directions are generated exactly.
    heading_deg = (180.0 + sign * (bearing_deg + 20.0)) % 360.0
    suffix = "positive" if sign > 0 else "negative"
    signature = f"pressure_{package_name}__disadvantage__{height_name}"
    return {
        "name": f"pressure_postmerge_{package_name}_{height_name}_{suffix}",
        "own_init": {
            "position_m": [0.0, 0.0, 5500.0],
            "velocity_mps": own_speed,
            "heading_deg": heading_deg,
        },
        "target_init": {
            "position_m": [horizontal * math.cos(bearing_rad), horizontal * math.sin(bearing_rad), 5500.0 + altitude_diff],
            "velocity_mps": target_speed,
            "heading_deg": heading_deg,
        },
        "metadata": {
            "manifest_family": "thesis_disadvantage_pressure_exposed_postmerge_v1",
            "initial_class": "disadvantage",
            "taxonomy_geometry_state": "disadvantage",
            "height_condition": height_name,
            "mirror_sign": suffix,
            "mirror_pair_id": signature,
            "distance_speed_package": package_name,
            "phase_at_reset": "pre_merge",
            "task_registry_key": "head_on",
            "ppo_task_bit": 0,
            "crossing_context": False,
            "scenario_seed": seed,
            "initial_range_m": range_m,
            "horizontal_range_m": horizontal,
            "altitude_diff_m": altitude_diff,
            "own_speed_mps": own_speed,
            "target_speed_mps": target_speed,
            "los_azimuth_abs_deg": bearing_deg,
            "own_to_target_los_angle_deg": 160.0,
            "target_velocity_to_own_los_angle_deg": 20.0,
        },
    }


def build_manifest() -> dict[str, Any]:
    scenarios = []
    seed = 871100
    for package in PACKAGES:
        for height in HEIGHTS:
            for sign in (-1, 1):
                scenarios.append(_scenario(package=package, height=height, sign=sign, seed=seed))
                seed += 1
    payload = {"scenarios": scenarios}
    payload_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_nonlearning_collection_not_authorized",
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "methods": ["frozen_fixed_head_on"],
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
            "continuous_run_in_required": True,
            "reset_after_initialization": False,
            "postmerge_pressure_selection": "first_40_step_continuous_window_starting_in_disadvantage",
            "episodes_total_if_authorized": 48,
        },
        "disjoint_against": [
            "config/experiment/manifests/thesis_dualskill_taxonomy60_v1.yaml",
            "config/experiment/manifests/thesis_temporal_reference_reachability_v1.yaml",
        ],
        "scenarios": scenarios,
        "integrity": {"scenario_count": len(scenarios), "payload_sha256": payload_sha256},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(build_manifest(), sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
