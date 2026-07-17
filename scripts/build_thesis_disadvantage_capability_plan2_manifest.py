#!/usr/bin/env python3
"""Build the frozen, disjoint opening-pressure Disadvantage12 for Plan 2."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison


SOURCE_ID = "THESIS-DISADVANTAGE-CAPABILITY-PLAN2-V1"
FAMILY = "thesis_disadvantage_capability_plan2_v1"
HEIGHTS = (
    ("own_below", 4600.0, 5750.0),
    ("co_altitude", 5250.0, 5250.0),
    ("own_above", 5900.0, 4750.0),
)
PACKAGES = (
    {
        "id": "dcv3_mid_press",
        "range_m": 3400.0,
        "los_azimuth_abs_deg": 18.0,
        "own_speed_mps": 235.0,
        "target_speed_mps": 278.0,
    },
    {
        "id": "dcv3_wide_fast",
        "range_m": 4550.0,
        "los_azimuth_abs_deg": 34.0,
        "own_speed_mps": 255.0,
        "target_speed_mps": 282.0,
    },
)
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "lateral_break_same_side",
    "range_extension_los",
    "vertical_separation",
)


def _wrap_deg(value: float) -> float:
    return value % 360.0


def _signed_angle_deg(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def _scenario(
    *,
    package: Mapping[str, Any],
    height: tuple[str, float, float],
    mirror_sign: str,
    index: int,
) -> dict[str, Any]:
    height_name, own_altitude_m, target_altitude_m = height
    sign = -1.0 if mirror_sign == "negative" else 1.0
    azimuth_deg = sign * float(package["los_azimuth_abs_deg"])
    heading_deg = _wrap_deg(azimuth_deg + (-160.0 if sign > 0.0 else 160.0))
    range_m = float(package["range_m"])
    azimuth_rad = math.radians(azimuth_deg)
    target_position = [
        round(range_m * math.cos(azimuth_rad), 9),
        round(range_m * math.sin(azimuth_rad), 9),
        target_altitude_m,
    ]
    own_to_target = abs(_signed_angle_deg(heading_deg, azimuth_deg))
    target_to_own = abs(_signed_angle_deg(heading_deg, _wrap_deg(azimuth_deg + 180.0)))
    if not math.isclose(own_to_target, 160.0, abs_tol=1e-9) or not math.isclose(
        target_to_own, 20.0, abs_tol=1e-9
    ):
        raise AssertionError("Plan 2 disadvantage angle construction drifted")
    package_id = str(package["id"])
    return {
        "name": f"{FAMILY}_{package_id}_{height_name}_{mirror_sign}",
        "own_init": {
            "position_m": [0.0, 0.0, own_altitude_m],
            "velocity_mps": float(package["own_speed_mps"]),
            "heading_deg": round(heading_deg, 9),
        },
        "target_init": {
            "position_m": target_position,
            "velocity_mps": float(package["target_speed_mps"]),
            "heading_deg": round(heading_deg, 9),
        },
        "metadata": {
            "manifest_family": FAMILY,
            "initial_class": "disadvantage",
            "taxonomy_geometry_state": "disadvantage",
            "height_condition": height_name,
            "mirror_sign": mirror_sign,
            "mirror_pair_id": f"{package_id}__disadvantage__{height_name}",
            "distance_speed_package": package_id,
            "phase_at_reset": "pre_merge",
            "task_registry_key": "head_on",
            "ppo_task_bit": 0,
            "crossing_context": False,
            "scenario_seed": 274100 + index,
            "initial_range_m": range_m,
            "horizontal_range_m": round(range_m, 9),
            "altitude_diff_m": target_altitude_m - own_altitude_m,
            "own_speed_mps": float(package["own_speed_mps"]),
            "target_speed_mps": float(package["target_speed_mps"]),
            "los_azimuth_abs_deg": float(package["los_azimuth_abs_deg"]),
            "own_to_target_los_angle_deg": own_to_target,
            "target_velocity_to_own_los_angle_deg": target_to_own,
        },
    }


def build_manifest() -> dict[str, Any]:
    scenarios = [
        _scenario(package=package, height=height, mirror_sign=mirror_sign, index=index)
        for index, (package, height, mirror_sign) in enumerate(
            (package, height, mirror_sign)
            for package in PACKAGES
            for height in HEIGHTS
            for mirror_sign in ("negative", "positive")
        )
    ]
    payload = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
            "methods": list(METHODS),
            "episodes_per_scenario_method_opponent": 1,
            "episodes_total": len(scenarios) * len(METHODS) * 3,
            "warmup_high_level_steps": 10,
            "motif_intervention_high_level_steps": 20,
            "future_state_or_outcome_routing": False,
        },
        "scenarios": scenarios,
    }
    payload["integrity"] = {"scenario_count": len(scenarios)}
    payload["integrity"]["payload_sha256"] = comparison._scenario_manifest_payload_sha256(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    scenarios = manifest.get("scenarios")
    if manifest.get("source_id") != SOURCE_ID or not isinstance(scenarios, list) or len(scenarios) != 12:
        raise ValueError("Plan 2 manifest must contain exactly 12 scenarios")
    if manifest.get("integrity", {}).get("payload_sha256") != comparison._scenario_manifest_payload_sha256(
        dict(manifest)
    ):
        raise ValueError("Plan 2 manifest integrity hash mismatch")
    names = [str(item["name"]) for item in scenarios]
    seeds = [int(item["metadata"]["scenario_seed"]) for item in scenarios]
    if len(set(names)) != 12 or len(set(seeds)) != 12:
        raise ValueError("Plan 2 names and seeds must be unique")
    for item in scenarios:
        metadata = item["metadata"]
        if metadata.get("initial_class") != "disadvantage" or metadata.get("task_registry_key") != "head_on":
            raise ValueError("Plan 2 taxonomy/task contract drifted")
        if not math.isclose(float(metadata["own_to_target_los_angle_deg"]), 160.0, abs_tol=1e-9):
            raise ValueError("Plan 2 ATA construction drifted")
        if not math.isclose(float(metadata["target_velocity_to_own_los_angle_deg"]), 20.0, abs_tol=1e-9):
            raise ValueError("Plan 2 AA construction drifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    manifest = build_manifest()
    validate_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(manifest["integrity"]["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
