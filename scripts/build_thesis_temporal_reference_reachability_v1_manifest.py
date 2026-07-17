#!/usr/bin/env python3
"""Build a frozen Disadvantage12 manifest for temporal-reference reachability."""

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


SOURCE_ID = "THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1"
FAMILY = "thesis_temporal_reference_reachability_v1"
METHODS = (
    "fixed_head_on",
    "lateral_extension",
    "vertical_extension",
    "combined_break_extension",
)
HEIGHTS = (
    ("own_below", 4550.0, 5900.0),
    ("co_altitude", 5250.0, 5250.0),
    ("own_above", 5950.0, 4600.0),
)
PACKAGES = (
    {
        "id": "trv1_long_escape",
        "range_m": 5200.0,
        "los_azimuth_abs_deg": 28.0,
        "own_speed_mps": 245.0,
        "target_speed_mps": 285.0,
    },
    {
        "id": "trv1_rapid_closure",
        "range_m": 3850.0,
        "los_azimuth_abs_deg": 32.0,
        "own_speed_mps": 270.0,
        "target_speed_mps": 305.0,
    },
)
DISJOINT_SOURCES = (
    "config/experiment/manifests/thesis_disadvantage_capability_plan1_v2.yaml",
    "config/experiment/manifests/thesis_disadvantage_capability_plan2_v1.yaml",
    "config/experiment/manifests/thesis_dualskill_taxonomy60_v1.yaml",
)


def _wrap_deg(value: float) -> float:
    return value % 360.0


def _signed_angle_deg(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def _signature(scenario: Mapping[str, Any]) -> tuple[float, ...]:
    """Return an exact reset-state signature shared by all legacy manifests.

    Taxonomy60 predates some newer convenience metadata such as
    ``los_azimuth_abs_deg``.  Comparing the actual reset geometry avoids
    treating absent descriptive fields as proof of disjointness.
    """

    own = scenario["own_init"]
    target = scenario["target_init"]
    own_position = own["position_m"]
    target_position = target["position_m"]
    return (
        *(round(float(value), 6) for value in own_position),
        round(float(own["velocity_mps"]), 6),
        round(float(own["heading_deg"]), 6),
        *(round(float(value), 6) for value in target_position),
        round(float(target["velocity_mps"]), 6),
        round(float(target["heading_deg"]), 6),
    )


def _scenario(*, package: Mapping[str, Any], height: tuple[str, float, float], mirror_sign: str, index: int) -> dict[str, Any]:
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
    if not math.isclose(own_to_target, 160.0, abs_tol=1e-9) or not math.isclose(target_to_own, 20.0, abs_tol=1e-9):
        raise AssertionError("temporal-reference disadvantage angle construction drifted")
    package_id = str(package["id"])
    return {
        "name": f"{FAMILY}_{package_id}_{height_name}_{mirror_sign}",
        "own_init": {"position_m": [0.0, 0.0, own_altitude_m], "velocity_mps": float(package["own_speed_mps"]), "heading_deg": round(heading_deg, 9)},
        "target_init": {"position_m": target_position, "velocity_mps": float(package["target_speed_mps"]), "heading_deg": round(heading_deg, 9)},
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
            "scenario_seed": 275100 + index,
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


def _existing_signatures() -> set[tuple[float, ...]]:
    signatures: set[tuple[float, ...]] = set()
    for relative_path in DISJOINT_SOURCES:
        payload = yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8")) or {}
        for scenario in payload.get("scenarios", []):
            signatures.add(_signature(scenario))
    return signatures


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
        "status": "preregistered_p0_p1_execution_not_authorized",
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
            "methods": list(METHODS),
            "episodes_per_scenario_method_opponent": 1,
            "episodes_total_if_authorized": len(scenarios) * len(METHODS) * 3,
            "future_state_or_outcome_routing": False,
            "one_formal_matrix_run": True,
        },
        "disjoint_against": list(DISJOINT_SOURCES),
        "scenarios": scenarios,
    }
    payload["integrity"] = {"scenario_count": len(scenarios)}
    payload["integrity"]["payload_sha256"] = comparison._scenario_manifest_payload_sha256(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    scenarios = manifest.get("scenarios")
    if manifest.get("source_id") != SOURCE_ID or not isinstance(scenarios, list) or len(scenarios) != 12:
        raise ValueError("temporal-reference manifest must contain exactly 12 scenarios")
    if manifest.get("integrity", {}).get("payload_sha256") != comparison._scenario_manifest_payload_sha256(dict(manifest)):
        raise ValueError("temporal-reference manifest integrity hash mismatch")
    names = [str(item["name"]) for item in scenarios]
    seeds = [int(item["metadata"]["scenario_seed"]) for item in scenarios]
    if len(set(names)) != len(scenarios) or len(set(seeds)) != len(scenarios):
        raise ValueError("temporal-reference names and seeds must be unique")
    if set(_signature(item) for item in scenarios).intersection(_existing_signatures()):
        raise ValueError("temporal-reference manifest overlaps a frozen source signature")
    for scenario in scenarios:
        metadata = scenario["metadata"]
        if metadata.get("initial_class") != "disadvantage" or metadata.get("task_registry_key") != "head_on":
            raise ValueError("temporal-reference taxonomy/task contract drifted")
        if not math.isclose(float(metadata["own_to_target_los_angle_deg"]), 160.0, abs_tol=1e-9):
            raise ValueError("temporal-reference ATA construction drifted")
        if not math.isclose(float(metadata["target_velocity_to_own_los_angle_deg"]), 20.0, abs_tol=1e-9):
            raise ValueError("temporal-reference AA construction drifted")


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
