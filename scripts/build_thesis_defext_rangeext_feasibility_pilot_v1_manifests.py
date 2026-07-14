#!/usr/bin/env python3
"""Build disjoint dev12 and heldout24 physical run-in pilot manifests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

FAMILY_SOURCE_ID = "THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1"
OWN_ALTITUDE_M = 5200.0
HEIGHT_OFFSETS_M = {"own_below": 200.0, "co_altitude": 0.0, "own_above": -200.0}
V2_SIGNATURES = {
    (1200.0, 185.0, 365.0),
    (1500.0, 200.0, 400.0),
}
DEV_PACKAGES = (
    {"id": "defext_dev_a", "initial_range_m": 1125.0, "own_speed_mps": 180.0, "target_speed_mps": 350.0, "own_angle_deg": 156.0, "target_angle_deg": 24.0, "los_azimuth_abs_deg": 18.0},
    {"id": "defext_dev_b", "initial_range_m": 1600.0, "own_speed_mps": 210.0, "target_speed_mps": 410.0, "own_angle_deg": 164.0, "target_angle_deg": 16.0, "los_azimuth_abs_deg": 26.0},
)
HELDOUT_PACKAGES = (
    {"id": "defext_hold_a", "initial_range_m": 1075.0, "own_speed_mps": 175.0, "target_speed_mps": 345.0, "own_angle_deg": 154.0, "target_angle_deg": 26.0, "los_azimuth_abs_deg": 14.0},
    {"id": "defext_hold_b", "initial_range_m": 1525.0, "own_speed_mps": 195.0, "target_speed_mps": 395.0, "own_angle_deg": 162.0, "target_angle_deg": 18.0, "los_azimuth_abs_deg": 22.0},
    {"id": "defext_hold_c", "initial_range_m": 1775.0, "own_speed_mps": 215.0, "target_speed_mps": 420.0, "own_angle_deg": 158.0, "target_angle_deg": 22.0, "los_azimuth_abs_deg": 30.0},
    {"id": "defext_hold_d", "initial_range_m": 2000.0, "own_speed_mps": 225.0, "target_speed_mps": 430.0, "own_angle_deg": 166.0, "target_angle_deg": 14.0, "los_azimuth_abs_deg": 10.0},
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(payload)
    prepared.get("integrity", {}).pop("payload_sha256", None)
    encoded = json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def package_signature(package: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(package["initial_range_m"]),
        float(package["own_speed_mps"]),
        float(package["target_speed_mps"]),
    )


def _scenario(package: Mapping[str, Any], height: str, mirror: str, seed: int, split: str) -> dict[str, Any]:
    from build_thesis_taxonomy_ablation30_manifest import (
        _horizontal_heading_for_3d_angle,
        _normalize_heading_deg,
    )
    from build_thesis_five_state_heldout40_manifest import initial_geometry

    altitude_diff_m = float(HEIGHT_OFFSETS_M[height])
    initial_range_m = float(package["initial_range_m"])
    horizontal_range_m = math.sqrt(initial_range_m**2 - altitude_diff_m**2)
    horizontal_fraction = horizontal_range_m / initial_range_m
    sign = -1 if mirror == "negative" else 1
    los_azimuth_deg = sign * float(package["los_azimuth_abs_deg"])
    target_position = (
        horizontal_range_m * math.cos(math.radians(los_azimuth_deg)),
        horizontal_range_m * math.sin(math.radians(los_azimuth_deg)),
        OWN_ALTITUDE_M + altitude_diff_m,
    )
    own_heading = _horizontal_heading_for_3d_angle(
        los_azimuth_deg,
        horizontal_fraction,
        float(package["own_angle_deg"]),
        -sign,
    )
    target_heading = _horizontal_heading_for_3d_angle(
        _normalize_heading_deg(los_azimuth_deg + 180.0),
        horizontal_fraction,
        float(package["target_angle_deg"]),
        sign,
    )
    short_mirror = "neg" if sign < 0 else "pos"
    scenario = {
        "name": f"defext_rangeext_{split}_{package['id']}_{height}_{short_mirror}",
        "own_init": {
            "position_m": [0.0, 0.0, OWN_ALTITUDE_M],
            "velocity_mps": float(package["own_speed_mps"]),
            "heading_deg": round(own_heading, 9),
        },
        "target_init": {
            "position_m": [round(value, 9) for value in target_position],
            "velocity_mps": float(package["target_speed_mps"]),
            "heading_deg": round(target_heading, 9),
        },
        "metadata": {
            "pilot_family": FAMILY_SOURCE_ID,
            "split": split,
            "initial_class": "disadvantage",
            "height_condition": height,
            "mirror_sign": mirror,
            "distance_speed_package": package["id"],
            "scenario_signature": f"{package['id']}|{height}|{mirror}",
            "scenario_seed": int(seed),
            "phase_at_reset": "pre_merge",
            "continuous_run_in": True,
            "task_registry_key": "head_on",
            "routing_enabled": False,
            "fixed_skill": "defensive_extension",
            "fixed_profile": "range_extension",
        },
    }
    scenario["metadata"].update(initial_geometry(scenario))
    if scenario["metadata"]["taxonomy_geometry_state"] != "disadvantage":
        raise ValueError(f"taxonomy drift: {scenario['name']}")
    return scenario


def build_manifest(split: str, packages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if split not in {"dev12", "heldout24"}:
        raise ValueError(f"unsupported split: {split}")
    signatures = [package_signature(package) for package in packages]
    if len(signatures) != len(set(signatures)) or set(signatures) & V2_SIGNATURES:
        raise ValueError(f"{split} package signatures are duplicated or overlap v2")
    scenarios: list[dict[str, Any]] = []
    seed = 81400 if split == "dev12" else 82600
    for package in packages:
        for height in HEIGHT_OFFSETS_M:
            for mirror in ("negative", "positive"):
                scenarios.append(_scenario(package, height, mirror, seed, split))
                seed += 1
    source_id = f"{FAMILY_SOURCE_ID}-{split.upper()}"
    manifest = {
        "source_id": source_id,
        "manifest_version": 1,
        "status": "preregistered_design_only",
        "protocol": {
            "training_permitted": False,
            "evaluation_permitted": False,
            "continuous_physical_run_in": True,
            "snapshot_restore": False,
            "future_state_injection": False,
            "routing_enabled": False,
            "fixed_skill": "defensive_extension",
            "fixed_profile": "range_extension",
            "opponents_reported_separately": True,
        },
        "independence": {
            "v2_source_id": "THESIS-PHASE-REACHABILITY-V2-RUN-IN-HANDOFF-R1",
            "v2_payload_sha256": "5ffdaa4bc248bf40c1532b13c8030dca43240711f5751cf8c4f26217525cfaae",
            "v2_signature_overlap_count": 0,
            "package_signatures": [list(signature) for signature in signatures],
        },
        "generation_contract": {
            "initial_class": "disadvantage",
            "own_altitude_m": OWN_ALTITUDE_M,
            "height_offsets_target_minus_own_m": HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "packages": [dict(package) for package in packages],
            "first_pass_requirement": "legal reset with range > 1000 m followed by one continuous JSBSim episode",
        },
        "scenarios": scenarios,
        "integrity": {"builder_code_sha256": _sha256_bytes(Path(__file__).read_bytes())},
    }
    manifest["integrity"]["payload_sha256"] = payload_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "config" / "experiment" / "manifests")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = {
        "dev12": ("thesis_defext_rangeext_feasibility_pilot_v1_dev12.yaml", DEV_PACKAGES),
        "heldout24": ("thesis_defext_rangeext_feasibility_pilot_v1_heldout24.yaml", HELDOUT_PACKAGES),
    }
    result = {}
    for split, (name, packages) in outputs.items():
        path = args.output_dir / name
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        manifest = build_manifest(split, packages)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        result[split] = {
            "path": str(path),
            "scenario_count": len(manifest["scenarios"]),
            "payload_sha256": manifest["integrity"]["payload_sha256"],
        }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
