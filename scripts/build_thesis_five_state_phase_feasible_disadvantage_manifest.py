"""Build a non-training disadvantage envelope intended to physically cross first pass."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SOURCE_ID = "THESIS-FIVE-STATE-PHASE-FEASIBLE-DISADVANTAGE-V1"
OWN_ALTITUDE_M = 5200.0
HEIGHT_OFFSETS_M = {"own_below": 200.0, "own_above": -200.0}
PACKAGES = (
    {"id": "phase_pass_a", "initial_range_m": 1500.0, "own_speed_mps": 205.0, "target_speed_mps": 315.0},
    {"id": "phase_pass_b", "initial_range_m": 1900.0, "own_speed_mps": 195.0, "target_speed_mps": 325.0},
    {"id": "phase_pass_c", "initial_range_m": 2300.0, "own_speed_mps": 215.0, "target_speed_mps": 340.0},
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(payload)
    prepared.get("integrity", {}).pop("payload_sha256", None)
    return _sha256_bytes(json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def build_manifest() -> dict[str, Any]:
    from build_thesis_taxonomy_ablation30_manifest import (
        _horizontal_heading_for_3d_angle,
        _normalize_heading_deg,
    )
    from build_thesis_five_state_heldout40_manifest import initial_geometry

    scenarios = []
    index = 0
    for package in PACKAGES:
        for height_condition in HEIGHT_OFFSETS_M:
            for mirror_sign in ("negative", "positive"):
                sign = -1 if mirror_sign == "negative" else 1
                altitude_diff_m = HEIGHT_OFFSETS_M[height_condition]
                initial_range_m = float(package["initial_range_m"])
                horizontal_range_m = math.sqrt(initial_range_m**2 - altitude_diff_m**2)
                horizontal_fraction = horizontal_range_m / initial_range_m
                los_azimuth_deg = sign * 24.0
                target_position = (
                    horizontal_range_m * math.cos(math.radians(los_azimuth_deg)),
                    horizontal_range_m * math.sin(math.radians(los_azimuth_deg)),
                    OWN_ALTITUDE_M + altitude_diff_m,
                )
                own_heading = _horizontal_heading_for_3d_angle(
                    los_azimuth_deg, horizontal_fraction, 160.0, -sign
                )
                target_heading = _horizontal_heading_for_3d_angle(
                    _normalize_heading_deg(los_azimuth_deg + 180.0), horizontal_fraction, 20.0, sign
                )
                scenario = {
                    "name": f"phase_feasible_{package['id']}_disadvantage_{height_condition}_{'neg' if sign < 0 else 'pos'}",
                    "own_init": {"position_m": [0.0, 0.0, OWN_ALTITUDE_M], "velocity_mps": package["own_speed_mps"], "heading_deg": round(own_heading, 9)},
                    "target_init": {"position_m": [round(value, 9) for value in target_position], "velocity_mps": package["target_speed_mps"], "heading_deg": round(target_heading, 9)},
                    "metadata": {
                        "manifest_family": "noncanonical_thesis_five_state_phase_feasible_disadvantage_v1",
                        "initial_class": "disadvantage",
                        "height_condition": height_condition,
                        "mirror_sign": mirror_sign,
                        "distance_speed_package": package["id"],
                        "phase_at_reset": "pre_merge",
                        "scenario_seed": 78400 + index,
                        "phase_feasible_intent": "target_overtake_then_range_opening",
                        "reference_only": True,
                        "task_registry_key": "head_on",
                        "ppo_task_bit": 0,
                        "crossing_context": False,
                    },
                }
                metadata = scenario["metadata"]
                metadata.update(initial_geometry(scenario))
                if metadata["taxonomy_geometry_state"] != "disadvantage":
                    raise ValueError(f"Taxonomy drift in {scenario['name']}: {metadata['taxonomy_geometry_state']}")
                scenarios.append(scenario)
                index += 1
    manifest = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_before_evaluation",
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "reference_controller": "legacy_static_oracle_task_gate",
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
            "episodes_per_scenario_opponent": 1,
            "episode_horizon_high_level_steps": 240,
            "high_level_dt_s": 0.2,
            "stop_rule": "No training, tuning, reward/VPP/guidance/PID change, scenario replacement, or result-driven rerun.",
        },
        "phase_gate": {
            "post_merge_step_minimum_per_episode": 10,
            "re_entry_step_minimum_per_episode": 10,
            "minimum_qualifying_episode_fraction_per_opponent": 0.5,
            "re_entry_rate_threshold_mps": -25.0,
            "re_entry_consecutive_post_merge_steps": 5,
            "failure_verdict": "phase_feasible_envelope_not_established",
        },
        "generation_contract": {
            "initial_class": "disadvantage",
            "height_offsets_target_minus_own_m": HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "distance_speed_packages": list(PACKAGES),
            "own_altitude_m": OWN_ALTITUDE_M,
            "target_overtake_design": "target is faster than ownship while preserving the frozen disadvantage angle definition",
        },
        "scenarios": scenarios,
        "integrity": {"builder_code_sha256": _sha256_bytes(Path(__file__).read_bytes())},
    }
    manifest["integrity"]["payload_sha256"] = _payload_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_phase_feasible_disadvantage_v1.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "scenario_count": len(manifest["scenarios"]), "payload_sha256": manifest["integrity"]["payload_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
