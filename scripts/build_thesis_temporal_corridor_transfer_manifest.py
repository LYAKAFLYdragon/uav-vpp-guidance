#!/usr/bin/env python3
"""Build the fresh Dev12 manifest for the temporal-corridor transfer probe."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V1"
PACKAGE_ID = "temporal_corridor_v1_unseen"


def _fingerprint(scenario: dict[str, Any]) -> tuple[Any, ...]:
    own, target = scenario["own_init"], scenario["target_init"]
    return (
        tuple(round(float(x), 6) for x in own["position_m"]), round(float(own["velocity_mps"]), 6), round(float(own["heading_deg"]), 6),
        tuple(round(float(x), 6) for x in target["position_m"]), round(float(target["velocity_mps"]), 6), round(float(target["heading_deg"]), 6),
    )


def _existing() -> set[tuple[Any, ...]]:
    """Parse only manifests that could share this unique range/speed package."""
    values: set[tuple[Any, ...]] = set()
    for path in (ROOT / "config" / "experiment" / "manifests").glob("*.yaml"):
        if path.name == "thesis_temporal_corridor_transfer_v1_dev12.yaml":
            continue
        text = path.read_text(encoding="utf-8")
        # A full-state collision must also share the unique 9150/310/370 package.
        if not all(token in text for token in ("9150.0", "310.0", "370.0")):
            continue
        for scenario in (yaml.safe_load(text) or {}).get("scenarios", []):
            if isinstance(scenario, dict) and "own_init" in scenario and "target_init" in scenario:
                values.add(_fingerprint(scenario))
    return values


def build_manifest() -> dict[str, Any]:
    """Create 2 initial geometries x 3 heights x 2 mirrors with no exact overlap."""
    range_m, azimuth_deg, own_speed, target_speed, altitude = 9150.0, 24.0, 310.0, 370.0, 5200.0
    target_x, target_y = range_m * math.cos(math.radians(azimuth_deg)), range_m * math.sin(math.radians(azimuth_deg))
    scenarios: list[dict[str, Any]] = []
    seed = 206160
    for initial_class, headings, crossing in (("head_on", (150.0, 210.0), False), ("crossing_entry", (78.0, 282.0), True)):
        for height, altitude_delta in (("own_below", 600.0), ("co_altitude", 0.0), ("own_above", -600.0)):
            for mirror, sign, own_heading, target_heading in (("negative", -1.0, 2.0, headings[0]), ("positive", 1.0, 358.0, headings[1])):
                cell = f"{initial_class}__{PACKAGE_ID}__{height}__{mirror}"
                metadata = {
                    "manifest_family": "thesis_temporal_corridor_transfer_v1", "split": "dev12_not_for_training_or_heldout", "initial_class": initial_class,
                    "height_condition": height, "mirror_sign": mirror, "mirror_pair_id": f"{PACKAGE_ID}__{initial_class}__{height}",
                    "distance_speed_package": PACKAGE_ID, "phase_at_reset": "pre_merge", "task_registry_key": "five_state_shared_intent", "crossing_context": crossing,
                    "scenario_seed": seed, "initial_range_m": range_m, "altitude_diff_m": altitude_delta, "own_speed_mps": own_speed, "target_speed_mps": target_speed,
                    "los_azimuth_abs_deg": azimuth_deg, "taxonomy_geometry_state": "transition" if crossing else "head_on", "pilot_family": SOURCE_ID,
                    "identity_kind": "evaluation", "geometry_cell_id": cell, "pair_key": f"{cell}::seed={seed}",
                    "scenario_signature": f"temporal_corridor_transfer_v1::{cell}::seed={seed}", "raw_si_first_pass_event": "three_closing_then_two_opening",
                    "reentry_event": "after_first_pass_two_consecutive_range_rate_lt_minus_25_mps",
                }
                scenarios.append({
                    "name": f"temporal_corridor_{cell}_s{seed}",
                    "own_init": {"position_m": [0.0, 0.0, altitude], "velocity_mps": own_speed, "heading_deg": own_heading},
                    "target_init": {"position_m": [target_x, sign * target_y, altitude + altitude_delta], "velocity_mps": target_speed, "heading_deg": target_heading},
                    "metadata": metadata,
                })
                seed += 1
    overlap = [item["metadata"]["pair_key"] for item in scenarios if _fingerprint(item) in _existing()]
    if overlap:
        raise RuntimeError(f"fresh temporal-corridor manifest overlaps existing states: {overlap}")
    return {
        "source_id": SOURCE_ID + "-MANIFEST12", "manifest_version": 1, "status": "preregistered_design_only",
        "protocol": {"evaluation_only": True, "training_permitted": False, "reward_recording": "prohibited", "opponents_reported_separately": True, "raw_si_only": True, "methods": ["frozen_fixed_head_on", "moderate_left", "moderate_right"]},
        "generation_contract": {"initial_states": ["head_on", "crossing_entry"], "package": {"id": PACKAGE_ID, "initial_range_m": range_m, "own_speed_mps": own_speed, "target_speed_mps": target_speed}, "height_offsets_target_minus_own_m": {"own_below": 600.0, "co_altitude": 0.0, "own_above": -600.0}, "mirror_signs": ["negative", "positive"], "physical_intersection_count": 0},
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "experiment" / "manifests" / "thesis_temporal_corridor_transfer_v1_dev12.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("temporal-corridor manifest output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(build_manifest(), sort_keys=False, allow_unicode=True), encoding="utf-8")
