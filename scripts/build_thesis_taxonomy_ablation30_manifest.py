"""Build the frozen 30-scenario taxonomy ablation manifest.

This is evaluation infrastructure only. It constructs explicit initial states,
recomputes their geometry from positions and headings, and can verify that the
resulting full-state signatures do not overlap earlier evidence manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import yaml

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (
    GeometryTaxonomyConfig,
    classify_relative_geometry,
)


SOURCE_ID = "THESIS-TAXONOMY-ABLATION30-V1"
FROZEN_GIT_SHA = "9a9f9bf6d68560afa81560f5260b78f318dcd9b1"
TAXONOMY_VERSION = "v1"
INITIAL_RANGE_M = 2600.0
OWN_ALTITUDE_M = 5000.0
OWN_SPEED_MPS = 220.0
TARGET_SPEED_MPS = 220.0
LOS_AZIMUTH_ABS_DEG = 12.0
TRANSITION_BAND_DEG = (85.0, 95.0)

HEIGHT_OFFSETS_M = {
    "own_below": 400.0,
    "co_altitude": 0.0,
    "own_above": -400.0,
}

CLASS_ANGLE_TARGETS_DEG = {
    "advantage": (20.0, 160.0),
    "head_on": (20.0, 20.0),
    "disadvantage": (160.0, 20.0),
    "neutral": (160.0, 160.0),
    # Crossing is intentionally separate from the four-quadrant taxonomy.
    "crossing_entry": (20.0, 90.0),
}


def _normalize_heading_deg(value: float) -> float:
    result = value % 360.0
    return 0.0 if math.isclose(result, 360.0, abs_tol=1e-12) else result


def _vector_angle_deg(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return math.degrees(math.acos(cosine))


def _horizontal_heading_for_3d_angle(
    los_azimuth_deg: float,
    los_horizontal_fraction: float,
    desired_angle_deg: float,
    signed_side: int,
) -> float:
    ratio = math.cos(math.radians(desired_angle_deg)) / los_horizontal_fraction
    if ratio < -1.0 - 1e-12 or ratio > 1.0 + 1e-12:
        raise ValueError(
            f"Desired angle {desired_angle_deg} is infeasible for the selected LOS elevation"
        )
    azimuth_delta_deg = math.degrees(math.acos(max(-1.0, min(1.0, ratio))))
    return _normalize_heading_deg(los_azimuth_deg + signed_side * azimuth_delta_deg)


def _heading_vector(heading_deg: float, speed_mps: float) -> tuple[float, float, float]:
    radians = math.radians(heading_deg)
    return (
        speed_mps * math.cos(radians),
        speed_mps * math.sin(radians),
        0.0,
    )


def initial_geometry(scenario: Mapping[str, Any]) -> dict[str, float | str]:
    """Recompute initial geometry without reading scenario names or labels."""

    own = scenario["own_init"]
    target = scenario["target_init"]
    own_position = tuple(float(value) for value in own["position_m"])
    target_position = tuple(float(value) for value in target["position_m"])
    los = tuple(target_position[index] - own_position[index] for index in range(3))
    reverse_los = tuple(-value for value in los)
    own_velocity = _heading_vector(float(own["heading_deg"]), float(own["velocity_mps"]))
    target_velocity = _heading_vector(
        float(target["heading_deg"]), float(target["velocity_mps"])
    )
    own_angle = _vector_angle_deg(own_velocity, los)
    target_angle = _vector_angle_deg(target_velocity, reverse_los)
    initial_range = math.sqrt(sum(value * value for value in los))
    altitude_diff = target_position[2] - own_position[2]
    return {
        "initial_range_m": initial_range,
        "altitude_diff_m": altitude_diff,
        "own_to_target_los_angle_deg": own_angle,
        "target_velocity_to_own_los_angle_deg": target_angle,
        "geometry_state": classify_relative_geometry(
            own_angle,
            target_angle,
            GeometryTaxonomyConfig(transition_half_width_deg=5.0),
        ),
    }


def scenario_signature(scenario: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the physical full-state signature used for evidence disjointness."""

    own = scenario["own_init"]
    target = scenario["target_init"]

    def rounded(values: Iterable[Any]) -> tuple[float, ...]:
        return tuple(round(float(value), 6) for value in values)

    return (
        rounded(own["position_m"]),
        round(float(own["velocity_mps"]), 6),
        round(_normalize_heading_deg(float(own["heading_deg"])), 6),
        rounded(target["position_m"]),
        round(float(target["velocity_mps"]), 6),
        round(_normalize_heading_deg(float(target["heading_deg"])), 6),
    )


def _build_scenario(
    initial_class: str,
    height_condition: str,
    mirror_sign: str,
    scenario_index: int,
) -> dict[str, Any]:
    sign = -1 if mirror_sign == "negative" else 1
    altitude_diff_m = HEIGHT_OFFSETS_M[height_condition]
    horizontal_range_m = math.sqrt(INITIAL_RANGE_M**2 - altitude_diff_m**2)
    los_azimuth_deg = sign * LOS_AZIMUTH_ABS_DEG
    los_azimuth_rad = math.radians(los_azimuth_deg)
    target_position = (
        horizontal_range_m * math.cos(los_azimuth_rad),
        horizontal_range_m * math.sin(los_azimuth_rad),
        OWN_ALTITUDE_M + altitude_diff_m,
    )
    horizontal_fraction = horizontal_range_m / INITIAL_RANGE_M
    own_angle_target, target_angle_target = CLASS_ANGLE_TARGETS_DEG[initial_class]

    # Opposite signed offsets produce exact reflection symmetry about the x-z plane.
    own_heading_deg = _horizontal_heading_for_3d_angle(
        los_azimuth_deg,
        horizontal_fraction,
        own_angle_target,
        -sign,
    )
    target_to_own_azimuth_deg = _normalize_heading_deg(los_azimuth_deg + 180.0)
    target_heading_deg = _horizontal_heading_for_3d_angle(
        target_to_own_azimuth_deg,
        horizontal_fraction,
        target_angle_target,
        sign,
    )

    height_token = {
        "own_below": "below400",
        "co_altitude": "level",
        "own_above": "above400",
    }[height_condition]
    mirror_token = "neg" if sign < 0 else "pos"
    mirror_pair_id = f"{initial_class}__{height_condition}"
    task_registry_key = "crossing_feasible" if initial_class == "crossing_entry" else "head_on"
    scenario = {
        "name": f"taxonomy30_{initial_class}_{height_token}_{mirror_token}",
        "own_init": {
            "position_m": [0.0, 0.0, OWN_ALTITUDE_M],
            "velocity_mps": OWN_SPEED_MPS,
            "heading_deg": round(own_heading_deg, 9),
        },
        "target_init": {
            "position_m": [round(value, 9) for value in target_position],
            "velocity_mps": TARGET_SPEED_MPS,
            "heading_deg": round(target_heading_deg, 9),
        },
        "metadata": {
            "scenario_type": (
                f"crossing_{mirror_sign}" if initial_class == "crossing_entry" else initial_class
            ),
            "manifest_family": "thesis_taxonomy_ablation30_v1",
            "initial_class": initial_class,
            "height_condition": height_condition,
            "mirror_sign": mirror_sign,
            "mirror_pair_id": mirror_pair_id,
            "task_registry_key": task_registry_key,
            "ppo_task_bit": 1 if initial_class == "crossing_entry" else 0,
            "crossing_context": initial_class == "crossing_entry",
            "scenario_seed": 73000 + scenario_index,
            "initial_range_m": INITIAL_RANGE_M,
            "horizontal_range_m": round(horizontal_range_m, 9),
            "altitude_diff_m": altitude_diff_m,
            "lateral_offset_m": round(target_position[1], 9),
        },
    }
    geometry = initial_geometry(scenario)
    scenario["metadata"].update(
        {
            "own_to_target_los_angle_deg": round(
                float(geometry["own_to_target_los_angle_deg"]), 9
            ),
            "target_velocity_to_own_los_angle_deg": round(
                float(geometry["target_velocity_to_own_los_angle_deg"]), 9
            ),
            "taxonomy_geometry_state": geometry["geometry_state"],
        }
    )
    return scenario


def build_manifest() -> dict[str, Any]:
    scenarios = []
    index = 0
    for initial_class in CLASS_ANGLE_TARGETS_DEG:
        for height_condition in HEIGHT_OFFSETS_M:
            for mirror_sign in ("negative", "positive"):
                scenarios.append(
                    _build_scenario(
                        initial_class,
                        height_condition,
                        mirror_sign,
                        index,
                    )
                )
                index += 1

    return {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_before_evaluation",
        "frozen_source": {"git_sha": FROZEN_GIT_SHA, "detached_worktree_required": True},
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "opponent_stages": ["expert", "end_to_end"],
            "methods": [
                "canonical_ppo_high_level_policy",
                "legacy_static_oracle_task_gate",
                "head_on_vpp_specialist_no_routing",
                "crossing_vpp_specialist_no_routing",
            ],
            "episodes_per_scenario_method_opponent": 1,
            "episodes_total": 240,
            "stop_rule": (
                "No training, tuning, scenario replacement, seed replacement, reward change, "
                "VPP change, guidance change, PID change, or result-driven rerun is permitted."
            ),
        },
        "generation_contract": {
            "taxonomy_version": TAXONOMY_VERSION,
            "initial_classes": list(CLASS_ANGLE_TARGETS_DEG),
            "height_offsets_target_minus_own_m": HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "initial_range_m": INITIAL_RANGE_M,
            "own_altitude_m": OWN_ALTITUDE_M,
            "own_speed_mps": OWN_SPEED_MPS,
            "target_speed_mps": TARGET_SPEED_MPS,
            "transition_band_deg": list(TRANSITION_BAND_DEG),
            "legacy_task_bit_contract": {
                "crossing_entry": 1,
                "advantage_head_on_disadvantage_neutral": 0,
            },
        },
        "scenarios": scenarios,
    }


def validate_disjointness(
    scenarios: Iterable[Mapping[str, Any]],
    historical_signatures: Mapping[str, set[tuple[Any, ...]]],
) -> dict[str, Any]:
    candidate = {scenario_signature(scenario) for scenario in scenarios}
    report = {"sources": [], "intersection_count": 0}
    for source_id, signatures in historical_signatures.items():
        overlap = candidate.intersection(signatures)
        report["sources"].append(
            {
                "source_id": source_id,
                "unique_signature_count": len(signatures),
                "intersection_count": len(overlap),
            }
        )
        report["intersection_count"] += len(overlap)
    if report["intersection_count"]:
        raise ValueError(
            f"Generated manifest is not disjoint from historical evidence: {report}"
        )
    return report


def _collect_scenarios(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "own_init" in value and "target_init" in value:
            found.append(value)
        for child in value.values():
            found.extend(_collect_scenarios(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_scenarios(child))
    return found


def load_config_signatures(path: Path, visited: set[Path] | None = None) -> set[tuple[Any, ...]]:
    """Read explicit scenarios from a YAML include chain without executing configs."""

    resolved_path = path.resolve()
    seen = visited if visited is not None else set()
    if resolved_path in seen:
        return set()
    seen.add(resolved_path)
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    signatures = {scenario_signature(scenario) for scenario in _collect_scenarios(payload)}
    for include in payload.get("includes", []) or []:
        include_path = (resolved_path.parent / include).resolve()
        if include_path.exists():
            signatures.update(load_config_signatures(include_path, seen))
    return signatures


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _parse_disjoint_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOURCE_ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not source_id or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid disjoint source: {value}")
    return source_id, path


def _write_frozen_outputs(
    manifest: dict[str, Any], output_path: Path, evidence_root: Path | None, overwrite: bool
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False).encode("utf-8")
    output_path.write_bytes(manifest_bytes)

    if evidence_root is None:
        return
    if evidence_root.exists() and any(evidence_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Evidence root is not empty: {evidence_root}")
    preregistration_dir = evidence_root / "preregistration"
    preregistration_dir.mkdir(parents=True, exist_ok=True)
    (preregistration_dir / output_path.name).write_bytes(manifest_bytes)
    freeze = {
        "source_id": SOURCE_ID,
        "base_git_sha": FROZEN_GIT_SHA,
        "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "manifest_file_sha256": _sha256_bytes(manifest_bytes),
        "scenario_count": len(manifest["scenarios"]),
        "expected_episode_count": manifest["protocol"]["episodes_total"],
        "output_isolation_root": str(evidence_root),
    }
    (preregistration_dir / "manifest_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "config" / "experiment" / "manifests" / "thesis_taxonomy_ablation30_v1.yaml",
    )
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument(
        "--disjoint-source",
        action="append",
        default=[],
        type=_parse_disjoint_source,
        metavar="SOURCE_ID=PATH",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    actual_head = _git_head(root)
    if actual_head != FROZEN_GIT_SHA:
        raise RuntimeError(f"Expected frozen SHA {FROZEN_GIT_SHA}, found {actual_head}")

    manifest = build_manifest()
    historical: dict[str, set[tuple[Any, ...]]] = {}
    source_details = []
    for source_id, path in args.disjoint_source:
        signatures = load_config_signatures(path)
        historical[source_id] = signatures
        source_details.append(
            {
                "source_id": source_id,
                "config_name": path.name,
                "config_sha256": _sha256_file(path),
                "unique_signature_count": len(signatures),
            }
        )
    disjointness = validate_disjointness(manifest["scenarios"], historical)
    disjointness["source_files"] = source_details
    manifest["disjointness"] = disjointness

    classifier_path = root / "src" / "uav_vpp_guidance" / "evaluation" / "engagement_geometry_taxonomy.py"
    manifest["integrity"] = {
        "classifier_code_sha256": _sha256_file(classifier_path),
        "builder_code_sha256": _sha256_file(Path(__file__)),
    }
    manifest["integrity"]["payload_sha256"] = _payload_sha256(manifest)
    _write_frozen_outputs(manifest, args.output, args.evidence_root, args.overwrite)
    print(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "scenario_count": len(manifest["scenarios"]),
                "payload_sha256": manifest["integrity"]["payload_sha256"],
                "disjoint_intersections": disjointness["intersection_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
