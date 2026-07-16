"""Build the frozen, evaluation-only 60-scenario dual-skill taxonomy manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Optional, Set, Tuple

import yaml

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (
    GeometryTaxonomyConfig,
    classify_relative_geometry,
)


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
TAXONOMY_VERSION = "v1"
TRANSITION_BAND_DEG = (85.0, 95.0)

CLASS_ANGLE_TARGETS_DEG = {
    "advantage": (20.0, 160.0),
    "head_on": (20.0, 20.0),
    "disadvantage": (160.0, 20.0),
    "neutral": (160.0, 160.0),
    # Crossing-entry remains a task/phase relation, not a fifth quadrant.
    "crossing_entry": (20.0, 90.0),
}

# Both packages differ from Taxonomy30 and from each other in full physical state.
PACKAGE_SPECS = {
    "wide_balanced": {
        "initial_range_m": 3400.0,
        "own_altitude_m": 5200.0,
        "own_speed_mps": 235.0,
        "target_speed_mps": 215.0,
        "los_azimuth_abs_deg": 16.0,
        "height_offsets_m": {
            "own_below": 500.0,
            "co_altitude": 0.0,
            "own_above": -500.0,
        },
    },
    "fast_closure": {
        "initial_range_m": 3850.0,
        "own_altitude_m": 5500.0,
        "own_speed_mps": 255.0,
        "target_speed_mps": 225.0,
        "los_azimuth_abs_deg": 18.0,
        "height_offsets_m": {
            "own_below": 650.0,
            "co_altitude": 0.0,
            "own_above": -650.0,
        },
    },
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


def _heading_vector(heading_deg: float, speed_mps: float) -> Tuple[float, float, float]:
    radians = math.radians(heading_deg)
    return (
        speed_mps * math.cos(radians),
        speed_mps * math.sin(radians),
        0.0,
    )


def _horizontal_heading_for_3d_angle(
    los_azimuth_deg: float,
    los_horizontal_fraction: float,
    desired_angle_deg: float,
    signed_side: int,
) -> float:
    ratio = math.cos(math.radians(desired_angle_deg)) / los_horizontal_fraction
    if ratio < -1.0 - 1e-12 or ratio > 1.0 + 1e-12:
        raise ValueError(
            "Desired angle {:.3f} is infeasible for the selected LOS elevation".format(
                desired_angle_deg
            )
        )
    azimuth_delta_deg = math.degrees(math.acos(max(-1.0, min(1.0, ratio))))
    return _normalize_heading_deg(los_azimuth_deg + signed_side * azimuth_delta_deg)


def initial_geometry(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recompute geometry from states, headings, and speeds, never scenario labels."""

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
    return {
        "initial_range_m": math.sqrt(sum(value * value for value in los)),
        "altitude_diff_m": target_position[2] - own_position[2],
        "own_to_target_los_angle_deg": own_angle,
        "target_velocity_to_own_los_angle_deg": target_angle,
        "geometry_state": classify_relative_geometry(
            own_angle,
            target_angle,
            GeometryTaxonomyConfig(transition_half_width_deg=5.0),
        ),
    }


def scenario_signature(scenario: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Return a rounded physical initial-state signature for disjointness checks."""

    own = scenario["own_init"]
    target = scenario["target_init"]

    def rounded(values: Iterable[Any]) -> Tuple[float, ...]:
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
    package_name: str,
    initial_class: str,
    height_condition: str,
    mirror_sign: str,
    scenario_index: int,
) -> Mapping[str, Any]:
    package = PACKAGE_SPECS[package_name]
    sign = -1 if mirror_sign == "negative" else 1
    initial_range_m = float(package["initial_range_m"])
    own_altitude_m = float(package["own_altitude_m"])
    altitude_diff_m = float(package["height_offsets_m"][height_condition])
    horizontal_range_m = math.sqrt(initial_range_m**2 - altitude_diff_m**2)
    los_azimuth_deg = sign * float(package["los_azimuth_abs_deg"])
    los_azimuth_rad = math.radians(los_azimuth_deg)
    target_position = (
        horizontal_range_m * math.cos(los_azimuth_rad),
        horizontal_range_m * math.sin(los_azimuth_rad),
        own_altitude_m + altitude_diff_m,
    )
    horizontal_fraction = horizontal_range_m / initial_range_m
    own_angle_target, target_angle_target = CLASS_ANGLE_TARGETS_DEG[initial_class]
    own_heading_deg = _horizontal_heading_for_3d_angle(
        los_azimuth_deg, horizontal_fraction, own_angle_target, -sign
    )
    target_to_own_azimuth_deg = _normalize_heading_deg(los_azimuth_deg + 180.0)
    target_heading_deg = _horizontal_heading_for_3d_angle(
        target_to_own_azimuth_deg,
        horizontal_fraction,
        target_angle_target,
        sign,
    )
    height_token = {
        "own_below": "below",
        "co_altitude": "level",
        "own_above": "above",
    }[height_condition]
    task_registry_key = "crossing_feasible" if initial_class == "crossing_entry" else "head_on"
    scenario = {
        "name": "taxonomy60_{}_{}_{}_{}".format(
            package_name, initial_class, height_token, "neg" if sign < 0 else "pos"
        ),
        "own_init": {
            "position_m": [0.0, 0.0, own_altitude_m],
            "velocity_mps": float(package["own_speed_mps"]),
            "heading_deg": round(own_heading_deg, 9),
        },
        "target_init": {
            "position_m": [round(value, 9) for value in target_position],
            "velocity_mps": float(package["target_speed_mps"]),
            "heading_deg": round(target_heading_deg, 9),
        },
        "metadata": {
            "manifest_family": "thesis_dualskill_taxonomy60_v1",
            "geometry_package": package_name,
            "initial_class": initial_class,
            "height_condition": height_condition,
            "mirror_sign": mirror_sign,
            "mirror_pair_id": "{}__{}__{}".format(
                package_name, initial_class, height_condition
            ),
            "task_registry_key": task_registry_key,
            "ppo_task_bit": 1 if initial_class == "crossing_entry" else 0,
            "crossing_context": initial_class == "crossing_entry",
            "scenario_seed": 79000 + scenario_index,
            "initial_range_m": initial_range_m,
            "own_speed_mps": float(package["own_speed_mps"]),
            "target_speed_mps": float(package["target_speed_mps"]),
            "altitude_diff_m": altitude_diff_m,
            "horizontal_range_m": round(horizontal_range_m, 9),
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


def build_manifest(frozen_git_sha: str) -> Mapping[str, Any]:
    scenarios = []
    scenario_index = 0
    for package_name in PACKAGE_SPECS:
        for initial_class in CLASS_ANGLE_TARGETS_DEG:
            for height_condition in PACKAGE_SPECS[package_name]["height_offsets_m"]:
                for mirror_sign in ("negative", "positive"):
                    scenarios.append(
                        _build_scenario(
                            package_name,
                            initial_class,
                            height_condition,
                            mirror_sign,
                            scenario_index,
                        )
                    )
                    scenario_index += 1
    return {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_before_evaluation",
        "frozen_source": {
            "generation_git_sha": frozen_git_sha,
            "detached_worktree_required": True,
        },
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "opponent_stages": ["expert", "end_to_end", "independent_ppo_vpp"],
            "methods": [
                "fixed_head_on",
                "fixed_crossing",
                "static_oracle_gate",
                "canonical_ppo",
                "ppo_unlocked_crossing",
            ],
            "episodes_per_scenario_method_opponent": 1,
            "episodes_total": 900,
            "stop_rule": (
                "No training, tuning, scenario replacement, seed replacement, reward change, "
                "VPP change, guidance change, PID change, or result-driven rerun is permitted."
            ),
        },
        "generation_contract": {
            "taxonomy_version": TAXONOMY_VERSION,
            "initial_classes": list(CLASS_ANGLE_TARGETS_DEG),
            "geometry_packages": PACKAGE_SPECS,
            "mirror_signs": ["negative", "positive"],
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
    historical_signatures: Mapping[str, Set[Tuple[Any, ...]]],
) -> Mapping[str, Any]:
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
        raise ValueError("Generated manifest is not disjoint: {}".format(report))
    return report


def _collect_scenarios(value: Any) -> list:
    found = []
    if isinstance(value, Mapping):
        if "own_init" in value and "target_init" in value:
            found.append(value)
        for child in value.values():
            found.extend(_collect_scenarios(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_scenarios(child))
    return found


def load_config_signatures(path: Path, visited: Optional[Set[Path]] = None) -> Set[Tuple[Any, ...]]:
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _parse_disjoint_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOURCE_ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not source_id or not path.is_file():
        raise argparse.ArgumentTypeError("Invalid disjoint source: {}".format(value))
    return source_id, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml",
    )
    parser.add_argument("--source-git-sha", default=None)
    parser.add_argument(
        "--disjoint-source",
        action="append",
        default=[],
        type=_parse_disjoint_source,
        metavar="SOURCE_ID=PATH",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite manifest: {}".format(args.output))
    manifest = dict(build_manifest(args.source_git_sha or _git_head(root)))
    historical = {}
    source_files = []
    for source_id, path in args.disjoint_source:
        signatures = load_config_signatures(path)
        historical[source_id] = signatures
        source_files.append(
            {
                "source_id": source_id,
                "config_name": path.name,
                "config_sha256": _sha256_file(path),
                "unique_signature_count": len(signatures),
            }
        )
    disjointness = dict(validate_disjointness(manifest["scenarios"], historical))
    disjointness["source_files"] = source_files
    manifest["disjointness"] = disjointness
    manifest["integrity"] = {
        "builder_code_sha256": _sha256_file(Path(__file__)),
        "classifier_code_sha256": _sha256_file(
            root / "src" / "uav_vpp_guidance" / "evaluation" / "engagement_geometry_taxonomy.py"
        ),
    }
    manifest["integrity"]["payload_sha256"] = _payload_sha256(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
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
