"""Build the independent five-state held-out 40-scenario envelope.

This evaluation-only manifest is disjoint from Taxonomy30 and the previous
five-state dev30/heldout60 manifests.  It deliberately retains the legacy
two-task key only as an input compatibility contract for frozen policies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_thesis_taxonomy_ablation30_manifest import (  # noqa: E402
    _heading_vector,
    _horizontal_heading_for_3d_angle,
    _normalize_heading_deg,
    _vector_angle_deg,
    load_config_signatures,
    scenario_signature,
)
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (  # noqa: E402
    GeometryTaxonomyConfig,
    classify_relative_geometry,
)


SOURCE_ID = "THESIS-FIVE-STATE-HELDOUT40-V1"
BASE_GIT_SHA = "9413834e96c60188bd8bef92c2d3177909ca53bb"
TAXONOMY_VERSION = "v1"
OWN_ALTITUDE_M = 5200.0
LOS_AZIMUTH_ABS_DEG = 24.0
EPISODE_HORIZON_HIGH_LEVEL_STEPS = 160
TRANSITION_BAND_DEG = (85.0, 95.0)

INITIAL_CLASS_ANGLES_DEG = {
    "advantage": (20.0, 160.0),
    "head_on": (20.0, 20.0),
    "disadvantage": (160.0, 20.0),
    "neutral": (160.0, 160.0),
    "crossing_entry": (20.0, 90.0),
}
HEIGHT_OFFSETS_M = {
    "own_below": 600.0,
    "own_above": -600.0,
}
DISTANCE_SPEED_PACKAGES = (
    {
        "id": "heldout40_closing_a",
        "initial_range_m": 3000.0,
        "own_speed_mps": 235.0,
        "target_speed_mps": 260.0,
    },
    {
        "id": "heldout40_closing_b",
        "initial_range_m": 3900.0,
        "own_speed_mps": 275.0,
        "target_speed_mps": 225.0,
    },
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def initial_geometry(scenario: Mapping[str, Any]) -> dict[str, float | str]:
    """Recompute geometry from physical state only, never from scenario labels."""

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
        "taxonomy_geometry_state": classify_relative_geometry(
            own_angle,
            target_angle,
            GeometryTaxonomyConfig(transition_half_width_deg=5.0),
        ),
    }


def _build_scenario(
    *,
    package: Mapping[str, Any],
    initial_class: str,
    height_condition: str,
    mirror_sign: str,
    scenario_index: int,
) -> dict[str, Any]:
    sign = -1 if mirror_sign == "negative" else 1
    initial_range_m = float(package["initial_range_m"])
    altitude_diff_m = HEIGHT_OFFSETS_M[height_condition]
    horizontal_range_m = math.sqrt(initial_range_m**2 - altitude_diff_m**2)
    los_azimuth_deg = sign * LOS_AZIMUTH_ABS_DEG
    los_azimuth_rad = math.radians(los_azimuth_deg)
    target_position = (
        horizontal_range_m * math.cos(los_azimuth_rad),
        horizontal_range_m * math.sin(los_azimuth_rad),
        OWN_ALTITUDE_M + altitude_diff_m,
    )
    horizontal_fraction = horizontal_range_m / initial_range_m
    own_angle, target_angle = INITIAL_CLASS_ANGLES_DEG[initial_class]
    own_heading = _horizontal_heading_for_3d_angle(
        los_azimuth_deg, horizontal_fraction, own_angle, -sign
    )
    target_heading = _horizontal_heading_for_3d_angle(
        _normalize_heading_deg(los_azimuth_deg + 180.0),
        horizontal_fraction,
        target_angle,
        sign,
    )
    scenario = {
        "name": (
            f"heldout40_{package['id']}_{initial_class}_{height_condition}_"
            f"{'neg' if sign < 0 else 'pos'}"
        ),
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
            "manifest_family": "noncanonical_thesis_five_state_heldout_envelope_v1",
            "initial_class": initial_class,
            "height_condition": height_condition,
            "mirror_sign": mirror_sign,
            "mirror_pair_id": f"{package['id']}__{initial_class}__{height_condition}",
            "distance_speed_package": package["id"],
            "phase_at_reset": "pre_merge",
            "task_registry_key": (
                "crossing_feasible" if initial_class == "crossing_entry" else "head_on"
            ),
            "ppo_task_bit": 1 if initial_class == "crossing_entry" else 0,
            "crossing_context": initial_class == "crossing_entry",
            "scenario_seed": 78100 + scenario_index,
            "initial_range_m": initial_range_m,
            "altitude_diff_m": altitude_diff_m,
            "own_speed_mps": float(package["own_speed_mps"]),
            "target_speed_mps": float(package["target_speed_mps"]),
            "los_azimuth_abs_deg": LOS_AZIMUTH_ABS_DEG,
        },
    }
    scenario["metadata"].update(
        {
            key: round(float(value), 9) if isinstance(value, float) else value
            for key, value in initial_geometry(scenario).items()
        }
    )
    return scenario


def build_manifest() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    index = 0
    for package in DISTANCE_SPEED_PACKAGES:
        for initial_class in INITIAL_CLASS_ANGLES_DEG:
            for height_condition in HEIGHT_OFFSETS_M:
                for mirror_sign in ("negative", "positive"):
                    scenarios.append(
                        _build_scenario(
                            package=package,
                            initial_class=initial_class,
                            height_condition=height_condition,
                            mirror_sign=mirror_sign,
                            scenario_index=index,
                        )
                    )
                    index += 1
    return {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_before_evaluation",
        "frozen_source": {
            "git_sha": BASE_GIT_SHA,
            "clean_worktree_required": True,
        },
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "episode_horizon_high_level_steps": EPISODE_HORIZON_HIGH_LEVEL_STEPS,
            "high_level_dt_s": 0.2,
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
            "methods": [
                "canonical_ppo_high_level_policy",
                "legacy_static_oracle_task_gate",
                "head_on_vpp_specialist_no_routing",
                "crossing_vpp_specialist_no_routing",
            ],
            "episodes_per_scenario_method_opponent": 1,
            "episodes_total": 480,
            "stop_rule": (
                "No training, tuning, scenario replacement, seed replacement, reward change, "
                "VPP change, guidance change, PID change, or result-driven rerun is permitted."
            ),
        },
        "generation_contract": {
            "taxonomy_version": TAXONOMY_VERSION,
            "initial_classes": list(INITIAL_CLASS_ANGLES_DEG),
            "height_offsets_target_minus_own_m": HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "distance_speed_packages": list(DISTANCE_SPEED_PACKAGES),
            "own_altitude_m": OWN_ALTITUDE_M,
            "los_azimuth_abs_deg": LOS_AZIMUTH_ABS_DEG,
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
    report: dict[str, Any] = {"sources": [], "intersection_count": 0}
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
        raise ValueError(f"Generated heldout40 overlaps earlier evidence: {report}")
    return report


def _load_manifest_signatures(path: Path) -> set[tuple[Any, ...]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise TypeError(f"Manifest scenarios must be a list: {path}")
    return {scenario_signature(scenario) for scenario in scenarios}


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _write_outputs(
    manifest: Mapping[str, Any],
    output_path: Path,
    preregistration_path: Path | None,
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False).encode("utf-8")
    output_path.write_bytes(output_bytes)
    if preregistration_path is not None:
        preregistration_path.parent.mkdir(parents=True, exist_ok=True)
        if preregistration_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite preregistration copy: {preregistration_path}")
        preregistration_path.write_bytes(output_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_heldout40_v1.yaml",
    )
    parser.add_argument(
        "--preregistration-copy",
        type=Path,
        default=REPO_ROOT / "reports" / "thesis_five_state_heldout40_v1_manifest_freeze.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    actual_head = _git_head(REPO_ROOT)
    if actual_head != BASE_GIT_SHA:
        raise RuntimeError(f"Expected clean base SHA {BASE_GIT_SHA}, found {actual_head}")

    historical_paths = {
        "THESIS-TAXONOMY-ABLATION30-V1": REPO_ROOT
        / "config"
        / "experiment"
        / "manifests"
        / "thesis_taxonomy_ablation30_v1.yaml",
        "THESIS-FIVE-STATE-SHARED-INTENT-V1-DEV30": REPO_ROOT
        / "config"
        / "experiment"
        / "manifests"
        / "thesis_five_state_shared_intent_v1_dev30.yaml",
        "THESIS-FIVE-STATE-SHARED-INTENT-V1-HELDOUT60": REPO_ROOT
        / "config"
        / "experiment"
        / "manifests"
        / "thesis_five_state_shared_intent_v1_heldout60.yaml",
    }
    missing = [str(path) for path in historical_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required disjointness source(s): " + "; ".join(missing))
    historical = {
        source_id: _load_manifest_signatures(path)
        for source_id, path in historical_paths.items()
    }
    manifest = build_manifest()
    manifest["disjointness"] = validate_disjointness(manifest["scenarios"], historical)
    manifest["disjointness"]["source_files"] = [
        {
            "source_id": source_id,
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for source_id, path in historical_paths.items()
    ]
    classifier_path = REPO_ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "engagement_geometry_taxonomy.py"
    manifest["integrity"] = {
        "builder_code_sha256": _sha256_file(Path(__file__)),
        "classifier_code_sha256": _sha256_file(classifier_path),
    }
    manifest["integrity"]["payload_sha256"] = _payload_sha256(manifest)
    _write_outputs(
        manifest,
        args.output,
        args.preregistration_copy,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "scenario_count": len(manifest["scenarios"]),
                "payload_sha256": manifest["integrity"]["payload_sha256"],
                "disjoint_intersections": manifest["disjointness"]["intersection_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
