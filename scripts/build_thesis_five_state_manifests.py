"""Build frozen dev30 and heldout60 manifests for the five-state v1 lane.

The evaluation packages intentionally sit outside the continuous training
distance-speed support.  The script creates explicit geometries, recomputes
their taxonomy angles, and records full-state-signature disjointness.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from build_thesis_taxonomy_ablation30_manifest import (
    _heading_vector,
    _horizontal_heading_for_3d_angle,
    _normalize_heading_deg,
    _vector_angle_deg,
    load_config_signatures,
    scenario_signature,
)
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (
    GeometryTaxonomyConfig,
    classify_relative_geometry,
)


MANIFEST_DIR = REPO_ROOT / "config" / "experiment" / "manifests"
TRAIN_DISTRIBUTION_PATH = (
    REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml"
)
SOURCE_GIT_SHA = "11cedce"
TAXONOMY_VERSION = "v1"
OWN_ALTITUDE_M = 5200.0
LOS_AZIMUTH_ABS_DEG = 18.0
TRANSITION_BAND_DEG = (85.0, 95.0)

INITIAL_CLASS_ANGLES_DEG = {
    "advantage": (20.0, 160.0),
    "head_on": (20.0, 20.0),
    "disadvantage": (160.0, 20.0),
    "neutral": (160.0, 160.0),
    # Crossing is an orthogonal task/phase label, retained at the explicit
    # transition boundary rather than silently reclassified as head-on.
    "crossing_entry": (20.0, 90.0),
}
HEIGHT_OFFSETS_M = {
    "own_below": 600.0,
    "co_altitude": 0.0,
    "own_above": -600.0,
}
DEV_PACKAGE = {
    "id": "dev_outside_train_midrange",
    "initial_range_m": 2800.0,
    "own_speed_mps": 235.0,
    "target_speed_mps": 230.0,
}
HELDOUT_PACKAGES = (
    {
        "id": "heldout_unseen_far_fast_a",
        "initial_range_m": 3200.0,
        "own_speed_mps": 250.0,
        "target_speed_mps": 240.0,
    },
    {
        "id": "heldout_unseen_far_fast_b",
        "initial_range_m": 3600.0,
        "own_speed_mps": 265.0,
        "target_speed_mps": 255.0,
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


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return loaded


def _initial_geometry(scenario: Mapping[str, Any]) -> dict[str, float | str]:
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
    split: str,
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
            f"five_state_{split}_{package['id']}_{initial_class}_"
            f"{height_condition}_{'neg' if sign < 0 else 'pos'}"
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
            "manifest_family": "thesis_five_state_shared_intent_v1",
            "split": split,
            "initial_class": initial_class,
            "height_condition": height_condition,
            "mirror_sign": mirror_sign,
            "mirror_pair_id": f"{package['id']}__{initial_class}__{height_condition}",
            "distance_speed_package": package["id"],
            "phase_at_reset": "pre_merge",
            "task_registry_key": "five_state_shared_intent",
            "crossing_context": initial_class == "crossing_entry",
            "scenario_seed": 76100 + scenario_index,
            "initial_range_m": initial_range_m,
            "altitude_diff_m": altitude_diff_m,
            "own_speed_mps": float(package["own_speed_mps"]),
            "target_speed_mps": float(package["target_speed_mps"]),
            "los_azimuth_abs_deg": LOS_AZIMUTH_ABS_DEG,
        },
    }
    scenario["metadata"].update(
        {key: round(float(value), 9) if isinstance(value, float) else value
         for key, value in _initial_geometry(scenario).items()}
    )
    return scenario


def _build_scenarios(split: str, packages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    index = 0
    for package in packages:
        for initial_class in INITIAL_CLASS_ANGLES_DEG:
            for height_condition in HEIGHT_OFFSETS_M:
                for mirror_sign in ("negative", "positive"):
                    scenarios.append(
                        _build_scenario(
                            split=split,
                            package=package,
                            initial_class=initial_class,
                            height_condition=height_condition,
                            mirror_sign=mirror_sign,
                            scenario_index=index,
                        )
                    )
                    index += 1
    return scenarios


def _outside_train_support(
    scenario: Mapping[str, Any], train_distribution: Mapping[str, Any]
) -> bool:
    support = train_distribution["sampling_contract"]["continuous_support"]
    metadata = scenario["metadata"]
    for field in ("initial_range_m", "own_speed_mps", "target_speed_mps"):
        lower, upper = (float(value) for value in support[field])
        value = float(metadata[field])
        if value < lower or value > upper:
            return True
    return False


def _validate_manifest(
    manifest: Mapping[str, Any], train_distribution: Mapping[str, Any]
) -> None:
    scenarios = manifest["scenarios"]
    expected_count = 30 if manifest["split"] == "dev30" else 60
    if len(scenarios) != expected_count:
        raise ValueError(f"Expected {expected_count} scenarios, found {len(scenarios)}")
    if len({scenario["name"] for scenario in scenarios}) != len(scenarios):
        raise ValueError("Scenario names must be unique")
    if len({scenario_signature(scenario) for scenario in scenarios}) != len(scenarios):
        raise ValueError("Full-state signatures must be unique")
    if not all(_outside_train_support(scenario, train_distribution) for scenario in scenarios):
        raise ValueError("Evaluation scenario leaks into the continuous training support")
    for scenario in scenarios:
        metadata = scenario["metadata"]
        own_angle = float(metadata["own_to_target_los_angle_deg"])
        target_angle = float(metadata["target_velocity_to_own_los_angle_deg"])
        if metadata["initial_class"] != "crossing_entry" and (
            TRANSITION_BAND_DEG[0] <= own_angle <= TRANSITION_BAND_DEG[1]
            or TRANSITION_BAND_DEG[0] <= target_angle <= TRANSITION_BAND_DEG[1]
        ):
            raise ValueError("Quadrant scenario intersects the transition band")
        if metadata["initial_class"] == "crossing_entry" and not math.isclose(
            target_angle, 90.0, abs_tol=1e-5
        ):
            raise ValueError("Crossing-entry scenario must retain the explicit 90 degree boundary")


def _build_manifest(
    split: str, packages: Iterable[Mapping[str, Any]], train_distribution: Mapping[str, Any]
) -> dict[str, Any]:
    scenarios = _build_scenarios(split, packages)
    protocol = {
        "training_permitted": False,
        "tuning_permitted": False,
        "opponents": ["expert_rule_based", "end_to_end_neural", "independent_ppo_vpp"],
        "opponents_reported_separately": True,
        "seeds_for_future_formal": [11, 29, 47],
        "selection_permitted": split == "dev30",
        "taxonomy30_usage": "diagnostic_baseline_only",
    }
    if split == "dev30":
        protocol["allowed_use"] = "checkpoint_selection_and_small_ablation_only"
        source_id = "THESIS-FIVE-STATE-SHARED-INTENT-V1-DEV30"
    else:
        protocol["allowed_use"] = "final_formal_evaluation_only"
        source_id = "THESIS-FIVE-STATE-SHARED-INTENT-V1-HELDOUT60"
    manifest = {
        "source_id": source_id,
        "manifest_version": 1,
        "lane": "noncanonical_thesis_extension",
        "family": "thesis_five_state_shared_intent_v1",
        "split": split,
        "frozen_source": {"git_sha": SOURCE_GIT_SHA, "taxonomy_version": TAXONOMY_VERSION},
        "protocol": protocol,
        "generation_contract": {
            "initial_classes": list(INITIAL_CLASS_ANGLES_DEG),
            "height_offsets_target_minus_own_m": HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "evaluation_packages": list(packages),
            "own_altitude_m": OWN_ALTITUDE_M,
            "transition_band_deg": list(TRANSITION_BAND_DEG),
            "train_distribution_config": str(TRAIN_DISTRIBUTION_PATH.relative_to(REPO_ROOT)),
            "unseen_distance_speed_packages": True,
        },
        "scenarios": scenarios,
    }
    _validate_manifest(manifest, train_distribution)
    return manifest


def _disjointness_report(
    manifests: Mapping[str, Mapping[str, Any]], sources: Mapping[str, Path]
) -> dict[str, Any]:
    signatures = {
        split: {scenario_signature(scenario) for scenario in manifest["scenarios"]}
        for split, manifest in manifests.items()
    }
    overlap = signatures["dev30"].intersection(signatures["heldout60"])
    if overlap:
        raise ValueError(f"dev30 and heldout60 overlap in {len(overlap)} full signatures")
    report: dict[str, Any] = {
        "dev30_heldout60_intersection_count": 0,
        "historical_sources": [],
    }
    for source_id, path in sources.items():
        historical = load_config_signatures(path)
        counts = {
            split: len(signature_set.intersection(historical))
            for split, signature_set in signatures.items()
        }
        if any(counts.values()):
            raise ValueError(f"P2 manifest overlaps historical source {source_id}: {counts}")
        report["historical_sources"].append(
            {
                "source_id": source_id,
                "path": str(path),
                "sha256": _sha256_file(path),
                "unique_signature_count": len(historical),
                "dev30_intersection_count": counts["dev30"],
                "heldout60_intersection_count": counts["heldout60"],
            }
        )
    return report


def build_manifests(
    train_distribution: Mapping[str, Any],
    historical_sources: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    manifests = {
        "dev30": _build_manifest("dev30", (DEV_PACKAGE,), train_distribution),
        "heldout60": _build_manifest("heldout60", HELDOUT_PACKAGES, train_distribution),
    }
    sources = historical_sources or {}
    disjointness = _disjointness_report(manifests, sources)
    for manifest in manifests.values():
        manifest["disjointness"] = copy.deepcopy(disjointness)
        manifest["integrity"] = {"builder_code_sha256": _sha256_file(Path(__file__))}
        manifest["integrity"]["payload_sha256"] = _payload_sha256(manifest)
    return manifests


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOURCE_ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not source_id or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid disjoint source: {value}")
    return source_id, path


def _write_manifest(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-distribution", type=Path, default=TRAIN_DISTRIBUTION_PATH)
    parser.add_argument("--dev-output", type=Path, default=MANIFEST_DIR / "thesis_five_state_shared_intent_v1_dev30.yaml")
    parser.add_argument("--heldout-output", type=Path, default=MANIFEST_DIR / "thesis_five_state_shared_intent_v1_heldout60.yaml")
    parser.add_argument("--disjoint-source", action="append", default=[], type=_parse_source)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = dict(args.disjoint_source)
    train_distribution = _load_yaml(args.train_distribution)
    manifests = build_manifests(train_distribution, sources)
    _write_manifest(args.dev_output, manifests["dev30"], args.overwrite)
    _write_manifest(args.heldout_output, manifests["heldout60"], args.overwrite)
    print(json.dumps({
        "dev30": {"count": len(manifests["dev30"]["scenarios"]), "sha256": manifests["dev30"]["integrity"]["payload_sha256"]},
        "heldout60": {"count": len(manifests["heldout60"]["scenarios"]), "sha256": manifests["heldout60"]["integrity"]["payload_sha256"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
