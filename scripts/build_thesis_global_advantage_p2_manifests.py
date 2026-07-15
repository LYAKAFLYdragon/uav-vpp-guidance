"""Build the independent heldout240 manifest for GLOBAL-ADVANTAGE-V1 P2.

This is a manifest-only builder.  It does not construct an environment, load a
policy, collect telemetry, or permit training.  The separate physical
reachability gate must validate the generated cells before any evaluation use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_thesis_five_state_manifests as v1
from build_thesis_taxonomy_ablation30_manifest import scenario_signature


MANIFEST_DIR = REPO_ROOT / "config" / "experiment" / "manifests"
TRAIN_DISTRIBUTION_PATH = (
    REPO_ROOT
    / "config"
    / "experiment"
    / "thesis_five_state_shared_intent_v1_train_distribution.yaml"
)
DEV30_PATH = MANIFEST_DIR / "thesis_five_state_shared_intent_v1_dev30.yaml"
HISTORICAL_HELDOUT60_PATH = (
    MANIFEST_DIR / "thesis_five_state_shared_intent_v1_heldout60.yaml"
)
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-HELDOUT240-V1"
EVALUATION_SEEDS = (11011, 11029, 11047, 11071)
GLOBAL_HELDOUT_PACKAGES = (
    {
        "id": "global_heldout_far_fast_c",
        "initial_range_m": 3000.0,
        "own_speed_mps": 245.0,
        "target_speed_mps": 235.0,
    },
    {
        "id": "global_heldout_far_fast_d",
        "initial_range_m": 3400.0,
        "own_speed_mps": 255.0,
        "target_speed_mps": 245.0,
    },
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _instance_signature(scenario: Mapping[str, Any]) -> str:
    """Hash a complete seeded episode contract, not only its geometry."""

    return hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scenario(
    *,
    package: Mapping[str, Any],
    initial_class: str,
    height_condition: str,
    mirror_sign: str,
    geometry_index: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    scenario = v1._build_scenario(
        split="heldout240",
        package=package,
        initial_class=initial_class,
        height_condition=height_condition,
        mirror_sign=mirror_sign,
        scenario_index=geometry_index,
    )
    metadata = scenario["metadata"]
    geometry_cell_id = (
        f"{package['id']}__{initial_class}__{height_condition}__{mirror_sign}"
    )
    scenario["name"] = f"global_advantage_heldout240_{geometry_cell_id}_s{evaluation_seed}"
    metadata.update(
        {
            "manifest_family": "thesis_global_advantage_v1",
            "split": "heldout240",
            "scenario_seed": int(evaluation_seed),
            "evaluation_seed": int(evaluation_seed),
            "geometry_cell_id": geometry_cell_id,
            "global_advantage_evaluation": True,
            "physical_geometry_signature": scenario_signature(scenario),
        }
    )
    return scenario


def _build_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    geometry_index = 0
    for package in GLOBAL_HELDOUT_PACKAGES:
        for initial_class in v1.INITIAL_CLASS_ANGLES_DEG:
            for height_condition in v1.HEIGHT_OFFSETS_M:
                for mirror_sign in ("negative", "positive"):
                    for evaluation_seed in EVALUATION_SEEDS:
                        scenarios.append(
                            _scenario(
                                package=package,
                                initial_class=initial_class,
                                height_condition=height_condition,
                                mirror_sign=mirror_sign,
                                geometry_index=geometry_index,
                                evaluation_seed=evaluation_seed,
                            )
                        )
                    geometry_index += 1
    return scenarios


def _validate(
    manifest: Mapping[str, Any],
    train_distribution: Mapping[str, Any],
    dev30: Mapping[str, Any],
    historical_heldout60: Mapping[str, Any],
) -> None:
    scenarios = list(manifest["scenarios"])
    if len(scenarios) != 240:
        raise ValueError(f"Expected 240 held-out scenarios, found {len(scenarios)}")
    if len({item["name"] for item in scenarios}) != 240:
        raise ValueError("heldout240 scenario names must be unique")
    instance_signatures = {_instance_signature(item) for item in scenarios}
    if len(instance_signatures) != 240:
        raise ValueError("heldout240 seeded episode signatures must be unique")
    physical_signatures = {scenario_signature(item) for item in scenarios}
    if len(physical_signatures) != 60:
        raise ValueError("heldout240 must contain exactly 60 physical geometry signatures")
    if not all(v1._outside_train_support(item, train_distribution) for item in scenarios):
        raise ValueError("heldout240 leaks into continuous training support")

    cells: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        metadata = scenario["metadata"]
        cells.setdefault(str(metadata["geometry_cell_id"]), []).append(scenario)
        own_angle = float(metadata["own_to_target_los_angle_deg"])
        target_angle = float(metadata["target_velocity_to_own_los_angle_deg"])
        if metadata["initial_class"] == "crossing_entry":
            if target_angle != 90.0 or metadata["taxonomy_geometry_state"] != "transition":
                raise ValueError("crossing-entry must retain the explicit transition geometry")
        elif (
            v1.TRANSITION_BAND_DEG[0] <= own_angle <= v1.TRANSITION_BAND_DEG[1]
            or v1.TRANSITION_BAND_DEG[0] <= target_angle <= v1.TRANSITION_BAND_DEG[1]
        ):
            raise ValueError("quadrant geometry intersects the transition band")
    if len(cells) != 60:
        raise ValueError(f"Expected 60 geometry cells, found {len(cells)}")
    for cell_id, members in cells.items():
        if len(members) != len(EVALUATION_SEEDS):
            raise ValueError(f"{cell_id} does not contain four evaluation seeds")
        if {int(item["metadata"]["evaluation_seed"]) for item in members} != set(
            EVALUATION_SEEDS
        ):
            raise ValueError(f"{cell_id} evaluation seeds drifted")

    for label, source in (("dev30", dev30), ("historical_heldout60", historical_heldout60)):
        source_instances = {_instance_signature(item) for item in source["scenarios"]}
        source_physical = {scenario_signature(item) for item in source["scenarios"]}
        instance_overlap = instance_signatures.intersection(source_instances)
        physical_overlap = physical_signatures.intersection(source_physical)
        if instance_overlap or physical_overlap:
            raise ValueError(
                f"heldout240 overlaps {label}: "
                f"instances={len(instance_overlap)}, physical={len(physical_overlap)}"
            )


def build_manifest(
    train_distribution: Mapping[str, Any],
    dev30: Mapping[str, Any],
    historical_heldout60: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "lane": "noncanonical_thesis_extension",
        "family": "thesis_five_state_shared_intent_v1",
        "split": "heldout240",
        "protocol": {
            "training_permitted": False,
            "tuning_permitted": False,
            "selection_permitted": False,
            "allowed_use": "formal_evaluation_only_after_p2_physical_reachability_pass",
            "opponents": ["expert_rule_based", "end_to_end_neural", "independent_ppo_vpp"],
            "opponents_reported_separately": True,
            "evaluation_seeds": list(EVALUATION_SEEDS),
        },
        "generation_contract": {
            "initial_classes": list(v1.INITIAL_CLASS_ANGLES_DEG),
            "height_offsets_target_minus_own_m": v1.HEIGHT_OFFSETS_M,
            "mirror_signs": ["negative", "positive"],
            "evaluation_packages": list(GLOBAL_HELDOUT_PACKAGES),
            "geometry_cell_count": 60,
            "evaluation_seed_count_per_cell": 4,
            "own_altitude_m": v1.OWN_ALTITUDE_M,
            "transition_band_deg": list(v1.TRANSITION_BAND_DEG),
            "train_distribution_config": str(TRAIN_DISTRIBUTION_PATH.relative_to(REPO_ROOT)),
            "unseen_distance_speed_packages": True,
        },
        "scenarios": _build_scenarios(),
        "disjointness": {
            "dev30_path": str(DEV30_PATH.relative_to(REPO_ROOT)),
            "dev30_sha256": _sha256_file(DEV30_PATH),
            "dev30_instance_intersection_count": 0,
            "dev30_physical_intersection_count": 0,
            "historical_heldout60_path": str(HISTORICAL_HELDOUT60_PATH.relative_to(REPO_ROOT)),
            "historical_heldout60_sha256": _sha256_file(HISTORICAL_HELDOUT60_PATH),
            "historical_heldout60_instance_intersection_count": 0,
            "historical_heldout60_physical_intersection_count": 0,
            "train_support_intersection_count": 0,
        },
    }
    _validate(manifest, train_distribution, dev30, historical_heldout60)
    manifest["integrity"] = {"builder_code_sha256": _sha256_file(Path(__file__))}
    manifest["integrity"]["payload_sha256"] = _payload_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MANIFEST_DIR / "thesis_global_advantage_v1_heldout240.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = build_manifest(
        _load_yaml(TRAIN_DISTRIBUTION_PATH),
        _load_yaml(DEV30_PATH),
        _load_yaml(HISTORICAL_HELDOUT60_PATH),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "count": len(manifest["scenarios"]), "sha256": manifest["integrity"]["payload_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
