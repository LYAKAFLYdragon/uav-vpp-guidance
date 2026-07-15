"""Build the preregistered B6 opponent-conditional reachability envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_thesis_five_state_manifests as v1
from build_thesis_taxonomy_ablation30_manifest import scenario_signature


MANIFEST_DIR = ROOT / "config" / "experiment" / "manifests"
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-MANIFEST60-R1"
SEED_BASE = 16201
PACKAGES = (
    {
        "id": "b6_midrange_fast_target_a",
        "initial_range_m": 3400.0,
        "own_speed_mps": 240.0,
        "target_speed_mps": 350.0,
    },
    {
        "id": "b6_farrange_fast_target_b",
        "initial_range_m": 4200.0,
        "own_speed_mps": 265.0,
        "target_speed_mps": 370.0,
    },
)
DISJOINT_MANIFESTS = (
    ("p2_b2", MANIFEST_DIR / "thesis_global_advantage_v1_p2_physical_preflight60.yaml"),
    ("p2_b3", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b3_phase_observability30.yaml"),
    ("p2_b4", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b4_raw_si_phase30.yaml"),
    ("p2_b5", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b5_phase_feasible_sampler12.yaml"),
    ("heldout240", MANIFEST_DIR / "thesis_global_advantage_v1_heldout240.yaml"),
    ("dev30", MANIFEST_DIR / "thesis_five_state_shared_intent_v1_dev30.yaml"),
    ("historical_heldout60", MANIFEST_DIR / "thesis_five_state_shared_intent_v1_heldout60.yaml"),
    ("phase_v2", MANIFEST_DIR / "thesis_phase_reachability_v2_run_in_handoff_r1.yaml"),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
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


def _scenario(package: Mapping[str, Any], initial_class: str, height: str, mirror: str, index: int) -> dict[str, Any]:
    scenario = v1._build_scenario(
        split="p2_b6_opponent_conditional_reachability60",
        package=package,
        initial_class=initial_class,
        height_condition=height,
        mirror_sign=mirror,
        scenario_index=index,
    )
    seed = SEED_BASE + index
    metadata = scenario["metadata"]
    cell = f"{package['id']}__{initial_class}__{height}__{mirror}"
    scenario["name"] = f"global_advantage_p2_b6_{cell}_s{seed}"
    metadata.update(
        {
            "manifest_family": "noncanonical_thesis_global_advantage_p2_b6_reachability",
            "split": "p2_b6_opponent_conditional_reachability60",
            "scenario_seed": seed,
            "geometry_cell_id": cell,
            "b6_protocol": "profile_free_dynamic_state_phase_reachability_only",
            "phase_at_reset": "pre_merge",
            "physical_geometry_signature": scenario_signature(scenario),
        }
    )
    expected_geometry = "transition" if initial_class == "crossing_entry" else initial_class
    if metadata.get("taxonomy_geometry_state") != expected_geometry:
        raise ValueError(f"taxonomy drift in {scenario['name']}")
    return scenario


def _validate(manifest: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> None:
    scenarios = list(manifest.get("scenarios") or [])
    expected_states = set(v1.INITIAL_CLASS_ANGLES_DEG)
    if len(scenarios) != 60 or len({item["name"] for item in scenarios}) != 60:
        raise ValueError("B6 requires sixty unique scenarios")
    if {item["metadata"]["initial_class"] for item in scenarios} != expected_states:
        raise ValueError("B6 must cover all five initial states")
    if {item["metadata"]["height_condition"] for item in scenarios} != set(v1.HEIGHT_OFFSETS_M):
        raise ValueError("B6 must cover all three height conditions")
    if {item["metadata"]["mirror_sign"] for item in scenarios} != {"negative", "positive"}:
        raise ValueError("B6 must preserve both mirror directions")
    if {item["metadata"]["distance_speed_package"] for item in scenarios} != {
        package["id"] for package in PACKAGES
    }:
        raise ValueError("B6 must retain both preregistered distance/speed packages")
    if not all(item["metadata"].get("phase_at_reset") == "pre_merge" for item in scenarios):
        raise ValueError("B6 may not reset directly into post-merge or re-entry")
    signatures = {scenario_signature(item) for item in scenarios}
    if len(signatures) != 60:
        raise ValueError("B6 physical signatures are not unique")
    for label, source in sources.items():
        source_signatures = {scenario_signature(item) for item in source.get("scenarios") or []}
        if signatures.intersection(source_signatures):
            raise ValueError(f"B6 overlaps frozen source: {label}")


def build_manifest(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    index = 0
    for package in PACKAGES:
        for initial_class in v1.INITIAL_CLASS_ANGLES_DEG:
            for height in v1.HEIGHT_OFFSETS_M:
                for mirror in ("negative", "positive"):
                    scenarios.append(_scenario(package, initial_class, height, mirror, index))
                    index += 1
    manifest = {
        "source_id": SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_before_b6_execution",
        "lane": "noncanonical_thesis_extension",
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "action_replacement_permitted": False,
            "reference_controller": "frozen_fixed_head_on_specialist",
            "opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
            "episodes_per_scenario_opponent": 1,
            "purpose": "profile_free_opponent_conditional_dynamic_state_phase_reachability",
            "stop_rule": "A B6 result never authorises training automatically; no same-source rerun, tuning, scenario replacement, snapshot restore, or future-state injection.",
        },
        "generation_contract": {
            "initial_states": list(v1.INITIAL_CLASS_ANGLES_DEG),
            "height_offsets_target_minus_own_m": dict(v1.HEIGHT_OFFSETS_M),
            "mirror_signs": ["negative", "positive"],
            "distance_speed_packages": [dict(package) for package in PACKAGES],
            "scenario_count": 60,
            "seed_base": SEED_BASE,
            "raw_si_phase_at_reset": "pre_merge",
        },
        "disjointness": {
            label: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256_file(path),
                "physical_intersection_count": 0,
            }
            for label, path in DISJOINT_MANIFESTS
        },
        "scenarios": scenarios,
    }
    _validate(manifest, sources)
    manifest["integrity"] = {"payload_sha256": _payload_sha256(manifest)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_DIR / "thesis_global_advantage_v1_p2_b6_opponent_conditional_reachability60.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sources = {label: _load_yaml(path) for label, path in DISJOINT_MANIFESTS}
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "scenario_count": len(manifest["scenarios"]),
                **manifest["integrity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
