#!/usr/bin/env python3
"""Build the disjoint V4 reward-free runtime-feasibility envelope."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import build_thesis_five_state_manifests as five_state
from build_thesis_taxonomy_ablation30_manifest import scenario_signature
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_contract import (
    FAMILY_ORDER,
    MANIFEST_SOURCE_ID,
    SOURCE_ID,
    evaluation_pair_key,
    validate_feasibility_manifest,
)


MANIFEST_DIR = ROOT / "config" / "experiment" / "manifests"
SEED_BASE = 119100
FAMILIES: Mapping[str, Sequence[Mapping[str, float | str]]] = {
    "close_balanced": (
        {"id": "v4_close_bal_a", "initial_range_m": 3280.0, "own_speed_mps": 266.0, "target_speed_mps": 374.0},
        {"id": "v4_close_bal_b", "initial_range_m": 3440.0, "own_speed_mps": 270.0, "target_speed_mps": 380.0},
    ),
    "close_high_closure": (
        {"id": "v4_close_close_a", "initial_range_m": 3370.0, "own_speed_mps": 232.0, "target_speed_mps": 390.0},
        {"id": "v4_close_close_b", "initial_range_m": 3530.0, "own_speed_mps": 240.0, "target_speed_mps": 404.0},
    ),
    "extended_balanced": (
        {"id": "v4_ext_bal_a", "initial_range_m": 5040.0, "own_speed_mps": 250.0, "target_speed_mps": 356.0},
        {"id": "v4_ext_bal_b", "initial_range_m": 5260.0, "own_speed_mps": 262.0, "target_speed_mps": 370.0},
    ),
    "extended_high_closure": (
        {"id": "v4_ext_close_a", "initial_range_m": 5160.0, "own_speed_mps": 220.0, "target_speed_mps": 380.0},
        {"id": "v4_ext_close_b", "initial_range_m": 5380.0, "own_speed_mps": 234.0, "target_speed_mps": 400.0},
    ),
}
DISJOINT_MANIFESTS = (
    ("v1_dev12", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_dev12.yaml"),
    ("v1_heldout24", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_heldout24.yaml"),
    ("v2_dev12", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_pilot_v2_dev12.yaml"),
    ("v2_heldout24", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_pilot_v2_heldout24.yaml"),
    ("v3_runtime_design", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility48.yaml"),
    ("p2_b2", MANIFEST_DIR / "thesis_global_advantage_v1_p2_physical_preflight60.yaml"),
    ("p2_b3", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b3_phase_observability30.yaml"),
    ("p2_b4", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b4_raw_si_phase30.yaml"),
    ("p2_b5", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b5_phase_feasible_sampler12.yaml"),
    ("p2_b6", MANIFEST_DIR / "thesis_global_advantage_v1_p2_b6_opponent_conditional_reachability60.yaml"),
    ("heldout240", MANIFEST_DIR / "thesis_global_advantage_v1_heldout240.yaml"),
    ("dev30", MANIFEST_DIR / "thesis_five_state_shared_intent_v1_dev30.yaml"),
    ("historical_heldout60", MANIFEST_DIR / "thesis_five_state_shared_intent_v1_heldout60.yaml"),
    ("phase_v2", MANIFEST_DIR / "thesis_phase_reachability_v2_run_in_handoff_r1.yaml"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(dict(payload))
    if isinstance(prepared.get("integrity"), dict):
        prepared["integrity"].pop("payload_sha256", None)
    return hashlib.sha256(json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _signature_set(manifest: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    return {scenario_signature(item) for item in manifest.get("scenarios") or []}


def _seed_set(manifest: Mapping[str, Any]) -> set[int]:
    return {
        int(item["metadata"]["scenario_seed"])
        for item in manifest.get("scenarios") or []
        if item.get("metadata", {}).get("scenario_seed") is not None
    }


def _scenario(family: str, package: Mapping[str, Any], height: str, mirror: str, index: int) -> dict[str, Any]:
    scenario = five_state._build_scenario(
        split="neutral_postmerge_reentry_v4_runtime_feasibility",
        package=package,
        initial_class="neutral",
        height_condition=height,
        mirror_sign=mirror,
        scenario_index=index,
    )
    metadata = scenario["metadata"]
    seed = SEED_BASE + index
    cell = f"{family}__{package['id']}__neutral__{height}__{mirror}"
    scenario["name"] = f"neutral_postmerge_v4_feasibility_{cell}_s{seed}"
    metadata.update(
        {
            "pilot_family": SOURCE_ID,
            "identity_kind": "evaluation",
            "split": "runtime_feasibility48",
            "scenario_seed": seed,
            "geometry_cell_id": cell,
            "pair_key": f"{cell}::seed={seed}",
            "scenario_signature": f"v4_feasibility::{cell}::seed={seed}",
            "feasibility_family_id": family,
            "phase_at_reset": "pre_merge",
            "continuous_run_in": True,
            "task_registry_key": "head_on",
            "routing_enabled": False,
            "fixed_run_in": "frozen_fixed_head_on",
            "physical_geometry_signature": repr(scenario_signature(scenario)),
        }
    )
    if metadata.get("taxonomy_geometry_state") != "neutral" or evaluation_pair_key(scenario) != metadata["pair_key"]:
        raise ValueError("V4 scenario taxonomy or evaluation identity drifted")
    return scenario


def build_manifest(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for package in FAMILIES[family]:
            for height in five_state.HEIGHT_OFFSETS_M:
                for mirror in ("negative", "positive"):
                    scenarios.append(_scenario(family, package, height, mirror, len(scenarios)))
    signatures = _signature_set({"scenarios": scenarios})
    seeds = _seed_set({"scenarios": scenarios})
    if len(scenarios) != 48 or len(signatures) != 48 or len(seeds) != 48:
        raise ValueError("V4 feasibility scenarios must have unique physical signatures and seeds")
    for label, source in sources.items():
        if signatures & _signature_set(source) or seeds & _seed_set(source):
            raise ValueError(f"V4 feasibility manifest overlaps frozen source: {label}")
    manifest: dict[str, Any] = {
        "source_id": MANIFEST_SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_design_only",
        "split": "runtime_feasibility48",
        "protocol": {
            "evaluation_only": True,
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
            "training_permitted": False,
            "reference_controller": "frozen_fixed_head_on",
            "continuous_physical_run_in": True,
            "snapshot_restore": False,
            "future_state_injection": False,
            "routing_enabled": False,
            "opponents_reported_separately": True,
        },
        "generation_contract": {
            "initial_class": "neutral",
            "height_offsets_target_minus_own_m": dict(five_state.HEIGHT_OFFSETS_M),
            "mirror_signs": ["negative", "positive"],
            "families": {family: [dict(item) for item in FAMILIES[family]] for family in FAMILY_ORDER},
            "future_pilot_must_not_reuse_v4_feasibility_states": True,
        },
        "identity_contract": {
            "evaluation_pair_key": "metadata.geometry_cell_id + metadata.scenario_seed",
            "train_identity": "separate_train_stream_id + scenario_seed",
            "train_records_ineligible_for_paired_delta": True,
        },
        "disjointness": {
            label: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "physical_intersection_count": 0,
                "seed_intersection_count": 0,
            }
            for label, path in DISJOINT_MANIFESTS
        },
        "scenarios": scenarios,
        "integrity": {"builder_code_sha256": sha256_file(Path(__file__))},
    }
    manifest["integrity"]["payload_sha256"] = payload_sha256(manifest)
    validate_feasibility_manifest(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility48.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sources = {label: _load_yaml(path) for label, path in DISJOINT_MANIFESTS}
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(manifest, sort_keys=False))
    print(json.dumps({"source_id": MANIFEST_SOURCE_ID, "scenario_count": len(manifest["scenarios"]), **manifest["integrity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
