#!/usr/bin/env python3
"""Build V2 neutral/post-merge manifests with explicit pairing metadata."""

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

import build_thesis_five_state_manifests as five_state
from build_thesis_taxonomy_ablation30_manifest import scenario_signature


FAMILY_SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2"
MANIFEST_DIR = ROOT / "config" / "experiment" / "manifests"
SEED_BASES = {"dev12": 96100, "heldout24": 97300}
DEV_PACKAGES = (
    {"id": "neutral_reentry_v2_dev_a", "initial_range_m": 3350.0, "own_speed_mps": 240.0, "target_speed_mps": 355.0},
    {"id": "neutral_reentry_v2_dev_b", "initial_range_m": 4600.0, "own_speed_mps": 290.0, "target_speed_mps": 405.0},
)
HELDOUT_PACKAGES = (
    {"id": "neutral_reentry_v2_hold_a", "initial_range_m": 3150.0, "own_speed_mps": 228.0, "target_speed_mps": 340.0},
    {"id": "neutral_reentry_v2_hold_b", "initial_range_m": 3750.0, "own_speed_mps": 255.0, "target_speed_mps": 370.0},
    {"id": "neutral_reentry_v2_hold_c", "initial_range_m": 4050.0, "own_speed_mps": 275.0, "target_speed_mps": 385.0},
    {"id": "neutral_reentry_v2_hold_d", "initial_range_m": 4850.0, "own_speed_mps": 305.0, "target_speed_mps": 420.0},
)
DISJOINT_MANIFESTS = (
    ("v1_dev12", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_dev12.yaml"),
    ("v1_heldout24", MANIFEST_DIR / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_heldout24.yaml"),
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


def pair_key(scenario: Mapping[str, Any]) -> str:
    metadata = scenario.get("metadata") or {}
    cell, seed = metadata.get("geometry_cell_id"), metadata.get("scenario_seed")
    if not isinstance(cell, str) or not cell or not isinstance(seed, int):
        raise ValueError("scenario lacks the V2 geometry_cell_id/scenario_seed pairing contract")
    return f"{cell}::seed={seed}"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _signature_set(manifest: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    return {scenario_signature(item) for item in manifest.get("scenarios") or []}


def _seed_set(manifest: Mapping[str, Any]) -> set[int]:
    return {int(item["metadata"]["scenario_seed"]) for item in manifest.get("scenarios") or [] if item.get("metadata", {}).get("scenario_seed") is not None}


def _scenario(package: Mapping[str, Any], height: str, mirror: str, index: int, split: str) -> dict[str, Any]:
    scenario = five_state._build_scenario(
        split=f"neutral_postmerge_reentry_v2_{split}", package=package, initial_class="neutral",
        height_condition=height, mirror_sign=mirror, scenario_index=index,
    )
    metadata = scenario["metadata"]
    seed = SEED_BASES[split] + index
    cell = f"{package['id']}__neutral__{height}__{mirror}"
    scenario["name"] = f"neutral_postmerge_reentry_v2_{split}_{cell}_s{seed}"
    metadata.update({
        "pilot_family": FAMILY_SOURCE_ID, "split": split, "scenario_seed": seed,
        "geometry_cell_id": cell, "pair_key": f"{cell}::seed={seed}",
        "phase_at_reset": "pre_merge", "continuous_run_in": True, "task_registry_key": "head_on",
        "routing_enabled": False, "fixed_skill": "reentry_recovery", "fixed_profile": "reentry_preparation",
        "physical_geometry_signature": repr(scenario_signature(scenario)),
    })
    if metadata.get("taxonomy_geometry_state") != "neutral" or pair_key(scenario) != metadata["pair_key"]:
        raise ValueError("V2 scenario taxonomy or pairing contract drifted")
    return scenario


def build_manifest(split: str, packages: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scenarios = [_scenario(package, height, mirror, index, split) for index, (package, height, mirror) in enumerate((package, height, mirror) for package in packages for height in five_state.HEIGHT_OFFSETS_M for mirror in ("negative", "positive"))]
    expected_count = 12 if split == "dev12" else 24
    if len(scenarios) != expected_count or len({pair_key(item) for item in scenarios}) != expected_count:
        raise ValueError("V2 split has duplicate scenarios or pair keys")
    signatures, seeds = _signature_set({"scenarios": scenarios}), _seed_set({"scenarios": scenarios})
    if len(signatures) != expected_count or len(seeds) != expected_count:
        raise ValueError("V2 split has duplicate physical signatures or seeds")
    for label, source in sources.items():
        if signatures & _signature_set(source) or seeds & _seed_set(source):
            raise ValueError(f"V2 split overlaps frozen source: {label}")
    manifest = {
        "source_id": f"{FAMILY_SOURCE_ID}-{split.upper()}", "manifest_version": 2,
        "status": "preregistered_design_only", "split": split,
        "protocol": {"training_permitted": False, "evaluation_permitted": False, "continuous_physical_run_in": True, "snapshot_restore": False, "future_state_injection": False, "routing_enabled": False, "candidate_skill": "reentry_recovery", "fixed_profile": "reentry_preparation", "opponents_reported_separately": True},
        "pairing_contract": {"key": "metadata.geometry_cell_id + metadata.scenario_seed", "serialized_field": "metadata.pair_key", "same_key_required_across_methods": True},
        "generation_contract": {"initial_class": "neutral", "height_offsets_target_minus_own_m": dict(five_state.HEIGHT_OFFSETS_M), "mirror_signs": ["negative", "positive"], "packages": [dict(item) for item in packages], "first_pass_requirement": "legal pre_merge reset followed by one continuous strict-JSBSim episode"},
        "disjointness": {label: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "physical_intersection_count": 0, "seed_intersection_count": 0} for label, path in DISJOINT_MANIFESTS},
        "scenarios": scenarios, "integrity": {"builder_code_sha256": sha256_file(Path(__file__))},
    }
    manifest["integrity"]["payload_sha256"] = payload_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sources = {label: _load_yaml(path) for label, path in DISJOINT_MANIFESTS}
    outputs = {"dev12": ("thesis_neutral_postmerge_reentry_recovery_pilot_v2_dev12.yaml", DEV_PACKAGES), "heldout24": ("thesis_neutral_postmerge_reentry_recovery_pilot_v2_heldout24.yaml", HELDOUT_PACKAGES)}
    result = {}
    for split, (name, packages) in outputs.items():
        path = args.output_dir / name
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")
        manifest = build_manifest(split, packages, sources)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        result[split] = {"path": str(path), "scenario_count": len(manifest["scenarios"]), "payload_sha256": manifest["integrity"]["payload_sha256"]}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
