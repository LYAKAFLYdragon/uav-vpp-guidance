#!/usr/bin/env python3
"""Build the V6 re-entry control-matrix manifest, disjoint from frozen lanes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_thesis_five_state_manifests as five_state
import build_thesis_five_state_raw_si_v5_manifest as v5
from build_thesis_taxonomy_ablation30_manifest import scenario_signature

SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-REACHABILITY-CONTROL-MATRIX-V6"
MANIFEST_SOURCE_ID = SOURCE_ID + "-MANIFEST12"
STATES = ("head_on", "crossing_entry")
PACKAGE = {"id": "v6_reentry_unseen_closure", "initial_range_m": 5270.0, "own_speed_mps": 272.0, "target_speed_mps": 397.0}
SEED_BASE = 130100


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("expected YAML mapping: {0}".format(path))
    return value


def _payload_hash(payload: dict) -> str:
    value = copy.deepcopy(payload)
    value.get("integrity", {}).pop("payload_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_manifest(sources: dict) -> dict:
    scenarios = []
    for state in STATES:
        for height in five_state.HEIGHT_OFFSETS_M:
            for mirror in ("negative", "positive"):
                index = len(scenarios)
                scenario = five_state._build_scenario(
                    split="five_state_reentry_v6", package=PACKAGE, initial_class=state,
                    height_condition=height, mirror_sign=mirror, scenario_index=index,
                )
                metadata = scenario["metadata"]
                seed = SEED_BASE + index
                cell = "{0}__{1}__{2}__{3}".format(state, PACKAGE["id"], height, mirror)
                metadata.update({
                    "pilot_family": SOURCE_ID, "identity_kind": "evaluation", "split": "reentry_v6_control12",
                    "scenario_seed": seed, "geometry_cell_id": cell, "pair_key": "{0}::seed={1}".format(cell, seed),
                    "scenario_signature": "v6_reentry::{0}::seed={1}".format(cell, seed), "phase_at_reset": "pre_merge",
                    "raw_si_first_pass_event": "three_closing_then_two_opening",
                    "reentry_event": "after_first_pass_two_consecutive_range_rate_lt_minus_25_mps",
                })
                scenario["name"] = "v6_reentry_{0}_s{1}".format(cell, seed)
                expected = "transition" if state == "crossing_entry" else state
                if metadata["taxonomy_geometry_state"] != expected:
                    raise ValueError("V6 taxonomy drift")
                scenarios.append(scenario)
    signatures = {scenario_signature(item) for item in scenarios}
    if len(scenarios) != 12 or len(signatures) != 12:
        raise ValueError("V6 must contain 12 unique scenarios")
    for label, source in sources.items():
        if signatures & {scenario_signature(item) for item in source.get("scenarios", [])}:
            raise ValueError("V6 overlaps {0}".format(label))
    manifest = {
        "source_id": MANIFEST_SOURCE_ID, "manifest_version": 1, "status": "preregistered_design_only",
        "protocol": {"evaluation_only": True, "training_permitted": False, "controllers": ["frozen_fixed_head_on", "frozen_fixed_crossing"], "opponents_reported_separately": True, "raw_si_only": True},
        "generation_contract": {"initial_states": list(STATES), "package": dict(PACKAGE), "height_offsets_target_minus_own_m": dict(five_state.HEIGHT_OFFSETS_M), "mirror_signs": ["negative", "positive"]},
        "disjointness": {label: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _hash(path), "physical_intersection_count": 0} for label, path in v5.DISJOINT_MANIFESTS},
        "scenarios": scenarios, "integrity": {"builder_code_sha256": _hash(Path(__file__))},
    }
    manifest["integrity"]["payload_sha256"] = _payload_hash(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite V6 manifest")
    sources = {label: _load(path) for label, path in v5.DISJOINT_MANIFESTS}
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(manifest, sort_keys=False))
    print(json.dumps({"source_id": SOURCE_ID, "scenario_count": len(manifest["scenarios"]), **manifest["integrity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
