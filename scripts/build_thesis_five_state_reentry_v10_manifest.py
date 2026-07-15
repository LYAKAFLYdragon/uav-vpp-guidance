#!/usr/bin/env python3
"""Build V10's physically disjoint action-authority manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_thesis_five_state_reentry_v6_manifest as v6


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V10"
MANIFEST_SOURCE_ID = SOURCE_ID + "-MANIFEST12"
EXTRA_MANIFESTS = (
    ("v6", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml"),
    ("v7", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v7_teacher12.yaml"),
    ("v8", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v8_phase_contract12.yaml"),
    ("v9", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v9_action_authority12.yaml"),
    ("reward_v1_dev", ROOT / "config/experiment/manifests/thesis_reentry_terminal_reward_pilot_v1_dev12.yaml"),
    ("reward_v1_heldout", ROOT / "config/experiment/manifests/thesis_reentry_terminal_reward_pilot_v1_heldout24.yaml"),
)


def build_manifest(sources: dict) -> dict:
    original = (v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE)
    try:
        v6.SOURCE_ID = SOURCE_ID
        v6.MANIFEST_SOURCE_ID = MANIFEST_SOURCE_ID
        v6.PACKAGE = {
            "id": "v10_action_authority_unseen_closure",
            "initial_range_m": 6150.0,
            "own_speed_mps": 281.0,
            "target_speed_mps": 374.0,
        }
        v6.SEED_BASE = 190100
        manifest = v6.build_manifest(sources)
    finally:
        v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE = original
    manifest["status"] = "preregistered_design_only"
    manifest["protocol"].update({
        "controllers": ["frozen_fixed_head_on_plus_predeclared_vpp_deltas"],
        "phase_contract": "raw_physical_first_pass_aware_v1",
        "reward_recording": "prohibited",
        "training_permitted": False,
    })
    for label, path in EXTRA_MANIFESTS:
        manifest["disjointness"][label] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": v6._hash(path),
            "physical_intersection_count": 0,
        }
    manifest["integrity"]["payload_sha256"] = v6._payload_hash(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "config/experiment/manifests/thesis_five_state_reentry_v10_action_authority12.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite V10 manifest")
    sources = {label: v6._load(path) for label, path in v6.v5.DISJOINT_MANIFESTS}
    for label, path in EXTRA_MANIFESTS:
        sources[label] = v6._load(path)
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(manifest, sort_keys=False))
    print(manifest["integrity"]["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
