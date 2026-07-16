#!/usr/bin/env python3
"""Build the V2 temporal-corridor transfer manifest disjoint from prior lanes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_thesis_five_state_reentry_v15_manifest as v15
import build_thesis_five_state_reentry_v6_manifest as v6


SOURCE_ID = "THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V2"
MANIFEST_SOURCE_ID = SOURCE_ID + "-MANIFEST12"
EXTRA_MANIFESTS = v15.EXTRA_MANIFESTS + (
    (
        "temporal_corridor_transfer_v1",
        ROOT / "config/experiment/manifests/thesis_temporal_corridor_transfer_v1_dev12.yaml",
    ),
    (
        "v15_fixed_encoder_corridor",
        ROOT / "config/experiment/manifests/thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor12.yaml",
    ),
)


def build_manifest(sources: dict) -> dict:
    original = (v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE)
    try:
        v6.SOURCE_ID = SOURCE_ID
        v6.MANIFEST_SOURCE_ID = MANIFEST_SOURCE_ID
        v6.PACKAGE = {
            "id": "temporal_corridor_v2_runtime_mapping_unseen",
            "initial_range_m": 8870.0,
            "own_speed_mps": 334.0,
            "target_speed_mps": 364.0,
        }
        v6.SEED_BASE = 208200
        manifest = v6.build_manifest(sources)
    finally:
        v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE = original

    manifest["status"] = "preregistered_design_only"
    manifest["protocol"].update(
        {
            "controllers": ["frozen_fixed_head_on_plus_bounded_temporal_corridor"],
            "phase_contract": "raw_physical_first_pass_aware_v1",
            "reward_recording": "prohibited",
            "training_permitted": False,
        }
    )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config/experiment/manifests/thesis_temporal_corridor_transfer_v2_dev12.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite the V2 transfer manifest")

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
