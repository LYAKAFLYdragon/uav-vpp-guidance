#!/usr/bin/env python3
"""Build V8's physically disjoint non-learning phase-contract manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_thesis_five_state_reentry_v6_manifest as v6


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-PHASE-CONTRACT-PROBE-V8"
MANIFEST_SOURCE_ID = SOURCE_ID + "-MANIFEST12"
V7_MANIFEST = ROOT / "config/experiment/manifests/thesis_five_state_reentry_v7_teacher12.yaml"


def build_manifest(sources: dict) -> dict:
    """Reuse the frozen five-state geometry generator with an unseen package."""

    original = (v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE)
    try:
        v6.SOURCE_ID = SOURCE_ID
        v6.MANIFEST_SOURCE_ID = MANIFEST_SOURCE_ID
        v6.PACKAGE = {
            "id": "v8_phase_contract_unseen_closure",
            "initial_range_m": 6760.0,
            "own_speed_mps": 254.0,
            "target_speed_mps": 383.0,
        }
        v6.SEED_BASE = 150100
        manifest = v6.build_manifest(sources)
    finally:
        v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE = original

    manifest["status"] = "preregistered_design_only"
    manifest["protocol"]["controllers"] = ["frozen_fixed_head_on"]
    manifest["protocol"]["phase_contract"] = "raw_physical_first_pass_aware_v1"
    manifest["protocol"]["reward_recording"] = "prohibited"
    for label, path in (
        ("v6", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml"),
        ("v7", V7_MANIFEST),
    ):
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
        "--output", type=Path,
        default=ROOT / "config/experiment/manifests/thesis_five_state_reentry_v8_phase_contract12.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite V8 manifest")

    sources = {label: v6._load(path) for label, path in v6.v5.DISJOINT_MANIFESTS}
    for label, path in (("v6", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml"),
                        ("v7", V7_MANIFEST)):
        sources[label] = v6._load(path)
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(manifest, sort_keys=False))
    print(manifest["integrity"]["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
