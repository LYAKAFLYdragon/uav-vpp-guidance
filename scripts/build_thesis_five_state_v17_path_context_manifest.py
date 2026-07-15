#!/usr/bin/env python3
"""Build V17's two-package, physically disjoint 24-scenario sampler manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_thesis_five_state_reentry_v13_manifest as v13
import build_thesis_five_state_reentry_v6_manifest as v6


SOURCE_ID = "THESIS-FIVE-STATE-PATH-CONTEXT-PREFLIGHT-V17"
MANIFEST_SOURCE_ID = SOURCE_ID + "-MANIFEST24"
PACKAGES = (
    ("v17_path_context_a", 8300.0, 315.0, 350.0, 199100),
    ("v17_path_context_b", 8700.0, 340.0, 370.0, 199200),
)
EXTRA_MANIFESTS = v13.EXTRA_MANIFESTS + (
    ("v14", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v14_temporal_corridor12.yaml"),
    ("v15", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor12.yaml"),
    ("v16", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v16_dynamic_reference_path12.yaml"),
)


def build_manifest(sources: dict[str, dict]) -> dict:
    """Compose two independently seeded V6-shaped packages into a 24-cell universe."""

    original = (v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE)
    scenarios = []
    try:
        for package_id, initial_range_m, own_speed_mps, target_speed_mps, seed_base in PACKAGES:
            v6.SOURCE_ID = SOURCE_ID
            v6.MANIFEST_SOURCE_ID = MANIFEST_SOURCE_ID
            v6.PACKAGE = {
                "id": package_id,
                "initial_range_m": initial_range_m,
                "own_speed_mps": own_speed_mps,
                "target_speed_mps": target_speed_mps,
            }
            v6.SEED_BASE = seed_base
            scenarios.extend(v6.build_manifest(sources)["scenarios"])
    finally:
        v6.SOURCE_ID, v6.MANIFEST_SOURCE_ID, v6.PACKAGE, v6.SEED_BASE = original
    if len(scenarios) != 24:
        raise AssertionError("V17 manifest must contain exactly 24 scenarios")
    manifest = {
        "source_id": MANIFEST_SOURCE_ID,
        "manifest_version": 1,
        "status": "preregistered_design_only",
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "reward_recording": "prohibited",
            "controllers": ["frozen_fixed_head_on_path_context_collector"],
            "opponents_reported_separately": True,
            "raw_si_only": True,
        },
        "generation_contract": {
            "initial_states": ["head_on", "crossing_entry"],
            "packages": [
                {
                    "id": package_id,
                    "initial_range_m": initial_range_m,
                    "own_speed_mps": own_speed_mps,
                    "target_speed_mps": target_speed_mps,
                    "seed_base": seed_base,
                }
                for package_id, initial_range_m, own_speed_mps, target_speed_mps, seed_base in PACKAGES
            ],
            "height_offsets_target_minus_own_m": {"own_below": 600.0, "co_altitude": 0.0, "own_above": -600.0},
            "mirror_signs": ["negative", "positive"],
        },
        "disjointness": {},
        "scenarios": scenarios,
    }
    for label, path in (*v6.v5.DISJOINT_MANIFESTS, *EXTRA_MANIFESTS):
        manifest["disjointness"][label] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": v6._hash(path),
            "physical_intersection_count": 0,
        }
    manifest["integrity"] = {"payload_sha256": v6._payload_hash(manifest)}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config/experiment/manifests/thesis_five_state_v17_path_context_preflight24.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite V17 manifest")
    sources = {label: v6._load(path) for label, path in (*v6.v5.DISJOINT_MANIFESTS, *EXTRA_MANIFESTS)}
    manifest = build_manifest(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(manifest["integrity"]["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
