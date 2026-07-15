#!/usr/bin/env python3
"""Build physically disjoint dev12 and heldout24 manifests for reward pilot V1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_thesis_five_state_reentry_v6_manifest as v6
import build_thesis_five_state_manifests as five_state
from build_thesis_taxonomy_ablation30_manifest import scenario_signature


SOURCE_ID = "THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1"
V7_MANIFEST = ROOT / "config/experiment/manifests/thesis_five_state_reentry_v7_teacher12.yaml"
V8_MANIFEST = ROOT / "config/experiment/manifests/thesis_five_state_reentry_v8_phase_contract12.yaml"
DEV_PACKAGES = (
    {"id": "reward_pilot_dev_unseen_a", "initial_range_m": 6380.0, "own_speed_mps": 263.0, "target_speed_mps": 389.0},
)
HELDOUT_PACKAGES = (
    {"id": "reward_pilot_heldout_unseen_a", "initial_range_m": 7040.0, "own_speed_mps": 241.0, "target_speed_mps": 374.0},
    {"id": "reward_pilot_heldout_unseen_b", "initial_range_m": 7520.0, "own_speed_mps": 297.0, "target_speed_mps": 418.0},
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _payload_hash(payload: dict) -> str:
    value = copy.deepcopy(payload)
    value.get("integrity", {}).pop("payload_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def frozen_sources() -> dict[str, dict]:
    sources = {label: _load(path) for label, path in v6.v5.DISJOINT_MANIFESTS}
    for label, path in (
        ("v6", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml"),
        ("v7", V7_MANIFEST),
        ("v8", V8_MANIFEST),
    ):
        sources[label] = _load(path)
    return sources


def build_manifest(*, split: str, packages: tuple[dict, ...], seed_base: int, sources: dict[str, dict]) -> dict:
    scenarios = []
    for package_index, package in enumerate(packages):
        for state in v6.STATES:
            for height in five_state.HEIGHT_OFFSETS_M:
                for mirror in ("negative", "positive"):
                    index = len(scenarios)
                    scenario = five_state._build_scenario(
                        split=split,
                        package=package,
                        initial_class=state,
                        height_condition=height,
                        mirror_sign=mirror,
                        scenario_index=index,
                    )
                    metadata = scenario["metadata"]
                    seed = seed_base + index
                    cell = f"{state}__{package['id']}__{height}__{mirror}"
                    metadata.update({
                        "pilot_family": SOURCE_ID,
                        "identity_kind": "evaluation",
                        "split": split,
                        "scenario_seed": seed,
                        "geometry_cell_id": cell,
                        "pair_key": f"{cell}::seed={seed}",
                        "scenario_signature": f"reward_pilot_v1::{cell}::seed={seed}",
                        "phase_at_reset": "pre_merge",
                        "continuous_run_in": True,
                        "routing_enabled": False,
                        "fixed_profile": "range_extension",
                        "raw_si_first_pass_event": "three_closing_then_two_opening",
                        "phase_contract": "raw_physical_first_pass_aware_v1",
                    })
                    scenario["name"] = f"reward_pilot_v1_{split}_{cell}_s{seed}"
                    expected = "transition" if state == "crossing_entry" else state
                    if metadata["taxonomy_geometry_state"] != expected:
                        raise ValueError("taxonomy drift")
                    scenarios.append(scenario)
    expected_count = 12 * len(packages)
    signatures = {scenario_signature(item) for item in scenarios}
    if len(scenarios) != expected_count or len(signatures) != expected_count:
        raise ValueError("manifest scenario uniqueness failure")
    for label, source in sources.items():
        if signatures & {scenario_signature(item) for item in source.get("scenarios", [])}:
            raise ValueError(f"physical overlap with {label}")
    return {
        "source_id": f"{SOURCE_ID}-{split.upper()}",
        "manifest_version": 1,
        "status": "preregistered_design_only",
        "protocol": {
            "evaluation_only": True,
            "training_permitted": False,
            "reward_pilot_execution_permitted": False,
            "opponents_reported_separately": True,
            "continuous_physical_run_in": True,
            "phase_contract": "raw_physical_first_pass_aware_v1",
        },
        "generation_contract": {
            "initial_states": list(v6.STATES),
            "packages": [dict(package) for package in packages],
            "height_offsets_target_minus_own_m": dict(five_state.HEIGHT_OFFSETS_M),
            "mirror_signs": ["negative", "positive"],
        },
        "disjointness": {
            label: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _hash(path),
                "physical_intersection_count": 0,
            }
            for label, path in v6.v5.DISJOINT_MANIFESTS
        },
        "scenarios": scenarios,
        "integrity": {"builder_code_sha256": _hash(Path(__file__))},
    }


def _add_recent_disjointness(manifest: dict) -> None:
    for label, path in (
        ("v6", ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml"),
        ("v7", V7_MANIFEST),
        ("v8", V8_MANIFEST),
    ):
        manifest["disjointness"][label] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _hash(path),
            "physical_intersection_count": 0,
        }
    manifest["integrity"]["payload_sha256"] = _payload_hash(manifest)


def write_manifest(path: Path, manifest: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(manifest, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-output", type=Path, default=ROOT / "config/experiment/manifests/thesis_reentry_terminal_reward_pilot_v1_dev12.yaml")
    parser.add_argument("--heldout-output", type=Path, default=ROOT / "config/experiment/manifests/thesis_reentry_terminal_reward_pilot_v1_heldout24.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sources = frozen_sources()
    dev = build_manifest(split="dev12", packages=DEV_PACKAGES, seed_base=160100, sources=sources)
    heldout = build_manifest(split="heldout24", packages=HELDOUT_PACKAGES, seed_base=160200, sources=sources)
    dev_signatures = {scenario_signature(item) for item in dev["scenarios"]}
    heldout_signatures = {scenario_signature(item) for item in heldout["scenarios"]}
    if dev_signatures & heldout_signatures:
        raise ValueError("dev/heldout physical overlap")
    _add_recent_disjointness(dev)
    _add_recent_disjointness(heldout)
    write_manifest(args.dev_output, dev, overwrite=args.overwrite)
    write_manifest(args.heldout_output, heldout, overwrite=args.overwrite)
    print(json.dumps({
        "dev": {"scenarios": len(dev["scenarios"]), "payload_sha256": dev["integrity"]["payload_sha256"]},
        "heldout": {"scenarios": len(heldout["scenarios"]), "payload_sha256": heldout["integrity"]["payload_sha256"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
