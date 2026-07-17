#!/usr/bin/env python3
"""Build paired task-context aliases for a non-learning interface preflight."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
SOURCE_MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_disadvantage_pressure_exposed_postmerge_v1.yaml"
DEFAULT_OUTPUT = ROOT / "config" / "experiment" / "manifests" / "thesis_task_interface_discriminability_preflight_v1.yaml"
PREFLIGHT_ID = "ENGINEERING-TASK-INTERFACE-DISCRIMINABILITY-PREFLIGHT-V1-NOT-A-SOURCE-ID"
SELECTION = (
    "pressure_postmerge_short_tight_own_below_negative",
    "pressure_postmerge_short_tight_own_below_positive",
    "pressure_postmerge_medium_wide_co_altitude_negative",
    "pressure_postmerge_medium_wide_co_altitude_positive",
)
CONTEXTS = (("head_on", 0, False), ("crossing_feasible", 1, True))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    payload.get("integrity", {}).pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build(source: Path = SOURCE_MANIFEST) -> dict[str, Any]:
    original = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    scenarios = {str(item.get("name")): item for item in original.get("scenarios", [])}
    if set(SELECTION).difference(scenarios):
        raise ValueError("the frozen pressure source lacks a selected physical scenario")
    rows = []
    for physical_name in SELECTION:
        physical = scenarios[physical_name]
        for task_key, task_bit, crossing_context in CONTEXTS:
            item = copy.deepcopy(physical)
            item["name"] = "{}__context_{}".format(physical_name, task_key)
            metadata = item.setdefault("metadata", {})
            metadata.update(
                {
                    "manifest_family": "thesis_task_interface_discriminability_preflight_v1",
                    "physical_scenario_id": physical_name,
                    "context_pair_id": physical_name,
                    "interface_context": task_key,
                    "task_registry_key": task_key,
                    "ppo_task_bit": task_bit,
                    "crossing_context": crossing_context,
                    "same_physical_reset_required": True,
                    "same_seed_required": True,
                }
            )
            rows.append(item)
    manifest = {
        "source_id": PREFLIGHT_ID,
        "manifest_version": 1,
        "status": "design_only_runtime_probe_not_authorized",
        "protocol": {
            "nonlearning_only": True,
            "training_permitted": False,
            "tuning_permitted": False,
            "performance_comparison_permitted": False,
            "physical_scenarios": len(SELECTION),
            "task_contexts_per_physical_scenario": 2,
            "expected_runtime_records_if_authorized": 48,
            "purpose": "verify task-bit and bootstrap treatment discriminability, not tactical task correctness",
        },
        "source_manifest": {"path": str(source), "sha256": _hash(source)},
        "scenarios": rows,
        "integrity": {"scenario_count": len(rows), "payload_sha256": ""},
    }
    manifest["integrity"]["payload_sha256"] = _payload_hash(manifest)
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "payload_sha256": manifest["integrity"]["payload_sha256"], "scenarios": len(manifest["scenarios"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
