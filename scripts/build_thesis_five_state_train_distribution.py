"""Freeze and validate the five-state v1 continuous training distribution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from build_thesis_taxonomy_ablation30_manifest import scenario_signature


DEFAULT_CONFIG = (
    REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml"
)
DEFAULT_DEV = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_dev30.yaml"
DEFAULT_HELDOUT = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_heldout60.yaml"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _outside_support(scenario: Mapping[str, Any], support: Mapping[str, Any]) -> bool:
    metadata = scenario["metadata"]
    return any(
        float(metadata[field]) < float(support[field][0])
        or float(metadata[field]) > float(support[field][1])
        for field in ("initial_range_m", "own_speed_mps", "target_speed_mps")
    )


def build_distribution_record(
    config: Mapping[str, Any], dev_manifest: Mapping[str, Any], heldout_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    record = copy.deepcopy(dict(config))
    record.pop("integrity", None)
    contract = record["sampling_contract"]
    weights = {
        name: float(item["weight"])
        for name, item in contract["initial_states"].items()
    }
    if set(weights) != {"advantage", "head_on", "disadvantage", "neutral", "crossing_entry"}:
        raise ValueError("Training distribution must balance exactly the five frozen states")
    if len(set(weights.values())) != 1:
        raise ValueError("Initial-state weights must be balanced in v1")
    if set(contract["height_conditions"]) != {"own_below", "co_altitude", "own_above"}:
        raise ValueError("Training distribution must include exactly three height conditions")
    support = contract["continuous_support"]
    if not all(
        _outside_support(scenario, support)
        for manifest in (dev_manifest, heldout_manifest)
        for scenario in manifest["scenarios"]
    ):
        raise ValueError("dev30 or heldout60 leaks into the training distance-speed support")
    dev_signatures = {scenario_signature(scenario) for scenario in dev_manifest["scenarios"]}
    heldout_signatures = {scenario_signature(scenario) for scenario in heldout_manifest["scenarios"]}
    if dev_signatures.intersection(heldout_signatures):
        raise ValueError("dev30 and heldout60 full signatures must be disjoint")
    record["integrity"] = {
        "dev30_payload_sha256": dev_manifest["integrity"]["payload_sha256"],
        "heldout60_payload_sha256": heldout_manifest["integrity"]["payload_sha256"],
        "dev30_training_support_overlap_count": 0,
        "heldout60_training_support_overlap_count": 0,
    }
    record["integrity"]["payload_sha256"] = _payload_sha256(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--heldout-manifest", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and args.output != args.config and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    record = build_distribution_record(
        _load_yaml(args.config), _load_yaml(args.dev_manifest), _load_yaml(args.heldout_manifest)
    )
    args.output.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "payload_sha256": record["integrity"]["payload_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
