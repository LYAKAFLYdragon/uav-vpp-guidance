#!/usr/bin/env python3
"""Fail-closed design preflight for the non-learning B6 reachability atlas."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-R1"
MANIFEST_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-MANIFEST60-R1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b6_opponent_conditional_reachability.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} must declare a SHA-256")
    if not path.is_file() or _sha256(path) != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch")


def _payload_sha256(manifest: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(dict(manifest))
    prepared.pop("integrity", None)
    return hashlib.sha256(
        json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_design(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "preregistered_design_execution_not_authorised":
        raise ValueError("unexpected B6 source/status")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("B6 authorization must be a mapping")
    if authorization.get("execution_permitted") is not False:
        raise ValueError("B6 design must remain execution-disabled")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "candidate_policy_loading_permitted",
        "action_replacement_permitted",
        "policy_change_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
        "heldout_claims_permitted",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"B6 authorization.{key} must remain false")
    contract = config.get("contract")
    if not isinstance(contract, Mapping) or (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("fresh_environment_per_episode") is not True
        or contract.get("phase_input_unit_contract") != "raw_SI_from_observation.relative_state"
        or tuple(contract.get("taxonomy_angle_order", ())) != ("ATA", "AA")
    ):
        raise ValueError("B6 physical/taxonomy contract drifted")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("B6 inputs must be a mapping")
    manifest_path = _repo_path(str(inputs.get("manifest", "")))
    runtime_path = _repo_path(str(inputs.get("runtime_config", "")))
    registry_path = _repo_path(str(inputs.get("runtime_registry", "")))
    encoder_path = Path(str(inputs.get("frozen_p3_encoder", "")))
    _require_hash(manifest_path, inputs.get("manifest_sha256"), "B6 manifest")
    _require_hash(runtime_path, inputs.get("runtime_config_sha256"), "B6 runtime config")
    _require_hash(registry_path, inputs.get("runtime_registry_sha256"), "B6 runtime registry")
    _require_hash(encoder_path, inputs.get("frozen_p3_encoder_sha256"), "B6 P3 encoder")
    manifest = _load_yaml(manifest_path)
    scenarios = list(manifest.get("scenarios") or [])
    states = {item.get("metadata", {}).get("initial_class") for item in scenarios}
    if (
        manifest.get("source_id") != MANIFEST_SOURCE_ID
        or manifest.get("source_id") != inputs.get("manifest_source_id")
        or len(scenarios) != 60
        or states != {"advantage", "head_on", "disadvantage", "neutral", "crossing_entry"}
        or _payload_sha256(manifest) != manifest.get("integrity", {}).get("payload_sha256")
    ):
        raise ValueError("B6 manifest identity/integrity/coverage drifted")
    if not all(item.get("metadata", {}).get("phase_at_reset") == "pre_merge" for item in scenarios):
        raise ValueError("B6 manifest must retain pre-merge reset semantics")
    for label, source in (manifest.get("disjointness") or {}).items():
        _require_hash(_repo_path(str(source.get("path", ""))), source.get("sha256"), f"B6 source {label}")
    registry = _load_yaml(registry_path)
    specialist = ((registry.get("specialists") or {}).get(inputs.get("reference_specialist")) or {})
    if specialist.get("checkpoint_sha256") != inputs.get("reference_specialist_sha256"):
        raise ValueError("B6 frozen reference specialist SHA drifted")
    if tuple((registry.get("opponents") or {}).keys())[:3] != OPPONENTS:
        raise ValueError("B6 opponent registry order drifted")
    output_root = Path(str(config.get("outputs", {}).get("root", "")))
    if not output_root.is_absolute():
        raise ValueError("B6 output root must be absolute")
    free_gb = shutil.disk_usage(output_root.parent).free / (1024**3)
    return {
        "source_id": SOURCE_ID,
        "mode": "design_preflight_only_no_jsbsim_no_training",
        "execution_permitted": False,
        "training_permitted": False,
        "scenario_count": len(scenarios),
        "planned_records": len(scenarios) * len(OPPONENTS),
        "opponents": list(OPPONENTS),
        "output_root_absent": not output_root.exists(),
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= float(contract["min_free_disk_gb"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_design(args.config), indent=2, ensure_ascii=False))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"INVALID B6 reachability design: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
