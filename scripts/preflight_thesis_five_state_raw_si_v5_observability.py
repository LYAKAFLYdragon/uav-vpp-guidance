#!/usr/bin/env python3
"""Fail-closed design preflight for the V5 raw-SI observability collector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_contract import (
    MANIFEST_SOURCE_ID,
    SOURCE_ID,
    FiveStateRawSIV5ContractError,
    build_plan,
    validate_manifest,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_five_state_raw_si_v5_observability.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("expected YAML mapping: {0}".format(path))
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or sha256_file(path) != expected.lower():
        raise ValueError("{0} SHA-256 mismatch".format(label))


def _payload_sha256(manifest: Mapping[str, Any]) -> str:
    prepared = dict(manifest)
    integrity = dict(prepared.get("integrity") or {})
    integrity.pop("payload_sha256", None)
    prepared["integrity"] = integrity
    return hashlib.sha256(json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_design_authorization(authorization: Mapping[str, Any]) -> None:
    if authorization.get("execution_permitted") is not False:
        raise ValueError("V5 design preflight requires execution_permitted=false")
    if any(value is not False for value in authorization.values()):
        raise ValueError("V5 design preflight requires every authorization value to be false")


def _validate_runtime_registry(config: Mapping[str, Any]) -> None:
    inputs = config["inputs"]
    registry_path = _repo_path(str(inputs["runtime_registry"]))
    _require_hash(registry_path, inputs.get("runtime_registry_sha256"), "runtime registry")
    registry = load_yaml(registry_path)
    opponents = tuple(config["opponents"]["order"])
    if tuple(registry.get("opponents", ())) != opponents:
        raise ValueError("V5 runtime registry opponent order drifted")
    for opponent in opponents:
        entry = registry["opponents"][opponent]
        if opponent != "expert":
            _require_hash(Path(str(entry["checkpoint"])), entry.get("checkpoint_sha256"), "opponent {0}".format(opponent))
    specialist = registry.get("specialists", {}).get("frozen_fixed_head_on")
    if not isinstance(specialist, Mapping):
        raise ValueError("V5 runtime registry lacks frozen_fixed_head_on")
    if specialist.get("checkpoint_sha256") != inputs.get("frozen_fixed_head_on_sha256"):
        raise ValueError("V5 fixed head-on specialist SHA drifted")
    _require_hash(Path(str(specialist["checkpoint"])), specialist.get("checkpoint_sha256"), "fixed head-on specialist")


def validate_design(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate frozen inputs without creating JSBSim, output, or a policy."""

    config = load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "preregistered_design_execution_not_authorised":
        raise ValueError("unexpected V5 design source or status")
    if "proposed_training" in config:
        raise ValueError("V5 observability design must not define proposed_training")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("V5 authorization is missing")
    _validate_design_authorization(authorization)
    plan = build_plan(config)
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V5 inputs are missing")
    manifest_path = _repo_path(str(inputs.get("observability_manifest", "")))
    template_path = _repo_path(str(inputs.get("runtime_template", "")))
    encoder_path = Path(str(inputs.get("frozen_p3_encoder", "")))
    _require_hash(manifest_path, inputs.get("observability_manifest_sha256"), "observability manifest")
    _require_hash(template_path, inputs.get("runtime_template_sha256"), "runtime template")
    _require_hash(encoder_path, inputs.get("frozen_p3_encoder_sha256"), "frozen P3 encoder")
    manifest = load_yaml(manifest_path)
    if manifest.get("source_id") != MANIFEST_SOURCE_ID or _payload_sha256(manifest) != manifest.get("integrity", {}).get("payload_sha256"):
        raise ValueError("V5 observability manifest identity/integrity drifted")
    pair_keys = validate_manifest(manifest)
    if manifest.get("protocol", {}).get("reference_controller") != "frozen_fixed_head_on":
        raise ValueError("V5 manifest controller drifted")
    for label, source in (manifest.get("disjointness") or {}).items():
        if not isinstance(source, Mapping):
            raise ValueError("V5 disjointness entry is invalid: {0}".format(label))
        _require_hash(_repo_path(str(source.get("path", ""))), source.get("sha256"), "V5 disjoint source {0}".format(label))
        if source.get("physical_intersection_count") != 0 or source.get("seed_intersection_count") != 0:
            raise ValueError("V5 manifest is not disjoint from {0}".format(label))
    _validate_runtime_registry(config)
    output = Path(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise ValueError("V5 output root must be absolute and absent")
    free_gb = shutil.disk_usage(output.parent).free / (1024 ** 3)
    return {
        "source_id": SOURCE_ID,
        "mode": "design_preflight_only_no_jsbsim_no_training",
        "scenario_count": len(pair_keys),
        "planned_records": len(pair_keys) * len(plan.opponents),
        "initial_states": list(plan.initial_states),
        "opponents": list(plan.opponents),
        "reference_controller": "frozen_fixed_head_on",
        "reference_profile": "front_intercept",
        "output_root_absent": True,
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= plan.min_free_disk_gb,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_design(args.config), indent=2, ensure_ascii=False))
    except (OSError, ValueError, yaml.YAMLError, FiveStateRawSIV5ContractError) as error:
        print("INVALID V5 observability design: {0}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
