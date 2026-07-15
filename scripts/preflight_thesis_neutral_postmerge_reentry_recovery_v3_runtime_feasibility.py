#!/usr/bin/env python3
"""Fail-closed, non-executing preflight for V3 runtime feasibility."""

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

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v3_contract import (
    MANIFEST_SOURCE_ID,
    SOURCE_ID,
    NeutralPostMergeV3ContractError,
    build_plan,
    episode_identity,
    validate_feasibility_manifest,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
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
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch")


def _payload_sha256(manifest: Mapping[str, Any]) -> str:
    prepared = dict(manifest)
    integrity = dict(prepared.get("integrity") or {})
    integrity.pop("payload_sha256", None)
    prepared["integrity"] = integrity
    return hashlib.sha256(
        json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_authorizations(authorization: Mapping[str, Any]) -> None:
    if authorization.get("execution_permitted") is not False:
        raise ValueError("V3 design preflight requires execution_permitted=false")
    for key, value in authorization.items():
        if key != "execution_permitted" and value is not False:
            raise ValueError(f"V3 design preflight requires authorization.{key}=false")


def _validate_runtime_registry(config: Mapping[str, Any]) -> None:
    inputs = config["inputs"]
    registry_path = _repo_path(str(inputs["runtime_registry"]))
    _require_hash(registry_path, inputs.get("runtime_registry_sha256"), "runtime registry")
    registry = load_yaml(registry_path)
    opponents = tuple(config["opponents"]["order"])
    if tuple(registry.get("opponents", ())) != opponents:
        raise ValueError("runtime registry opponent order drifted")
    for opponent in opponents:
        entry = registry["opponents"][opponent]
        if opponent != "expert":
            _require_hash(Path(str(entry["checkpoint"])), entry.get("checkpoint_sha256"), f"opponent {opponent}")
    specialist = registry["specialists"].get("run_in_head_on")
    if not isinstance(specialist, Mapping):
        raise ValueError("runtime registry lacks run_in_head_on specialist")
    if specialist.get("checkpoint_sha256") != inputs.get("frozen_run_in_head_on_specialist_sha256"):
        raise ValueError("frozen run-in specialist SHA drifted")
    _require_hash(Path(str(specialist["checkpoint"])), specialist.get("checkpoint_sha256"), "run-in specialist")


def validate_design(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Verify all V3 inputs without instantiating JSBSim or creating output."""

    config = load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "preregistered_design_execution_not_authorised":
        raise ValueError("unexpected V3 design source or status")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("V3 authorization is missing")
    _validate_authorizations(authorization)
    plan = build_plan(config)
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V3 inputs are missing")
    manifest_path = _repo_path(str(inputs.get("feasibility_manifest", "")))
    train_path = _repo_path(str(inputs.get("train_distribution", "")))
    runtime_path = _repo_path(str(inputs.get("runtime_template", "")))
    encoder_path = Path(str(inputs.get("frozen_p3_encoder", "")))
    _require_hash(manifest_path, inputs.get("feasibility_manifest_sha256"), "feasibility manifest")
    _require_hash(train_path, inputs.get("train_distribution_sha256"), "train identity contract")
    _require_hash(runtime_path, inputs.get("runtime_template_sha256"), "runtime template")
    _require_hash(encoder_path, inputs.get("frozen_p3_encoder_sha256"), "frozen P3 encoder")
    manifest = load_yaml(manifest_path)
    if manifest.get("source_id") != MANIFEST_SOURCE_ID or _payload_sha256(manifest) != manifest.get("integrity", {}).get("payload_sha256"):
        raise ValueError("V3 feasibility manifest identity/integrity drifted")
    pair_keys = validate_feasibility_manifest(manifest)
    for label, source in (manifest.get("disjointness") or {}).items():
        if not isinstance(source, Mapping):
            raise ValueError(f"V3 manifest disjointness entry {label} is invalid")
        _require_hash(_repo_path(str(source.get("path", ""))), source.get("sha256"), f"V3 disjointness source {label}")
        if source.get("physical_intersection_count") != 0 or source.get("seed_intersection_count") != 0:
            raise ValueError(f"V3 manifest is not disjoint from {label}")
    train = load_yaml(train_path)
    train_identity = episode_identity(
        {
            "metadata": {
                "identity_kind": train.get("identity_contract", {}).get("identity_kind"),
                "train_stream_id": train.get("identity_contract", {}).get("train_stream_id"),
                "scenario_seed": train.get("sampling_contract", {}).get("seed"),
            }
        }
    )
    if train.get("training_permitted") is not False or train_identity["paired_delta_eligible"]:
        raise ValueError("V3 train identity must remain unpaired and non-executing")
    _validate_runtime_registry(config)
    output = Path(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise ValueError("V3 output root must be absolute and absent")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    return {
        "source_id": SOURCE_ID,
        "mode": "design_preflight_only_no_jsbsim_no_training",
        "execution_permitted": False,
        "training_permitted": False,
        "scenario_count": len(pair_keys),
        "planned_records": len(pair_keys) * len(plan.opponents),
        "families": list(plan.family_order),
        "opponents": list(plan.opponents),
        "train_identity": train_identity,
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
    except (OSError, ValueError, yaml.YAMLError, NeutralPostMergeV3ContractError) as error:
        print(f"INVALID V3 runtime-feasibility design: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
