#!/usr/bin/env python3
"""Fail-closed design preflight for the V2 neutral/post-merge pilot."""

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
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_pairing_v2 import (  # noqa: E402
    NeutralPostMergeV2PairingError,
    load_yaml,
    validate_manifest,
)


SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_pilot_v2.yaml"


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
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or _sha256(path) != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch")


def validate_design(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "preregistered_design_only_not_authorised":
        raise ValueError("unexpected V2 source or status")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping) or not all(value is False for value in authorization.values()):
        raise ValueError("V2 design must keep every authorization permission false")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("V2 sources are missing")
    for name in ("train_distribution", "dev_manifest", "heldout_manifest", "manifest_builder", "pairing_contract"):
        _require_hash(_repo_path(str(sources.get(name, ""))), sources.get(f"{name}_sha256"), name)
    _require_hash(
        _repo_path(str(sources.get("execution_runtime_registry", ""))),
        sources.get("execution_runtime_registry_sha256"),
        "execution_runtime_registry",
    )
    dev = load_yaml(_repo_path(str(sources["dev_manifest"])))
    heldout = load_yaml(_repo_path(str(sources["heldout_manifest"])))
    dev_keys = validate_manifest(dev, split="dev12", expected_count=12)
    heldout_keys = validate_manifest(heldout, split="heldout24", expected_count=24)
    if set(dev_keys) & set(heldout_keys):
        raise ValueError("V2 dev and heldout pair keys overlap")
    pairing = config.get("pairing_contract")
    if not isinstance(pairing, Mapping) or pairing.get("primary_key") != "metadata.geometry_cell_id + metadata.scenario_seed" or pairing.get("serialized_field") != "metadata.pair_key" or pairing.get("required_before_paired_delta") is not True:
        raise ValueError("V2 pairing contract drifted")
    train = load_yaml(_repo_path(str(sources["train_distribution"])))
    support = train.get("sampling_contract") or {}
    if train.get("source_id") != f"{SOURCE_ID}-TRAIN" or support.get("initial_class") != "neutral" or support.get("candidate_skill") != "reentry_recovery" or support.get("fixed_profile") != "reentry_preparation" or support.get("own_to_target_los_angle_deg", [0, 999])[1] > 165.0:
        raise ValueError("V2 train distribution drifted")
    encoder = config.get("fixed_contract", {}).get("encoder") or {}
    checkpoint = Path(str(encoder.get("checkpoint", "")))
    if encoder.get("trainable") is not False or encoder.get("fallback") != "prohibited":
        raise ValueError("V2 encoder must remain frozen without fallback")
    _require_hash(checkpoint, encoder.get("sha256"), "frozen P3 encoder")
    output = Path(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise ValueError("V2 output root must be absolute and absent")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    return {"source_id": SOURCE_ID, "mode": "design_preflight_only_no_jsbsim_no_training", "execution_permitted": False, "training_permitted": False, "dev_pair_keys": len(dev_keys), "heldout_pair_keys": len(heldout_keys), "output_root_absent": True, "free_disk_gb": round(free_gb, 3), "disk_gate_would_pass": free_gb >= 120.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_design(args.config), indent=2, ensure_ascii=False))
    except (OSError, ValueError, yaml.YAMLError, NeutralPostMergeV2PairingError) as error:
        print(f"INVALID V2 neutral post-merge design: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
