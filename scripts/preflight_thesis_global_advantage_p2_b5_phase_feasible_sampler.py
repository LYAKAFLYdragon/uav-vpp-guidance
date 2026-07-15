#!/usr/bin/env python3
"""Validate the P2-B5 design without starting JSBSim or training."""

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

from uav_vpp_guidance.evaluation.global_advantage_p2_b5_phase_feasible_sampler import (  # noqa: E402
    MANIFEST_SOURCE_ID,
    PhaseFeasibleSamplerError,
    build_phase_feasible_sampler_plan,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b5_phase_feasible_sampler.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PhaseFeasibleSamplerError(f"expected YAML mapping: {path}")
    return payload


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise PhaseFeasibleSamplerError(f"{label} must declare SHA-256")
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise PhaseFeasibleSamplerError(f"{label} SHA-256 mismatch")


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(dict(manifest))
    prepared.pop("integrity", None)
    encoded = json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_design(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Check all immutable inputs while refusing to execute the sampler."""

    config = _load_yaml(config_path)
    plan = build_phase_feasible_sampler_plan(config)
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PhaseFeasibleSamplerError("B5 inputs must be a mapping")
    manifest_path = _repo_path(str(inputs.get("manifest", "")))
    runtime_path = _repo_path(str(inputs.get("runtime_config", "")))
    registry_path = _repo_path(str(inputs.get("runtime_registry", "")))
    _require_hash(manifest_path, inputs.get("manifest_sha256"), "B5 manifest")
    _require_hash(runtime_path, inputs.get("runtime_config_sha256"), "B5 runtime config")
    _require_hash(registry_path, inputs.get("runtime_registry_sha256"), "B5 runtime registry")
    _require_hash(plan.p3_checkpoint, plan.p3_checkpoint_sha256, "B5 frozen P3 encoder")
    manifest = _load_yaml(manifest_path)
    scenarios = list(manifest.get("scenarios") or [])
    if (
        manifest.get("source_id") != MANIFEST_SOURCE_ID
        or manifest.get("source_id") != inputs.get("manifest_source_id")
        or len(scenarios) != 12
        or _manifest_payload_sha256(manifest) != manifest.get("integrity", {}).get("payload_sha256")
    ):
        raise PhaseFeasibleSamplerError("B5 manifest identity/integrity drifted")
    if not all(
        (scenario.get("metadata") or {}).get("taxonomy_geometry_state") == "disadvantage"
        and (scenario.get("metadata") or {}).get("phase_at_reset") == "pre_merge"
        for scenario in scenarios
    ):
        raise PhaseFeasibleSamplerError("B5 manifest leaves the preregistered target boundary")
    for label, entry in (manifest.get("disjointness") or {}).items():
        source = _repo_path(str(entry.get("path", "")))
        _require_hash(source, entry.get("sha256"), f"B5 disjointness source {label}")
    registry = _load_yaml(registry_path)
    specialist = ((registry.get("specialists") or {}).get(plan.reference_specialist) or {})
    if specialist.get("checkpoint_sha256") != plan.reference_specialist_sha256:
        raise PhaseFeasibleSamplerError("B5 frozen reference specialist SHA drifted")
    for opponent in plan.opponents:
        if opponent not in (registry.get("opponents") or {}):
            raise PhaseFeasibleSamplerError(f"B5 registry is missing opponent: {opponent}")
    output_root = Path(str(config.get("outputs", {}).get("root", "")))
    if not output_root.is_absolute():
        raise PhaseFeasibleSamplerError("B5 output root must be absolute")
    free_gb = shutil.disk_usage(output_root.parent).free / (1024**3)
    return {
        "source_id": plan.source_id,
        "mode": "preflight_only_no_jsbsim_no_training",
        "execution_permitted": plan.execution_permitted,
        "training_permitted": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scenario_count": len(scenarios),
        "opponents": list(plan.opponents),
        "raw_si_phase_contract": True,
        "p3_checkpoint_sha256": plan.p3_checkpoint_sha256,
        "reference_specialist_sha256": plan.reference_specialist_sha256,
        "output_root": str(output_root),
        "output_root_absent": not output_root.exists(),
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= plan.min_free_disk_gb,
        "next_action": "independent_preregistration_review_then_one_time_authorization_or_stop",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        result = validate_design(args.config)
    except (OSError, ValueError, yaml.YAMLError, PhaseFeasibleSamplerError) as error:
        print(f"INVALID B5 phase-feasible sampler design: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["output_root_absent"] and result["disk_gate_would_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
