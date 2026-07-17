#!/usr/bin/env python3
"""Validate the design-only selected-lateral post-merge runtime probe."""

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

from uav_vpp_guidance.evaluation.tactical_basis_lateral_runtime_probe_contract import (
    SOURCE_ID,
    TacticalBasisLateralRuntimeProbeDesignError,
    validate_design_spec,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"YAML mapping required: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"JSON mapping required: {path}")
    return value


def _require_input(config: Mapping[str, Any], key: str, *, loader: str) -> dict[str, Any] | None:
    inputs = config["inputs"]
    path = _resolve(str(inputs[key]))
    if not path.is_file() or _hash(path) != str(inputs[key + "_sha256"]):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"frozen input SHA mismatch: {key}")
    if loader == "json":
        return _load_json(path)
    if loader == "yaml":
        return _load_yaml(path)
    if loader == "file":
        return None
    raise TacticalBasisLateralRuntimeProbeDesignError(f"unsupported input loader: {loader}")


def validate_design(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "design_frozen_execution_not_authorized":
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe identity or design status drifted")
    p2_summary = _require_input(config, "p2_summary", loader="json")
    p2_verification = _require_input(config, "p2_verification", loader="json")
    p2_config = _require_input(config, "p2_config", loader="yaml")
    continuity_gate = _require_input(config, "continuity_gate", loader="json")
    probe_manifest = _require_input(config, "probe_manifest", loader="yaml")
    parent_manifest = _require_input(config, "parent_manifest", loader="yaml")
    _require_input(config, "runtime_config", loader="yaml")
    _require_input(config, "runtime_registry", loader="yaml")
    _require_input(config, "frozen_p3_encoder", loader="file")
    _require_input(config, "frozen_head_on_specialist", loader="file")
    _require_input(config, "temporal_reference_design", loader="yaml")
    _require_input(config, "temporal_reference_compiler", loader="file")
    output = _resolve(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe output root must be absolute and absent")
    if not isinstance(p2_summary, dict) or not isinstance(p2_verification, dict):
        raise TacticalBasisLateralRuntimeProbeDesignError("P2 JSON inputs are malformed")
    if not isinstance(p2_config, dict) or p2_config.get("source_id") != p2_summary.get("source_id"):
        raise TacticalBasisLateralRuntimeProbeDesignError("P2 config provenance is malformed")
    if not isinstance(continuity_gate, dict) or continuity_gate.get("passes") is not True:
        raise TacticalBasisLateralRuntimeProbeDesignError("V3 continuity gate is not valid")
    if not isinstance(probe_manifest, dict) or not isinstance(parent_manifest, dict):
        raise TacticalBasisLateralRuntimeProbeDesignError("scenario manifests are malformed")
    result = validate_design_spec(config, p2_summary, p2_verification, probe_manifest, parent_manifest)
    result.update(
        {
            "mode": "design_only_preflight",
            "output_root_absent": True,
            "free_disk_gb": round(shutil.disk_usage(output.parent).free / 1024**3, 3),
            "runtime_execution": "prohibited_pending_one_off_authorization",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise TacticalBasisLateralRuntimeProbeDesignError(
            "this command is design-only; an independent authorized runner is required before execution"
        )
    print(json.dumps(validate_design(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
