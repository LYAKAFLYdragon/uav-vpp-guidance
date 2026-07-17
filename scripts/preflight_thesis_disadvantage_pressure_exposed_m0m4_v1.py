#!/usr/bin/env python3
"""Fail-closed preflight for the pressure-exposed M0--M4 formal source."""

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
sys.path.insert(0, str(ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison  # noqa: E402


SOURCE_ID = "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-M0M4-3OPP-V1"
PROTOCOL_KEY = "pressure_exposed_m0m4_compact_ablation"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_disadvantage_pressure_exposed_postmerge_v1.yaml"
DECISION = ROOT / "config" / "experiment" / "decision_contracts" / "thesis_disadvantage_pressure_exposed_m0m4_no_go_v1.yaml"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_disadvantage_pressure_exposed_m0m4_v1.yaml"
DEFAULT_OUTPUT = Path("E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1")
MIN_FREE_BYTES = 50 * 1024**3


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_manifest_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_static(config: Mapping[str, Any], manifest: Mapping[str, Any], decision: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    protocol = config.get(PROTOCOL_KEY, {})
    if protocol.get("source_id") != SOURCE_ID:
        errors.append("Source ID mismatch")
    if protocol.get("evaluation_only") is not True or protocol.get("training_permitted") is not False or protocol.get("tuning_permitted") is not False:
        errors.append("evaluation-only authorization mismatch")
    if tuple(protocol.get("opponents") or ()) != OPPONENTS or tuple(protocol.get("methods") or ()) != METHODS:
        errors.append("frozen opponent or method matrix mismatch")
    if int(protocol.get("expected_episodes", 0)) != 240 or int(protocol.get("expected_episodes_per_opponent", 0)) != 80:
        errors.append("expected episode count mismatch")
    if config.get("env", {}).get("strict_backend") is not True:
        errors.append("strict JSBSim is not required")
    if tuple(config.get("run_defaults", {}).get("tasks") or ()) != ("head_on",):
        errors.append("compact source must contain only the frozen head_on task key")
    if tuple(config.get("run_defaults", {}).get("ablation_methods") or ()) != METHODS:
        errors.append("run-default method matrix mismatch")
    if manifest.get("integrity", {}).get("scenario_count") != 16:
        errors.append("pressure manifest does not contain 16 scenarios")
    scenarios = manifest.get("scenarios") or []
    if len(scenarios) != 16 or any(item.get("metadata", {}).get("initial_class") != "disadvantage" for item in scenarios):
        errors.append("manifest is not the fixed initial-disadvantage cohort")
    if any(item.get("metadata", {}).get("task_registry_key") != "head_on" for item in scenarios):
        errors.append("manifest task mapping drifted")
    signatures = {item.get("metadata", {}).get("mirror_pair_id") for item in scenarios}
    mirrors = {item.get("metadata", {}).get("mirror_sign") for item in scenarios}
    if len(signatures) != 8 or mirrors != {"negative", "positive"}:
        errors.append("manifest signature or mirror coverage mismatch")
    selection = config.get("scenario_manifest", {})
    generic_manifest_hash = comparison._scenario_manifest_payload_sha256(dict(manifest))
    scenario_only_hash = manifest.get("integrity", {}).get("payload_sha256")
    if selection.get("payload_sha256") != generic_manifest_hash:
        errors.append("config does not bind the runner-compatible manifest payload")
    if scenario_only_hash != "9c80977500700a43a279231f8b7556a67a682f002eff5fdfc2a4372f5450cf4b":
        errors.append("frozen manifest scenario-only integrity receipt drifted")
    if decision.get("source_id") != SOURCE_ID or int(decision.get("frozen_matrix", {}).get("expected_records", 0)) != 240:
        errors.append("decision contract does not bind this frozen source")
    return errors


def _telemetry_prerequisite(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    spec = config[PROTOCOL_KEY]["telemetry_prerequisite"]
    path = Path(str(spec["path"]))
    if not path.is_file():
        return {"path": str(path), "exists": False}, ["telemetry-completion prerequisite is missing"]
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("passes") is not bool(spec.get("required_passes")) or int(gate.get("records", 0)) != 48:
        errors.append("telemetry-completion prerequisite did not pass its 48-record gate")
    for opponent in OPPONENTS:
        details = gate.get("per_opponent", {}).get(opponent, {})
        if (
            int(details.get("records", 0)) != 16
            or int(details.get("reset_receipts_passed", 0)) != 16
            or int(details.get("physical_trace_passed", 0)) != 16
            or len(details.get("signatures", [])) != 8
            or set(details.get("mirrors", [])) != {"negative", "positive"}
        ):
            errors.append("telemetry-completion prerequisite failed for {}".format(opponent))
    return {"path": str(path), "sha256": _hash(path), "gate": gate}, errors


def run_preflight(*, config_path: Path = DEFAULT_CONFIG, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = comparison._materialize_scenario_manifest(
        comparison._load_config_with_includes(config_path), config_path
    )
    manifest = _load_yaml(MANIFEST)
    decision = _load_yaml(DECISION)
    errors = _validate_static(config, manifest, decision)
    prerequisite, prerequisite_errors = _telemetry_prerequisite(config)
    errors.extend(prerequisite_errors)

    config_cells: list[dict[str, Any]] = []
    checkpoint_checks: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        checks, check_errors = comparison._checkpoint_checks(config, list(METHODS), opponent, False)
        checkpoint_checks.extend(checks)
        errors.extend("{}: {}".format(opponent, item) for item in check_errors)
        for method in METHODS:
            try:
                resolved = comparison.build_eval_config(config, method, "head_on", "jsbsim", opponent)
                config_cells.append(
                    {
                        "opponent": opponent,
                        "method": method,
                        "strict_backend": resolved.get("env", {}).get("strict_backend"),
                        "opponent_stage": resolved.get("opponent_stage"),
                    }
                )
            except Exception as error:
                errors.append("config build {}/{}: {}".format(opponent, method, error))
    if len(config_cells) != 15 or any(not row["strict_backend"] or row["opponent_stage"] != row["opponent"] for row in config_cells):
        errors.append("one or more method/opponent runtime configs drifted")

    assets: list[dict[str, Any]] = []
    for item in config.get("frozen_assets", {}).get("artifacts", []):
        path = Path(str(item.get("path", "")))
        actual = _hash(path) if path.is_file() else None
        assets.append({"id": item.get("id"), "path": str(path), "exists": path.is_file(), "expected_sha256": item.get("sha256"), "actual_sha256": actual})
    if any(not item["exists"] or item["expected_sha256"] != item["actual_sha256"] for item in assets):
        errors.append("frozen asset hash mismatch")

    jsbsim_root = Path(str(config[PROTOCOL_KEY]["jsbsim_legacy_root"]))
    jsbsim_data = jsbsim_root / "envs" / "JSBSim" / "data"
    data_hash = _directory_manifest_sha256(jsbsim_data) if jsbsim_data.is_dir() else None
    if data_hash != config[PROTOCOL_KEY]["jsbsim_data_manifest_sha256"]:
        errors.append("JSBSim data manifest hash mismatch")
    drive = output_root.drive or output_root.anchor
    free_bytes = shutil.disk_usage(drive).free
    if output_root.exists():
        errors.append("formal output root already exists")
    if free_bytes < MIN_FREE_BYTES:
        errors.append("output drive has less than the 50 GiB reserve")
    return {
        "source_id": SOURCE_ID,
        "config_path": str(config_path),
        "config_sha256": _hash(config_path),
        "manifest_path": str(MANIFEST),
        "manifest_sha256": _hash(MANIFEST),
        "manifest_runner_payload_sha256": comparison._scenario_manifest_payload_sha256(dict(manifest)),
        "manifest_scenario_only_payload_sha256": manifest.get("integrity", {}).get("payload_sha256"),
        "decision_contract_path": str(DECISION),
        "decision_contract_sha256": _hash(DECISION),
        "telemetry_prerequisite": prerequisite,
        "config_cells": config_cells,
        "checkpoint_checks": checkpoint_checks,
        "assets": assets,
        "jsbsim_data": {"path": str(jsbsim_data), "exists": jsbsim_data.is_dir(), "sha256": data_hash},
        "output_root": str(output_root),
        "output_root_exists": output_root.exists(),
        "free_bytes": free_bytes,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "errors": errors,
        "verdict": "pass" if not errors else "fail",
        "execution_authorized": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = run_preflight(config_path=args.config, output_root=args.output_root)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
