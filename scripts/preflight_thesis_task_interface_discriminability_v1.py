#!/usr/bin/env python3
"""Static, non-learning contract check for the paired task-interface probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison  # noqa: E402


PREFLIGHT_ID = "ENGINEERING-TASK-INTERFACE-DISCRIMINABILITY-PREFLIGHT-V1-NOT-A-SOURCE-ID"
PROTOCOL_KEY = "task_interface_discriminability_preflight"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = ("canonical_ppo", "ppo_unbootstrapped_crossing")
CONTEXTS = ("head_on", "crossing_feasible")
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_task_interface_discriminability_preflight_v1.yaml"
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_task_interface_discriminability_preflight_v1.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            result.update(_flatten_config(item, child))
        return result
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True)}
    return {prefix: value}


def _paired_manifest_errors(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("source_id") != PREFLIGHT_ID or manifest.get("status") != "design_only_runtime_probe_not_authorized":
        errors.append("manifest source or design-only status mismatch")
    protocol = manifest.get("protocol", {})
    if protocol.get("nonlearning_only") is not True or protocol.get("training_permitted") is not False or protocol.get("performance_comparison_permitted") is not False:
        errors.append("manifest permits learning or performance comparison")
    if int(manifest.get("integrity", {}).get("scenario_count", 0)) != 8:
        errors.append("manifest must contain eight paired task-context aliases")
    if comparison._scenario_manifest_payload_sha256(copy.deepcopy(dict(manifest))) != manifest.get("integrity", {}).get("payload_sha256"):
        errors.append("manifest payload hash mismatch")
    paired: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for scenario in manifest.get("scenarios", []):
        metadata = scenario.get("metadata", {})
        paired[str(metadata.get("context_pair_id"))].append(scenario)
    if len(paired) != 4:
        errors.append("manifest does not contain four physical reset pairs")
    for pair_id, rows in paired.items():
        if len(rows) != 2:
            errors.append("{} does not have exactly two contexts".format(pair_id))
            continue
        by_context = {str(row.get("metadata", {}).get("interface_context")): row for row in rows}
        if set(by_context) != set(CONTEXTS):
            errors.append("{} context keys are incomplete".format(pair_id))
            continue
        head_on, crossing = by_context["head_on"], by_context["crossing_feasible"]
        for key in ("own_init", "target_init"):
            if head_on.get(key) != crossing.get(key):
                errors.append("{} changed physical {} across contexts".format(pair_id, key))
        head_meta, crossing_meta = head_on.get("metadata", {}), crossing.get("metadata", {})
        if head_meta.get("scenario_seed") != crossing_meta.get("scenario_seed"):
            errors.append("{} changed the paired scenario seed".format(pair_id))
        if (head_meta.get("task_registry_key"), head_meta.get("ppo_task_bit"), head_meta.get("crossing_context")) != ("head_on", 0, False):
            errors.append("{} head_on context contract mismatch".format(pair_id))
        if (crossing_meta.get("task_registry_key"), crossing_meta.get("ppo_task_bit"), crossing_meta.get("crossing_context")) != ("crossing_feasible", 1, True):
            errors.append("{} crossing context contract mismatch".format(pair_id))
    return errors


def run_preflight(config_path: Path = CONFIG) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = comparison._materialize_scenario_manifest(comparison._load_config_with_includes(config_path), config_path)
    manifest = _load_yaml(MANIFEST)
    errors = _paired_manifest_errors(manifest)
    protocol = config.get(PROTOCOL_KEY, {})
    if protocol.get("preflight_id") != PREFLIGHT_ID or protocol.get("execution_permitted") is not False:
        errors.append("config does not remain design-only")
    if protocol.get("training_permitted") is not False or protocol.get("tuning_permitted") is not False or protocol.get("performance_comparison_permitted") is not False:
        errors.append("config permits learning or performance comparison")
    if tuple(protocol.get("opponents") or ()) != OPPONENTS or tuple(protocol.get("methods") or ()) != METHODS or tuple(protocol.get("contexts") or ()) != CONTEXTS:
        errors.append("config matrix mismatch")
    if int(protocol.get("expected_runtime_records_if_separately_authorized", 0)) != 48:
        errors.append("config runtime-probe size mismatch")
    if config.get("env", {}).get("strict_backend") is not True:
        errors.append("strict JSBSim is not required")
    resolved_counts = config.get("scenario_manifest_resolution", {}).get("task_counts")
    if resolved_counts != {"head_on": 4, "crossing_feasible": 4}:
        errors.append("paired context task groups did not materialize as 4+4")
    canonical_overrides = config.get("methods", {}).get("canonical_ppo", {}).get("config_overrides", {})
    ablated_overrides = config.get("methods", {}).get("ppo_unbootstrapped_crossing", {}).get("config_overrides", {})
    if canonical_overrides.get("commander.crossing_initial_task_bootstrap.enabled") is not True or ablated_overrides.get("commander.crossing_initial_task_bootstrap.enabled") is not False:
        errors.append("M3/M4 bootstrap switch is not explicitly contrasted")
    if canonical_overrides.get("commander.crossing_pre_merge_mode_lock.enabled") is not False or ablated_overrides.get("commander.crossing_pre_merge_mode_lock.enabled") is not False:
        errors.append("pre-merge lock is not consistently disabled")

    config_cells: list[dict[str, Any]] = []
    checkpoint_checks: list[dict[str, Any]] = []
    resolved_configs: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for opponent in OPPONENTS:
        checks, check_errors = comparison._checkpoint_checks(config, list(METHODS), opponent, False)
        checkpoint_checks.extend(checks)
        errors.extend("{}: {}".format(opponent, error) for error in check_errors)
        for context in CONTEXTS:
            for method in METHODS:
                try:
                    resolved = comparison.build_eval_config(config, method, context, "jsbsim", opponent)
                    resolved_configs[(opponent, method, context)] = resolved
                    config_cells.append({"opponent": opponent, "context": context, "method": method, "strict_backend": resolved.get("env", {}).get("strict_backend"), "opponent_stage": resolved.get("opponent_stage"), "max_high_level_steps": resolved.get("env", {}).get("max_high_level_steps"), "high_level_dt_s": resolved.get("env", {}).get("high_level_dt")})
                except Exception as error:
                    errors.append("config build {}/{}/{}: {}".format(opponent, context, method, error))
    if len(config_cells) != 12 or any(not row["strict_backend"] or row["opponent_stage"] != row["opponent"] for row in config_cells):
        errors.append("runtime config matrix drifted")
    if any(row["max_high_level_steps"] != 20 or row["high_level_dt_s"] != 0.2 for row in config_cells):
        errors.append("runtime probe horizon or decision period drifted")
    context_config_differences: dict[str, list[str]] = {}
    for opponent in OPPONENTS:
        for method in METHODS:
            head_on = _flatten_config(resolved_configs[(opponent, method, "head_on")])
            crossing = _flatten_config(resolved_configs[(opponent, method, "crossing_feasible")])
            differences = sorted(
                key for key in set(head_on).union(crossing) if head_on.get(key) != crossing.get(key)
            )
            receipt_key = "{}:{}".format(opponent, method)
            context_config_differences[receipt_key] = differences
            allowed = ["provenance.config_overrides", "task.name", "task.scenarios"]
            if differences != allowed:
                errors.append("{} changes config outside task.name: {}".format(receipt_key, differences))
    return {
        "preflight_id": PREFLIGHT_ID,
        "config_path": str(config_path),
        "config_sha256": _hash(config_path),
        "manifest_path": str(MANIFEST),
        "manifest_sha256": _hash(MANIFEST),
        "manifest_payload_sha256": manifest.get("integrity", {}).get("payload_sha256"),
        "config_cells": config_cells,
        "context_config_differences": context_config_differences,
        "checkpoint_checks": checkpoint_checks,
        "errors": errors,
        "verdict": "pass" if not errors else "fail",
        "runtime_probe_design_ready": not errors,
        "execution_authorized": False,
        "next_action_if_pass": "separately preregister and authorize one 48-episode non-learning runtime probe",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = run_preflight(args.config)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
