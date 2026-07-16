#!/usr/bin/env python3
"""Independently verify the frozen 900-record Taxonomy60 formal universe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison  # noqa: E402


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
PRIMARY_OPPONENTS = ("expert", "end_to_end")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
PPO_METHODS = {"canonical_ppo", "ppo_unbootstrapped_crossing"}
FORMAL_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_formal.yaml"
MANIFEST_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
DECISION_CONTRACT_PATH = (
    REPO_ROOT
    / "config"
    / "experiment"
    / "decision_contracts"
    / "thesis_dualskill_taxonomy60_no_go_v3.yaml"
)


class FormalVerificationError(RuntimeError):
    """Raised internally for a record-universe contract violation."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise FormalVerificationError("{} is missing {}".format(context, key))
    return mapping[key]


def _run_dir(root: Path, opponent: str) -> Path:
    return root / "{}-{}".format(SOURCE_ID, opponent)


def _expected_universe(manifest: Mapping[str, Any]) -> Dict[Tuple[str, str, str], str]:
    expected: Dict[Tuple[str, str, str], str] = {}
    scenarios = manifest.get("scenarios", [])
    if len(scenarios) != 60:
        raise FormalVerificationError("formal manifest does not contain 60 scenarios")
    for scenario in scenarios:
        name = scenario.get("name")
        task = scenario.get("metadata", {}).get("task_registry_key")
        if not isinstance(name, str) or task not in {"head_on", "crossing_feasible"}:
            raise FormalVerificationError("formal manifest scenario lacks name/task contract")
        for opponent in OPPONENTS:
            for method in METHODS:
                expected[(name, opponent, method)] = task
    if len(expected) != 900:
        raise FormalVerificationError("formal universe does not expand to 900 records")
    return expected


def _validate_execution_manifest(
    root: Path,
    *,
    formal_config_path: Path,
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    execution = _load_json(root / "formal_execution_manifest.json")
    if execution.get("source_id") != SOURCE_ID:
        raise FormalVerificationError("formal execution Source ID mismatch")
    if execution.get("attempt_kind") != "initial":
        raise FormalVerificationError("formal execution attempt kind is not frozen initial")
    if execution.get("execution_status") != "completed":
        raise FormalVerificationError("formal execution did not complete")
    if execution.get("formal_config_sha256") != _sha256(formal_config_path):
        raise FormalVerificationError("formal execution config hash mismatch")
    if execution.get("manifest_payload_sha256") != manifest.get("integrity", {}).get("payload_sha256"):
        raise FormalVerificationError("formal execution manifest payload mismatch")
    if execution.get("decision_contract_sha256") != _sha256(DECISION_CONTRACT_PATH):
        raise FormalVerificationError("formal execution decision-contract hash mismatch")
    if execution.get("decision_contract_id") != decision.get("contract_id"):
        raise FormalVerificationError("formal execution decision-contract ID mismatch")
    if int(execution.get("expected_records", 0)) != 900:
        raise FormalVerificationError("formal execution expected record count mismatch")
    executions = execution.get("executions")
    if not isinstance(executions, list) or len(executions) != len(OPPONENTS):
        raise FormalVerificationError("formal execution child receipt count mismatch")
    if any(item.get("returncode") != 0 for item in executions if isinstance(item, Mapping)):
        raise FormalVerificationError("formal execution contains a failed child run")
    return execution


def _validate_root_contract(root: Path) -> None:
    for relative in (
        "formal_execution_manifest.json",
        "formal_execution_plan.json",
        "formal_artifact_contract.json",
    ):
        if not (root / relative).is_file():
            raise FormalVerificationError("formal root is missing {}".format(relative))
    contract = _load_json(root / "formal_artifact_contract.json")
    children = contract.get("required_child_runs")
    if not isinstance(children, list) or len(children) != len(OPPONENTS):
        raise FormalVerificationError("formal root contract child matrix mismatch")
    expected = {"{}-{}".format(SOURCE_ID, opponent) for opponent in OPPONENTS}
    if {item.get("run_id") for item in children} != expected:
        raise FormalVerificationError("formal root contract run IDs mismatch")


def _verify_prediction_receipt(trajectory: Sequence[Mapping[str, Any]], *, context: str) -> None:
    for index, step in enumerate(trajectory):
        valid = _require(step, "prediction_valid", context=context)
        fallback = _require(step, "prediction_fallback", context=context)
        if not isinstance(valid, bool) or not isinstance(fallback, bool):
            raise FormalVerificationError("{} has non-boolean prediction receipt".format(context))
        if index < 9:
            if valid or not fallback:
                raise FormalVerificationError("{} violates the fixed 9-step prediction warm-up".format(context))
        elif fallback or not valid:
            raise FormalVerificationError("{} has invalid prediction after warm-up".format(context))


def _verify_step(
    step: Mapping[str, Any],
    *,
    method: str,
    context: str,
) -> bool:
    action = _require(step, "vpp_action", context=context)
    if not isinstance(action, list) or len(action) != 3 or any(not _is_finite(value) for value in action):
        raise FormalVerificationError("{} has non-finite 3-D VPP action".format(context))
    if not _is_finite(_require(step, "aoa_deg", context=context)):
        raise FormalVerificationError("{} has non-finite physical AoA".format(context))
    for key in ("nz_cmd", "nz_actual"):
        if not _is_finite(_require(step, key, context=context)):
            raise FormalVerificationError("{} has non-finite {}".format(context, key))

    guidance = _require(step, "guidance_telemetry", context=context)
    if not isinstance(guidance, Mapping) or not isinstance(guidance.get("effective_guidance_mode"), str):
        raise FormalVerificationError("{} has unreadable guidance telemetry".format(context))
    for key in ("nz_cmd", "roll_rate_cmd", "throttle_cmd", "nz_actual"):
        if not _is_finite(guidance.get(key)):
            raise FormalVerificationError("{} has non-finite guidance {}".format(context, key))

    clipping = _require(step, "clipping", context=context)
    if not isinstance(clipping, Mapping) or any(
        not isinstance(clipping.get(key), bool)
        for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated", "any_saturated")
    ):
        raise FormalVerificationError("{} has unreadable clipping telemetry".format(context))

    for key in (
        "commander_requested_mode_id",
        "commander_mode_id",
        "commander_mode_constraint_triggered",
        "commander_mode_constraint_reason",
        "commander_crossing_initial_task_bootstrap_enabled",
        "commander_crossing_initial_task_bootstrap_active",
        "commander_crossing_pre_merge_mode_lock_active",
    ):
        _require(step, key, context=context)
    if method in PPO_METHODS:
        if not isinstance(step["commander_requested_mode_id"], int) or not isinstance(step["commander_mode_id"], int):
            raise FormalVerificationError("{} has unreadable PPO mode telemetry".format(context))
        for key in (
            "commander_mode_constraint_triggered",
            "commander_crossing_initial_task_bootstrap_enabled",
            "commander_crossing_initial_task_bootstrap_active",
            "commander_crossing_pre_merge_mode_lock_active",
        ):
            if not isinstance(step[key], bool):
                raise FormalVerificationError("{} has non-boolean {}".format(context, key))
    requested = step.get("commander_requested_mode_id")
    effective = step.get("commander_mode_id")
    return isinstance(requested, int) and isinstance(effective, int) and requested != effective


def _validate_record(
    record: Mapping[str, Any],
    *,
    opponent: str,
    expected_task: str,
) -> Tuple[Tuple[str, str, str], bool]:
    context = "{}/{}".format(opponent, record.get("controller", "unknown"))
    for key in (
        "controller",
        "task",
        "opponent_stage",
        "scenario",
        "seed",
        "backend",
        "strict_backend",
        "backend_fallback_occurred",
        "run_status",
        "terminal_reason",
        "outcome",
        "trajectory",
    ):
        _require(record, key, context=context)
    method = str(record["controller"])
    if method not in METHODS:
        raise FormalVerificationError("{} has unexpected method".format(context))
    if str(record["opponent_stage"]) != opponent:
        raise FormalVerificationError("{} has mismatched opponent".format(context))
    if str(record["task"]) != expected_task:
        raise FormalVerificationError("{} has mismatched task".format(context))
    if record["backend"] != "jsbsim" or record["strict_backend"] is not True:
        raise FormalVerificationError("{} is not strict JSBSim".format(context))
    if bool(record["backend_fallback_occurred"]):
        raise FormalVerificationError("{} reports backend fallback".format(context))
    if bool(record.get("checkpoint_fallback_occurred", False)):
        raise FormalVerificationError("{} reports checkpoint fallback".format(context))
    if record["run_status"] != "formal":
        raise FormalVerificationError("{} has non-formal run status".format(context))
    if not isinstance(record["terminal_reason"], str) or not record["terminal_reason"]:
        raise FormalVerificationError("{} has unreadable terminal reason".format(context))
    if not isinstance(record["outcome"], str) or not record["outcome"]:
        raise FormalVerificationError("{} has unreadable outcome".format(context))
    trajectory = record["trajectory"]
    if not isinstance(trajectory, list) or not trajectory:
        raise FormalVerificationError("{} has no raw trajectory".format(context))
    guard_changed = False
    for index, step in enumerate(trajectory):
        if not isinstance(step, Mapping):
            raise FormalVerificationError("{} trajectory step {} is invalid".format(context, index))
        guard_changed = _verify_step(
            step,
            method=method,
            context="{} step {}".format(context, index),
        ) or guard_changed
    _verify_prediction_receipt(trajectory, context=context)
    if method == "canonical_ppo" and expected_task == "crossing_feasible":
        first = trajectory[0]
        if first["commander_crossing_initial_task_bootstrap_enabled"] is not True or first[
            "commander_crossing_initial_task_bootstrap_active"
        ] is not True:
            raise FormalVerificationError("{} canonical crossing bootstrap is not active".format(context))
    if method == "ppo_unbootstrapped_crossing" and expected_task == "crossing_feasible":
        first = trajectory[0]
        if first["commander_crossing_initial_task_bootstrap_enabled"] is not False or first[
            "commander_crossing_initial_task_bootstrap_active"
        ] is not False:
            raise FormalVerificationError("{} crossing ablation bootstrap is not disabled".format(context))
    if method in PPO_METHODS and any(
        step["commander_crossing_pre_merge_mode_lock_active"] for step in trajectory
    ):
        raise FormalVerificationError("{} activates the frozen-disabled pre-merge lock".format(context))
    return (str(record["scenario"]), opponent, method), guard_changed


def _load_complete_records(
    root: Path,
    *,
    expected: Mapping[Tuple[str, str, str], str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    guard_confounded: List[str] = []
    actual_keys: List[Tuple[str, str, str]] = []
    for opponent in OPPONENTS:
        run_dir = _run_dir(root, opponent)
        for relative in (
            "run_manifest.json",
            "artifact_contract.json",
            "failures.json",
            "aggregate/episode_records.json",
            "observation_audit.json",
        ):
            if not (run_dir / relative).is_file():
                raise FormalVerificationError("{} is missing {}".format(opponent, relative))
        child_manifest = _load_json(run_dir / "run_manifest.json")
        failures = _load_json(run_dir / "failures.json")
        child_contract = _load_json(run_dir / "artifact_contract.json")
        if child_manifest.get("status") != "completed" or child_manifest.get("paper_safe") is not True:
            raise FormalVerificationError("{} child run is not completed paper-safe evidence".format(opponent))
        extra = child_manifest.get("extra", {})
        if extra.get("backend") != "jsbsim" or extra.get("opponent_stage") != opponent:
            raise FormalVerificationError("{} child manifest backend/opponent mismatch".format(opponent))
        if int(extra.get("total_episodes", 0)) != 300:
            raise FormalVerificationError("{} child record count is not 300".format(opponent))
        checkpoint_checks = extra.get("checkpoint_checks")
        if not isinstance(checkpoint_checks, list) or not checkpoint_checks or any(
            not isinstance(item, Mapping) or item.get("exists") is not True
            for item in checkpoint_checks
        ):
            raise FormalVerificationError("{} child checkpoint receipt is incomplete".format(opponent))
        if not extra.get("artifact_validation", {}).get("valid", False):
            raise FormalVerificationError("{} child artifact contract is invalid".format(opponent))
        if failures.get("failures"):
            raise FormalVerificationError("{} child run recorded failures".format(opponent))
        if "aggregate/episode_records.json" not in child_contract.get("required_files", []):
            raise FormalVerificationError("{} child contract lacks raw episode records".format(opponent))

        payload = _load_json(run_dir / "aggregate" / "episode_records.json")
        child_records = payload.get("episodes") if isinstance(payload, Mapping) else None
        if payload.get("n_episodes") != 300 or not isinstance(child_records, list) or len(child_records) != 300:
            raise FormalVerificationError("{} child aggregate count is not 300".format(opponent))
        for record in child_records:
            if not isinstance(record, Mapping):
                raise FormalVerificationError("{} contains a non-mapping record".format(opponent))
            scenario = str(record.get("scenario", ""))
            method = str(record.get("controller", ""))
            key = (scenario, opponent, method)
            expected_task = expected.get(key)
            if expected_task is None:
                raise FormalVerificationError("{} record is outside the frozen universe".format(opponent))
            record_key, guard_changed = _validate_record(
                record,
                opponent=opponent,
                expected_task=expected_task,
            )
            actual_keys.append(record_key)
            copied = dict(record)
            copied["_taxonomy60_guard_changed_requested_mode"] = guard_changed
            records.append(copied)
            if guard_changed:
                guard_confounded.append("{}|{}|{}".format(*record_key))

    duplicates = [key for key, count in Counter(actual_keys).items() if count != 1]
    if duplicates:
        raise FormalVerificationError("formal universe contains duplicate record identities")
    if set(actual_keys) != set(expected):
        raise FormalVerificationError("formal universe is not exactly the frozen 900-record matrix")
    return records, guard_confounded


def _first_mode(record: Mapping[str, Any]) -> Any:
    trajectory = record.get("trajectory", [])
    return trajectory[0].get("commander_requested_mode_id") if trajectory else None


def _classify_cells(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_cell: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        by_cell[(str(record["scenario"]), str(record["opponent_stage"]))][
            str(record["controller"])
        ] = record
    classifications: List[Dict[str, Any]] = []
    for (scenario, opponent), methods in sorted(by_cell.items()):
        labels: List[str] = []
        canonical = methods["canonical_ppo"]
        fixed_head_on = methods["fixed_head_on"]
        fixed_crossing = methods["fixed_crossing"]
        unbootstrapped = methods["ppo_unbootstrapped_crossing"]
        if canonical.get("_taxonomy60_guard_changed_requested_mode") or unbootstrapped.get(
            "_taxonomy60_guard_changed_requested_mode"
        ):
            labels.append("mixed_interface_guard_bound")
        if canonical.get("outcome") == "loss" and any(
            fixed.get("outcome") == "win" for fixed in (fixed_head_on, fixed_crossing)
        ):
            labels.append("routing_sensitive_candidate")
        if all(item.get("outcome") == "loss" for item in (fixed_head_on, fixed_crossing, canonical)):
            labels.append("two_skill_capacity_boundary_candidate")
        outcomes = {item.get("outcome") for item in methods.values()}
        terminals = {item.get("terminal_reason") for item in methods.values()}
        if len(outcomes) == 1 and len(terminals) == 1:
            labels.append("opponent_dependent_or_non_discriminative")
        if scenario.startswith("taxonomy60_crossing") and _first_mode(canonical) != _first_mode(unbootstrapped):
            labels.append("crossing_constraint_mechanism")
        if len(terminals) > 1 or any("crash" in str(reason).lower() or "out_of_bounds" in str(reason).lower() for reason in terminals):
            labels.append("terminal_semantic_qualification")
        if not labels:
            labels.append("paper_closure")
        classifications.append(
            {
                "scenario_id": scenario,
                "opponent": opponent,
                "labels": labels,
                "method_outcomes": {method: item.get("outcome") for method, item in methods.items()},
                "terminal_reasons": {method: item.get("terminal_reason") for method, item in methods.items()},
            }
        )
    return classifications


def _evidence_by_opponent(classifications: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for opponent in OPPONENTS:
        cells = [item for item in classifications if item.get("opponent") == opponent]
        output[opponent] = {
            "complete_cells": len(cells),
            "classification_counts": dict(
                Counter(label for cell in cells for label in cell.get("labels", []))
            ),
        }
    return output


def _route_decision(
    *,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    validity: str,
    error: str | None,
    classifications: List[Dict[str, Any]],
    guard_confounded: List[str],
) -> Dict[str, Any]:
    routes = contract["allowed_routes"]
    precedence = contract["classification"]["precedence"]
    if validity != "pass":
        selected = "implementation_invalid"
        route_reason = "Formal record universe is invalid; no partial performance result may be interpreted."
        counts = Counter({"implementation_invalid": 1})
    else:
        selected = "paper_closure"
        route_reason = (
            "The complete formal universe is valid. All positive, mixed, and negative cell findings "
            "remain on the pre-registered paper-closure route; no training is authorized."
        )
        counts = Counter(label for cell in classifications for label in cell["labels"])
    candidates = [
        {
            "classification": label,
            "matched_cell_count": int(counts.get(label, 0)),
            "selected_study_route": contract["classification_routes"][label]["selected_study_route"],
        }
        for label in precedence
    ]
    selected_route = routes[selected]
    evidence = _evidence_by_opponent(classifications) if validity == "pass" else {}
    return {
        "source_id": SOURCE_ID,
        "decision_contract_sha256": _sha256(DECISION_CONTRACT_PATH),
        "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "formal_validity_verdict": validity,
        "primary_opponent_evidence": {key: evidence.get(key, {}) for key in PRIMARY_OPPONENTS},
        "supporting_opponent_evidence": {"independent_ppo_vpp": evidence.get("independent_ppo_vpp", {})},
        "cell_classifications": classifications,
        "guard_confounded_episode_ids": sorted(guard_confounded),
        "route_candidates_in_precedence_order": candidates,
        "selected_study_route": selected,
        "route_transition_reason": route_reason,
        "post_closure_eligibility": {
            "eligible": False,
            "reason": (
                "A future optional_new_interface_memo requires all independent conditions and is never "
                "authorized by this verifier."
            ),
        },
        "allowed_next_action": selected_route["next_action"],
        "prohibited_actions": selected_route.get("prohibited", []),
        "recovery_state": {
            "attempt_kind": "initial",
            "formal_source_started": True,
            "r1_consumed": False,
            "next_route_if_recovery_proof_is_absent": "paper_two_opponent_only",
        },
        "reviewer_signoff": {"status": "pending_independent_reviewer", "signed_by": None},
        "verification_error": error,
    }


def verify_formal_root(
    root: Path,
    *,
    formal_config_path: Path = FORMAL_CONFIG,
) -> Dict[str, Any]:
    """Return a pass/fail receipt; only pass paths inspect comparative outcomes."""

    root = Path(root)
    manifest = _load_yaml(MANIFEST_PATH)
    decision = _load_yaml(DECISION_CONTRACT_PATH)
    try:
        _validate_execution_manifest(
            root,
            formal_config_path=formal_config_path,
            manifest=manifest,
            decision=decision,
        )
        _validate_root_contract(root)
        expected = _expected_universe(manifest)
        records, guard_confounded = _load_complete_records(root, expected=expected)
        classifications = _classify_cells(records)
        decision_payload = _route_decision(
            contract=decision,
            manifest=manifest,
            validity="pass",
            error=None,
            classifications=classifications,
            guard_confounded=guard_confounded,
        )
        result = {
            "source_id": SOURCE_ID,
            "verdict": "pass",
            "expected_records": 900,
            "records": len(records),
            "guard_confounded_records": len(guard_confounded),
            "route_decision": decision_payload,
        }
    except (OSError, ValueError, FormalVerificationError, KeyError, TypeError) as error:
        decision_payload = _route_decision(
            contract=decision,
            manifest=manifest,
            validity="fail",
            error=str(error),
            classifications=[],
            guard_confounded=[],
        )
        result = {
            "source_id": SOURCE_ID,
            "verdict": "fail",
            "expected_records": 900,
            "records": 0,
            "error": str(error),
            "route_decision": decision_payload,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, default=FORMAL_CONFIG)
    args = parser.parse_args()
    root = args.output_root.resolve()
    result = verify_formal_root(root, formal_config_path=args.formal_config.resolve())
    root.mkdir(parents=True, exist_ok=True)
    (root / "taxonomy60_route_decision.json").write_text(
        json.dumps(result["route_decision"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "formal_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
