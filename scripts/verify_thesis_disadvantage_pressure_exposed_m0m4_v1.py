#!/usr/bin/env python3
"""Independently verify the complete pressure-exposed M0--M4 evidence matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry  # noqa: E402


SOURCE_ID = "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-M0M4-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = ("fixed_head_on", "fixed_crossing", "static_oracle_gate", "canonical_ppo", "ppo_unbootstrapped_crossing")
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_disadvantage_pressure_exposed_postmerge_v1.yaml"
DECISION = ROOT / "config" / "experiment" / "decision_contracts" / "thesis_disadvantage_pressure_exposed_m0m4_no_go_v1.yaml"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_disadvantage_pressure_exposed_m0m4_v1.yaml"


class VerificationError(RuntimeError):
    pass


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _vector(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(_finite(item) for item in value)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _expected_universe() -> dict[tuple[str, str, str], str]:
    manifest = _load_yaml(MANIFEST)
    scenarios = manifest.get("scenarios") or []
    expected: dict[tuple[str, str, str], str] = {}
    for scenario in scenarios:
        name = scenario.get("name")
        task = scenario.get("metadata", {}).get("task_registry_key")
        if not isinstance(name, str) or task != "head_on":
            raise VerificationError("frozen manifest identity or task mapping is invalid")
        for opponent in OPPONENTS:
            for method in METHODS:
                expected[(name, opponent, method)] = task
    if len(expected) != 240:
        raise VerificationError("frozen universe is not 240 records")
    return expected


def _verify_prediction(trajectory: list[Mapping[str, Any]], context: str) -> None:
    for index, step in enumerate(trajectory):
        valid, fallback = step.get("prediction_valid"), step.get("prediction_fallback")
        if not isinstance(valid, bool) or not isinstance(fallback, bool):
            raise VerificationError("{} has unreadable prediction receipt".format(context))
        if index < 9 and (valid or not fallback):
            raise VerificationError("{} violates the fixed nine-step prediction warm-up".format(context))
        if index >= 9 and (not valid or fallback):
            raise VerificationError("{} has prediction fallback after warm-up".format(context))


def _verify_step(step: Mapping[str, Any], context: str) -> str:
    if not _vector(step.get("vpp_action")):
        raise VerificationError("{} lacks a finite normalized 3-D VPP action".format(context))
    if not _vector(step.get("own_pos_m")) or not _vector(step.get("target_pos_m")):
        raise VerificationError("{} lacks finite raw JSBSim positions".format(context))
    for key in ("range_m", "aa_deg", "ata_deg", "aoa_deg", "nz_cmd", "nz_actual"):
        if not _finite(step.get(key)):
            raise VerificationError("{} has non-finite {}".format(context, key))
    for key in ("tactical_basis_world_offset_x", "tactical_basis_world_offset_y", "tactical_basis_world_offset_z"):
        if not _finite(step.get(key)):
            raise VerificationError("{} lacks tactical-basis world-offset telemetry".format(context))
    guidance = step.get("guidance_telemetry")
    if not isinstance(guidance, Mapping) or not isinstance(guidance.get("effective_guidance_mode"), str):
        raise VerificationError("{} has unreadable guidance telemetry".format(context))
    clipping = step.get("clipping")
    if not isinstance(clipping, Mapping) or any(not isinstance(clipping.get(key), bool) for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated", "any_saturated")):
        raise VerificationError("{} has unreadable actuator-clipping telemetry".format(context))
    taxonomy = classify_relative_geometry(step["aa_deg"], step["ata_deg"])
    if taxonomy == "unknown":
        raise VerificationError("{} cannot be assigned the frozen (AA, ATA) taxonomy".format(context))
    return taxonomy


def _verify_record(record: Mapping[str, Any], expected_task: str, opponent: str) -> tuple[tuple[str, str, str], Counter[str]]:
    context = "{}/{}/{}".format(opponent, record.get("scenario"), record.get("controller"))
    method = str(record.get("controller") or "")
    if method not in METHODS or str(record.get("opponent_stage")) != opponent or str(record.get("task")) != expected_task:
        raise VerificationError("{} has an invalid method, opponent, or task identity".format(context))
    if record.get("backend") != "jsbsim" or record.get("strict_backend") is not True or record.get("backend_fallback_occurred") is not False:
        raise VerificationError("{} violates the strict-JSBSim backend contract".format(context))
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory or not all(isinstance(step, Mapping) for step in trajectory):
        raise VerificationError("{} has no usable raw trajectory".format(context))
    taxonomy = Counter(_verify_step(step, "{} step {}".format(context, index + 1)) for index, step in enumerate(trajectory))
    _verify_prediction(trajectory, context)
    return (str(record["scenario"]), opponent, method), taxonomy


def verify(root: Path, *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    root = Path(root)
    expected = _expected_universe()
    config_hash = _hash(config_path)
    decision_hash = _hash(DECISION)
    execution_path = root / "formal_execution_manifest.json"
    contract_path = root / "formal_artifact_contract.json"
    if not execution_path.is_file() or not contract_path.is_file():
        raise VerificationError("formal root lacks execution manifest or root contract")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("source_id") != SOURCE_ID or execution.get("execution_status") != "completed":
        raise VerificationError("execution source or completion receipt mismatch")
    if execution.get("config_sha256") != config_hash or execution.get("decision_contract_sha256") != decision_hash:
        raise VerificationError("execution config or decision-contract hash mismatch")
    if int(execution.get("expected_records", 0)) != 240 or any(item.get("returncode") != 0 for item in execution.get("executions", [])):
        raise VerificationError("execution contains a failed or incomplete child run")

    actual: list[tuple[str, str, str]] = []
    taxonomy_counts: dict[str, Counter[str]] = {opponent: Counter() for opponent in OPPONENTS}
    terminal_counts: dict[str, Counter[str]] = {opponent: Counter() for opponent in OPPONENTS}
    for opponent in OPPONENTS:
        child = root / "{}-{}".format(SOURCE_ID, opponent)
        for name in ("run_manifest.json", "artifact_contract.json", "failures.json", "aggregate/episode_records.json", "observation_audit.json"):
            if not (child / name).is_file():
                raise VerificationError("{} is missing {}".format(opponent, name))
        run_manifest = json.loads((child / "run_manifest.json").read_text(encoding="utf-8"))
        failures = json.loads((child / "failures.json").read_text(encoding="utf-8"))
        episodes = json.loads((child / "aggregate" / "episode_records.json").read_text(encoding="utf-8"))
        if run_manifest.get("status") != "completed" or run_manifest.get("paper_safe") is not True:
            raise VerificationError("{} child run is not a completed paper-safe artifact".format(opponent))
        if failures.get("failures") or int(run_manifest.get("extra", {}).get("total_episodes", 0)) != 80:
            raise VerificationError("{} child reports failures or the wrong record count".format(opponent))
        records = episodes.get("episodes")
        if not isinstance(records, list) or len(records) != 80:
            raise VerificationError("{} aggregate does not contain 80 records".format(opponent))
        for record in records:
            scenario = str(record.get("scenario") or "")
            method = str(record.get("controller") or "")
            key = (scenario, opponent, method)
            task = expected.get(key)
            if task is None:
                raise VerificationError("{} record is outside the frozen universe".format(opponent))
            record_key, counter = _verify_record(record, task, opponent)
            actual.append(record_key)
            taxonomy_counts[opponent].update(counter)
            terminal_counts[opponent][str(record.get("terminal_reason") or "unknown")] += 1
    duplicates = [key for key, count in Counter(actual).items() if count != 1]
    if duplicates or set(actual) != set(expected):
        raise VerificationError("actual records are not exactly the frozen 240-record matrix")
    return {
        "source_id": SOURCE_ID,
        "verdict": "pass",
        "records": len(actual),
        "taxonomy_angle_order": ["AA", "ATA"],
        "dynamic_taxonomy_step_counts": {key: dict(value) for key, value in taxonomy_counts.items()},
        "terminal_reason_counts": {key: dict(value) for key, value in terminal_counts.items()},
        "boundary": "The verifier establishes matrix and telemetry validity. It does not turn the static gate into a dynamic oracle or prove universal generalization.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        result = verify(args.output_root.resolve(), config_path=args.config.resolve())
    except (OSError, ValueError, TypeError, VerificationError) as error:
        result = {"source_id": SOURCE_ID, "verdict": "fail", "records": 0, "error": str(error)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "formal_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
