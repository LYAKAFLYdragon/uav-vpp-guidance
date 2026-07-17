#!/usr/bin/env python3
"""Screen frozen temporal references on V3 post-merge baseline windows only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.tactical_basis_continuity_contract import record_passes
from uav_vpp_guidance.evaluation.tactical_basis_postmerge_p2 import (
    candidate_variants,
    compile_candidate_row,
    summarize_rows,
)
from uav_vpp_guidance.evaluation.temporal_reference_compiler import TemporalReferenceConfig


SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1"
CONTINUITY_SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V3"
CONTINUITY_GATES = {
    "required_window_steps": 40,
    "max_basis_condition_number": 100.0,
    "min_qualifying_handoffs_per_opponent": 4,
    "min_distinct_signatures_per_opponent": 4,
    "opponents_required": 2,
}


class TacticalBasisPostmergeP2RunnerError(RuntimeError):
    """Raised when the read-only P2 source contract is violated."""


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
        raise TacticalBasisPostmergeP2RunnerError(f"YAML mapping required: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TacticalBasisPostmergeP2RunnerError(f"JSON mapping required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _require_input(config: Mapping[str, Any], key: str) -> Path:
    inputs = config["inputs"]
    path = _resolve(str(inputs[key]))
    expected = str(inputs[key + "_sha256"])
    if not path.is_file() or _hash(path) != expected:
        raise TacticalBasisPostmergeP2RunnerError(f"frozen input SHA mismatch: {key}")
    return path


def _compiler_config(reference: Mapping[str, Any]) -> TemporalReferenceConfig:
    return TemporalReferenceConfig(
        horizon_steps=int(reference["horizon_steps"]),
        high_level_dt_s=float(reference["high_level_dt_s"]),
        max_action_delta=float(reference["max_action_delta"]),
        max_action_slew=float(reference["max_action_slew"]),
        max_basis_condition_number=float(reference["max_basis_condition_number"]),
    )


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    config = _load_yaml(path)
    if config.get("source_id") != SOURCE_ID or config.get("stage") != "postmerge_offline_envelope_screen":
        raise TacticalBasisPostmergeP2RunnerError("P2 source identity drifted")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise TacticalBasisPostmergeP2RunnerError("authorization mapping is required")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "candidate_action_execution_permitted",
        "high_level_ppo_training_permitted",
        "shared_skill_training_permitted",
        "guidance_pid_change_permitted",
        "action_envelope_expansion_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
        "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise TacticalBasisPostmergeP2RunnerError(f"P2 must prohibit {key}")
    if config.get("contract", {}).get("candidate_actions_enter_env") is not False:
        raise TacticalBasisPostmergeP2RunnerError("offline P2 must not execute candidate actions")
    if config["contract"].get("phase_scope") != "post_merge_only":
        raise TacticalBasisPostmergeP2RunnerError("P2 may only claim post-merge coverage")

    gate_path = _require_input(config, "continuity_gate")
    verification_path = _require_input(config, "continuity_verification")
    index_path = _require_input(config, "continuity_records_index")
    continuity_config_path = _require_input(config, "continuity_config")
    design_path = _require_input(config, "temporal_reference_design")
    _require_input(config, "temporal_reference_compiler")
    gate = _load_json(gate_path)
    verification = _load_json(verification_path)
    index = _load_json(index_path)
    continuity_config = _load_yaml(continuity_config_path)
    design = _load_yaml(design_path)
    if (
        gate.get("source_id") != CONTINUITY_SOURCE_ID
        or gate.get("passes") is not True
        or verification.get("source_id") != CONTINUITY_SOURCE_ID
        or verification.get("passes") is not True
        or continuity_config.get("source_id") != CONTINUITY_SOURCE_ID
        or design.get("source_id") != "THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1"
    ):
        raise TacticalBasisPostmergeP2RunnerError("V3/P1 provenance contract drifted")
    reference = design.get("temporal_reference")
    if not isinstance(reference, Mapping):
        raise TacticalBasisPostmergeP2RunnerError("P1 temporal-reference design is missing")
    compiler = _compiler_config(reference)
    if (
        compiler.horizon_steps != int(config["contract"]["horizon_steps"])
        or compiler.high_level_dt_s != float(config["contract"]["high_level_dt_s"])
        or compiler.max_action_delta != float(config["contract"]["max_action_delta"])
        or compiler.max_action_slew != float(config["contract"]["max_action_slew"])
        or compiler.max_basis_condition_number != float(config["contract"]["max_basis_condition_number"])
    ):
        raise TacticalBasisPostmergeP2RunnerError("P2 compiler envelope differs from P1")
    if not isinstance(index.get("records"), list) or len(index["records"]) != 36:
        raise TacticalBasisPostmergeP2RunnerError("V3 records index is incomplete")
    root = _resolve(str(config["inputs"]["continuity_root"]))
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for item in index["records"]:
        record_path = root / str(item.get("path", ""))
        record_hash = str(item.get("sha256", ""))
        if not record_path.is_file() or _hash(record_path) != record_hash:
            raise TacticalBasisPostmergeP2RunnerError("V3 record hash mismatch")
        payload = _load_json(record_path)
        record = payload.get("record")
        if not isinstance(record, dict) or record.get("source_id") != CONTINUITY_SOURCE_ID:
            raise TacticalBasisPostmergeP2RunnerError("V3 record identity drifted")
        if record_passes(record, CONTINUITY_GATES):
            records.append(record)
            hashes[str(record_path)] = record_hash
    expected = config.get("input_cohort", {}).get("qualifying_records_per_opponent")
    actual = {opponent: sum(record["opponent"] == opponent for record in records) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    if actual != expected or len(records) != int(config["input_cohort"]["qualifying_records_total"]):
        raise TacticalBasisPostmergeP2RunnerError("V3 qualifying cohort differs from the frozen P2 input")
    output = _resolve(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise TacticalBasisPostmergeP2RunnerError("P2 output root must be absolute and absent")
    return config, dict(reference), records, hashes


def validate_design(path: Path) -> dict[str, Any]:
    config, reference, records, _hashes = _validate_config(path)
    return {
        "source_id": SOURCE_ID,
        "stage": config["stage"],
        "qualifying_v3_records": len(records),
        "candidate_variants": len(candidate_variants(reference)),
        "planned_rows": len(records) * len(candidate_variants(reference)),
        "candidate_actions_enter_env": False,
        "training_permitted": False,
        "execution_permitted": bool(config["authorization"]["execution_permitted"]),
        "output_root_absent": True,
    }


def _authorize_execution(config: Mapping[str, Any], expected_rows: int) -> None:
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or int(execution.get("expected_rows", -1)) != expected_rows:
        raise TacticalBasisPostmergeP2RunnerError("P2 row universe is not authorized")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if len(required_sha) != 40 or subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise TacticalBasisPostmergeP2RunnerError("P2 implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise TacticalBasisPostmergeP2RunnerError("P2 authorized source file list is missing")
    for item in files:
        source = _resolve(str(item.get("path") or ""))
        if not source.is_file() or _hash(source) != str(item.get("sha256") or ""):
            raise TacticalBasisPostmergeP2RunnerError(f"P2 source hash mismatch: {source}")


def run(path: Path) -> dict[str, Any]:
    config, reference, records, source_hashes = _validate_config(path)
    variants = candidate_variants(reference)
    expected_rows = len(records) * len(variants)
    if config["authorization"].get("execution_permitted") is not True:
        raise TacticalBasisPostmergeP2RunnerError("P2 execution is not authorized")
    if config["authorization"].get("require_clean_worktree") is True and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise TacticalBasisPostmergeP2RunnerError("P2 requires a clean worktree")
    _authorize_execution(config, expected_rows)
    output = _resolve(str(config["outputs"]["root"]))
    output.mkdir(parents=True)
    try:
        compiler = _compiler_config(reference)
        rows: list[dict[str, Any]] = []
        for candidate in variants:
            for record in sorted(records, key=lambda item: (str(item["opponent"]), str(item["scenario_name"]))):
                record_path = _resolve(str(config["inputs"]["continuity_root"])) / "episodes" / str(record["opponent"]) / (str(record["scenario_name"]) + ".json")
                row = compile_candidate_row(
                    record,
                    candidate,
                    compiler,
                    source_record_sha256=source_hashes[str(record_path)],
                )
                rows.append(row)
        summary = summarize_rows(rows, config["gates"])
        _write_json(output / "offline_envelope_matrix.json", {"source_id": SOURCE_ID, "rows": rows})
        _write_json(output / "offline_envelope_summary.json", {"source_id": SOURCE_ID, **summary})
        fields = [key for key in rows[0] if key != "receipts"]
        with (output / "offline_envelope_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in fields} for row in rows])
        _write_json(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "stage": config["stage"],
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config_sha256": _hash(path),
                "v3_continuity_gate_sha256": _hash(_resolve(str(config["inputs"]["continuity_gate"]))),
                "v3_records_index_sha256": _hash(_resolve(str(config["inputs"]["continuity_records_index"]))),
                "input_qualified_records": len(records),
                "rows": len(rows),
                "training": "prohibited",
                "candidate_actions_enter_env": False,
            },
        )
        return {"output": str(output), **summary}
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(validate_design(args.config), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
