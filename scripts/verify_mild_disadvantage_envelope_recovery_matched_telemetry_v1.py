#!/usr/bin/env python
"""Verify the telemetry-complete matched-contract probe without rerunning JSBSim."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-TELEMETRY-COLLECTOR-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_CANDIDATES = ("x625_y200", "x750_y250")
REPETITIONS = 3
REQUIRED_COLUMNS = (
    "state_hash_before_action", "action_0", "action_1", "action_2",
    "vp_position_neu", "anchor_pos", "vp_offset", "vp_world_offset",
    "raw_nz_cmd", "raw_roll_rate_cmd", "raw_throttle_cmd",
    "guidance_nz_cmd", "guidance_roll_rate_cmd", "guidance_throttle_cmd",
    "elevator_cmd", "aileron_cmd", "rudder_cmd", "roll_rad", "p_rps",
    "nz_g", "range_m", "range_rate_mps", "ata_deg", "aa_deg",
    "backend", "backend_fallback_occurred", "prediction_fallback",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_json_vector(value: Any) -> bool:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(parsed, list)
        and len(parsed) == 3
        and all(_finite(item) for item in parsed)
    )


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "pilot_manifest.json"
    result_path = root / "episode_results.csv"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != SOURCE_ID:
        errors.append("wrong source_id")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if manifest.get("task_name") != "crossing_feasible":
        errors.append("task context is not crossing_feasible")
    contract = manifest.get("matched_contract", {})
    if contract.get("action_semantics") != "tactical_basis_v1":
        errors.append("action semantics mismatch")
    if contract.get("anchor_mode") != "predicted_target":
        errors.append("anchor mode mismatch")
    telemetry = manifest.get("telemetry_contract", {})
    if telemetry.get("hash_before_action") is not True:
        errors.append("state hash before action is not enabled")
    if tuple(manifest.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("opponent set mismatch")
    if tuple(manifest.get("candidates", {}).keys()) != EXPECTED_CANDIDATES:
        errors.append("candidate set mismatch")
    if manifest.get("repetitions_per_candidate_and_opponent") != REPETITIONS:
        errors.append("repetition count mismatch")
    rows = _read_csv(result_path) if result_path.is_file() else []
    if len(rows) != 12:
        errors.append(f"episode result row count is {len(rows)}, expected 12")
    counts = {(opponent, candidate): 0 for opponent in EXPECTED_OPPONENTS for candidate in EXPECTED_CANDIDATES}
    trace_rows = 0
    for row in rows:
        key = (row.get("opponent", ""), row.get("candidate", ""))
        if key not in counts:
            errors.append(f"unexpected episode key {key}")
            continue
        counts[key] += 1
        if row.get("source_id") != SOURCE_ID:
            errors.append(f"{key}: episode source id mismatch")
        if row.get("reset_calls") != "1":
            errors.append(f"{key}: reset_calls is not 1")
        if row.get("backend_effective") != "jsbsim":
            errors.append(f"{key}: effective backend is not jsbsim")
        if row.get("backend_fallback_occurred", "").lower() != "false":
            errors.append(f"{key}: backend fallback occurred")
        if row.get("prediction_fallback_occurred", "").lower() != "false":
            errors.append(f"{key}: predictor fallback occurred")
        if row.get("finite_telemetry", "").lower() != "true":
            errors.append(f"{key}: finite telemetry gate failed")
        if row.get("reason", "") == "":
            errors.append(f"{key}: terminal reason missing")
        trace_path = root / row["opponent"] / "trajectories" / f"{row['candidate']}_{counts[key]:02d}.csv"
        if not trace_path.is_file() or trace_path.stat().st_size <= 0:
            errors.append(f"{key}: missing trace {trace_path}")
            continue
        trace = _read_csv(trace_path)
        trace_rows += len(trace)
        if not trace:
            errors.append(f"{key}: empty trace")
            continue
        missing = [column for column in REQUIRED_COLUMNS if column not in trace[0]]
        if missing:
            errors.append(f"{key}: missing telemetry columns {missing}")
        for index, sample in enumerate(trace):
            if not sample.get("state_hash_before_action") or len(sample["state_hash_before_action"]) != 64:
                errors.append(f"{key}: invalid state hash at trace row {index}")
                break
            if sample.get("backend") != "jsbsim":
                errors.append(f"{key}: trace backend is not jsbsim")
                break
            if sample.get("backend_fallback_occurred", "").lower() != "false":
                errors.append(f"{key}: trace backend fallback occurred")
                break
            if sample.get("prediction_fallback", "").lower() != "false":
                errors.append(f"{key}: trace predictor fallback occurred")
                break
            for column in REQUIRED_COLUMNS:
                if column in {"state_hash_before_action", "vp_position_neu", "anchor_pos", "vp_offset", "vp_world_offset"}:
                    continue
                if not _finite(sample.get(column)):
                    errors.append(f"{key}: non-finite {column} at trace row {index}")
                    break
            for column in ("vp_position_neu", "anchor_pos", "vp_offset", "vp_world_offset"):
                if not _finite_json_vector(sample.get(column)):
                    errors.append(f"{key}: invalid finite vector {column} at trace row {index}")
                    break
    if any(value != REPETITIONS for value in counts.values()):
        errors.append(f"unbalanced cells: {counts}")
    strict_runtime_gate = not errors
    return {
        "ok": not errors,
        "source_id": manifest.get("source_id"),
        "errors": errors,
        "episode_count": len(rows),
        "trace_row_count": trace_rows,
        "strict_runtime_gate": "pass" if strict_runtime_gate else "fail",
        "telemetry_gate": "pass" if strict_runtime_gate else "fail",
        "interpretation": "telemetry_complete_strict_matched_probe" if strict_runtime_gate else "telemetry_contract_or_runtime_integrity_failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.output:
        args.output.resolve().write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
