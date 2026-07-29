#!/usr/bin/env python3
"""Verify the opt-in F01/F02 strict-JSBSim paired regression output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

SOURCE_ID = "THESIS-CONTROL-GUIDANCE-F01-F02-OPTIN-PAIRED-REGRESSION-V1"
WINDOW_STEPS = 48
REQUIRED = ("pre_action_state_hash", "target_state_hash", "raw_command", "filtered_command", "fcs_command", "actuator_actual", "own_state", "target_state", "geometry", "p_rps", "nz_g", "clipping")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def max_delta(left: List[Mapping[str, Any]], right: List[Mapping[str, Any]], container: str, field: str) -> float:
    values = []
    for a, b in zip(left, right):
        av = (a.get(container) or {}).get(field) if container else a.get(field)
        bv = (b.get(container) or {}).get(field) if container else b.get(field)
        if finite(av) and finite(bv):
            values.append(abs(float(bv) - float(av)))
    return max(values, default=0.0)


def verify_pair(pair: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    baseline, candidate = pair.get("baseline") or {}, pair.get("candidate") or {}
    for label, episode, expected_mode in (("baseline", baseline, "legacy"), ("candidate", candidate, "f01_f02_remediated")):
        if episode.get("mode") != expected_mode:
            errors.append(f"{label}:wrong_mode")
        if episode.get("backend") != "jsbsim":
            errors.append(f"{label}:backend_not_jsbsim")
        if episode.get("strict_backend") is not True:
            errors.append(f"{label}:strict_backend_false")
        if episode.get("backend_fallback_occurred") is not False:
            errors.append(f"{label}:backend_fallback")
        if episode.get("reset_count") != 1:
            errors.append(f"{label}:reset_count")
        if episode.get("mid_episode_reset_count") != 0:
            errors.append(f"{label}:mid_episode_reset")
        rows = episode.get("trajectory") or []
        if len(rows) != WINDOW_STEPS:
            errors.append(f"{label}:trajectory_length")
        for index, row in enumerate(rows):
            errors.extend(f"{label}:step_{index}:missing_{key}" for key in REQUIRED if key not in row)
            if not finite_tree(row):
                errors.append(f"{label}:step_{index}:nonfinite")
            geometry = row.get("geometry") or {}
            if any(not finite(geometry.get(key)) for key in ("range_m", "range_rate_mps", "closing_speed_mps", "ata_deg", "aa_deg")):
                errors.append(f"{label}:step_{index}:geometry")
            actuator = row.get("actuator_actual") or {}
            if any(not finite(actuator.get(key)) for key in ("elevator_pos_rad", "aileron_pos_rad", "left_aileron_pos_rad", "right_aileron_pos_rad", "rudder_pos_rad", "throttle_pos_norm")):
                errors.append(f"{label}:step_{index}:actuator")
            if not finite(row.get("p_rps")) or not finite(row.get("nz_g")):
                errors.append(f"{label}:step_{index}:physical_response")
    baseline_rows, candidate_rows = baseline.get("trajectory") or [], candidate.get("trajectory") or []
    if baseline.get("initial_state_hash") != candidate.get("initial_state_hash"):
        errors.append("initial_state_hash_mismatch")
    if len(baseline_rows) == len(candidate_rows) == WINDOW_STEPS:
        if any(a.get("target_state_hash") != b.get("target_state_hash") for a, b in zip(baseline_rows, candidate_rows)):
            errors.append("target_trajectory_hash_mismatch")
    return {
        "pair_id": pair.get("pair_id"), "scenario_id": pair.get("scenario_id"), "variant": pair.get("variant"),
        "errors": errors,
        "initial_state_hash_match": baseline.get("initial_state_hash") == candidate.get("initial_state_hash"),
        "target_trajectory_hash_match": not any(a.get("target_state_hash") != b.get("target_state_hash") for a, b in zip(baseline_rows, candidate_rows)),
        "raw_nz_max_delta": max_delta(baseline_rows, candidate_rows, "raw_command", "nz_cmd"),
        "raw_roll_rate_max_delta": max_delta(baseline_rows, candidate_rows, "raw_command", "roll_rate_cmd"),
        "actuator_elevator_max_delta": max_delta(baseline_rows, candidate_rows, "actuator_actual", "elevator_pos_rad"),
        "actuator_aileron_max_delta": max_delta(baseline_rows, candidate_rows, "actuator_actual", "aileron_pos_rad"),
        "nz_response_max_delta": max_delta(baseline_rows, candidate_rows, "", "nz_g"),
        "roll_rate_response_max_delta": max_delta(baseline_rows, candidate_rows, "", "p_rps"),
    }


def verify(output_root: Path) -> Dict[str, Any]:
    errors: List[str] = []
    manifest_path, records_path = output_root / "run_manifest.json", output_root / "paired_records.json"
    if not manifest_path.is_file() or not records_path.is_file():
        return {"status": "fail", "source_id": SOURCE_ID, "errors": ["required_artifact_missing"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "fail", "source_id": SOURCE_ID, "errors": [f"parse_error:{exc}"]}
    if manifest.get("source_id") != SOURCE_ID:
        errors.append("manifest_source_id")
    if manifest.get("backend") != "jsbsim" or manifest.get("strict_backend") is not True:
        errors.append("manifest_not_strict_jsbsim")
    if manifest.get("backend_fallback_occurred") is not False:
        errors.append("manifest_backend_fallback")
    if manifest.get("baseline_mode") != "legacy" or manifest.get("candidate_mode") != "f01_f02_remediated":
        errors.append("manifest_mode_contract")
    if not isinstance(records, list) or len(records) != 12:
        errors.append("pair_count_not_12")
        records = records if isinstance(records, list) else []
    pair_results = [verify_pair(pair) for pair in records]
    errors.extend(error for result in pair_results for error in result["errors"])
    f01_activity = any(result["raw_nz_max_delta"] > 1.0e-9 for result in pair_results)
    f02_activity = any(result["raw_roll_rate_max_delta"] > 1.0e-9 for result in pair_results)
    if not f01_activity:
        errors.append("candidate_f01_axis_never_distinct")
    if not f02_activity:
        errors.append("candidate_f02_axis_never_distinct")
    return {
        "status": "pass" if not errors else "fail", "source_id": SOURCE_ID,
        "reviewer_approved": False, "runtime_modification_authorized": False,
        "errors": errors, "pair_count": len(pair_results),
        "f01_axis_activity": f01_activity, "f02_axis_activity": f02_activity,
        "pair_results": pair_results,
        "interpretation": "opt-in physical response witness only; no superiority or production authorization claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    result = verify(root)
    out = (args.out or root / "optin_paired_verification.json").resolve()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_modification_authorized": False, "output": str(out)}, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
