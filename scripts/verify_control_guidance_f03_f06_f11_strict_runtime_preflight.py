#!/usr/bin/env python3
"""Verify the F03/F06/F11 strict-JSBSim preflight without selecting a fix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

SOURCE_ID = "THESIS-CONTROL-GUIDANCE-F03-F06-F11-STRICT-RUNTIME-PREFLIGHT-V1"
WINDOW_TICKS = 200
EXPECTED_PAIRS = 12
REQUIRED = ("pre_action_state_hash", "target_state_hash", "raw_command", "fcs_command", "actuator_actual", "own_state", "target_state", "geometry", "p_rps", "nz_g", "controller_saturated", "rate_limited")


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
    for label, episode in (("baseline", baseline), ("candidate", candidate)):
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
        if len(rows) != WINDOW_TICKS:
            errors.append(f"{label}:trajectory_length")
        for index, row in enumerate(rows):
            errors.extend(f"{label}:tick_{index}:missing_{key}" for key in REQUIRED if key not in row)
            if not finite_tree(row):
                errors.append(f"{label}:tick_{index}:nonfinite")
            geometry = row.get("geometry") or {}
            if any(not finite(geometry.get(key)) for key in ("range_m", "range_rate_mps", "ata_deg", "aa_deg")):
                errors.append(f"{label}:tick_{index}:geometry")
            actuator = row.get("actuator_actual") or {}
            if any(not finite(actuator.get(key)) for key in ("elevator_pos_rad", "aileron_pos_rad", "left_aileron_pos_rad", "right_aileron_pos_rad", "throttle_pos_norm")):
                errors.append(f"{label}:tick_{index}:actuator")
            if not finite(row.get("p_rps")) or not finite(row.get("nz_g")):
                errors.append(f"{label}:tick_{index}:physical_response")
    left, right = baseline.get("trajectory") or [], candidate.get("trajectory") or []
    if baseline.get("initial_state_hash") != candidate.get("initial_state_hash"):
        errors.append("initial_state_hash_mismatch")
    if len(left) == len(right) == WINDOW_TICKS and any(a.get("target_state_hash") != b.get("target_state_hash") for a, b in zip(left, right)):
        errors.append("target_trajectory_hash_mismatch")
    return {
        "pair_id": pair.get("pair_id"), "finding_id": pair.get("finding_id"), "comparison_id": pair.get("comparison_id"), "scenario_id": pair.get("scenario_id"),
        "errors": errors, "initial_state_hash_match": baseline.get("initial_state_hash") == candidate.get("initial_state_hash"),
        "target_trajectory_hash_match": len(left) == len(right) == WINDOW_TICKS and not any(a.get("target_state_hash") != b.get("target_state_hash") for a, b in zip(left, right)),
        "raw_nz_max_delta": max_delta(left, right, "raw_command", "nz_cmd"),
        "raw_roll_rate_max_delta": max_delta(left, right, "raw_command", "roll_rate_cmd"),
        "fcs_elevator_max_delta": max_delta(left, right, "fcs_command", "fcs/elevator-cmd-norm"),
        "fcs_aileron_max_delta": max_delta(left, right, "fcs_command", "fcs/aileron-cmd-norm"),
        "actuator_elevator_max_delta": max_delta(left, right, "actuator_actual", "elevator_pos_rad"),
        "actuator_aileron_max_delta": max_delta(left, right, "actuator_actual", "aileron_pos_rad"),
        "nz_response_max_delta": max_delta(left, right, "", "nz_g"),
        "roll_response_max_delta": max_delta(left, right, "", "p_rps"),
        "candidate_rate_limited_ticks": sum(bool(row.get("rate_limited")) for row in (candidate.get("trajectory") or [])),
        "candidate_controller_saturated_ticks": sum(bool(row.get("controller_saturated")) for row in (candidate.get("trajectory") or [])),
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
    if not isinstance(records, list) or len(records) != EXPECTED_PAIRS:
        errors.append(f"pair_count_not_{EXPECTED_PAIRS}")
        records = records if isinstance(records, list) else []
    pair_results = [verify_pair(pair) for pair in records]
    errors.extend(error for result in pair_results for error in result["errors"])
    return {
        "status": "pass" if not errors else "fail", "source_id": SOURCE_ID,
        "reviewer_approved": False, "runtime_modification_authorized": False,
        "errors": errors, "pair_count": len(pair_results), "pair_results": pair_results,
        "interpretation": "strict-JSBSim response/telemetry preflight only; no candidate selection or protected-runtime authorization",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.output_root.resolve())
    output = (args.out or args.output_root / "strict_runtime_preflight_verification.json").resolve()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_modification_authorized": False, "output": str(output)}, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
