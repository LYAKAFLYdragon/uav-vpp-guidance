#!/usr/bin/env python3
"""Verify paired handoff local-response collection without rerunning JSBSim."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V1-20260903"
CONDITIONS = {
    "baseline": (None, 0),
    "axis0_plus": (0, 1), "axis0_minus": (0, -1),
    "axis1_plus": (1, 1), "axis1_minus": (1, -1),
    "axis2_plus": (2, 1), "axis2_minus": (2, -1),
}
OFFSETS = {4, 12, 48}
METRICS = ("range_m", "range_rate_mps", "ata_deg", "aa_deg", "specific_energy_m2ps2", "altitude_m")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(root: Path, expected_source_id: str = SOURCE_ID) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "run_manifest.json"
    condition_path = root / "condition_results.csv"
    response_path = root / "local_response_records.csv"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != expected_source_id:
        errors.append("wrong source_id")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if manifest.get("compiler_runtime_probe_started") is not False:
        errors.append("compiler runtime probe was started")
    if float(manifest.get("pulse_amplitude", 0.0)) != 0.08:
        errors.append("pulse amplitude is not 0.08")
    if set(manifest.get("response_offsets", [])) != OFFSETS:
        errors.append("response offsets mismatch")
    conditions = read_csv(condition_path) if condition_path.is_file() else []
    if not conditions:
        errors.append("condition_results.csv is missing or empty")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in conditions:
        key = row.get("source_trace_id", "")
        groups.setdefault(key, []).append(row)
        label = row.get("condition", "")
        if row.get("source_id") != expected_source_id:
            errors.append(f"{key}: source id mismatch")
        if label not in CONDITIONS:
            errors.append(f"{key}: unexpected condition {label}")
            continue
        axis, sign = CONDITIONS[label]
        expected_axis = "" if axis is None else str(axis)
        if (row.get("pulse_axis") or "") != expected_axis:
            errors.append(f"{key}/{label}: pulse axis mismatch")
        if int(float(row.get("pulse_sign", 0))) != sign:
            errors.append(f"{key}/{label}: pulse sign mismatch")
        if row.get("backend_effective") != "jsbsim":
            errors.append(f"{key}/{label}: backend is not jsbsim")
        if row.get("backend_fallback_occurred", "").lower() != "false":
            errors.append(f"{key}/{label}: backend fallback occurred")
        if row.get("reset_calls") != "1":
            errors.append(f"{key}/{label}: reset count is not one")
        if row.get("handoff_hash_match", "").lower() != "true":
            errors.append(f"{key}/{label}: handoff hash mismatch")
        if row.get("pulse_clipped", "").lower() == "true":
            errors.append(f"{key}/{label}: pulse clipped")
        if label != "baseline" and abs(float(row.get("pulse_amplitude", "nan")) - 0.08) > 1e-12:
            errors.append(f"{key}/{label}: pulse amplitude mismatch")
        if row.get("response_window_prediction_fallback", "").lower() == "true":
            errors.append(f"{key}/{label}: prediction fallback in response window")
    for key, rows in groups.items():
        if len(rows) != len(CONDITIONS) or {row.get("condition") for row in rows} != set(CONDITIONS):
            errors.append(f"{key}: expected baseline plus six symmetric pulses")
    response = read_csv(response_path) if response_path.is_file() else []
    if len(response) != len(conditions) * len(OFFSETS):
        errors.append(f"response row count {len(response)} is not {len(conditions) * len(OFFSETS)}")
    for row in response:
        if int(row.get("response_offset", -1)) not in OFFSETS:
            errors.append("unexpected response offset")
        if row.get("response_available", "").lower() != "true":
            errors.append(f"missing response at {row.get('source_trace_id')}/{row.get('condition')}")
        for metric in METRICS:
            if not finite(row.get(metric)):
                errors.append(f"non-finite {metric} at {row.get('source_trace_id')}/{row.get('condition')}/{row.get('response_offset')}")
                break
    return {"ok": not errors, "source_id": manifest.get("source_id"), "errors": errors, "source_trace_count": len(groups), "condition_count": len(conditions), "response_count": len(response), "paired_local_response_gate": "pass" if not errors else "fail", "runtime_probe_authorized": not errors, "runtime_probe_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-id", default=SOURCE_ID)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.source_id)
    if args.output:
        args.output.resolve().write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
