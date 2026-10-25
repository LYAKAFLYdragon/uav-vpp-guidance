#!/usr/bin/env python3
"""Independently verify the paired two-node runtime probe and its preregistered gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-RUNTIME-PROBE-V1-20260904"
OPPONENTS = ("expert", "end_to_end")
CONDITIONS = ("baseline", "candidate")
OFFSETS = (4, 12, 48)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "run_manifest.json"
    results_path = root / "condition_results.csv"
    pairs_path = root / "paired_response_records.csv"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != SOURCE_ID:
        errors.append("wrong source_id")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if manifest.get("node0_delta") != [0.0, 0.08, 0.0] or manifest.get("node1_delta") != [-0.08, 0.0, 0.0]:
        errors.append("fixed sequence mismatch")
    if manifest.get("response_offsets") != list(OFFSETS):
        errors.append("response horizon mismatch")
    if not results_path.is_file() or not pairs_path.is_file():
        errors.append("missing result file")
        return {"ok": False, "errors": errors}
    rows = read_csv(results_path)
    pairs = read_csv(pairs_path)
    if len(rows) != 24:
        errors.append(f"condition rows {len(rows)} != 24")
    if len(pairs) != 36:
        errors.append(f"paired rows {len(pairs)} != 36")
    keys = {(row.get("source_trace_id", ""), row.get("condition", "")) for row in rows}
    if len(keys) != 24:
        errors.append("condition/source trace keys are not unique")
    for row in rows:
        label = f"{row.get('source_trace_id')}/{row.get('condition')}"
        if row.get("backend_effective") != "jsbsim":
            errors.append(f"{label}: backend is not jsbsim")
        if truth(row.get("backend_fallback_occurred")):
            errors.append(f"{label}: backend fallback")
        if row.get("reset_calls") != "1":
            errors.append(f"{label}: reset count is not 1")
        if not truth(row.get("finite_telemetry")):
            errors.append(f"{label}: telemetry not finite")
        if not truth(row.get("handoff_hash_match")):
            errors.append(f"{label}: handoff hash mismatch")
        if not truth(row.get("response_window_complete")):
            errors.append(f"{label}: incomplete response window")
        if truth(row.get("response_window_backend_fallback")):
            errors.append(f"{label}: response-window backend fallback")
        if truth(row.get("response_window_prediction_fallback")):
            errors.append(f"{label}: response-window predictor fallback")
        if truth(row.get("response_window_clipping")):
            errors.append(f"{label}: response-window clipping")
        if truth(row.get("action_out_of_bounds")):
            errors.append(f"{label}: action out of bounds")
        if row.get("condition") == "candidate" and (not truth(row.get("node0_observed")) or not truth(row.get("node1_observed"))):
            errors.append(f"{label}: candidate nodes were not observed")
    pair_valid = [row for row in pairs if truth(row.get("pair_valid"))]
    if len(pair_valid) != 36:
        errors.append(f"valid paired horizons {len(pair_valid)} != 36")
    gate = manifest.get("preregistered_gate", {})
    primary = int(gate.get("primary_horizon", 12))
    by_opponent: dict[str, dict[str, Any]] = {}
    for opponent in OPPONENTS:
        scoped = [row for row in pair_valid if row.get("opponent") == opponent and int(number(row.get("response_offset"))) == primary]
        range_deltas = [number(row.get("delta_range_m")) for row in scoped]
        rate_deltas = [number(row.get("delta_range_rate_mps")) for row in scoped]
        mean_range = sum(range_deltas) / len(range_deltas) if range_deltas else float("nan")
        mean_rate = sum(rate_deltas) / len(rate_deltas) if rate_deltas else float("nan")
        candidate_rows = [row for row in rows if row.get("opponent") == opponent and row.get("condition") == "candidate"]
        baseline_rows = [row for row in rows if row.get("opponent") == opponent and row.get("condition") == "baseline"]
        new_crash_oob = any((truth(c.get("is_crash")) or truth(c.get("is_out_of_bounds"))) and not (truth(b.get("is_crash")) or truth(b.get("is_out_of_bounds"))) for c, b in zip(sorted(candidate_rows, key=lambda x: x.get("source_trace_id")), sorted(baseline_rows, key=lambda x: x.get("source_trace_id"))))
        safety = not new_crash_oob and all(not truth(row.get("response_window_clipping")) and not truth(row.get("response_window_backend_fallback")) and not truth(row.get("response_window_prediction_fallback")) and not truth(row.get("action_out_of_bounds")) for row in candidate_rows)
        passed = bool(scoped and safety and math.isfinite(mean_range) and math.isfinite(mean_rate) and mean_range >= float(gate.get("minimum_mean_range_delta_m", 0.001)) and mean_rate >= float(gate.get("minimum_mean_range_rate_delta_mps", 0.002)))
        by_opponent[opponent] = {"n_pairs": len(scoped), "mean_h12_range_delta_m": mean_range, "mean_h12_range_rate_delta_mps": mean_rate, "new_crash_or_oob": new_crash_oob, "safety_gate": safety, "practical_gate": passed}
    runtime_gate = not errors
    practical_gate = runtime_gate and all(item["practical_gate"] for item in by_opponent.values())
    result = {"ok": runtime_gate, "source_id": manifest.get("source_id"), "errors": errors, "runtime_integrity_gate": "pass" if runtime_gate else "fail", "practical_improvement_gate": "pass" if practical_gate else "fail", "by_opponent": by_opponent, "decision": "candidate_has_preregistered_runtime_improvement" if practical_gate else "sequence_not_confirmed_runtime_improvement"}
    return result


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
