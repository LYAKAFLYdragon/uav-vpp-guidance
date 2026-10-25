#!/usr/bin/env python
"""Verify the bounded envelope-recovery physical probe without rerunning it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-PHYSICAL-PROBE-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_CANDIDATES = ("x625_y200", "x750_y250")
REPETITIONS = 3


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "pilot_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != SOURCE_ID:
        errors.append("wrong source_id")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if manifest.get("backend_requested") != "jsbsim":
        errors.append("backend was not requested as jsbsim")
    if tuple(manifest.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("opponent set mismatch")
    candidates = manifest.get("candidates", {})
    if tuple(candidates.keys()) != EXPECTED_CANDIDATES:
        errors.append("candidate set mismatch")
    if manifest.get("repetitions_per_candidate_and_opponent") != REPETITIONS:
        errors.append("repetition count mismatch")
    if manifest.get("raw_episode_count") != len(EXPECTED_OPPONENTS) * len(EXPECTED_CANDIDATES) * REPETITIONS:
        errors.append("expected exactly 12 episodes")
    results_path = root / "episode_results.csv"
    rows: list[dict[str, str]] = []
    if not results_path.is_file():
        errors.append("missing episode_results.csv")
    else:
        with results_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 12:
            errors.append(f"episode result row count is {len(rows)}, expected 12")
    counts = {(opponent, candidate): 0 for opponent in EXPECTED_OPPONENTS for candidate in EXPECTED_CANDIDATES}
    for row in rows:
        key = (row.get("opponent", ""), row.get("candidate", ""))
        if key not in counts:
            errors.append(f"unexpected episode key {key}")
            continue
        counts[key] += 1
        if row.get("reset_calls") != "1":
            errors.append(f"{key}: reset_calls is not 1")
        if row.get("backend_effective") != "jsbsim":
            errors.append(f"{key}: effective backend is not jsbsim")
        if row.get("backend_fallback_occurred", "").lower() != "false":
            errors.append(f"{key}: backend fallback occurred")
        if row.get("finite_telemetry", "").lower() != "true":
            errors.append(f"{key}: telemetry is not finite/nonempty")
        if row.get("reason", "") == "":
            errors.append(f"{key}: terminal reason missing")
        for name in ("min_range_m", "min_ata_deg", "length", "recovery_steps"):
            try:
                float(row[name])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{key}: invalid numeric field {name}")
    if any(value != REPETITIONS for value in counts.values()):
        errors.append(f"unbalanced candidate/opponent cells: {counts}")
    for opponent in EXPECTED_OPPONENTS:
        run_manifest_path = root / opponent / "run_manifest.json"
        if not run_manifest_path.is_file():
            errors.append(f"{opponent}: missing run_manifest.json")
            continue
        run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run.get("backend_effective") != "jsbsim":
            errors.append(f"{opponent}: run effective backend is not jsbsim")
        if run.get("backend_fallback_occurred") is not False:
            errors.append(f"{opponent}: run backend fallback occurred")
        if run.get("episode_count") != 6:
            errors.append(f"{opponent}: expected six episodes")
        if run.get("reset_calls_total") != 6:
            errors.append(f"{opponent}: unexpected reset count")
        for candidate in EXPECTED_CANDIDATES:
            for repetition in range(1, REPETITIONS + 1):
                trajectory = root / opponent / "trajectories" / f"{candidate}_{repetition:02d}.csv"
                if not trajectory.is_file() or trajectory.stat().st_size <= 0:
                    errors.append(f"{opponent}/{candidate}/{repetition}: missing trajectory telemetry")
    entered_counts = {}
    for opponent in EXPECTED_OPPONENTS:
        for candidate in EXPECTED_CANDIDATES:
            key = (opponent, candidate)
            entered_counts[f"{opponent}/{candidate}"] = sum(
                row.get("entered_stable_envelope", "").lower() == "true"
                for row in rows
                if (row.get("opponent", ""), row.get("candidate", "")) == key
            )
    strict_runtime_pass = not errors and all(
        row.get("backend_effective") == "jsbsim"
        and row.get("backend_fallback_occurred", "").lower() == "false"
        and row.get("reset_calls") == "1"
        and row.get("finite_telemetry", "").lower() == "true"
        for row in rows
    )
    recovery_gate_pass = strict_runtime_pass and all(
        entered_counts[f"{opponent}/{candidate}"] >= 2
        for opponent in EXPECTED_OPPONENTS
        for candidate in EXPECTED_CANDIDATES
    )
    return {
        "ok": not errors,
        "source_id": manifest.get("source_id"),
        "errors": errors,
        "episode_count": len(rows),
        "strict_runtime_gate": "pass" if strict_runtime_pass else "fail",
        "recovery_gate": "pass" if recovery_gate_pass else "fail",
        "entered_stable_envelope_counts": entered_counts,
        "interpretation": (
            "runtime_chain_valid_but_recovery_gate_failed"
            if strict_runtime_pass and not recovery_gate_pass
            else "all_gates_passed"
            if recovery_gate_pass
            else "runtime_integrity_failed"
        ),
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
