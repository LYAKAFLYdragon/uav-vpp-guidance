#!/usr/bin/env python
"""Verify the matched-contract recovery probe without rerunning JSBSim."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-CONTRACT-PROBE-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_CANDIDATES = ("x625_y200", "x750_y250")
REPETITIONS = 3


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
        errors.append("matched task context is not crossing_feasible")
    if manifest.get("matched_contract", {}).get("action_semantics") != "tactical_basis_v1":
        errors.append("matched action semantics are not tactical_basis_v1")
    if manifest.get("matched_contract", {}).get("anchor_mode") != "predicted_target":
        errors.append("matched anchor mode is not predicted_target")
    if tuple(manifest.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("opponent set mismatch")
    if tuple(manifest.get("candidates", {}).keys()) != EXPECTED_CANDIDATES:
        errors.append("candidate set mismatch")
    if manifest.get("repetitions_per_candidate_and_opponent") != REPETITIONS:
        errors.append("repetition count mismatch")
    if manifest.get("raw_episode_count") != 12:
        errors.append("expected exactly 12 episodes")
    rows: list[dict[str, str]] = []
    if not result_path.is_file():
        errors.append("missing episode_results.csv")
    else:
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 12:
            errors.append(f"episode result row count is {len(rows)}, expected 12")
    counts = {(opponent, candidate): 0 for opponent in EXPECTED_OPPONENTS for candidate in EXPECTED_CANDIDATES}
    entered_counts = {f"{opponent}/{candidate}": 0 for opponent in EXPECTED_OPPONENTS for candidate in EXPECTED_CANDIDATES}
    for row in rows:
        key = (row.get("opponent", ""), row.get("candidate", ""))
        if key not in counts:
            errors.append(f"unexpected episode key {key}")
            continue
        counts[key] += 1
        name = f"{key[0]}/{key[1]}"
        entered_counts[name] += row.get("entered_stable_envelope", "").lower() == "true"
        if row.get("reset_calls") != "1":
            errors.append(f"{name}: reset_calls is not 1")
        if row.get("backend_effective") != "jsbsim":
            errors.append(f"{name}: effective backend is not jsbsim")
        if row.get("backend_fallback_occurred", "").lower() != "false":
            errors.append(f"{name}: backend fallback occurred")
        if row.get("finite_telemetry", "").lower() != "true":
            errors.append(f"{name}: telemetry is not finite/nonempty")
        if row.get("matched_contract", "").lower() != "true":
            errors.append(f"{name}: matched contract marker missing")
        if row.get("reason", "") == "":
            errors.append(f"{name}: terminal reason missing")
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
        if run.get("task_name") != "crossing_feasible":
            errors.append(f"{opponent}: task context mismatch")
        if run.get("episode_count") != 6 or run.get("reset_calls_total") != 6:
            errors.append(f"{opponent}: episode/reset count mismatch")
        for candidate in EXPECTED_CANDIDATES:
            for repetition in range(1, REPETITIONS + 1):
                trajectory = root / opponent / "trajectories" / f"{candidate}_{repetition:02d}.csv"
                if not trajectory.is_file() or trajectory.stat().st_size <= 0:
                    errors.append(f"{opponent}/{candidate}/{repetition}: missing trajectory")
    strict_runtime_pass = not errors and all(
        row.get("backend_effective") == "jsbsim"
        and row.get("backend_fallback_occurred", "").lower() == "false"
        and row.get("reset_calls") == "1"
        and row.get("finite_telemetry", "").lower() == "true"
        for row in rows
    )
    recovery_gate_pass = strict_runtime_pass and all(value >= 2 for value in entered_counts.values())
    return {
        "ok": not errors,
        "source_id": manifest.get("source_id"),
        "errors": errors,
        "episode_count": len(rows),
        "strict_runtime_gate": "pass" if strict_runtime_pass else "fail",
        "recovery_gate": "pass" if recovery_gate_pass else "fail",
        "entered_stable_envelope_counts": entered_counts,
        "interpretation": (
            "runtime_chain_valid_and_matched_contract_but_recovery_gate_failed"
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
