#!/usr/bin/env python3
"""Read-only terminal, horizon, and turn-back audit for the V5--V7 ledgers.

The raw-SI ledgers retain normalized range and range-rate in the 16-D base
observation.  This tool converts only those two fields using the frozen
observation contract so that the attribution report remains physically
interpretable.  It never changes a manifest, controller, checkpoint, or raw
episode record.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-TERMINAL-HORIZON-AUDIT-R1"
RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0
REAPPROACH_THRESHOLD_MPS = -25.0
ADEQUATE_POST_PASS_SECONDS = 10.0


@dataclass(frozen=True)
class StudySpec:
    label: str
    root: Path


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")
    return result


def _terminal_reason(record: Mapping[str, Any], steps: list[Mapping[str, Any]]) -> str:
    result = record.get("result")
    if isinstance(result, Mapping) and result.get("terminal_reason"):
        return str(result["terminal_reason"])
    if record.get("terminal"):
        return str(record["terminal"])
    if steps:
        return str(steps[-1].get("terminal_reason") or "unknown")
    return "unknown"


def _method_label(study: str, record: Mapping[str, Any], steps: list[Mapping[str, Any]]) -> str:
    if study == "V7":
        gain = record.get("gain")
        return f"turnback_gain_{_finite(gain, 'gain'):+.1f}"
    if record.get("controller"):
        return str(record["controller"])
    if steps and steps[0].get("controller_method"):
        return str(steps[0]["controller_method"])
    header = record.get("header")
    if isinstance(header, Mapping) and header.get("reference_controller"):
        return str(header["reference_controller"])
    return "unknown_controller"


def _longest_reapproach(steps: Iterable[Mapping[str, Any]]) -> int:
    longest = 0
    current = 0
    for step in steps:
        raw = step.get("raw_si_phase_input")
        if not isinstance(raw, Mapping):
            raise ValueError("ledger step lacks raw_si_phase_input")
        rate_mps = _finite(raw.get("range_rate_mps"), "range_rate_mps") * RANGE_RATE_REFERENCE_MPS
        if rate_mps < REAPPROACH_THRESHOLD_MPS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _physical_reentry_candidate_steps(steps: Iterable[Mapping[str, Any]]) -> int:
    """Count valid 66-D samples after a raw first pass and physical re-approach.

    The legacy PhaseTracker was given normalized observation values although
    its thresholds are specified in metres and metres per second.  This
    read-only diagnostic deliberately avoids that tracker: a candidate needs
    the independently latched raw first-pass event plus a physical closing
    rate at or below the frozen P4 threshold.
    """
    count = 0
    for step in steps:
        raw = step.get("raw_si_phase_input")
        if not isinstance(raw, Mapping):
            raise ValueError("ledger step lacks raw_si_phase_input")
        rate_mps = _finite(raw.get("range_rate_mps"), "range_rate_mps") * RANGE_RATE_REFERENCE_MPS
        if rate_mps <= REAPPROACH_THRESHOLD_MPS and step.get("observation_66d") is not None:
            count += 1
    return count


def summarize_payload(study: str, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    record = payload.get("record")
    ledger = payload.get("ledger")
    if not isinstance(record, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError(f"{path}: missing record or ledger")
    steps = ledger.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: ledger has no steps")
    if not all(isinstance(step, Mapping) for step in steps):
        raise ValueError(f"{path}: ledger steps are malformed")

    first_pass_index = next(
        (index for index, step in enumerate(steps) if bool(step.get("raw_si_first_pass_latched"))),
        None,
    )
    post_pass_steps = steps[first_pass_index:] if first_pass_index is not None else []
    ranges = [
        _finite(step["raw_si_phase_input"].get("range_m"), "range_m") * RANGE_REFERENCE_M
        for step in post_pass_steps
        if isinstance(step.get("raw_si_phase_input"), Mapping)
    ]
    if first_pass_index is not None and len(ranges) != len(post_pass_steps):
        raise ValueError(f"{path}: post-pass step lacks range")

    initial_state = str(record.get("initial_state") or record.get("metadata", {}).get("initial_class") or "unknown")
    opponent = str(record.get("opponent") or "unknown")
    first_pass_time = (
        _finite(post_pass_steps[0].get("own_state", {}).get("sim_time"), "first_pass_sim_time")
        if post_pass_steps
        else None
    )
    final_time = _finite(steps[-1].get("own_state", {}).get("sim_time"), "final_sim_time")
    saturated_steps = sum(
        bool(step.get("nz_saturated"))
        or bool(step.get("roll_rate_saturated"))
        or bool(step.get("throttle_saturated"))
        for step in post_pass_steps
    )
    max_abs_nz_delta = max(
        abs(
            _finite(step.get("actual_nz_g"), "actual_nz_g")
            - _finite(step.get("nz_cmd"), "nz_cmd")
        )
        for step in post_pass_steps
    ) if post_pass_steps else 0.0
    ledger_reentry_steps = sum(
        step.get("phase") == "re_entry" and step.get("observation_66d") is not None
        for step in post_pass_steps
    )
    physical_reentry_candidate_steps = _physical_reentry_candidate_steps(post_pass_steps)
    terminal_reason = _terminal_reason(record, steps)

    result = {
        "study": study,
        "episode_path": str(path),
        "opponent": opponent,
        "method": _method_label(study, record, steps),
        "initial_state": initial_state,
        "terminal_reason": terminal_reason,
        "first_pass_reached": first_pass_index is not None,
        "first_pass_step": None if first_pass_index is None else int(steps[first_pass_index].get("step", first_pass_index + 1)),
        "remaining_steps_after_first_pass": 0 if first_pass_index is None else len(steps) - first_pass_index - 1,
        "post_pass_seconds": 0.0 if first_pass_time is None else max(0.0, final_time - first_pass_time),
        "min_range_after_first_pass_m": None if not ranges else min(ranges),
        "max_separation_after_first_pass_m": None if not ranges else max(ranges),
        "longest_reapproach_steps": _longest_reapproach(post_pass_steps),
        "ledger_reentry_66d_steps": int(ledger_reentry_steps),
        "physical_reentry_candidate_66d_steps": int(physical_reentry_candidate_steps),
        "saturated_post_pass_steps": saturated_steps,
        "post_pass_step_count": len(post_pass_steps),
        "max_abs_nz_command_response_delta_g": max_abs_nz_delta,
        "ego_crash_or_oob": terminal_reason == "ego_crash_or_out_of_bounds",
        "target_crash_or_oob": terminal_reason == "target_crash_or_out_of_bounds",
    }
    return result


def load_study(spec: StudySpec) -> list[dict[str, Any]]:
    episode_root = spec.root / "episodes"
    if not episode_root.is_dir():
        raise ValueError(f"{spec.label}: missing episodes directory: {episode_root}")
    rows = []
    for path in sorted(episode_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summarize_payload(spec.label, path, payload))
    if not rows:
        raise ValueError(f"{spec.label}: no episode records found")
    return rows


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def aggregate(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["study"]), str(row["method"]), str(row["opponent"]))].append(row)

    result = []
    for (study, method, opponent), group in sorted(groups.items()):
        terminal_counts = Counter(str(row["terminal_reason"]) for row in group)
        first_pass = [row for row in group if bool(row["first_pass_reached"])]
        result.append({
            "study": study,
            "method": method,
            "opponent": opponent,
            "episodes": len(group),
            "first_pass_episodes": len(first_pass),
            "ledger_reentry_episodes": sum(int(row["ledger_reentry_66d_steps"]) > 0 for row in group),
            "ledger_reentry_66d_steps": sum(int(row["ledger_reentry_66d_steps"]) for row in group),
            "physical_reentry_candidate_episodes": sum(
                int(row["physical_reentry_candidate_66d_steps"]) > 0 for row in group
            ),
            "physical_reentry_candidate_66d_steps": sum(
                int(row["physical_reentry_candidate_66d_steps"]) for row in group
            ),
            "median_remaining_steps_after_first_pass": _median(float(row["remaining_steps_after_first_pass"]) for row in first_pass),
            "median_post_pass_seconds": _median(float(row["post_pass_seconds"]) for row in first_pass),
            "minimum_range_after_first_pass_m": min(
                (float(row["min_range_after_first_pass_m"]) for row in first_pass if row["min_range_after_first_pass_m"] is not None),
                default=0.0,
            ),
            "maximum_separation_after_first_pass_m": max(
                (float(row["max_separation_after_first_pass_m"]) for row in first_pass if row["max_separation_after_first_pass_m"] is not None),
                default=0.0,
            ),
            "episodes_with_reapproach": sum(int(row["longest_reapproach_steps"]) > 0 for row in first_pass),
            "maximum_consecutive_reapproach_steps": max(
                (int(row["longest_reapproach_steps"]) for row in first_pass), default=0
            ),
            "ego_crash_or_oob_episodes": sum(bool(row["ego_crash_or_oob"]) for row in group),
            "target_crash_or_oob_episodes": sum(bool(row["target_crash_or_oob"]) for row in group),
            "saturated_post_pass_steps": sum(int(row["saturated_post_pass_steps"]) for row in group),
            "max_abs_nz_command_response_delta_g": max(
                (float(row["max_abs_nz_command_response_delta_g"]) for row in group), default=0.0
            ),
            "terminal_counts": dict(sorted(terminal_counts.items())),
        })
    return result


def attribution_v6_v7(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    evidence = [row for row in rows if row["study"] in {"V6", "V7"}]
    if not evidence:
        raise ValueError("V6/V7 evidence is required for attribution")
    all_first_pass = all(bool(row["first_pass_reached"]) for row in evidence)
    median_seconds = _median(float(row["post_pass_seconds"]) for row in evidence)
    all_zero_ledger_reentry = all(int(row["ledger_reentry_66d_steps"]) == 0 for row in evidence)
    physical_candidate_episodes = sum(
        int(row["physical_reentry_candidate_66d_steps"]) > 0 for row in evidence
    )
    physical_candidate_steps = sum(int(row["physical_reentry_candidate_66d_steps"]) for row in evidence)
    total_saturation = sum(int(row["saturated_post_pass_steps"]) for row in evidence)
    ego_failures = sum(bool(row["ego_crash_or_oob"]) for row in evidence)

    if all_zero_ledger_reentry and physical_candidate_episodes:
        verdict = "phase-semantics/unit-contract"
    elif not all_first_pass or median_seconds < ADEQUATE_POST_PASS_SECONDS:
        verdict = "terminal/horizon"
    elif total_saturation or ego_failures:
        verdict = "execution/safety"
    elif all_zero_ledger_reentry:
        verdict = "geometry/behavior"
    else:
        verdict = "indeterminate"
    result = {
        "verdict": verdict,
        "scope": "V6 and V7 only; V5 is retained as supporting observability evidence.",
        "records": len(evidence),
        "all_records_reached_first_pass": all_first_pass,
        "median_post_pass_seconds": median_seconds,
        "adequate_post_pass_seconds_threshold": ADEQUATE_POST_PASS_SECONDS,
        "all_records_zero_ledger_reentry_66d_steps": all_zero_ledger_reentry,
        "physical_reentry_candidate_episodes": physical_candidate_episodes,
        "physical_reentry_candidate_66d_steps": physical_candidate_steps,
        "saturated_post_pass_steps": total_saturation,
        "ego_crash_or_oob_episodes": ego_failures,
        "interpretation": (
            "The frozen controllers and turn-back teacher had adequate post-pass time without recorded "
            "saturation or ego crash/OOB, but did not create re-entry. This supports a geometry/behavior "
            "attribution within the tested controllers and task dynamics; it is not a general aerodynamic limit."
            if verdict == "geometry/behavior" else "The evidence does not support a geometry/behavior attribution."
        ),
    }
    if verdict == "phase-semantics/unit-contract":
        result["interpretation"] = (
            "The ledger reports zero re-entry while the raw first-pass contract identifies valid 66-D "
            "samples during physical re-approach. The zero is therefore a phase-semantics/unit-contract "
            "failure, not evidence about reward, training duration, or flight-control capability."
        )
    return result


def build_audit(specs: Iterable[StudySpec]) -> dict[str, Any]:
    rows = [row for spec in specs for row in load_study(spec)]
    groups = aggregate(rows)
    return {
        "source_id": SOURCE_ID,
        "read_only": True,
        "unit_conversion": {
            "range_m": f"normalized range_m x {RANGE_REFERENCE_M}",
            "range_rate_mps": f"normalized range_rate_mps x {RANGE_RATE_REFERENCE_MPS}",
            "reapproach_threshold_mps": REAPPROACH_THRESHOLD_MPS,
        },
        "episode_rows": rows,
        "group_summary": groups,
        "attribution": attribution_v6_v7(rows),
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "study", "method", "opponent", "initial_state", "terminal_reason", "first_pass_reached",
        "first_pass_step", "remaining_steps_after_first_pass", "post_pass_seconds",
        "min_range_after_first_pass_m", "max_separation_after_first_pass_m", "longest_reapproach_steps",
        "ledger_reentry_66d_steps", "physical_reentry_candidate_66d_steps", "saturated_post_pass_steps", "post_pass_step_count",
        "max_abs_nz_command_response_delta_g", "ego_crash_or_oob", "target_crash_or_oob", "episode_path",
    ]
    lines = [",".join(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = _csv_value(row.get(field)).replace('"', '""')
            values.append(f'"{value}"')
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(audit: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# V5--V7 Re-entry Terminal/Horizon Read-only Audit",
        "",
        "## Scope",
        "",
        "This is a read-only analysis of the existing V5, V6, and V7 ledgers. It changes no environment, task, reward, controller, checkpoint, or raw artifact.",
        "",
        "## Attribution",
        "",
        f"- Verdict: `{audit['attribution']['verdict']}`",
        f"- V6/V7 records: {audit['attribution']['records']}",
        f"- All reached first pass: {audit['attribution']['all_records_reached_first_pass']}",
        f"- Median post-pass time: {audit['attribution']['median_post_pass_seconds']:.1f} s",
        f"- Ledger re-entry 66-D steps in every V6/V7 record: {audit['attribution']['all_records_zero_ledger_reentry_66d_steps']}",
        f"- Physical re-entry candidate episodes/steps: {audit['attribution']['physical_reentry_candidate_episodes']} / {audit['attribution']['physical_reentry_candidate_66d_steps']}",
        f"- Recorded post-pass saturation steps: {audit['attribution']['saturated_post_pass_steps']}",
        f"- Ego crash/OOB episodes: {audit['attribution']['ego_crash_or_oob_episodes']}",
        "",
        audit['attribution']['interpretation'],
        "",
        "## Group Summary",
        "",
        "| Study | Method | Opponent | N | First pass | Terminal reasons | Median time after pass (s) | Ledger re-entry episodes | Physical candidate episodes | Saturated steps |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in audit["group_summary"]:
        terminals = "; ".join(f"{name}: {count}" for name, count in row["terminal_counts"].items())
        lines.append(
            f"| {row['study']} | {row['method']} | {row['opponent']} | {row['episodes']} | "
            f"{row['first_pass_episodes']} | {terminals} | {row['median_post_pass_seconds']:.1f} | "
            f"{row['ledger_reentry_episodes']} | {row['physical_reentry_candidate_episodes']} | {row['saturated_post_pass_steps']} |"
        )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "The verdict distinguishes the tested task dynamics and controllers only. The ledgers do not record angle of attack, so this audit cannot independently attribute the result to an AoA limit. The existing `PhaseTracker` is not a valid re-entry labeler here: its metre and metre-per-second thresholds were supplied normalized observations, and it has no raw first-pass latch. Correct the phase contract and verify it on a fresh non-learning manifest before considering any reward experiment. Reward shaping cannot repair an invalid training label.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit((
        StudySpec("V5", args.v5_root),
        StudySpec("V6", args.v6_root),
        StudySpec("V7", args.v7_root),
    ))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "terminal_horizon_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_csv(audit["episode_rows"], args.out_dir / "episode_terminal_horizon_audit.csv")
    write_report(audit, args.out_dir / "terminal_horizon_audit.md")
    print(json.dumps(audit["attribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
