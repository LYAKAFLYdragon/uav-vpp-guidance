#!/usr/bin/env python3
"""Analyze multi-waypoint route quality from step telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_step_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            active_idx = _safe_int(row.get("active_waypoint_index"))
            segment_idx = _safe_int(row.get("segment_waypoint_index"), active_idx)
            rows.append(
                {
                    "seed": _safe_int(row.get("seed")),
                    "step": _safe_int(row.get("step")),
                    "time_s": _safe_float(row.get("time_s")),
                    "range_m": _safe_float(row.get("range_m")),
                    "waypoint_range_m": _safe_float(row.get("waypoint_range_m")),
                    "segment_min_waypoint_range_m": _safe_float(
                        row.get("segment_min_waypoint_range_m")
                    ),
                    "segment_min_waypoint_range_step": _safe_int(
                        row.get("segment_min_waypoint_range_step"), -1
                    ),
                    "segment_range_receded_m": _safe_float(
                        row.get("segment_range_receded_m")
                    ),
                    "synthetic_target_range_m": _safe_float(
                        row.get("synthetic_target_range_m")
                    ),
                    "backend_range_m": _safe_float(row.get("backend_range_m")),
                    "capture_radius_m": _safe_float(row.get("capture_radius_m")),
                    "capture_reason": row.get("capture_reason") or "",
                    "near_miss_capture_active": _safe_bool(
                        row.get("near_miss_capture_active")
                    ),
                    "near_miss_capture_radius_m": _safe_float(
                        row.get("near_miss_capture_radius_m")
                    ),
                    "segment_elapsed_s": _safe_float(row.get("segment_elapsed_s")),
                    "heading_error_to_waypoint_deg": _safe_float(
                        row.get("heading_error_to_waypoint_deg")
                    ),
                    "multi_waypoint_roll_moderation_active": _safe_bool(
                        row.get("multi_waypoint_roll_moderation_active")
                    ),
                    "multi_waypoint_roll_recovery_active": _safe_bool(
                        row.get("multi_waypoint_roll_recovery_active")
                    ),
                    "multi_waypoint_same_bank_guard_active": _safe_bool(
                        row.get("multi_waypoint_same_bank_guard_active")
                    ),
                    "multi_waypoint_same_bank_guard_soft_cap_active": _safe_bool(
                        row.get("multi_waypoint_same_bank_guard_soft_cap_active")
                    ),
                    "multi_waypoint_roll_rate_scale": _safe_float(
                        row.get("multi_waypoint_roll_rate_scale")
                    ),
                    "multi_waypoint_static_waypoint_blend": _safe_float(
                        row.get("multi_waypoint_static_waypoint_blend")
                    ),
                    "altitude_m": _safe_float(row.get("altitude_m")),
                    "roll_deg": _safe_float(row.get("roll_deg")),
                    "segment_waypoint_index": segment_idx,
                    "active_waypoint_index": active_idx,
                    "completed_waypoints": _safe_int(row.get("completed_waypoints")),
                }
            )
    rows.sort(key=lambda item: (item["seed"], item["step"]))
    return rows


def _finalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    end_range = segment.get("range_at_segment_end_m", float("nan"))
    min_range = segment.get("min_waypoint_range_m", float("nan"))
    segment["range_receded_from_closest_m"] = (
        end_range - min_range
        if math.isfinite(end_range) and math.isfinite(min_range)
        else float("nan")
    )
    return segment


def _extract_segments(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    prev_seed = None
    prev_idx = None

    for row in rows:
        seed = row["seed"]
        idx = row.get("segment_waypoint_index", row["active_waypoint_index"])
        waypoint_range = row["waypoint_range_m"]
        heading_error = row["heading_error_to_waypoint_deg"]
        abs_heading_error = abs(heading_error) if math.isfinite(heading_error) else float("nan")
        capture_reason = row.get("capture_reason") or ""
        if current is None or seed != prev_seed or idx != prev_idx:
            if current is not None:
                segments.append(_finalize_segment(current))
            current = {
                "seed": seed,
                "active_waypoint_index": idx,
                "segment_waypoint_index": idx,
                "start_step": row["step"],
                "end_step": row["step"],
                "steps": 1,
                "start_completed_waypoints": row["completed_waypoints"],
                "end_completed_waypoints": row["completed_waypoints"],
                "min_altitude_m": row["altitude_m"],
                "max_abs_roll_deg": abs(row["roll_deg"]) if math.isfinite(row["roll_deg"]) else float("nan"),
                "range_at_segment_start_m": waypoint_range,
                "range_at_segment_end_m": waypoint_range,
                "min_waypoint_range_m": waypoint_range,
                "closest_approach_step": row["step"] if math.isfinite(waypoint_range) else None,
                "min_recorded_segment_waypoint_range_m": row[
                    "segment_min_waypoint_range_m"
                ],
                "segment_elapsed_at_end_s": row["segment_elapsed_s"],
                "heading_error_at_segment_end_deg": heading_error,
                "max_abs_heading_error_to_waypoint_deg": abs_heading_error,
                "roll_moderation_active_steps": int(
                    row["multi_waypoint_roll_moderation_active"]
                ),
                "roll_recovery_active_steps": int(
                    row["multi_waypoint_roll_recovery_active"]
                ),
                "same_bank_guard_active_steps": int(
                    row["multi_waypoint_same_bank_guard_active"]
                ),
                "same_bank_guard_soft_cap_steps": int(
                    row["multi_waypoint_same_bank_guard_soft_cap_active"]
                ),
                "same_bank_guard_zero_steps": int(
                    row["multi_waypoint_same_bank_guard_active"]
                    and not row["multi_waypoint_same_bank_guard_soft_cap_active"]
                ),
                "max_static_waypoint_blend": row[
                    "multi_waypoint_static_waypoint_blend"
                ],
                "near_miss_capture_active_steps": int(
                    row["near_miss_capture_active"]
                ),
                "capture_reasons": [capture_reason] if capture_reason else [],
            }
        else:
            current["end_step"] = row["step"]
            current["steps"] += 1
            current["end_completed_waypoints"] = row["completed_waypoints"]
            current["range_at_segment_end_m"] = waypoint_range
            current["segment_elapsed_at_end_s"] = row["segment_elapsed_s"]
            current["heading_error_at_segment_end_deg"] = heading_error
            if math.isfinite(row["altitude_m"]):
                if not math.isfinite(current["min_altitude_m"]):
                    current["min_altitude_m"] = row["altitude_m"]
                else:
                    current["min_altitude_m"] = min(current["min_altitude_m"], row["altitude_m"])
            if math.isfinite(row["roll_deg"]):
                abs_roll = abs(row["roll_deg"])
                if not math.isfinite(current["max_abs_roll_deg"]):
                    current["max_abs_roll_deg"] = abs_roll
                else:
                    current["max_abs_roll_deg"] = max(current["max_abs_roll_deg"], abs_roll)
            if math.isfinite(waypoint_range):
                if not math.isfinite(current["min_waypoint_range_m"]) or waypoint_range < current["min_waypoint_range_m"]:
                    current["min_waypoint_range_m"] = waypoint_range
                    current["closest_approach_step"] = row["step"]
            recorded_min = row["segment_min_waypoint_range_m"]
            if math.isfinite(recorded_min):
                if not math.isfinite(current["min_recorded_segment_waypoint_range_m"]):
                    current["min_recorded_segment_waypoint_range_m"] = recorded_min
                else:
                    current["min_recorded_segment_waypoint_range_m"] = min(
                        current["min_recorded_segment_waypoint_range_m"],
                        recorded_min,
                    )
            if math.isfinite(abs_heading_error):
                if not math.isfinite(current["max_abs_heading_error_to_waypoint_deg"]):
                    current["max_abs_heading_error_to_waypoint_deg"] = abs_heading_error
                else:
                    current["max_abs_heading_error_to_waypoint_deg"] = max(
                        current["max_abs_heading_error_to_waypoint_deg"],
                        abs_heading_error,
                    )
            current["roll_moderation_active_steps"] += int(
                row["multi_waypoint_roll_moderation_active"]
            )
            current["roll_recovery_active_steps"] += int(
                row["multi_waypoint_roll_recovery_active"]
            )
            current["same_bank_guard_active_steps"] += int(
                row["multi_waypoint_same_bank_guard_active"]
            )
            current["same_bank_guard_soft_cap_steps"] += int(
                row["multi_waypoint_same_bank_guard_soft_cap_active"]
            )
            current["same_bank_guard_zero_steps"] += int(
                row["multi_waypoint_same_bank_guard_active"]
                and not row["multi_waypoint_same_bank_guard_soft_cap_active"]
            )
            static_blend = row["multi_waypoint_static_waypoint_blend"]
            if math.isfinite(static_blend):
                if not math.isfinite(current["max_static_waypoint_blend"]):
                    current["max_static_waypoint_blend"] = static_blend
                else:
                    current["max_static_waypoint_blend"] = max(
                        current["max_static_waypoint_blend"],
                        static_blend,
                    )
            current["near_miss_capture_active_steps"] += int(
                row["near_miss_capture_active"]
            )
            if capture_reason and capture_reason not in current["capture_reasons"]:
                current["capture_reasons"].append(capture_reason)
        prev_seed = seed
        prev_idx = idx

    if current is not None:
        segments.append(_finalize_segment(current))
    return segments


def analyze_route_quality(
    summary_payload: Dict[str, Any],
    step_rows: List[Dict[str, Any]],
    *,
    segment_timeout_s: float = 20.0,
    high_level_dt: float = 0.2,
    timeout_tolerance_steps: int = 1,
    high_roll_deg: float = 85.0,
    severe_roll_deg: float = 120.0,
) -> Dict[str, Any]:
    episodes = {
        int(item["seed"]): item for item in summary_payload.get("episodes", [])
    }
    expected_timeout_steps = int(round(segment_timeout_s / max(high_level_dt, 1e-6))) + 1
    segments = _extract_segments(step_rows)

    per_seed_rows: Dict[int, List[Dict[str, Any]]] = {}
    per_seed_segments: Dict[int, List[Dict[str, Any]]] = {}
    for row in step_rows:
        per_seed_rows.setdefault(int(row["seed"]), []).append(row)
    for seg in segments:
        per_seed_segments.setdefault(int(seg["seed"]), []).append(seg)

    seed_reports: Dict[str, Any] = {}
    for seed, ep in sorted(episodes.items()):
        rows = per_seed_rows.get(seed, [])
        segs = per_seed_segments.get(seed, [])
        meaningful_segments = [seg for seg in segs if seg["steps"] > 1]
        max_roll_row = max(
            rows,
            key=lambda item: abs(item["roll_deg"]) if math.isfinite(item["roll_deg"]) else -1.0,
            default=None,
        )
        timeout_like_segments = [
            seg["active_waypoint_index"]
            for seg in meaningful_segments
            if seg["steps"] >= expected_timeout_steps - int(timeout_tolerance_steps)
        ]
        high_roll_segments = [
            seg["active_waypoint_index"]
            for seg in meaningful_segments
            if math.isfinite(seg["max_abs_roll_deg"]) and seg["max_abs_roll_deg"] >= high_roll_deg
        ]
        severe_roll_segments = [
            seg["active_waypoint_index"]
            for seg in meaningful_segments
            if math.isfinite(seg["max_abs_roll_deg"]) and seg["max_abs_roll_deg"] >= severe_roll_deg
        ]
        near_miss_capture_segments = [
            seg["active_waypoint_index"]
            for seg in meaningful_segments
            if "near_miss_passed_waypoint" in seg.get("capture_reasons", [])
        ]

        issues = []
        if ep.get("crashed"):
            issues.append("crash")
        if ep.get("completed_waypoints", 0) <= 0:
            issues.append("no_waypoint_completion")
        if timeout_like_segments:
            issues.append("segment_timeout")
        if high_roll_segments:
            issues.append("high_roll")
        if severe_roll_segments:
            issues.append("severe_roll")

        seed_reports[str(seed)] = {
            "seed": seed,
            "termination_reason": ep.get("termination_reason"),
            "crashed": bool(ep.get("crashed", False)),
            "completed_waypoints": int(ep.get("completed_waypoints", 0)),
            "max_abs_roll_deg": (
                abs(max_roll_row["roll_deg"]) if max_roll_row and math.isfinite(max_roll_row["roll_deg"]) else float("nan")
            ),
            "max_roll_active_waypoint_index": (
                max_roll_row["active_waypoint_index"] if max_roll_row is not None else None
            ),
            "max_roll_step": max_roll_row["step"] if max_roll_row is not None else None,
            "timeout_like_segments": timeout_like_segments,
            "high_roll_segments": high_roll_segments,
            "severe_roll_segments": severe_roll_segments,
            "near_miss_capture_segments": near_miss_capture_segments,
            "issues": issues,
            "segments": segs,
        }

    overall_issue_counts: Dict[str, int] = {}
    for seed_report in seed_reports.values():
        for issue in seed_report["issues"]:
            overall_issue_counts[issue] = overall_issue_counts.get(issue, 0) + 1

    return {
        "thresholds": {
            "segment_timeout_s": float(segment_timeout_s),
            "high_level_dt": float(high_level_dt),
            "expected_timeout_steps": expected_timeout_steps,
            "timeout_tolerance_steps": int(timeout_tolerance_steps),
            "high_roll_deg": float(high_roll_deg),
            "severe_roll_deg": float(severe_roll_deg),
        },
        "overall_issue_counts": overall_issue_counts,
        "seeds": seed_reports,
    }


def render_markdown(report: Dict[str, Any], summary_path: Path, csv_path: Path) -> str:
    thresholds = report["thresholds"]
    lines = [
        "# Multi-Waypoint Route Quality",
        "",
        f"Summary: `{summary_path}`",
        f"Steps: `{csv_path}`",
        "",
        "## Thresholds",
        "",
        f"- expected_timeout_steps: {thresholds['expected_timeout_steps']}",
        f"- high_roll_deg: {thresholds['high_roll_deg']:.3g}",
        f"- severe_roll_deg: {thresholds['severe_roll_deg']:.3g}",
        "",
        "## Seeds",
        "",
        "| seed | completed_wp | termination | max_roll | timeout_segments | high_roll_segments | severe_roll_segments | near_miss_segments | issues |",
        "|---:|---:|---|---:|---|---|---|---|---|",
    ]
    for seed_key, item in report["seeds"].items():
        issues_text = ",".join(item["issues"]) if item["issues"] else "-"
        lines.append(
            "| {seed} | {completed_waypoints} | {termination_reason} | {max_abs_roll_deg:.3g} | "
            "{timeout_like_segments} | {high_roll_segments} | {severe_roll_segments} | "
            "{near_miss_capture_segments} | {issues} |".format(
                seed=item["seed"],
                completed_waypoints=item["completed_waypoints"],
                termination_reason=item["termination_reason"],
                max_abs_roll_deg=item["max_abs_roll_deg"],
                timeout_like_segments=item["timeout_like_segments"],
                high_roll_segments=item["high_roll_segments"],
                severe_roll_segments=item["severe_roll_segments"],
                near_miss_capture_segments=item.get("near_miss_capture_segments", []),
                issues=issues_text,
            )
        )
    return "\n".join(lines) + "\n"


def _resolve_csv_path(summary_path: Path, csv_arg: str | None, payload: Dict[str, Any]) -> Path:
    candidates: List[Path] = []
    if csv_arg:
        candidates.append(Path(csv_arg))
    csv_from_payload = payload.get("csv_path")
    if csv_from_payload:
        candidates.extend(
            [
                Path(str(csv_from_payload)),
                summary_path.parent / str(csv_from_payload),
                Path.cwd() / str(csv_from_payload),
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve CSV path from {csv_arg!r} / {csv_from_payload!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--steps-csv")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--segment-timeout-s", type=float, default=20.0)
    parser.add_argument("--high-level-dt", type=float, default=0.2)
    parser.add_argument("--timeout-tolerance-steps", type=int, default=1)
    parser.add_argument("--high-roll-deg", type=float, default=85.0)
    parser.add_argument("--severe-roll-deg", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary_json)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_path = _resolve_csv_path(summary_path, args.steps_csv, payload)
    rows = load_step_rows(csv_path)
    report = analyze_route_quality(
        payload,
        rows,
        segment_timeout_s=args.segment_timeout_s,
        high_level_dt=args.high_level_dt,
        timeout_tolerance_steps=args.timeout_tolerance_steps,
        high_roll_deg=args.high_roll_deg,
        severe_roll_deg=args.severe_roll_deg,
    )

    output_json = (
        Path(args.output_json)
        if args.output_json
        else summary_path.with_name(summary_path.stem + "_route_quality.json")
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else summary_path.with_name(summary_path.stem + "_route_quality.md")
    )
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report, summary_path, csv_path), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
