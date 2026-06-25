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


def load_step_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "seed": _safe_int(row.get("seed")),
                    "step": _safe_int(row.get("step")),
                    "time_s": _safe_float(row.get("time_s")),
                    "altitude_m": _safe_float(row.get("altitude_m")),
                    "roll_deg": _safe_float(row.get("roll_deg")),
                    "active_waypoint_index": _safe_int(row.get("active_waypoint_index")),
                    "completed_waypoints": _safe_int(row.get("completed_waypoints")),
                }
            )
    rows.sort(key=lambda item: (item["seed"], item["step"]))
    return rows


def _extract_segments(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    prev_seed = None
    prev_idx = None

    for row in rows:
        seed = row["seed"]
        idx = row["active_waypoint_index"]
        if current is None or seed != prev_seed or idx != prev_idx:
            if current is not None:
                segments.append(current)
            current = {
                "seed": seed,
                "active_waypoint_index": idx,
                "start_step": row["step"],
                "end_step": row["step"],
                "steps": 1,
                "start_completed_waypoints": row["completed_waypoints"],
                "end_completed_waypoints": row["completed_waypoints"],
                "min_altitude_m": row["altitude_m"],
                "max_abs_roll_deg": abs(row["roll_deg"]) if math.isfinite(row["roll_deg"]) else float("nan"),
            }
        else:
            current["end_step"] = row["step"]
            current["steps"] += 1
            current["end_completed_waypoints"] = row["completed_waypoints"]
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
        prev_seed = seed
        prev_idx = idx

    if current is not None:
        segments.append(current)
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
        "| seed | completed_wp | termination | max_roll | timeout_segments | high_roll_segments | severe_roll_segments | issues |",
        "|---:|---:|---|---:|---|---|---|---|",
    ]
    for seed_key, item in report["seeds"].items():
        issues_text = ",".join(item["issues"]) if item["issues"] else "-"
        lines.append(
            "| {seed} | {completed_waypoints} | {termination_reason} | {max_abs_roll_deg:.3g} | "
            "{timeout_like_segments} | {high_roll_segments} | {severe_roll_segments} | {issues} |".format(
                seed=item["seed"],
                completed_waypoints=item["completed_waypoints"],
                termination_reason=item["termination_reason"],
                max_abs_roll_deg=item["max_abs_roll_deg"],
                timeout_like_segments=item["timeout_like_segments"],
                high_roll_segments=item["high_roll_segments"],
                severe_roll_segments=item["severe_roll_segments"],
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
