#!/usr/bin/env python3
"""Compare head-on post-merge anchor-mode drift between two pilot runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _format_float(value: Any) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.3f}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_segments(points: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_step: Optional[int] = None
    for point in points:
        step = int(point.get("step", 0))
        active = bool(point.get("post_merge_anchor_mode_active", False))
        if active and start is None:
            start = step
        if not active and start is not None and prev_step is not None:
            segments.append((start, prev_step))
            start = None
        prev_step = step
    if start is not None and prev_step is not None:
        segments.append((start, prev_step))
    return segments


def _first_step(points: Sequence[Mapping[str, Any]], field: str) -> float:
    for index, point in enumerate(points, start=1):
        if not bool(point.get(field, False)):
            continue
        step = _safe_float(point.get("step"))
        if math.isfinite(step):
            return step
        return float(index)
    return float("nan")


def summarize_episode_anchor_mode_drift(episode: Mapping[str, Any]) -> Dict[str, Any]:
    trajectory = [
        point for point in episode.get("trajectory", []) if isinstance(point, dict)
    ]
    active_points = [
        point
        for point in trajectory
        if bool(point.get("post_merge_anchor_mode_active", False))
    ]
    active_segments = _active_segments(trajectory)
    first_active = active_points[0] if active_points else {}
    last_active = active_points[-1] if active_points else {}
    active_count = len(active_points)

    def _mean(values: Iterable[float]) -> float:
        clean = [value for value in values if math.isfinite(value)]
        return float(sum(clean) / len(clean)) if clean else float("nan")

    damage_margin = (
        (100.0 - _safe_float(episode.get("target_hp")))
        - (100.0 - _safe_float(episode.get("ego_hp")))
    )
    if not math.isfinite(damage_margin):
        damage_margin = _safe_float(episode.get("hp_advantage"))

    return {
        "seed": int(episode.get("seed", 0)),
        "episode": int(episode.get("episode", 0)),
        "scenario": episode.get("scenario"),
        "termination_reason": episode.get("termination_reason"),
        "success": bool(episode.get("success", False)),
        "damage_margin": damage_margin,
        "first_pass_step": _first_step(trajectory, "post_merge"),
        "active_segments": active_segments,
        "active_segment_count": len(active_segments),
        "total_active_steps": active_count,
        "first_active_step": _safe_float(first_active.get("step")),
        "last_active_step": _safe_float(last_active.get("step")),
        "first_release_step": _first_step(trajectory, "post_merge_anchor_mode_released"),
        "first_active_vp_forward_bias_m": _safe_float(
            first_active.get("vp_forward_bias_m")
        ),
        "first_active_vp_lateral_bias_m": _safe_float(
            first_active.get("vp_lateral_bias_m")
        ),
        "last_active_vp_forward_bias_m": _safe_float(
            last_active.get("vp_forward_bias_m")
        ),
        "last_active_vp_lateral_bias_m": _safe_float(
            last_active.get("vp_lateral_bias_m")
        ),
        "mean_active_vp_forward_bias_m": _mean(
            _safe_float(point.get("vp_forward_bias_m")) for point in active_points
        ),
        "mean_active_vp_lateral_bias_m": _mean(
            _safe_float(point.get("vp_lateral_bias_m")) for point in active_points
        ),
        "first_active_range_m": _safe_float(first_active.get("range_m")),
        "last_active_range_m": _safe_float(last_active.get("range_m")),
        "max_active_range_m": (
            max(
                (
                    _safe_float(point.get("range_m"))
                    for point in active_points
                    if math.isfinite(_safe_float(point.get("range_m")))
                ),
                default=float("nan"),
            )
        ),
        "min_active_range_m": (
            min(
                (
                    _safe_float(point.get("range_m"))
                    for point in active_points
                    if math.isfinite(_safe_float(point.get("range_m")))
                ),
                default=float("nan"),
            )
        ),
        "latch_world_offset_x": _safe_float(
            first_active.get("offensive_anchor_lateral_world_offset_x")
        ),
        "latch_world_offset_y": _safe_float(
            first_active.get("offensive_anchor_lateral_world_offset_y")
        ),
        "latch_world_offset_z": _safe_float(
            first_active.get("offensive_anchor_lateral_world_offset_z")
        ),
    }


def _load_run_summaries(
    run_dir: Path,
    *,
    task: str,
    method: Optional[str],
) -> List[Dict[str, Any]]:
    payload = _load_json(run_dir / "aggregate" / "episode_records.json")
    summaries: List[Dict[str, Any]] = []
    for episode in payload.get("episodes", []):
        if str(episode.get("task")) != task:
            continue
        if method is not None and str(episode.get("controller")) != method:
            continue
        summary = summarize_episode_anchor_mode_drift(episode)
        summary["run_dir"] = str(run_dir)
        summaries.append(summary)
    return summaries


def build_comparison_report(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    baseline_index = {
        (int(row["seed"]), int(row["episode"])): dict(row) for row in baseline_rows
    }
    rows: List[Dict[str, Any]] = []
    for candidate in candidate_rows:
        key = (int(candidate["seed"]), int(candidate["episode"]))
        baseline = baseline_index[key]
        row = {
            "seed": key[0],
            "episode": key[1],
            "scenario": candidate.get("scenario"),
            "baseline": baseline,
            "candidate": dict(candidate),
            "delta": {
                "damage_margin": _safe_float(candidate.get("damage_margin"))
                - _safe_float(baseline.get("damage_margin")),
                "first_active_step": _safe_float(candidate.get("first_active_step"))
                - _safe_float(baseline.get("first_active_step")),
                "total_active_steps": _safe_float(candidate.get("total_active_steps"))
                - _safe_float(baseline.get("total_active_steps")),
                "mean_active_vp_forward_bias_m": _safe_float(
                    candidate.get("mean_active_vp_forward_bias_m")
                )
                - _safe_float(baseline.get("mean_active_vp_forward_bias_m")),
                "last_active_vp_forward_bias_m": _safe_float(
                    candidate.get("last_active_vp_forward_bias_m")
                )
                - _safe_float(baseline.get("last_active_vp_forward_bias_m")),
            },
        }
        rows.append(row)
    rows.sort(key=lambda item: item["seed"])
    return {"rows": rows}


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Head-On Anchor Mode Drift",
        "",
        "| seed | baseline_damage | candidate_damage | delta_damage | baseline_active_steps | candidate_active_steps | delta_active_steps | baseline_first_step | candidate_first_step | baseline_mean_fwd | candidate_mean_fwd | delta_mean_fwd | baseline_last_fwd | candidate_last_fwd | delta_last_fwd | candidate_latch_world_xy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        baseline = row["baseline"]
        candidate = row["candidate"]
        delta = row["delta"]
        latch_xy = (
            f"({_format_float(candidate.get('latch_world_offset_x'))}, "
            f"{_format_float(candidate.get('latch_world_offset_y'))})"
        )
        lines.append(
            (
                f"| {row['seed']} | "
                f"{_format_float(baseline.get('damage_margin'))} | "
                f"{_format_float(candidate.get('damage_margin'))} | "
                f"{_format_float(delta.get('damage_margin'))} | "
                f"{_format_float(baseline.get('total_active_steps'))} | "
                f"{_format_float(candidate.get('total_active_steps'))} | "
                f"{_format_float(delta.get('total_active_steps'))} | "
                f"{_format_float(baseline.get('first_active_step'))} | "
                f"{_format_float(candidate.get('first_active_step'))} | "
                f"{_format_float(baseline.get('mean_active_vp_forward_bias_m'))} | "
                f"{_format_float(candidate.get('mean_active_vp_forward_bias_m'))} | "
                f"{_format_float(delta.get('mean_active_vp_forward_bias_m'))} | "
                f"{_format_float(baseline.get('last_active_vp_forward_bias_m'))} | "
                f"{_format_float(candidate.get('last_active_vp_forward_bias_m'))} | "
                f"{_format_float(delta.get('last_active_vp_forward_bias_m'))} | "
                f"{latch_xy} |"
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--task", default="head_on")
    parser.add_argument("--method", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_rows = _load_run_summaries(
        Path(args.baseline_run_dir),
        task=args.task,
        method=args.method,
    )
    candidate_rows = _load_run_summaries(
        Path(args.candidate_run_dir),
        task=args.task,
        method=args.method,
    )
    report = build_comparison_report(baseline_rows, candidate_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
