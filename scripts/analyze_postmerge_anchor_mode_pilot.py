#!/usr/bin/env python3
"""Compare a post-merge anchor-mode pilot lane against the current directtrack baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SUMMARY_KEYS = (
    "win_rate",
    "hp_advantage",
    "damaging_win_rate",
)

DIAGNOSTIC_KEYS = (
    "damage_margin",
    "post_merge_attack_zone_advantage_s",
    "post_merge_anchor_mode_active_fraction",
    "post_merge_anchor_mode_first_active_step",
    "post_merge_anchor_mode_offensive_anchor_blend",
    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend",
    "post_merge_anchor_mode_offensive_anchor_lateral_blend",
    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation",
    "post_merge_anchor_mode_lateral_world_offset_latch_active_fraction",
    "post_merge_anchor_mode_lateral_world_offset_latch_first_active_step",
    "post_merge_anchor_mode_active_vp_forward_bias_m",
    "post_merge_anchor_mode_active_vp_lateral_bias_m",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_active_vp_forward_bias_m",
    "post_merge_offensive_anchor_active_vp_lateral_bias_m",
    "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m",
    "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m",
    "merge_min_range_m",
    "offensive_anchor_lateral_sign_mode",
    "offensive_anchor_lateral_frame",
)


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


def _select_row(payload: Mapping[str, Any], *, task: str, method: Optional[str]) -> Dict[str, Any]:
    rows = [
        row
        for row in payload.get("rows", [])
        if str(row.get("task")) == task
        and (method is None or str(row.get("method")) == method)
    ]
    if not rows:
        raise ValueError(f"No row for task={task!r}, method={method!r}")
    if len(rows) > 1:
        raise ValueError(f"Expected one row for task={task!r}, method={method!r}, found {len(rows)}")
    return dict(rows[0])


def load_run_rows(
    *,
    label: str,
    run_dir: Path,
    method: Optional[str],
    tasks: Sequence[str],
) -> List[Dict[str, Any]]:
    summary_payload = _load_json(run_dir / "aggregate" / "method_task_summary.json")
    diagnostic_payload = _load_json(run_dir / "aggregate" / "combat_geometry_diagnostics.json")
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        summary_row = _select_row(summary_payload, task=task, method=method)
        diagnostic_row = _select_row(diagnostic_payload, task=task, method=method)
        merged = {
            "label": label,
            "run_dir": str(run_dir),
            "task": task,
            "opponent_stage": str(summary_row.get("opponent_stage")),
        }
        for key in SUMMARY_KEYS:
            merged[key] = summary_row.get(key)
        for key in DIAGNOSTIC_KEYS:
            merged[key] = diagnostic_row.get(key)
        rows.append(merged)
    return rows


def build_report(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    baseline_index = {
        (str(row["opponent_stage"]), str(row["task"])): dict(row)
        for row in baseline_rows
    }
    merged_rows: List[Dict[str, Any]] = []
    for candidate in candidate_rows:
        key = (str(candidate["opponent_stage"]), str(candidate["task"]))
        baseline = baseline_index[key]
        row = {
            "opponent_stage": key[0],
            "task": key[1],
            "baseline": baseline,
            "candidate": dict(candidate),
            "delta": {
                "win_rate": _safe_float(candidate.get("win_rate")) - _safe_float(baseline.get("win_rate")),
                "damage_margin": _safe_float(candidate.get("damage_margin")) - _safe_float(baseline.get("damage_margin")),
                "post_merge_attack_zone_advantage_s": _safe_float(candidate.get("post_merge_attack_zone_advantage_s"))
                - _safe_float(baseline.get("post_merge_attack_zone_advantage_s")),
            },
        }
        merged_rows.append(row)

    expert_head_on = next(
        row for row in merged_rows if row["opponent_stage"] == "expert" and row["task"] == "head_on"
    )
    end_to_end_head_on = next(
        row for row in merged_rows if row["opponent_stage"] == "end_to_end" and row["task"] == "head_on"
    )
    crossing_rows = [
        row for row in merged_rows if row["task"] == "crossing_feasible"
    ]
    return {
        "rows": merged_rows,
        "headline": {
            "expert_head_on": expert_head_on,
            "end_to_end_head_on": end_to_end_head_on,
            "crossing_rows": crossing_rows,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Merge Anchor Mode Pilot",
        "",
        "## Variant Matrix",
        "",
        "| split | task | baseline_win | candidate_win | delta_win | baseline_damage | candidate_damage | delta_damage | baseline_post_adv | candidate_post_adv | anchor_mode_blend | anchor_mode_long_blend | anchor_mode_lat_blend | latch_enabled | latch_active_frac | latch_first_step | lateral_sign_mode | lateral_frame | anchor_mode_active_frac | anchor_mode_active_step | anchor_mode_active_vp_forward_m | anchor_mode_active_vp_lateral_m | blend_active_frac | merge_min_range_m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        baseline = row["baseline"]
        candidate = row["candidate"]
        delta = row["delta"]
        lines.append(
            (
                f"| {row['opponent_stage']} | {row['task']} | "
                f"{_format_float(baseline.get('win_rate'))} | "
                f"{_format_float(candidate.get('win_rate'))} | "
                f"{_format_float(delta.get('win_rate'))} | "
                f"{_format_float(baseline.get('damage_margin'))} | "
                f"{_format_float(candidate.get('damage_margin'))} | "
                f"{_format_float(delta.get('damage_margin'))} | "
                f"{_format_float(baseline.get('post_merge_attack_zone_advantage_s'))} | "
                f"{_format_float(candidate.get('post_merge_attack_zone_advantage_s'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_offensive_anchor_blend'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_offensive_anchor_longitudinal_blend'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_offensive_anchor_lateral_blend'))} | "
                f"{'true' if bool(candidate.get('post_merge_anchor_mode_lateral_world_offset_latch_on_activation')) else 'false'} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_lateral_world_offset_latch_active_fraction'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_lateral_world_offset_latch_first_active_step'))} | "
                f"{candidate.get('offensive_anchor_lateral_sign_mode') or 'n/a'} | "
                f"{candidate.get('offensive_anchor_lateral_frame') or 'n/a'} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_active_fraction'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_first_active_step'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_active_vp_forward_bias_m'))} | "
                f"{_format_float(candidate.get('post_merge_anchor_mode_active_vp_lateral_bias_m'))} | "
                f"{_format_float(candidate.get('post_merge_offensive_anchor_blend_active_fraction'))} | "
                f"{_format_float(candidate.get('merge_min_range_m'))} |"
            )
        )

    expert = report["headline"]["expert_head_on"]
    e2e = report["headline"]["end_to_end_head_on"]
    lines.extend(
        [
            "",
            "## Headline",
            "",
            (
                f"- expert/head_on delta: win_rate={_format_float(expert['delta']['win_rate'])}, "
                f"damage_margin={_format_float(expert['delta']['damage_margin'])}, "
                f"post_merge_advantage_s={_format_float(expert['delta']['post_merge_attack_zone_advantage_s'])}"
            ),
            (
                f"- end_to_end/head_on delta: win_rate={_format_float(e2e['delta']['win_rate'])}, "
                f"damage_margin={_format_float(e2e['delta']['damage_margin'])}, "
                f"post_merge_advantage_s={_format_float(e2e['delta']['post_merge_attack_zone_advantage_s'])}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-expert-run-dir", required=True)
    parser.add_argument("--baseline-end-to-end-run-dir", required=True)
    parser.add_argument("--candidate-expert-run-dir", required=True)
    parser.add_argument("--candidate-end-to-end-run-dir", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--tasks", nargs="+", default=["head_on", "crossing_feasible"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_rows: List[Dict[str, Any]] = []
    baseline_rows.extend(
        load_run_rows(
            label="baseline",
            run_dir=Path(args.baseline_expert_run_dir),
            method=args.method,
            tasks=args.tasks,
        )
    )
    baseline_rows.extend(
        load_run_rows(
            label="baseline",
            run_dir=Path(args.baseline_end_to_end_run_dir),
            method=args.method,
            tasks=args.tasks,
        )
    )
    candidate_rows: List[Dict[str, Any]] = []
    candidate_rows.extend(
        load_run_rows(
            label="candidate",
            run_dir=Path(args.candidate_expert_run_dir),
            method=args.method,
            tasks=args.tasks,
        )
    )
    candidate_rows.extend(
        load_run_rows(
            label="candidate",
            run_dir=Path(args.candidate_end_to_end_run_dir),
            method=args.method,
            tasks=args.tasks,
        )
    )

    report = build_report(baseline_rows, candidate_rows)
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
