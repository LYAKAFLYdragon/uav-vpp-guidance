#!/usr/bin/env python3
"""Compare rear-quarter geometry factor pilot runs against a chosen baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUMMARY_KEYS = (
    "win_rate",
    "hp_advantage",
    "damaging_win_rate",
)

DIAGNOSTIC_KEYS = (
    "damage_margin",
    "post_merge_attack_zone_advantage_s",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_first_active_step",
    "post_merge_offensive_anchor_active_longitudinal_scale",
    "post_merge_offensive_anchor_active_lateral_scale",
    "post_merge_active_vp_longitudinal_scale",
    "post_merge_active_vp_lateral_scale",
    "post_merge_offensive_anchor_active_override_longitudinal_scale",
    "post_merge_offensive_anchor_active_override_lateral_scale",
    "post_merge_offensive_anchor_active_vp_forward_bias_m",
    "post_merge_offensive_anchor_active_vp_lateral_bias_m",
    "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m",
    "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction",
    "post_merge_offensive_anchor_lateral_world_offset_latch_first_active_step",
    "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction",
    "post_merge_offensive_anchor_blend_release_lateral_only_first_step",
    "post_merge_offensive_anchor_blend_release_recovery_active_fraction",
    "post_merge_offensive_anchor_blend_release_recovery_first_step",
    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m",
    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max",
    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend",
    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend",
    "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active_fraction",
    "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step",
    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction",
    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step",
    "offensive_anchor_frame",
    "offensive_anchor_encounter_stable_max_heading_delta_deg",
    "merge_min_range_m",
)

EXPERT_HEAD_ON_EQUIVALENCE_TOLERANCES = {
    "win_rate": 1e-9,
    "damage_margin": 1e-9,
    "post_merge_attack_zone_advantage_s": 1e-9,
    "post_merge_active_vp_longitudinal_scale": 1e-9,
    "post_merge_active_vp_lateral_scale": 1e-9,
    "post_merge_offensive_anchor_active_vp_forward_bias_m": 1e-6,
    "post_merge_offensive_anchor_active_vp_lateral_bias_m": 1e-6,
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": 1e-9,
}

CONFIG_DIFFERENCE_KEYS = (
    "offensive_anchor_frame",
    "offensive_anchor_encounter_stable_max_heading_delta_deg",
    "configured_post_merge_offensive_anchor_longitudinal_scale",
    "configured_post_merge_offensive_anchor_lateral_scale",
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


def _numbers_close(lhs: Any, rhs: Any, *, tol: float) -> bool:
    lhs_number = _safe_float(lhs)
    rhs_number = _safe_float(rhs)
    if math.isfinite(lhs_number) and math.isfinite(rhs_number):
        return abs(lhs_number - rhs_number) <= tol
    return not math.isfinite(lhs_number) and not math.isfinite(rhs_number)


def _row_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": row["label"],
        "opponent_stage": row["opponent_stage"],
        "task": row["task"],
        "win_rate": row.get("win_rate"),
        "damage_margin": row.get("damage_margin"),
        "post_merge_attack_zone_advantage_s": row.get("post_merge_attack_zone_advantage_s"),
        "offensive_anchor_frame": row.get("offensive_anchor_frame"),
        "offensive_anchor_encounter_stable_max_heading_delta_deg": row.get(
            "offensive_anchor_encounter_stable_max_heading_delta_deg"
        ),
        "configured_post_merge_offensive_anchor_longitudinal_scale": row.get(
            "configured_post_merge_offensive_anchor_longitudinal_scale"
        ),
        "configured_post_merge_offensive_anchor_lateral_scale": row.get(
            "configured_post_merge_offensive_anchor_lateral_scale"
        ),
        "post_merge_active_vp_longitudinal_scale": row.get(
            "post_merge_active_vp_longitudinal_scale"
        ),
        "post_merge_active_vp_lateral_scale": row.get(
            "post_merge_active_vp_lateral_scale"
        ),
        "post_merge_offensive_anchor_active_vp_forward_bias_m": row.get(
            "post_merge_offensive_anchor_active_vp_forward_bias_m"
        ),
        "post_merge_offensive_anchor_active_vp_lateral_bias_m": row.get(
            "post_merge_offensive_anchor_active_vp_lateral_bias_m"
        ),
        "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": row.get(
            "post_merge_offensive_anchor_blend_release_direct_track_active_fraction"
        ),
        "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction": row.get(
            "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction"
        ),
        "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction": row.get(
            "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_active_fraction": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_active_fraction"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_first_step": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_first_step"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active_fraction": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active_fraction"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction"
        ),
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step"
        ),
        "recovery_active_fraction": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_active_fraction"
        ),
        "recovery_first_step": row.get(
            "post_merge_offensive_anchor_blend_release_recovery_first_step"
        ),
        "configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": row.get(
            "configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
        ),
        "configured_post_merge_offensive_anchor_blend_release_lateral_only": row.get(
            "configured_post_merge_offensive_anchor_blend_release_lateral_only"
        ),
        "win_rate_delta_vs_baseline": row.get("win_rate_delta_vs_baseline"),
        "damage_margin_delta_vs_baseline": row.get("damage_margin_delta_vs_baseline"),
        "post_merge_attack_zone_advantage_s_delta_vs_baseline": row.get(
            "post_merge_attack_zone_advantage_s_delta_vs_baseline"
        ),
    }


def _config_differs(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    return any(
        not _numbers_close(lhs.get(key), rhs.get(key), tol=1e-9)
        if key != "offensive_anchor_frame"
        else str(lhs.get(key) or "") != str(rhs.get(key) or "")
        for key in CONFIG_DIFFERENCE_KEYS
    )


def _find_equivalent_expert_head_on_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    sorted_rows = sorted(rows, key=lambda row: str(row["label"]))
    for index, lhs in enumerate(sorted_rows):
        for rhs in sorted_rows[index + 1 :]:
            if not _config_differs(lhs, rhs):
                continue
            if all(
                _numbers_close(lhs.get(key), rhs.get(key), tol=tol)
                for key, tol in EXPERT_HEAD_ON_EQUIVALENCE_TOLERANCES.items()
            ):
                pairs.append(
                    {
                        "lhs": _row_summary(lhs),
                        "rhs": _row_summary(rhs),
                    }
                )
    return pairs


def _build_crossing_neutral_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_label: str,
) -> List[str]:
    labels = sorted({str(row["label"]) for row in rows if str(row["label"]) != baseline_label})
    neutral_labels: List[str] = []
    for label in labels:
        crossing_rows = [
            row
            for row in rows
            if str(row["label"]) == label and str(row["task"]) == "crossing_feasible"
        ]
        if not crossing_rows:
            continue
        if all(
            _numbers_close(row.get("win_rate_delta_vs_baseline"), 0.0, tol=1e-9)
            and _numbers_close(row.get("damage_margin_delta_vs_baseline"), 0.0, tol=1e-9)
            and _numbers_close(
                row.get("post_merge_attack_zone_advantage_s_delta_vs_baseline"),
                0.0,
                tol=1e-9,
            )
            for row in crossing_rows
        ):
            neutral_labels.append(label)
    return neutral_labels


def _build_executive_diagnosis(
    *,
    expert_head_on_rows: Sequence[Mapping[str, Any]],
    baseline_label: str,
    all_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    nonbaseline_rows = [
        row for row in expert_head_on_rows if str(row["label"]) != str(baseline_label)
    ]
    ranked_nonbaseline = sorted(
        nonbaseline_rows,
        key=lambda row: (
            -_safe_float(row.get("win_rate")),
            -_safe_float(row.get("damage_margin")),
            -_safe_float(row.get("post_merge_attack_zone_advantage_s")),
        ),
    )
    lowest_directtrack = sorted(
        nonbaseline_rows,
        key=lambda row: (
            _safe_float(
                row.get("post_merge_offensive_anchor_blend_release_direct_track_active_fraction")
            ),
            -_safe_float(row.get("damage_margin")),
            -_safe_float(row.get("post_merge_attack_zone_advantage_s")),
        ),
    )
    recovery_rows = [
        row
        for row in nonbaseline_rows
        if _safe_float(
            row.get("post_merge_offensive_anchor_blend_release_recovery_active_fraction")
        )
        > 0.0
    ]
    strongest_recovery = sorted(
        recovery_rows,
        key=lambda row: (
            -_safe_float(
                row.get(
                    "post_merge_offensive_anchor_blend_release_recovery_active_fraction"
                )
            ),
            _safe_float(
                row.get(
                    "post_merge_offensive_anchor_blend_release_recovery_first_step"
                )
            ),
            -_safe_float(row.get("damage_margin")),
        ),
    )
    return {
        "best_nonbaseline_expert_head_on": (
            _row_summary(ranked_nonbaseline[0]) if ranked_nonbaseline else None
        ),
        "lowest_directtrack_expert_head_on": (
            _row_summary(lowest_directtrack[0]) if lowest_directtrack else None
        ),
        "strongest_recovery_signal": (
            _row_summary(strongest_recovery[0]) if strongest_recovery else None
        ),
        "crossing_neutral_labels": _build_crossing_neutral_labels(
            all_rows,
            baseline_label=baseline_label,
        ),
        "geometry_equivalence_pairs": _find_equivalent_expert_head_on_pairs(
            nonbaseline_rows
        ),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return _load_json(manifest_path)


def _resolve_task_override(
    config_overrides: Sequence[Mapping[str, Any]],
    *,
    task: str,
    by_task_key: str,
    scalar_key: Optional[str] = None,
) -> Any:
    for item in reversed(list(config_overrides)):
        key = str(item.get("key"))
        if key == by_task_key:
            value = item.get("new_value")
            if isinstance(value, Mapping) and task in value:
                return value[task]
        if scalar_key is not None and key == scalar_key:
            return item.get("new_value")
    return None


def _select_single_row(
    payload: Mapping[str, Any],
    *,
    task: str,
    method: Optional[str],
) -> Dict[str, Any]:
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


def load_run_record(
    *,
    label: str,
    run_dir: Path,
    method: Optional[str],
    tasks: Sequence[str],
) -> List[Dict[str, Any]]:
    summary_payload = _load_json(run_dir / "aggregate" / "method_task_summary.json")
    diagnostic_payload = _load_json(run_dir / "aggregate" / "combat_geometry_diagnostics.json")
    manifest_payload = _load_run_manifest(run_dir)
    config_overrides = list(manifest_payload.get("config_overrides", []))
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        summary_row = _select_single_row(summary_payload, task=task, method=method)
        diagnostic_row = _select_single_row(diagnostic_payload, task=task, method=method)
        configured_anchor_frame = _resolve_task_override(
            config_overrides,
            task=task,
            by_task_key="virtual_point.offensive_anchor_frame_by_task",
            scalar_key="virtual_point.offensive_anchor_frame",
        )
        configured_anchor_stable_deg = _resolve_task_override(
            config_overrides,
            task=task,
            by_task_key=(
                "virtual_point."
                "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
            ),
            scalar_key=(
                "virtual_point."
                "offensive_anchor_encounter_stable_max_heading_delta_deg"
            ),
        )
        configured_anchor_long_scale = _resolve_task_override(
            config_overrides,
            task=task,
            by_task_key=(
                "virtual_point."
                "post_merge_offensive_anchor_longitudinal_scale_by_task"
            ),
            scalar_key="virtual_point.post_merge_offensive_anchor_longitudinal_scale",
        )
        configured_anchor_lat_scale = _resolve_task_override(
            config_overrides,
            task=task,
            by_task_key=(
                "virtual_point.post_merge_offensive_anchor_lateral_scale_by_task"
            ),
            scalar_key="virtual_point.post_merge_offensive_anchor_lateral_scale",
        )
        configured_latch_on_activation = _resolve_task_override(
            config_overrides,
            task=task,
            by_task_key=(
                "virtual_point."
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
            ),
            scalar_key=(
                "virtual_point."
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
            ),
        )
        configured_release_lateral_only = _resolve_task_override(
            config_overrides,
            task=task,
            by_task_key=(
                "virtual_point."
                "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
            ),
            scalar_key=(
                "virtual_point."
                "post_merge_offensive_anchor_blend_release_lateral_only"
            ),
        )
        merged = {
            "label": label,
            "run_dir": str(run_dir),
            "task": task,
            "opponent_stage": str(summary_row.get("opponent_stage")),
            "configured_offensive_anchor_frame": configured_anchor_frame,
            "configured_offensive_anchor_encounter_stable_max_heading_delta_deg": (
                configured_anchor_stable_deg
            ),
            "configured_post_merge_offensive_anchor_longitudinal_scale": (
                configured_anchor_long_scale
            ),
            "configured_post_merge_offensive_anchor_lateral_scale": (
                configured_anchor_lat_scale
            ),
            "configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": (
                configured_latch_on_activation
            ),
            "configured_post_merge_offensive_anchor_blend_release_lateral_only": (
                configured_release_lateral_only
            ),
        }
        for key in SUMMARY_KEYS:
            merged[key] = summary_row.get(key)
        for key in DIAGNOSTIC_KEYS:
            merged[key] = diagnostic_row.get(key)
        if merged.get("post_merge_active_vp_longitudinal_scale") is None:
            merged["post_merge_active_vp_longitudinal_scale"] = merged.get(
                "post_merge_offensive_anchor_active_longitudinal_scale"
            )
        if merged.get("post_merge_active_vp_lateral_scale") is None:
            merged["post_merge_active_vp_lateral_scale"] = merged.get(
                "post_merge_offensive_anchor_active_lateral_scale"
            )
        if merged.get("offensive_anchor_frame") is None:
            merged["offensive_anchor_frame"] = configured_anchor_frame
        if merged.get("offensive_anchor_encounter_stable_max_heading_delta_deg") is None:
            merged["offensive_anchor_encounter_stable_max_heading_delta_deg"] = (
                configured_anchor_stable_deg
            )
        if (
            merged.get("post_merge_offensive_anchor_active_override_longitudinal_scale")
            is None
        ):
            merged["post_merge_offensive_anchor_active_override_longitudinal_scale"] = (
                configured_anchor_long_scale
            )
        if (
            merged.get("post_merge_offensive_anchor_active_override_lateral_scale")
            is None
        ):
            merged["post_merge_offensive_anchor_active_override_lateral_scale"] = (
                configured_anchor_lat_scale
            )
        rows.append(merged)
    return rows


def _index_by_label_split_task(
    rows: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {
        (str(row["label"]), str(row["opponent_stage"]), str(row["task"])): dict(row)
        for row in rows
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_label: str,
) -> Dict[str, Any]:
    indexed = _index_by_label_split_task(rows)
    enriched_rows: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        baseline_key = (baseline_label, str(row["opponent_stage"]), str(row["task"]))
        baseline_row = indexed.get(baseline_key)
        if baseline_row is not None:
            enriched["baseline_label"] = baseline_label
            for key in ("win_rate", "damage_margin", "post_merge_attack_zone_advantage_s"):
                enriched[f"{key}_delta_vs_baseline"] = (
                    _safe_float(row.get(key)) - _safe_float(baseline_row.get(key))
                )
        enriched_rows.append(enriched)

    expert_head_on = [
        row for row in enriched_rows if row["opponent_stage"] == "expert" and row["task"] == "head_on"
    ]
    sorted_expert_head_on = sorted(
        expert_head_on,
        key=lambda row: (
            -_safe_float(row.get("win_rate")),
            -_safe_float(row.get("damage_margin")),
            -_safe_float(row.get("post_merge_attack_zone_advantage_s")),
        ),
    )
    executive_diagnosis = _build_executive_diagnosis(
        expert_head_on_rows=expert_head_on,
        baseline_label=baseline_label,
        all_rows=enriched_rows,
    )
    return {
        "baseline_label": baseline_label,
        "rows": enriched_rows,
        "executive_diagnosis": executive_diagnosis,
        "expert_head_on_ranking": [
            {
                "label": row["label"],
                "win_rate": row["win_rate"],
                "damage_margin": row["damage_margin"],
                "post_merge_attack_zone_advantage_s": row["post_merge_attack_zone_advantage_s"],
                "offensive_anchor_frame": row.get("offensive_anchor_frame"),
                "offensive_anchor_encounter_stable_max_heading_delta_deg": row.get(
                    "offensive_anchor_encounter_stable_max_heading_delta_deg"
                ),
                "configured_post_merge_offensive_anchor_longitudinal_scale": row.get(
                    "configured_post_merge_offensive_anchor_longitudinal_scale"
                ),
                "configured_post_merge_offensive_anchor_lateral_scale": row.get(
                    "configured_post_merge_offensive_anchor_lateral_scale"
                ),
                "post_merge_active_vp_longitudinal_scale": row.get(
                    "post_merge_active_vp_longitudinal_scale"
                ),
                "post_merge_active_vp_lateral_scale": row.get(
                    "post_merge_active_vp_lateral_scale"
                ),
                "post_merge_offensive_anchor_active_vp_forward_bias_m": row.get(
                    "post_merge_offensive_anchor_active_vp_forward_bias_m"
                ),
                "post_merge_offensive_anchor_active_vp_lateral_bias_m": row.get(
                    "post_merge_offensive_anchor_active_vp_lateral_bias_m"
                ),
                "configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": row.get(
                    "configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
                ),
                "configured_post_merge_offensive_anchor_blend_release_lateral_only": row.get(
                    "configured_post_merge_offensive_anchor_blend_release_lateral_only"
                ),
                "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction": row.get(
                    "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction"
                ),
                "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction": row.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_active_fraction": row.get(
                    "post_merge_offensive_anchor_blend_release_recovery_active_fraction"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_first_step": row.get(
                    "post_merge_offensive_anchor_blend_release_recovery_first_step"
                ),
            }
            for row in sorted_expert_head_on
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    diagnosis = report.get("executive_diagnosis", {})
    best_nonbaseline = diagnosis.get("best_nonbaseline_expert_head_on")
    lowest_directtrack = diagnosis.get("lowest_directtrack_expert_head_on")
    strongest_recovery = diagnosis.get("strongest_recovery_signal")
    crossing_neutral_labels = diagnosis.get("crossing_neutral_labels", [])
    equivalence_pairs = diagnosis.get("geometry_equivalence_pairs", [])
    lines = [
        "# Rear-Quarter Geometry Factor Pilot",
        "",
        f"- baseline_label: `{report['baseline_label']}`",
        "",
        "## Diagnosis",
        "",
    ]

    if best_nonbaseline is not None:
        lines.append(
            (
                f"- best nearby `expert/head_on` variant is `{best_nonbaseline['label']}`: "
                f"win_rate={_format_float(best_nonbaseline.get('win_rate'))}, "
                f"damage_margin={_format_float(best_nonbaseline.get('damage_margin'))}, "
                f"post_merge_adv_s={_format_float(best_nonbaseline.get('post_merge_attack_zone_advantage_s'))}, "
                f"directtrack_frac={_format_float(best_nonbaseline.get('post_merge_offensive_anchor_blend_release_direct_track_active_fraction'))}"
            )
        )
    if lowest_directtrack is not None:
        lines.append(
            (
                f"- strongest direct-track suppression in `expert/head_on` is `{lowest_directtrack['label']}`: "
                f"directtrack_frac={_format_float(lowest_directtrack.get('post_merge_offensive_anchor_blend_release_direct_track_active_fraction'))}, "
                f"damage_margin={_format_float(lowest_directtrack.get('damage_margin'))}, "
                f"post_merge_adv_s={_format_float(lowest_directtrack.get('post_merge_attack_zone_advantage_s'))}"
            )
        )
    if strongest_recovery is not None:
        lines.append(
            (
                f"- strongest recovery-stage signal is `{strongest_recovery['label']}` "
                f"on `{strongest_recovery['opponent_stage']}/{strongest_recovery['task']}`: "
                f"recovery_frac={_format_float(strongest_recovery.get('post_merge_offensive_anchor_blend_release_recovery_active_fraction'))}, "
                f"recovery_step={_format_float(strongest_recovery.get('post_merge_offensive_anchor_blend_release_recovery_first_step'))}, "
                f"alt_trigger_step={_format_float(strongest_recovery.get('post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step'))}, "
                f"forward_trigger_step={_format_float(strongest_recovery.get('post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step'))}"
            )
        )
    lateral_release_rows = [
        row
        for row in report["rows"]
        if row["task"] == "head_on"
        and _safe_float(
            row.get("post_merge_offensive_anchor_blend_release_lateral_only_active_fraction")
        )
        > 0.0
    ]
    if lateral_release_rows:
        strongest_lateral_release = max(
            lateral_release_rows,
            key=lambda row: (
                _safe_float(
                    row.get(
                        "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction"
                    )
                ),
                _safe_float(
                    row.get(
                        "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction"
                    )
                ),
                _safe_float(row.get("damage_margin")),
            ),
        )
        lines.append(
            (
                f"- strongest lateral-body persistence signal is `{strongest_lateral_release['label']}` "
                f"on `{strongest_lateral_release['opponent_stage']}/head_on`: "
                f"latch_cfg={strongest_lateral_release.get('configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation')}, "
                f"release_lateral_only_cfg={strongest_lateral_release.get('configured_post_merge_offensive_anchor_blend_release_lateral_only')}, "
                f"latch_frac={_format_float(strongest_lateral_release.get('post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction'))}, "
                f"lateral_only_frac={_format_float(strongest_lateral_release.get('post_merge_offensive_anchor_blend_release_lateral_only_active_fraction'))}"
            )
        )
    if crossing_neutral_labels:
        labels_joined = ", ".join(f"`{label}`" for label in crossing_neutral_labels)
        lines.append(
            f"- crossing stays baseline-neutral for: {labels_joined}"
        )
    if equivalence_pairs:
        for pair in equivalence_pairs:
            lhs = pair["lhs"]
            rhs = pair["rhs"]
            lines.append(
                (
                    f"- realized-geometry equivalent despite different rear-quarter config: "
                    f"`{lhs['label']}` vs `{rhs['label']}` -> "
                    f"win_rate={_format_float(lhs.get('win_rate'))}, "
                    f"damage_margin={_format_float(lhs.get('damage_margin'))}, "
                    f"post_merge_adv_s={_format_float(lhs.get('post_merge_attack_zone_advantage_s'))}, "
                    f"rear_active_scales=({_format_float(lhs.get('post_merge_active_vp_longitudinal_scale'))}, "
                    f"{_format_float(lhs.get('post_merge_active_vp_lateral_scale'))})"
                )
            )

    lines.extend(
        [
            "",
        "## Variant Matrix",
        "",
        "| variant | split | task | win_rate | damage_margin | delta_win | delta_damage | post_merge_adv_s | rear_active_frac | rear_active_step | rear_frame | rear_stable_deg | rear_anchor_cfg_long_scale | rear_anchor_cfg_lat_scale | rear_latch_cfg | rear_lateralonly_cfg | rear_latch_frac | rear_lateralonly_frac | recovery_frac | recovery_step | recovery_alt_step | recovery_forward_step | recovery_forward_thresh | rear_active_vp_long_scale | rear_active_vp_lat_scale | rear_active_vp_forward_m | rear_active_vp_lateral_m | predscale_active_vp_forward_m | directtrack_frac | merge_min_range_m |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    sorted_rows = sorted(
        report["rows"],
        key=lambda row: (str(row["label"]), str(row["opponent_stage"]), str(row["task"])),
    )
    for row in sorted_rows:
        lines.append(
            (
                f"| {row['label']} | {row['opponent_stage']} | {row['task']} | "
                f"{_format_float(row.get('win_rate'))} | "
                f"{_format_float(row.get('damage_margin'))} | "
                f"{_format_float(row.get('win_rate_delta_vs_baseline'))} | "
                f"{_format_float(row.get('damage_margin_delta_vs_baseline'))} | "
                f"{_format_float(row.get('post_merge_attack_zone_advantage_s'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_first_active_step'))} | "
                f"{row.get('offensive_anchor_frame') or 'nan'} | "
                f"{_format_float(row.get('offensive_anchor_encounter_stable_max_heading_delta_deg'))} | "
                f"{_format_float(row.get('configured_post_merge_offensive_anchor_longitudinal_scale'))} | "
                f"{_format_float(row.get('configured_post_merge_offensive_anchor_lateral_scale'))} | "
                f"{row.get('configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation')} | "
                f"{row.get('configured_post_merge_offensive_anchor_blend_release_lateral_only')} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_lateral_only_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_first_step'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max'))} | "
                f"{_format_float(row.get('post_merge_active_vp_longitudinal_scale'))} | "
                f"{_format_float(row.get('post_merge_active_vp_lateral_scale'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_active_vp_forward_bias_m'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_active_vp_lateral_bias_m'))} | "
                f"{_format_float(row.get('post_merge_predicted_target_forward_scale_active_vp_forward_bias_m'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_direct_track_active_fraction'))} | "
                f"{_format_float(row.get('merge_min_range_m'))} |"
            )
        )

    lines.extend(
        [
            "",
            "## Expert Head-On Ranking",
            "",
            "| rank | variant | win_rate | damage_margin | post_merge_adv_s | rear_frame | rear_stable_deg | rear_anchor_cfg_long_scale | rear_anchor_cfg_lat_scale | rear_latch_cfg | rear_lateralonly_cfg | rear_latch_frac | rear_lateralonly_frac | recovery_frac | recovery_step | rear_active_vp_long_scale | rear_active_vp_lat_scale | rear_active_vp_forward_m | rear_active_vp_lateral_m |",
            "|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(report["expert_head_on_ranking"], start=1):
        lines.append(
            (
                f"| {idx} | {row['label']} | {_format_float(row.get('win_rate'))} | "
                f"{_format_float(row.get('damage_margin'))} | "
                f"{_format_float(row.get('post_merge_attack_zone_advantage_s'))} | "
                f"{row.get('offensive_anchor_frame') or 'nan'} | "
                f"{_format_float(row.get('offensive_anchor_encounter_stable_max_heading_delta_deg'))} | "
                f"{_format_float(row.get('configured_post_merge_offensive_anchor_longitudinal_scale'))} | "
                f"{_format_float(row.get('configured_post_merge_offensive_anchor_lateral_scale'))} | "
                f"{row.get('configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation')} | "
                f"{row.get('configured_post_merge_offensive_anchor_blend_release_lateral_only')} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_lateral_only_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_active_fraction'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_blend_release_recovery_first_step'))} | "
                f"{_format_float(row.get('post_merge_active_vp_longitudinal_scale'))} | "
                f"{_format_float(row.get('post_merge_active_vp_lateral_scale'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_active_vp_forward_bias_m'))} | "
                f"{_format_float(row.get('post_merge_offensive_anchor_active_vp_lateral_bias_m'))} |"
            )
        )
    return "\n".join(lines) + "\n"


def _parse_run_specs(items: Iterable[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for item in items:
        entry = str(item).strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError("--run entries must be label=run_dir pairs")
        label, path_raw = (part.strip() for part in entry.split("=", 1))
        if not label:
            raise ValueError("--run entries must include a non-empty label")
        specs.append((label, Path(path_raw)))
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--tasks", nargs="+", default=["head_on", "crossing_feasible"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    specs = _parse_run_specs(args.run)
    rows: List[Dict[str, Any]] = []
    for label, run_dir in specs:
        rows.extend(
            load_run_record(
                label=label,
                run_dir=run_dir,
                method=args.method,
                tasks=args.tasks,
            )
        )

    report = build_report(rows, baseline_label=str(args.baseline_label))
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
