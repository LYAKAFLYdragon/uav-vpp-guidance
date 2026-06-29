#!/usr/bin/env python3
"""Merge multiple formal summary payloads into one top-level engagement gate report."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from analyze_formal_engagement_gate import analyze_engagement_gate, render_markdown

REQUIRED_COMBAT_TASKS = ("head_on", "crossing_feasible")
PREDICTION_COMPARISON_METHODS = ("no_prediction_vpp", "prediction_vpp")
PREDICTION_COMPARISON_OPPONENTS = ("expert", "end_to_end")
MODE_PRIORITY = {
    "formal": 4,
    "adversarial": 3,
    "hp_pilot": 2,
    "formal-small": 1,
}
OPPONENT_PRIORITY = {
    "end_to_end": 3,
    "expert": 2,
    "curriculum": 1,
    "none": 0,
}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _iter_rows(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if isinstance(payload.get("summary"), dict):
        for key, item in sorted(payload["summary"].items()):
            if not isinstance(item, dict):
                continue
            row = copy.deepcopy(item)
            row.setdefault("group", str(key))
            yield row
        return

    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            row = copy.deepcopy(item)
            row.setdefault(
                "group",
                "::".join(str(row.get(part, "unknown")) for part in ("task", "method", "opponent_stage"))
                or str(idx),
            )
            yield row
        return

    for key, item in sorted(payload.items()):
        if not isinstance(item, dict):
            continue
        row = copy.deepcopy(item)
        row.setdefault("group", str(key))
        yield row


def merge_summary_payloads(summary_paths: Iterable[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    for path in summary_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sources.append(str(path))
        for row in _iter_rows(payload):
            row.setdefault("source_summary", str(path))
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("task", "")),
            str(row.get("mode", row.get("method", ""))),
            str(row.get("opponent_stage", "")),
            str(row.get("damage_per_step", "")),
            str(row.get("close_range_max_km", "")),
            str(row.get("close_range_max_aoa_deg", "")),
            str(row.get("group", "")),
        )
    )
    return {
        "sources": sources,
        "rows": rows,
    }


def filter_merged_payload(
    merged_payload: Dict[str, Any],
    *,
    include_methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    method_filter = {str(item) for item in (include_methods or []) if str(item)}
    if not method_filter:
        return copy.deepcopy(merged_payload)

    rows = []
    for row in merged_payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("method", "")) not in method_filter:
            continue
        rows.append(copy.deepcopy(row))

    filtered = {
        "sources": copy.deepcopy(merged_payload.get("sources", [])),
        "rows": rows,
        "filters": {
            "include_methods": sorted(method_filter),
        },
    }
    if "executive_summary" in merged_payload:
        filtered["executive_summary"] = copy.deepcopy(merged_payload["executive_summary"])
    return filtered


def _damage_margin(row: Dict[str, Any]) -> float:
    dealt = _safe_float(row.get("mean_damage_dealt"))
    taken = _safe_float(row.get("mean_damage_taken"))
    if math.isfinite(dealt) and math.isfinite(taken):
        return dealt - taken
    return float("nan")


def _candidate_selection_key(candidate: Dict[str, Any]) -> tuple:
    return (
        int(bool(candidate.get("ready_for_scale"))),
        MODE_PRIORITY.get(str(candidate.get("mode", "")), 0),
        OPPONENT_PRIORITY.get(str(candidate.get("opponent_stage", "")), 0),
        int(candidate.get("episodes", 0)),
        _safe_float(candidate.get("damaging_win_rate")),
        _safe_float(candidate.get("effective_engagement_rate")),
        _safe_float(candidate.get("win_rate")),
        _damage_margin(candidate),
    )


def _prediction_comparison_selection_key(candidate: Dict[str, Any]) -> tuple:
    mode = str(candidate.get("run_status") or candidate.get("mode") or "")
    return (
        int(mode == "formal"),
        MODE_PRIORITY.get(mode, 0),
        int(candidate.get("episodes", 0)),
        _safe_float(candidate.get("win_rate")),
        _safe_float(candidate.get("damaging_win_rate")),
        _damage_margin(candidate),
    )


def _coerce_group_value(value: str) -> Any:
    if value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _group_config_parts(group: Any) -> Dict[str, Any]:
    if not isinstance(group, str):
        return {}
    parts = group.split("::")
    if len(parts) < 3:
        return {}
    parsed: Dict[str, Any] = {}
    for prefix, key in (("d", "damage_per_step"), ("r", "close_range_max_km"), ("a", "close_range_max_aoa_deg")):
        for part in parts:
            if part.startswith(prefix) and len(part) > 1:
                parsed[key] = _coerce_group_value(part[1:])
    return parsed


def _config_view(candidate: Dict[str, Any]) -> Dict[str, Any]:
    parsed = _group_config_parts(candidate.get("group"))
    return {
        "group": candidate.get("group"),
        "method": candidate.get("method"),
        "mode": candidate.get("mode"),
        "run_status": candidate.get("run_status"),
        "opponent_stage": candidate.get("opponent_stage"),
        "damage_per_step": candidate.get("damage_per_step", parsed.get("damage_per_step")) or parsed.get("damage_per_step"),
        "close_range_max_km": candidate.get("close_range_max_km", parsed.get("close_range_max_km")) or parsed.get("close_range_max_km"),
        "close_range_max_aoa_deg": candidate.get("close_range_max_aoa_deg", parsed.get("close_range_max_aoa_deg")) or parsed.get("close_range_max_aoa_deg"),
        "source_summary": candidate.get("source_summary"),
    }


def build_executive_summary(
    merged_payload: Dict[str, Any],
    gate_report: Dict[str, Any],
) -> Dict[str, Any]:
    task_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    groups = gate_report.get("groups", {})
    for row in merged_payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", ""))
        gate_item = groups.get(group, {})
        candidate = copy.deepcopy(row)
        candidate.update(copy.deepcopy(gate_item))
        task = str(candidate.get("task", "unknown"))
        task_candidates[task].append(candidate)

    tasks_summary: Dict[str, Any] = {}
    for task, candidates in sorted(task_candidates.items()):
        best = max(candidates, key=_candidate_selection_key)
        ready_groups = [item.get("group") for item in candidates if item.get("ready_for_scale")]
        next_action = str(best.get("recommended_action") or "hold")
        if bool(best.get("ready_for_scale")):
            next_action = (
                "formal_expand_with_explicit_aoa60"
                if task == "crossing_feasible"
                else "formal_expand"
            )
        tasks_summary[task] = {
            "task": task,
            "ready_for_scale": bool(best.get("ready_for_scale")),
            "best_config": _config_view(best),
            "best_metrics": {
                "episodes": int(best.get("episodes", 0)),
                "win_rate": _safe_float(best.get("win_rate")),
                "effective_engagement_rate": _safe_float(best.get("effective_engagement_rate")),
                "damaging_win_rate": _safe_float(best.get("damaging_win_rate")),
                "mean_damage_dealt": _safe_float(best.get("mean_damage_dealt")),
                "mean_damage_taken": _safe_float(best.get("mean_damage_taken")),
                "ego_crashes": int(best.get("ego_crashes", 0)),
                "target_crash_or_oob": int(best.get("target_crash_or_oob", 0)),
                "timeouts": int(best.get("timeouts", 0)),
                "backend_fallbacks": int(best.get("backend_fallbacks", 0)),
            },
            "candidate_count": len(candidates),
            "ready_groups": ready_groups,
            "next_action": next_action,
        }

    missing_required = [task for task in REQUIRED_COMBAT_TASKS if task not in tasks_summary]
    not_ready_required = [
        task for task in REQUIRED_COMBAT_TASKS
        if task in tasks_summary and not tasks_summary[task]["ready_for_scale"]
    ]
    can_run_large_scale = not missing_required and not not_ready_required

    overall = {
        "required_combat_tasks": list(REQUIRED_COMBAT_TASKS),
        "missing_required_tasks": missing_required,
        "not_ready_required_tasks": not_ready_required,
        "can_run_large_scale_air_combat_experiments": can_run_large_scale,
        "scope": "combat_only_head_on_crossing",
        "recommended_action": (
            "proceed_large_scale_combat_experiment"
            if can_run_large_scale
            else "hold_until_required_combat_tasks_are_ready"
        ),
        "notes": [
            "Assessment is limited to the combat tasks present in the merged formal summaries.",
            "Non-combat or route-tracking tasks such as sustained_turn or multi_waypoint are outside this large-scale air-combat readiness scope.",
        ],
    }
    filters = merged_payload.get("filters")
    if isinstance(filters, dict) and filters:
        overall["filters"] = copy.deepcopy(filters)
    return {
        "tasks": tasks_summary,
        "overall": overall,
    }


def build_prediction_comparison_summary(
    merged_payload: Dict[str, Any],
    *,
    baseline_method: str = "no_prediction_vpp",
    methods: Sequence[str] = PREDICTION_COMPARISON_METHODS,
    tasks: Sequence[str] = REQUIRED_COMBAT_TASKS,
    opponent_stages: Sequence[str] = PREDICTION_COMPARISON_OPPONENTS,
) -> Dict[str, Any]:
    selected_rows: Dict[tuple, Dict[str, Any]] = {}
    methods_set = {str(method) for method in methods}
    tasks_set = {str(task) for task in tasks}
    opponent_set = {str(stage) for stage in opponent_stages}

    for row in merged_payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        task = str(row.get("task", ""))
        opponent_stage = str(row.get("opponent_stage", ""))
        method = str(row.get("method", ""))
        if task not in tasks_set or opponent_stage not in opponent_set or method not in methods_set:
            continue
        key = (task, opponent_stage, method)
        current = selected_rows.get(key)
        if current is None or _prediction_comparison_selection_key(row) > _prediction_comparison_selection_key(current):
            selected_rows[key] = copy.deepcopy(row)

    rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, str]] = []
    for task in tasks:
        for opponent_stage in opponent_stages:
            baseline = selected_rows.get((task, opponent_stage, baseline_method))
            baseline_win_rate = _safe_float(baseline.get("win_rate")) if baseline else float("nan")
            baseline_damaging_win_rate = (
                _safe_float(baseline.get("damaging_win_rate")) if baseline else float("nan")
            )
            baseline_damage_margin = _damage_margin(baseline) if baseline else float("nan")
            for method in methods:
                row = selected_rows.get((task, opponent_stage, method))
                if row is None:
                    missing_rows.append(
                        {
                            "task": str(task),
                            "opponent_stage": str(opponent_stage),
                            "method": str(method),
                        }
                    )
                    continue
                damage_margin = _damage_margin(row)
                rows.append(
                    {
                        "task": str(task),
                        "opponent_stage": str(opponent_stage),
                        "method": str(method),
                        "group": row.get("group"),
                        "mode": row.get("mode"),
                        "run_status": row.get("run_status"),
                        "episodes": int(row.get("episodes", 0)),
                        "win_rate": _safe_float(row.get("win_rate")),
                        "effective_engagement_rate": _safe_float(row.get("effective_engagement_rate")),
                        "damaging_win_rate": _safe_float(row.get("damaging_win_rate")),
                        "mean_damage_dealt": _safe_float(row.get("mean_damage_dealt")),
                        "mean_damage_taken": _safe_float(row.get("mean_damage_taken")),
                        "damage_margin": damage_margin,
                        "ego_crashes": int(row.get("ego_crashes", 0)),
                        "target_crash_or_oob": int(row.get("target_crash_or_oob", 0)),
                        "delta_win_rate_vs_no_prediction": _safe_float(row.get("win_rate")) - baseline_win_rate,
                        "delta_damaging_win_rate_vs_no_prediction": (
                            _safe_float(row.get("damaging_win_rate")) - baseline_damaging_win_rate
                        ),
                        "delta_damage_margin_vs_no_prediction": damage_margin - baseline_damage_margin,
                    }
                )

    return {
        "baseline_method": baseline_method,
        "methods": [str(method) for method in methods],
        "tasks": [str(task) for task in tasks],
        "opponent_stages": [str(stage) for stage in opponent_stages],
        "rows": rows,
        "missing_rows": missing_rows,
    }


def render_executive_markdown(executive_summary: Dict[str, Any], summary_path: Path) -> str:
    overall = executive_summary["overall"]
    lines = [
        "# Formal Executive Summary",
        "",
        f"Source: `{summary_path}`",
        "",
        "## Overall",
        "",
        f"- can_run_large_scale_air_combat_experiments: {overall['can_run_large_scale_air_combat_experiments']}",
        f"- scope: {overall['scope']}",
        f"- recommended_action: {overall['recommended_action']}",
        f"- missing_required_tasks: {','.join(overall['missing_required_tasks']) or '-'}",
        f"- not_ready_required_tasks: {','.join(overall['not_ready_required_tasks']) or '-'}",
        f"- filters: {json.dumps(overall.get('filters', {}), ensure_ascii=False) if overall.get('filters') else '-'}",
        "",
        "## Tasks",
        "",
        "| task | ready | best_config | win | effective | damaging | damage_dealt | damage_taken | ego/target/timeout | next_action |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task, item in executive_summary["tasks"].items():
        cfg = item["best_config"]
        metrics = item["best_metrics"]
        config_kind = cfg.get("method") or cfg.get("mode")
        config_label = (
            f"{config_kind}::{cfg.get('opponent_stage')}"
            f"::d{cfg.get('damage_per_step')}"
            f"::r{cfg.get('close_range_max_km')}"
            f"::a{cfg.get('close_range_max_aoa_deg')}"
        )
        lines.append(
            "| {task} | {ready_for_scale} | {config_label} | {win_rate:.3g} | "
            "{effective_engagement_rate:.3g} | {damaging_win_rate:.3g} | "
            "{mean_damage_dealt:.3g} | {mean_damage_taken:.3g} | "
            "{ego_crashes}/{target_crash_or_oob}/{timeouts} | {next_action} |".format(
                task=task,
                config_label=config_label,
                next_action=item["next_action"],
                ready_for_scale=item["ready_for_scale"],
                **metrics,
            )
        )
    return "\n".join(lines) + "\n"


def render_prediction_comparison_markdown(
    prediction_comparison: Dict[str, Any],
    summary_path: Path,
) -> str:
    lines = [
        "# Prediction Comparison Summary",
        "",
        f"Source: `{summary_path}`",
        "",
        "## Scope",
        "",
        f"- baseline_method: {prediction_comparison['baseline_method']}",
        f"- methods: {','.join(prediction_comparison['methods'])}",
        f"- tasks: {','.join(prediction_comparison['tasks'])}",
        f"- opponent_stages: {','.join(prediction_comparison['opponent_stages'])}",
        f"- missing_rows: {json.dumps(prediction_comparison['missing_rows'], ensure_ascii=False) if prediction_comparison['missing_rows'] else '-'}",
        "",
        "## Rows",
        "",
        "| task | opponent_stage | method | win_rate | effective_engagement | damaging_win | damage_dealt | damage_taken | damage_margin | ego_crashes | target_crash_or_oob | delta_win_vs_no_pred | delta_damaging_vs_no_pred | delta_margin_vs_no_pred |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prediction_comparison["rows"]:
        lines.append(
            "| {task} | {opponent_stage} | {method} | {win_rate:.3g} | {effective_engagement_rate:.3g} | "
            "{damaging_win_rate:.3g} | {mean_damage_dealt:.3g} | {mean_damage_taken:.3g} | {damage_margin:.3g} | "
            "{ego_crashes} | {target_crash_or_oob} | {delta_win_rate_vs_no_prediction:.3g} | "
            "{delta_damaging_win_rate_vs_no_prediction:.3g} | {delta_damage_margin_vs_no_prediction:.3g} |".format(
                **row,
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", nargs="+", required=True, help="Input formal summary JSON paths")
    parser.add_argument("--output-json", required=True, help="Merged top-level summary JSON path")
    parser.add_argument("--output-gate-json", required=True, help="Merged gate JSON path")
    parser.add_argument("--output-gate-md", required=True, help="Merged gate markdown path")
    parser.add_argument("--output-executive-json", help="Optional executive summary JSON path")
    parser.add_argument("--output-executive-md", help="Optional executive summary markdown path")
    parser.add_argument(
        "--output-prediction-comparison-json",
        help="Optional dedicated prediction-vs-no-prediction comparison JSON path.",
    )
    parser.add_argument(
        "--output-prediction-comparison-md",
        help="Optional dedicated prediction-vs-no-prediction comparison markdown path.",
    )
    parser.add_argument(
        "--include-method",
        action="append",
        default=[],
        help="Optional method filter. Repeat to keep only selected methods in the merged/gate/executive outputs.",
    )
    parser.add_argument("--min-win-rate", type=float, default=0.5)
    parser.add_argument("--min-effective-engagement-rate", type=float, default=0.5)
    parser.add_argument("--min-damaging-win-rate", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_paths = [Path(path) for path in args.summary_json]
    merged = merge_summary_payloads(summary_paths)
    merged = filter_merged_payload(merged, include_methods=args.include_method)
    report = analyze_engagement_gate(
        merged,
        min_win_rate=args.min_win_rate,
        min_effective_engagement_rate=args.min_effective_engagement_rate,
        min_damaging_win_rate=args.min_damaging_win_rate,
    )
    executive_summary = build_executive_summary(merged, report)
    merged["executive_summary"] = executive_summary

    output_json = Path(args.output_json)
    output_gate_json = Path(args.output_gate_json)
    output_gate_md = Path(args.output_gate_md)
    output_executive_json = (
        Path(args.output_executive_json)
        if args.output_executive_json
        else output_json.with_name(output_json.stem + "_executive_summary.json")
    )
    output_executive_md = (
        Path(args.output_executive_md)
        if args.output_executive_md
        else output_json.with_name(output_json.stem + "_executive_summary.md")
    )
    output_prediction_comparison_json = (
        Path(args.output_prediction_comparison_json)
        if args.output_prediction_comparison_json
        else None
    )
    output_prediction_comparison_md = (
        Path(args.output_prediction_comparison_md)
        if args.output_prediction_comparison_md
        else None
    )
    prediction_comparison = None
    if output_prediction_comparison_json or output_prediction_comparison_md:
        prediction_comparison = build_prediction_comparison_summary(merged)
        if output_prediction_comparison_json is None:
            output_prediction_comparison_json = output_json.with_name(
                output_json.stem + "_prediction_comparison.json"
            )
        if output_prediction_comparison_md is None:
            output_prediction_comparison_md = output_prediction_comparison_json.with_suffix(".md")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_gate_json.parent.mkdir(parents=True, exist_ok=True)
    output_gate_md.parent.mkdir(parents=True, exist_ok=True)
    output_executive_json.parent.mkdir(parents=True, exist_ok=True)
    output_executive_md.parent.mkdir(parents=True, exist_ok=True)
    if output_prediction_comparison_json is not None:
        output_prediction_comparison_json.parent.mkdir(parents=True, exist_ok=True)
    if output_prediction_comparison_md is not None:
        output_prediction_comparison_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_gate_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_gate_md.write_text(render_markdown(report, output_json), encoding="utf-8")
    output_executive_json.write_text(
        json.dumps(executive_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_executive_md.write_text(
        render_executive_markdown(executive_summary, output_json),
        encoding="utf-8",
    )
    if prediction_comparison is not None:
        output_prediction_comparison_json.write_text(
            json.dumps(prediction_comparison, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_prediction_comparison_md.write_text(
            render_prediction_comparison_markdown(prediction_comparison, output_json),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "gate_report": report,
                "executive_summary": executive_summary,
                "prediction_comparison": prediction_comparison,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
