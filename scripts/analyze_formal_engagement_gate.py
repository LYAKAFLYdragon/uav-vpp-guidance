#!/usr/bin/env python3
"""Analyze formal combat summaries with engagement-quality gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def analyze_engagement_gate(
    summary_payload: Dict[str, Any],
    *,
    min_win_rate: float = 0.5,
    min_effective_engagement_rate: float = 0.5,
    min_damaging_win_rate: float = 0.25,
) -> Dict[str, Any]:
    summary_map = summary_payload.get("summary", summary_payload)
    groups: Dict[str, Any] = {}

    for key, item in sorted(summary_map.items()):
        task = str(item.get("task", "unknown"))
        mode = str(item.get("mode", "unknown"))
        opponent_stage = str(item.get("opponent_stage", "unknown"))
        win_rate = _safe_float(item.get("win_rate"))
        effective_engagement_rate = _safe_float(item.get("effective_engagement_rate"))
        damaging_win_rate = _safe_float(item.get("damaging_win_rate"))
        damage_exchange_rate = _safe_float(item.get("damage_exchange_rate"))

        combat_ready = math.isfinite(win_rate) and win_rate >= min_win_rate
        engagement_ready = (
            math.isfinite(effective_engagement_rate)
            and effective_engagement_rate >= min_effective_engagement_rate
            and math.isfinite(damaging_win_rate)
            and damaging_win_rate >= min_damaging_win_rate
        )

        issues = []
        if not combat_ready:
            issues.append("win_rate_below_gate")
        if not math.isfinite(effective_engagement_rate):
            issues.append("effective_engagement_rate_missing")
        elif effective_engagement_rate < min_effective_engagement_rate:
            issues.append("effective_engagement_rate_below_gate")
        if not math.isfinite(damaging_win_rate):
            issues.append("damaging_win_rate_missing")
        elif damaging_win_rate < min_damaging_win_rate:
            issues.append("damaging_win_rate_below_gate")
        if math.isfinite(damage_exchange_rate) and damage_exchange_rate <= 0.0:
            issues.append("no_damage_exchange")

        recommended_action = "hold"
        if combat_ready and engagement_ready:
            recommended_action = "formal_expand"
        elif task == "crossing_feasible" and combat_ready and not engagement_ready:
            recommended_action = "hold_fix_engagement_quality"
        elif task == "head_on" and not combat_ready:
            recommended_action = "hold_fix_combat_outcome"

        groups[key] = {
            "task": task,
            "mode": mode,
            "opponent_stage": opponent_stage,
            "episodes": int(item.get("episodes", 0)),
            "win_rate": win_rate,
            "effective_engagement_rate": effective_engagement_rate,
            "damaging_win_rate": damaging_win_rate,
            "damage_exchange_rate": damage_exchange_rate,
            "combat_ready": combat_ready,
            "engagement_ready": engagement_ready,
            "ready_for_scale": combat_ready and engagement_ready,
            "recommended_action": recommended_action,
            "gate_issues": issues,
        }

    return {
        "thresholds": {
            "min_win_rate": float(min_win_rate),
            "min_effective_engagement_rate": float(min_effective_engagement_rate),
            "min_damaging_win_rate": float(min_damaging_win_rate),
        },
        "groups": groups,
    }


def render_markdown(report: Dict[str, Any], summary_path: Path) -> str:
    thresholds = report["thresholds"]
    lines = [
        "# Formal Engagement Gate",
        "",
        f"Source: `{summary_path}`",
        "",
        "## Thresholds",
        "",
        f"- min_win_rate: {thresholds['min_win_rate']:.3g}",
        f"- min_effective_engagement_rate: {thresholds['min_effective_engagement_rate']:.3g}",
        f"- min_damaging_win_rate: {thresholds['min_damaging_win_rate']:.3g}",
        "",
        "## Groups",
        "",
        "| group | eps | win_rate | effective_engagement | damaging_win | ready | action | issues |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for key, item in report["groups"].items():
        issues = ",".join(item["gate_issues"]) if item["gate_issues"] else "-"
        lines.append(
            "| {key} | {episodes} | {win_rate:.3g} | {effective_engagement_rate:.3g} | "
            "{damaging_win_rate:.3g} | {ready_for_scale} | {recommended_action} | {issues} |".format(
                key=key,
                issues=issues,
                **item,
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, help="Path to formal summary JSON")
    parser.add_argument("--output-json", help="Optional output JSON path")
    parser.add_argument("--output-md", help="Optional output markdown path")
    parser.add_argument("--min-win-rate", type=float, default=0.5)
    parser.add_argument("--min-effective-engagement-rate", type=float, default=0.5)
    parser.add_argument("--min-damaging-win-rate", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary_json)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    report = analyze_engagement_gate(
        payload,
        min_win_rate=args.min_win_rate,
        min_effective_engagement_rate=args.min_effective_engagement_rate,
        min_damaging_win_rate=args.min_damaging_win_rate,
    )

    output_json = (
        Path(args.output_json)
        if args.output_json
        else summary_path.with_name(summary_path.stem + "_engagement_gate.json")
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else summary_path.with_name(summary_path.stem + "_engagement_gate.md")
    )

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report, summary_path), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
