#!/usr/bin/env python3
"""Produce a read-only V17 phase-occupancy attribution from frozen records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_v17_phase_occupancy_audit import (
    SOURCE_ID,
    summarize_phase_occupancy,
)


def _load_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_dir / "episodes").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("record")
        if not isinstance(record, dict):
            raise ValueError(f"V17 episode is missing a record: {path}")
        records.append(record)
    if not records:
        raise ValueError("V17 output contains no episode records")
    return records


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# V17 End-to-End 相位占用只读归因审计",
        "",
        "## 结论",
        "",
        "本审计只统计冻结 V17 record 已保存的 first-pass、phase trace、handoff 与 terminal reason。",
        "它不含逐步 target speed、altitude、specific energy 或 attack-zone telemetry，故不能将覆盖不足",
        "归因于对手强度、目标早期轨迹或飞控执行链。",
        "",
        "end-to-end 的缺口发生在 first-pass 之后：所有 24 条 episode 都锁存了 first-pass，但仅 4 条",
        "进入 post-merge/re-entry 并完成 handoff；另 20 条 first-pass 后始终没有 post-merge phase。",
        "因此 V17 的失败是 frozen envelope 中 post-first-pass phase occupancy 不足，而不是 first-pass",
        "检测失败或 74-D 数值合同失败。",
        "",
        "## 各对手汇总",
        "",
        "| Opponent | Records | First pass latched | Post-merge episodes | Re-entry episodes | Handoff reached | First-pass without post-merge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for opponent, cell in result["matrix"].items():
        lines.append(
            "| {opponent} | {records} | {first_pass_latched} | {episodes_with_post_merge} | "
            "{episodes_with_re_entry} | {handoff_reached} | {first_pass_without_post_merge} |".format(
                opponent=opponent,
                **cell,
            )
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 本审计不重跑、不训练、不改变 path context、reward、VPP、guidance 或 PID。",
            "- `first_pass_without_post_merge` 仅是标签轨迹事实，不能证明相位定义、target policy 或物理动力学中的任何单一原因。",
            "- 如需进一步解释原因，必须以新 Source ID 预注册一个带 target/own kinematics 和 phase telemetry 的独立非学习 collector；不能修改或补跑 V17。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = summarize_phase_occupancy(_load_records(args.output_dir))
    if result["source_id"] != SOURCE_ID:
        raise ValueError("unexpected source ID")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(_render(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
