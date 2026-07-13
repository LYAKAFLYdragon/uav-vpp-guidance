#!/usr/bin/env python3
"""Render the non-training P4 readiness JSON as a compact Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_p4_readiness_20260713.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_p4_readiness_20260713_zh.md"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P4 四共享技能训练前 Readiness 报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 是否启动训练：`{report['training_launched']}`",
        "- 解释：本报告只验证冻结资产与接口；四个技能仍未训练，尚未通过任何机动性能 readiness gate。",
        "",
        "## 冻结契约",
        "",
        "- P3 v2 encoder 已按 SHA 加载并保持冻结。",
        "- 共享输入严格为 66-D：几何、显式历史、profile targets/weights 与 32-D temporal embedding。",
        "- 输出严格为 3-D normalized VPP bias：forward、lateral、vertical。",
        "- 三对手采用 fail-closed balanced round-robin；禁止跨对手汇总。",
        "",
        "## 校验结果",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    for item in report["checks"]:
        result = "PASS" if item["passed"] is True else "SKIPPED" if item["passed"] is None else "FAIL"
        lines.append(f"| `{item['name']}` | {result} |")
    lines.extend(["", "## 各技能状态", "", "| Skill | Readiness |", "|---|---|"])
    for skill, status in report["skill_readiness_status"].items():
        lines.append(f"| `{skill}` | `{status}` |")
    lines.extend(
        [
            "",
            "## 下一道门",
            "",
            "本轮未运行 geometry pretraining 或 combat finetuning。后续必须获得显式授权、保持本 registry/config hash 不变、使用新的 P4 输出根，并在进入高层 PPO 前逐技能完成全部 P4 readiness 指标。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"input": str(args.input_json), "output": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
