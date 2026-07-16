#!/usr/bin/env python3
"""Replay V16 telemetry to find unclipped three-knot path candidates; no JSBSim run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_three_knot_envelope import (
    OPPONENTS,
    evaluate_step,
    summarize_rows,
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _records(root: Path):
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("record", {})
        if record.get("method") != "progress_triggered_reference_path" or not record.get("window"):
            continue
        if str(record.get("opponent")) not in OPPONENTS:
            continue
        yield record


def analyze(config_path: Path) -> dict[str, Any]:
    config = _load(config_path)
    if config.get("kind") != "read_only_three_knot_envelope_analysis":
        raise RuntimeError("three-knot envelope config identity drifted")
    root = Path(config["inputs"]["v16_output_root"])
    if not root.is_dir():
        raise RuntimeError("frozen V16 output root is unavailable")
    rows: list[dict[str, Any]] = []
    for record in _records(root):
        for candidate in config["candidates"]:
            for step in record["window"]:
                receipt = evaluate_step(step, candidate, config["raw_action_ranges"])
                rows.append(
                    {
                        "candidate_id": candidate["id"],
                        "opponent": record["opponent"],
                        "pair_key": record["pair_key"],
                        "local_step": step["local_step"],
                        **receipt,
                    }
                )
    if not rows:
        raise RuntimeError("V16 contains no replayable dynamic-path telemetry")
    gate = summarize_rows(
        rows,
        min_fully_unclipped_episodes=int(config["gate"]["min_fully_unclipped_episodes_per_opponent"]),
        min_unclipped_step_fraction=float(config["gate"]["min_unclipped_step_fraction"]),
    )
    return {"config": config, "rows": rows, "gate": gate}


def write_outputs(result: dict[str, Any], output: Path, report: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows = result["rows"]
    with (output / "three_knot_envelope_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    payload = {"analysis_kind": "read_only_not_a_scientific_source", "gate": result["gate"], "config": result["config"]}
    (output / "three_knot_envelope_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = result["gate"]
    lines = [
        "# 三锚点动态参考路径可执行包线分析（只读）",
        "",
        "该分析仅重放 V16 已冻结 telemetry，不运行 JSBSim、不训练，也不估计未来 AA/ATA/能量响应。",
        "",
        "## 预注册前置结论",
        "",
        f"`passing_candidates={gate['passing_candidates']}`；`passes={gate['passes']}`。",
        "",
        "候选通过条件：同一候选至少两个对手分别具有不少于 4 条全程无裁剪 episode，且无裁剪 step 比例不少于 0.90。",
        "",
        "## 分对手矩阵",
        "",
        "| Candidate | Opponent | Episodes | Steps | Unclipped-step fraction | Fully unclipped episodes | Clipped axes (x/y/z) | Pass |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for candidate_id, opponent_matrix in gate["matrix"].items():
        for opponent, cell in opponent_matrix.items():
            lines.append(
                f"| {candidate_id} | {opponent} | {cell['qualifying_episodes']} | {cell['steps']} | {cell['unclipped_step_fraction']:.3f} | {cell['fully_unclipped_episodes']} | {cell['clipped_axis_counts_xyz']} | {cell['passes']} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "通过只说明候选空间未被冻结 action head 大量裁剪，可据此起草新的非学习物理传递 probe；不通过则禁止创建该 probe。",
            "该离线分析不能声称六维 intent 改善，六维改善必须在全新、预注册的 JSBSim 对比中测量。",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise RuntimeError("three-knot envelope outputs must be absent")
    result = analyze(args.config)
    write_outputs(result, args.output, args.report)
    print(json.dumps({"output": str(args.output), **result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
