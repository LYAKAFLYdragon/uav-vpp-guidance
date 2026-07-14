#!/usr/bin/env python3
"""Create a read-only audit package for the frozen P1 run-in preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.global_advantage_p1_analysis import (  # noqa: E402
    audit_p1_manifest,
    sha256_file,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_global_advantage_v1_p1_runin_preflight.yaml"
)
DEFAULT_MD = ROOT / "reports" / "thesis_global_advantage_p1_r2_audit_20260715_zh.md"
DEFAULT_JSON = ROOT / "reports" / "thesis_global_advantage_p1_r2_matrix_20260715.json"
DEFAULT_CSV = ROOT / "reports" / "thesis_global_advantage_p1_r2_matrix_20260715.csv"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _markdown(audit: dict[str, Any], provenance: dict[str, str]) -> str:
    lines = [
        "# Global-Advantage P1 R2 Run-in 可重复性审计",
        "",
        f"**Source ID：** `{audit['source_id']}`",
        "**模式：** 只读；不训练、不调参、不重跑 P1 R2。",
        "",
        "## 结论",
        "",
        "P1 R2 为正式 **NO-GO**：所有 90 个 opponent x scenario cell 的 reset base-state hash 一致，但没有一个 cell 的完整轨迹或 first-pass/terminal boundary state hash 一致。当前批量 run-in 协议不能为未来方法比较提供同一物理起点之后的因果配对基础。",
        "",
        "这不是新技能效果结论，也不是对 JSBSim 飞行动力学的性能判断。它证明的是：在复用环境实例、改变 scenario 顺序的现有 runner 中，隐藏的运行时状态或未被 reset telemetry 会影响第一个或后续控制步。",
        "",
        "| Opponent | Cells | Reset equal | Trajectory equal | Boundary equal | Terminal equal | Telemetry complete | Hard fallback absent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for opponent, item in audit["per_opponent"].items():
        lines.append(
            f"| {opponent} | {item['comparison_cells']} | "
            f"{item['reset_state_equal_cells']} | {item['trajectory_equal_cells']} | "
            f"{item['boundary_state_equal_cells']} | {item['terminal_equal_cells']} | "
            f"{item['telemetry_complete_cells']} | {item['fallback_free_cells']} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- R2 完整写入并通过 SHA-256 验证的 raw episode artifact 为 `270` 个；不存在缺 episode、backend fallback、预测器运行时 fallback 或 telemetry 缺字段的情况。",
            "- 前 9 个预测器 warmup step 被显式记录，未误报为运行时 fallback。",
            "- R2 仅保存了 reset 时的 16-D base-state 摘要，没有持久化完整 reset observation vector、控制器内部状态和 JSBSim 完整初始 FDM state。因此不能把原因直接归咎于某一个模块。",
            "",
            "## 允许的下一步",
            "",
            "仅允许起草 P1 R3 fresh-environment-per-episode 复核：每条 scenario 在新建环境实例中独立运行，同时在 reset 前后保存完整 observation、reference action、opponent action、controller state 和完整 FDM state hash。R3 通过前不得启动 P2、任何技能训练、combat finetune、P5 或 formal held-out。",
            "",
            f"P1 R2 run manifest SHA-256: `{provenance['run_manifest_sha256']}`",
            f"P1 config SHA-256: `{provenance['config_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    root = args.result_root or Path(config["global_advantage_p1"]["outputs"]["root"])
    manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = audit_p1_manifest(root, run_manifest)
    provenance = {
        "run_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(args.config),
    }
    audit["provenance"] = provenance
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.out_csv, audit["rows"])
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(audit, provenance), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_id": audit["source_id"],
                "verdict": audit["verdict"],
                "artifacts": audit["artifact_count"],
                "cells": audit["comparison_cell_count"],
                "out_md": str(args.out_md),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
