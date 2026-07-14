#!/usr/bin/env python3
"""Analyze the preregistered single-motif 66-D collector without rerunning it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.single_motif_continuous_66d import (  # noqa: E402
    build_single_motif_plan,
)
from uav_vpp_guidance.evaluation.single_motif_continuous_66d_analysis import (  # noqa: E402
    evaluate_gate,
    load_collector_runs,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_single_motif_continuous_66d_r1.yaml"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "opponent",
        "scenario",
        "scenario_signature",
        "mirror_sign",
        "seed",
        "episode",
        "recorded_steps",
        "contract_ready_steps",
        "target_motif_steps",
        "valid_target_steps",
        "episode_contract_valid",
        "v2_continuity_valid",
        "qualifying_episode",
        "artifact_path",
        "artifact_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(result: dict) -> str:
    lines = [
        "# Single-Motif Continuous 66-D Collector R1 Gate",
        "",
        "## 结论",
        "",
    ]
    if result["gate_passed"]:
        lines.extend(
            [
                "三种对手均通过预注册 gate。真实连续 JSBSim rollout 已建立可执行的 `66-D -> normalized 3-D VPP` 输入输出契约。",
                "",
                "因此当前证据不再支持‘连可学习数据契约都没有建立’这一阻塞；后续缺口可以收窄为 **defensive-extension 新技能候选缺口**。这仍不证明新技能有效，也不自动授权训练，只允许起草单技能 feasibility pilot 预注册。",
            ]
        )
        if not result["provenance_clean"]:
            lines.extend(
                [
                    "",
                    "本轮仅为 **研发 gate 正证据**，不是可直接投稿的 formal evidence：三条 run 的 `run_manifest` 均记录 dirty worktree，因此 `paper_safe=false`。该 provenance caveat 不改变物理/数据契约结论，但在 clean SHA 冻结前不得写入论文主结果。",
                ]
            )
    else:
        lines.extend(
            [
                "至少一个对手未通过预注册 gate。当前仍未建立跨三对手可执行的 `66-D -> normalized 3-D VPP` 输入输出契约。",
                "",
                "因此不能把现有失败归因于‘缺少新技能’，五态势训练扩展应停止，且不得用 pooled 结果、调参或重跑修补本轮负证据。",
            ]
        )
    lines.extend(
        [
            "",
            "## 分对手 Gate",
            "",
            "| Opponent | Episodes | Qualifying episodes | Valid target steps | Signatures | Mirrors | Contract | Gate |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for opponent, item in result["opponents"].items():
        lines.append(
            "| {opponent} | {episodes} | {qualifying} | {steps} | {signatures} | {mirrors} | {contract} | {gate} |".format(
                opponent=opponent,
                episodes=item["episodes"],
                qualifying=item["qualifying_episodes"],
                steps=item["valid_target_steps"],
                signatures=len(item["distinct_scenario_signatures"]),
                mirrors=len(item["distinct_mirror_signs"]),
                contract="pass" if item["checks"]["all_episode_contracts_valid"] else "fail",
                gate="pass" if item["passed"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## 固定边界",
            "",
            "- 策略、VPP、制导、PID、reward 与 checkpoint 均未改变。",
            "- 仅使用冻结 `legacy_static_oracle_task_gate` 的原始动作；collector 不替换 action。",
            "- 目标样本必须是 first-pass 后的 dynamic disadvantage，且 P4 phase 为 post_merge/re_entry。",
            "- 每个可用样本均含真实 16-D base、10 个真实历史帧、冻结 P3 32-D embedding、固定 range_extension profile、validity mask 和原始 3-D action。",
            "- 通过只允许起草单一 defensive-extension pilot 预注册；`training_authorized` 始终为 false。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    plan = build_single_motif_plan(config)
    episodes = load_collector_runs(args.run_dir)
    result = evaluate_gate(episodes, plan)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "single_motif_continuous_66d_gate.json"
    csv_path = args.out_dir / "single_motif_continuous_66d_episode_matrix.csv"
    report_path = args.out_dir / "single_motif_continuous_66d_gate_zh.md"
    provenance_path = args.out_dir / "single_motif_continuous_66d_source_provenance.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, result["episode_rows"])
    report_path.write_text(_report(result), encoding="utf-8")
    source_paths = {
        "config": args.config,
        "collector": ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "single_motif_continuous_66d.py",
        "analysis_core": ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "single_motif_continuous_66d_analysis.py",
        "runner": ROOT / "scripts" / "run_jsbsim_hrl_comparison.py",
        "analyzer": Path(__file__).resolve(),
        "p3_checkpoint": plan.p3_checkpoint,
    }
    provenance_path.write_text(
        json.dumps(
            {
                "source_id": plan.source_id,
                "paper_safe": result["paper_safe"],
                "provenance_clean": result["provenance_clean"],
                "files": {
                    name: {
                        "path": str(path),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for name, path in source_paths.items()
                },
                "run_provenance": result["run_provenance"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "gate_passed": result["gate_passed"],
        "verdict": result["verdict"],
        "paper_safe": result["paper_safe"],
        "json": str(json_path),
        "csv": str(csv_path),
        "report": str(report_path),
        "provenance": str(provenance_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
