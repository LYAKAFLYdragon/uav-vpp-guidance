#!/usr/bin/env python3
"""Read-only corrected analysis for the completed defensive-extension pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.training.thesis_defext_rangeext_pilot import (  # noqa: E402
    analyze_heldout,
    load_yaml,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_defext_rangeext_feasibility_pilot_v1.yaml"
DEFAULT_RESULT_ROOT = Path("E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/defensive_extension_range_extension_feasibility_v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(decision: dict, original_sha: str) -> str:
    lines = [
        "# Defensive-Extension / Range-Extension Pilot V1 执行结论",
        "",
        "## 结论",
        "",
        f"预注册判定为 **`{decision['verdict']}`**。现有证据不支持‘现有技能库明确缺少 defensive-extension’。",
        "",
        "候选技能完成固定 50,000-step 训练，所有 Dev checkpoint safety gate 均通过，并按 minimax Dev 规则自动选择 20k checkpoint。但在一次性 Heldout24 中，independent PPO/VPP 对手下出现 `2/24` ego crash/OOB，而 fixed head-on 与 fixed crossing 均为 `0/24`，安全差值 `+0.0833` 超过预注册 `+0.05` 门。",
        "",
        "此外，independent PPO/VPP 下候选与两种 baseline 的 paired qualifying episode 均只有 `6`，低于 claim-ready 门槛 `8`。原始 decision 实现只检查 candidate qualifying 数量；该分析错误已只读纠正，原文件未覆盖，原 SHA-256 为 `" + original_sha + "`。",
        "",
        "## 分对手结果",
        "",
        "| Opponent | Candidate qualifying | Paired H/C | Delta vs head-on | Delta vs best existing | Crash/OOB delta | Contract | Safety |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for opponent, item in decision["per_opponent"].items():
        lines.append(
            "| {opponent} | {qualifying} | {head}/{crossing} | {head_delta:+.4f} | {best_delta:+.4f} | {crash:+.4f} | {contract} | {safety} |".format(
                opponent=opponent,
                qualifying=item["candidate"]["qualifying_episodes"],
                head=item["paired_head_count"],
                crossing=item["paired_crossing_count"],
                head_delta=float(item["mean_paired_delta_vs_head_on"]),
                best_delta=float(item["mean_paired_delta_vs_best_existing"]),
                crash=float(item["ego_crash_oob_delta_vs_head_on"]),
                contract="pass" if item["contract_pass"] else "fail",
                safety="pass" if item["safety_pass"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## 科学解释",
            "",
            "- 三个对手下 candidate 相对 head-on 的 paired mean delta 分别为约 `-0.0081/-0.0108/-0.0082`，均通过 `+0.02` 非劣门，但没有一个达到 `-0.05` practical-improvement 门。",
            "- 相对 best existing specialist，仅 independent PPO/VPP 达到 `-0.02` practical-improvement；expert 与 end-to-end 均未达到。",
            "- Fixed crossing 在 expert 与 independent PPO/VPP 中本身就是更好的 existing specialist，说明现有技能库已经覆盖一部分 extension-like geometry，不能把问题简单归因为‘缺一个新技能’。",
            "- 候选在高速度、positive-mirror heldout 条件中发生早期 crash/OOB，而两种 frozen baseline 在相同场景均存活，说明当前 range-extension policy 的几何改善不足以抵消执行安全风险。",
            "",
            "## Stop Rule",
            "",
            "结果冻结为负证据。不得在同一 Source ID 下增加训练步数、调整 reward、删除失败场景、更换 seed、放宽 safety/paired coverage 门或重跑 Heldout24。四共享技能、combat finetune 和 P5 继续锁定。",
            "",
            "当前最准确的回答是：**输入输出契约已经建立，但本 pilot 没有证明新 defensive-extension 技能优于现有技能库；当前候选还引入了 opponent-dependent 的安全退化。**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "thesis_defext_rangeext_feasibility_pilot_v1_execution_20260714_zh.md")
    args = parser.parse_args()
    config = load_yaml(args.config)
    records_path = args.result_root / "heldout" / "records.json"
    original_path = args.result_root / "heldout" / "decision.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    original_sha = _sha256(original_path)
    decision = analyze_heldout(config, records)
    corrected_path = args.result_root / "heldout" / "decision_corrected.json"
    corrected_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report(decision, original_sha), encoding="utf-8")
    evidence_paths = {
        "authorization_preflight": args.result_root / "authorization_preflight.json",
        "resolved_config": args.result_root / "resolved_config.json",
        "training_summary": args.result_root / "training_summary.json",
        "best_checkpoint": args.result_root / "checkpoints" / "best.pt",
        "heldout_records": records_path,
        "original_decision": original_path,
        "corrected_decision": corrected_path,
    }
    manifest_path = args.result_root / "evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_id": config["source_id"],
                "simulation_git_sha": json.loads((args.result_root / "authorization_preflight.json").read_text(encoding="utf-8"))["git_sha"],
                "original_decision_preserved": True,
                "original_decision_sha256": original_sha,
                "analysis_correction": "paired qualifying episode gate enforced for both baselines",
                "files": {
                    name: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
                    for name, path in evidence_paths.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": decision["verdict"], "corrected_decision": str(corrected_path), "report": str(args.report), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
