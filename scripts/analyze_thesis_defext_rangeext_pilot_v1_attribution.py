#!/usr/bin/env python3
"""Build a read-only post-pilot causal-attribution evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_defext_rangeext_attribution import (  # noqa: E402
    SOURCE_ID,
    analyze_records,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_defext_rangeext_feasibility_pilot_v1.yaml"
)
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "thesis_defext_rangeext_pilot_v1_attribution_audit_20260714_zh.md"
)
DEFAULT_CSV = (
    ROOT
    / "reports"
    / "thesis_defext_rangeext_pilot_v1_attribution_matrix_20260714.csv"
)
DEFAULT_JSON = (
    ROOT
    / "reports"
    / "thesis_defext_rangeext_pilot_v1_attribution_matrix_20260714.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return data


def _result_root(config: dict[str, Any], cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.environ.get("DEFEXT_PILOT_RESULT_ROOT")
    if env_value:
        return Path(env_value).resolve()
    return Path(config["outputs"]["root"]).resolve()


def _scenario_metadata(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for scenario in manifest.get("scenarios", []):
        scenario_meta = dict(scenario.get("metadata") or {})
        signature = str(scenario_meta.get("scenario_signature") or "")
        if not signature:
            continue
        metadata[signature] = {
            **scenario_meta,
            "own_speed_mps": scenario.get("own_init", {}).get("velocity_mps"),
            "target_speed_mps": scenario.get("target_init", {}).get(
                "velocity_mps"
            ),
        }
    return metadata


def _verify_frozen_inputs(result_root: Path) -> dict[str, Any]:
    manifest_path = result_root / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "heldout_records": result_root / "heldout" / "records.json",
        "original_decision": result_root / "heldout" / "decision.json",
        "corrected_decision": result_root / "heldout" / "decision_corrected.json",
    }
    verification: dict[str, Any] = {}
    for key, path in required.items():
        expected = str(manifest["files"][key]["sha256"]).lower()
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen input SHA mismatch for {key}: {actual} != {expected}"
            )
        verification[key] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return {
        "evidence_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "simulation_git_sha": manifest["simulation_git_sha"],
        "files": verification,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def _markdown(payload: dict[str, Any], provenance: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Defensive-Extension Pilot V1 失败机制归因审计",
        "",
        f"**Source ID：** `{SOURCE_ID}`",
        "**模式：** 只读；不训练、不调参、不重跑 Heldout24。",
        f"**Formal simulation SHA：** `{provenance['simulation_git_sha']}`",
        "",
        "## 1. 单一结论",
        "",
        "本轮能够确认候选策略存在可重复的、方法特异的 terminal safety signal，但无法从冻结产物识别该信号究竟起源于 VPP 动作、制导转换、PID 响应还是闭环对手轨迹分歧。正式归因结论为 **`causal_mechanism_not_identifiable_from_frozen_artifacts`**。",
        "",
        "该结论不改变原 pilot 的 `safety_or_contract_no_go`，也不支持‘现有技能库明确缺少 defensive-extension’。",
        "",
        "## 2. 新发现：预 handoff 配对并不等价",
        "",
        "三个方法在 handoff 之前使用相同 run-in controller，理论上同一 opponent/scenario/seed 应具有一致的 handoff metadata。冻结记录却显示多处 handoff 到达状态、进入步数或未到达时的 terminal reason 不同，因此按 scenario signature 配对不等于从相同物理状态开始比较。",
        "",
        "| Opponent | 场景数 | Handoff mismatch | Candidate-head loose/strict | Candidate-crossing loose/strict |",
        "|---|---:|---:|---:|---:|",
    ]
    for opponent, item in summary["per_opponent"].items():
        lines.append(
            f"| {opponent} | {item['scenarios']} | "
            f"{item['pre_handoff_mismatch_scenarios']} | "
            f"{item['candidate_head_loose_pairs']}/{item['candidate_head_strict_pairs']} | "
            f"{item['candidate_crossing_loose_pairs']}/{item['candidate_crossing_strict_pairs']} |"
        )
    lines.extend(
        [
            "",
            "按最低限度的 handoff metadata 等价条件筛选后，三组对手的有效数量均低于预注册 claim-ready 门槛 `8`。冻结记录没有 handoff 时刻的完整状态哈希，因此 strict subset 仍不等同于已证明物理状态逐值一致；这些 delta 只作为敏感性诊断，不用于替换或重算原 formal gate。",
            "",
            "| Opponent | Δ vs head loose/strict | Δ vs crossing loose/strict |",
            "|---|---:|---:|",
        ]
    )
    for opponent, item in summary["per_opponent"].items():
        lines.append(
            f"| {opponent} | "
            f"{_fmt(item['mean_candidate_minus_head_on_delta_loose'])}/"
            f"{_fmt(item['mean_candidate_minus_head_on_delta_strict'])} | "
            f"{_fmt(item['mean_candidate_minus_crossing_delta_loose'])}/"
            f"{_fmt(item['mean_candidate_minus_crossing_delta_strict'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Candidate-specific safety cases",
            "",
            "以下 episode 在三个方法中具有相同 handoff step；candidate 发生 ego crash/OOB，而两种 frozen baseline 均存活。因此可以确认风险发生于 handoff 后的候选闭环，但不能进一步定位到 VPP/guidance/PID 中的某一层。",
            "",
            "| Opponent | Scenario | Speeds own/target | Handoff | Candidate terminal | Head-on terminal | Crossing terminal |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in summary["candidate_specific_safety_cases"]:
        lines.append(
            f"| {row['opponent']} | {row['scenario_signature']} | "
            f"{_fmt(row['own_speed_mps'], 0)}/{_fmt(row['target_speed_mps'], 0)} | "
            f"{row['candidate_handoff_step']} | "
            f"{row['candidate_terminal_reason']} | "
            f"{row['head_on_terminal_reason']} | "
            f"{row['crossing_terminal_reason']} |"
        )
    lines.extend(
        [
            "",
            "三条风险记录都位于 positive-mirror、高速度 `defext_hold_c/d` 条件；这是局部模式线索，不是统计泛化结论。",
            "",
            "## 4. Telemetry evidence boundary",
            "",
            "冻结 `heldout/records.json` 只保存 episode-level 汇总。以下逐步字段没有落盘：",
            "",
            "`" + "`, `".join(summary["missing_step_telemetry_fields"]) + "`。",
            "",
            "因此无法恢复首个 action divergence、VPP 三轴变化、n_z command/actual、AoA、速度-高度裕度和 PID saturation 时间线。重新运行相同 Heldout24 会违反一次性使用和 stop rule，不能用于修补该证据缺口。",
            "",
            "## 5. 第一性原理决策",
            "",
            "- **技能库缺口：未证明。** Candidate 的几何变化幅度不足以跨对手达到预注册实用改善，同时存在安全退化。",
            "- **组合/路由假设：仅为 plausible。** Frozen crossing 在失败场景中保持存活并提供相近几何能力，但现有记录不足以证明 phase-aware composition 能解决问题。",
            "- **执行安全假设：存在 terminal signal，但机制未定位。** 不得直接归因于 PID、制导律或 VPP action。",
            "- **下一道门：协议修复而非性能训练。** 任何新 pilot 前，必须先在全新 dev-only 场景证明 pre-handoff deterministic equivalence，并逐步持久化 VPP/guidance/PID telemetry。",
            "",
            "四共享技能、combat finetune、P5/P6/P7 继续锁定；不得增加本候选训练步数、调整 reward、删除失败场景或重跑 Heldout24。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    result_root = _result_root(config, args.result_root)
    provenance = _verify_frozen_inputs(result_root)
    records_path = result_root / "heldout" / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["sources"]["heldout_manifest"]
    manifest = _load_yaml(manifest_path)
    payload = analyze_records(records, _scenario_metadata(manifest))
    payload["provenance"] = {
        **provenance,
        "heldout_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
    }

    _write_csv(args.out_csv, payload["rows"])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(payload, provenance), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "verdict": payload["summary"]["posthoc_attribution_verdict"],
                "rows": len(payload["rows"]),
                "out_md": str(args.out_md),
                "out_csv": str(args.out_csv),
                "out_json": str(args.out_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
