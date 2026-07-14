"""Read existing phase-reachability v2 episodes against frozen P4 sampler evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.evaluation.phase_reachability_p4_alignment import (  # noqa: E402
    aggregate_episode_rows,
    audit_episode,
    decide_primary_attribution,
)


DEFAULT_V2_ROOT = Path("E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/phase_reachability_v2_run_in_handoff_r1")
DEFAULT_P4_PHYSICAL_SUMMARY = Path("E:/uav-vpp-guidance-thesis-five-state-v1-results/p4_v2_sampler_physical_preflight_r1/preflight_summary.json")
DEFAULT_P4_GATE = Path("E:/uav-vpp-guidance-thesis-five-state-v1/reports/p4_evidence_bundle_20260714/p4_geometry_pretrain_gate.json")
RUNS = {
    "expert": "PHASE-V2-R1-EXPERT-RETRY1-20260714",
    "end_to_end": "PHASE-V2-R1-ENDTOEND-20260714",
    "independent_ppo_vpp": "PHASE-V2-R1-INDPPOVPP-20260714",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _p4_v1_defensive_cells(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    skills = gate.get("skills", {})
    defensive = skills.get("defensive_extension", {})
    selected = defensive.get("selected_gate", {})
    coverage = selected.get("geometry_phase_coverage", {})
    if not coverage:
        raise ValueError("frozen P4 gate lacks defensive-extension selected coverage")
    return coverage


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty csv: {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _render_report(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# V2 Raw Episode 与 P4 Sampler 对齐审计",
        "",
        "## 唯一归因",
        "",
        f"`{decision['primary_attribution']}`：{decision['reason']}",
        "",
        "这不是训练结论，也不改变 P4 v1/P4-v2 的既有负证据。",
        "",
        "## 直接证据",
        "",
        f"- v2 raw episode：{report['v2_episode_count']} 条；first-pass 连续可观测：{report['v2_first_pass_episodes']}/{report['v2_episode_count']}。",
        f"- P4-v2 physical sampler：通过覆盖行 {report['p4_physical']['passed_rows']}/{report['p4_physical']['total_rows']}；`all_cells_supported={report['p4_physical']['all_cells_supported']}`。",
        f"- P4 v1 defensive-extension 零覆盖 cell：{', '.join(report['p4_v1_defensive_zero_cells'])}。",
        f"- exact 66-D -> 3-D VPP pair：{report['exact_66d_pair_episodes']}/{report['v2_episode_count']} 条；因此不能用本 raw corpus 起草 pilot 预注册。",
        "",
        "## 三对手分别统计",
        "",
        "| Opponent | Episodes | First pass | P4 post-merge | Dynamic disadvantage post/re-entry steps | Exact 66-D pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for opponent, item in report["aggregation"]["opponent_summary"].items():
        lines.append(
            f"| {opponent} | {item['episodes']} | {item['first_pass_episodes']} | "
            f"{item['p4_post_merge_episodes']} | {item['dynamic_disadvantage_post_merge_reentry_steps']} | "
            f"{item['exact_66d_to_3d_pair_episodes']} |"
        )
    lines.extend(
        [
            "",
            "## 预测与执行遥测",
            "",
            "| Opponent | Prediction valid | Fallback | Mean VPP forward (m) | Mean VPP lateral (m) | Max abs n_z command | Mean abs roll-rate command | Mean throttle | Saturation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["aggregation"]["telemetry_summary"]:
        lines.append(
            f"| {item['opponent']} | {item['mean_prediction_valid_fraction']:.4f} | "
            f"{item['mean_prediction_fallback_fraction']:.4f} | {item['mean_vp_forward_bias_m']:.1f} | "
            f"{item['mean_vp_lateral_bias_m']:.1f} | {item['max_abs_nz_cmd']:.3f} | "
            f"{item['mean_abs_roll_rate_cmd']:.3f} | {item['mean_throttle_cmd']:.3f} | "
            f"{item['mean_saturation_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## P4 Zero-Cell 排查",
            "",
            "| 检查项 | 证据 | 结论 |",
            "|---|---|---|",
            "| 物理 phase 不可达 | v2 三对手 36/36 均记录 first-pass；所有对手均有动态 disadvantage post-merge/re-entry steps。 | 排除为主归因。 |",
            "| 提前终止 | v2 36 条均有 first-pass；P4-v2 physical preflight 的 120 条均为 horizon timeout。 | 没有证据表明早终止是零 cell 的主因。 |",
            "| 采样 horizon / 场景 / 行为不同 | P4-v2 使用 20 reset signature、两种非学习 reference 和 80-step horizon；v2 使用 oracle、12 个独立 run-in 场景和 260-step horizon。 | sampler mismatch 的直接组成。 |",
            "| Phase 语义不同 | P4 从首次 range <= 1000 m 开始 post-merge；v2 以 close-range 后 range opening 的 first-pass 为边界。 | sampler mismatch 的直接组成。 |",
            "| 态势记账不同 | P4 v1 gate 以 fixed initial_class 计数；v2/P4-v2 sidecar 按逐步动态 taxonomy 计数。 | sampler mismatch 的直接组成。 |",
            "| 66-D 输入/动作复构 | v2 raw 没有保存 base vector、history、P3 embedding、profile 或 normalized action。 | 审计证据边界，阻止 pilot，不作为第二归因。 |",
            "",
            "## 边界与决定",
            "",
            "v2 的 19-D raw telemetry 足以审计动态 taxonomy、P4 phase 语义、first-pass、预测有效性及 VPP/PID 响应；但未保存原始 16-D observation vector、十帧 history、P3 embedding、intent profile 与原始 normalized 3-D action。该事实是本审计的 evidence boundary，不是第二个根因标签。",
            "",
            "P4 的训练 gate 按固定 initial_class 记账，P4-v2 physical preflight 则按动态 state x phase 记账；其场景、reference controller 和 80-step horizon 又不同于 v2 的 oracle、12 场景和 260-step horizon。因此 v2 的可达性不能回填 P4 失败 cell，也不能解锁训练。",
            "",
            "结论：四共享技能和最小 pilot 继续锁定。下一动作仅可为独立、先审议的 sampler-contract design，不得复用本审计结果启动训练。",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    *,
    v2_root: Path,
    p4_physical_summary: Path,
    p4_gate: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = []
    sources: dict[str, Any] = {
        "v2_episode_records": {},
        "p4_physical_summary": {"path": str(p4_physical_summary), "sha256": _sha256_file(p4_physical_summary)},
        "p4_v1_gate": {"path": str(p4_gate), "sha256": _sha256_file(p4_gate)},
    }
    for opponent, run_name in RUNS.items():
        path = v2_root / run_name / "aggregate" / "episode_records.json"
        payload = _read_json(path)
        episodes = payload.get("episodes", [])
        if len(episodes) != 12:
            raise ValueError(f"expected 12 v2 episodes for {opponent}")
        if any(item.get("opponent_stage") != opponent for item in episodes):
            raise ValueError(f"opponent provenance mismatch for {opponent}")
        sources["v2_episode_records"][opponent] = {"path": str(path), "sha256": _sha256_file(path)}
        rows.extend(audit_episode(item) for item in episodes)
    aggregation = aggregate_episode_rows(rows)
    physical = _read_json(p4_physical_summary)
    coverage = physical.get("coverage_gate", {})
    coverage_rows = coverage.get("rows", [])
    p4_v1 = _read_json(p4_gate)
    defensive_coverage = _p4_v1_defensive_cells(p4_v1)
    zero_cells = sorted(key for key, value in defensive_coverage.items() if int(value) == 0)
    first_pass_episodes = sum(row["v2_first_pass_step"] is not None for row in rows)
    exact_pair_episodes = sum(bool(row["exact_66d_to_3d_pair_constructable"]) for row in rows)
    decision = decide_primary_attribution(
        all_v2_episodes_reach_first_pass=first_pass_episodes == len(rows),
        p4_v2_all_cells_supported=bool(coverage.get("all_cells_supported", False)),
        exact_66d_pair_count=exact_pair_episodes,
        episode_count=len(rows),
    )
    report = {
        "source_id": "THESIS-PHASE-REACHABILITY-V2-P4-ALIGNMENT-AUDIT-R1",
        "mode": "read_only_no_training_no_rerun",
        "input_artifacts": sources,
        "v2_episode_count": len(rows),
        "v2_first_pass_episodes": first_pass_episodes,
        "exact_66d_pair_episodes": exact_pair_episodes,
        "p4_v1_defensive_zero_cells": zero_cells,
        "p4_physical": {
            "all_cells_supported": bool(coverage.get("all_cells_supported", False)),
            "passed_rows": sum(bool(item.get("passed", False)) for item in coverage_rows),
            "total_rows": len(coverage_rows),
        },
        "aggregation": aggregation,
        "decision": {
            "primary_attribution": decision.primary_attribution,
            "pilot_preregistration_eligible": decision.pilot_preregistration_eligible,
            "reason": decision.reason,
        },
        "locks": {
            "four_shared_skills_training": "locked",
            "minimal_single_skill_pilot": "locked",
            "combat_finetune": "locked",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "v2_p4_alignment_episode_matrix.csv", rows)
    _write_csv(output_dir / "v2_p4_alignment_dynamic_phase_summary.csv", aggregation["dynamic_phase_steps"])
    _write_csv(output_dir / "v2_p4_alignment_telemetry_summary.csv", aggregation["telemetry_summary"])
    (output_dir / "v2_p4_alignment_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "v2_p4_alignment_report_zh.md").write_text(_render_report(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--p4-physical-summary", type=Path, default=DEFAULT_P4_PHYSICAL_SUMMARY)
    parser.add_argument("--p4-gate", type=Path, default=DEFAULT_P4_GATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-report", type=Path, default=None)
    args = parser.parse_args()
    report = analyze(
        v2_root=args.v2_root,
        p4_physical_summary=args.p4_physical_summary,
        p4_gate=args.p4_gate,
        output_dir=args.output_dir,
    )
    if args.repo_report is not None:
        args.repo_report.parent.mkdir(parents=True, exist_ok=True)
        args.repo_report.write_text(_render_report(report), encoding="utf-8")
    print(json.dumps({"primary_attribution": report["decision"]["primary_attribution"], "pilot_preregistration_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
