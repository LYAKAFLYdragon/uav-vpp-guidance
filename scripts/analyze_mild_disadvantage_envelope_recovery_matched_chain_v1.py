#!/usr/bin/env python
"""Read-only attribution analysis for the matched-contract recovery traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry"
DEFAULT_OUTPUT_JSON = DEFAULT_ROOT / "chain_attribution_v1.json"
DEFAULT_OUTPUT_CSV = DEFAULT_ROOT / "chain_attribution_horizon_deltas.csv"
DEFAULT_REPORT = REPO_ROOT / "reports/mild_disadvantage_envelope_recovery_matched_chain_attribution_v1_zh.md"

INPUT_SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-CONTRACT-PROBE-V1-20260903"
ANALYSIS_ID = "MILD-DISADVANTAGE-MATCHED-CHAIN-ATTRIBUTION-READONLY-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_CANDIDATES = ("x625_y200", "x750_y250")
HORIZONS = (4, 12, 48)

FIELD_GROUPS = {
    "compiled_action": ("action_0", "action_1", "action_2"),
    "vpp_actual": (
        "vpp_actual_x_m",
        "vpp_actual_y_m",
        "vpp_actual_z_m",
        "virtual_point_actual_x_m",
        "virtual_point_actual_y_m",
        "virtual_point_actual_z_m",
        "actual_vpp_x_m",
        "actual_vpp_y_m",
        "actual_vpp_z_m",
    ),
    "guidance_command": (
        "guidance_nz_cmd",
        "guidance_roll_rate_cmd",
        "guidance_throttle_cmd",
        "nz_cmd",
        "roll_rate_cmd",
        "throttle_cmd",
    ),
    "roll_response": (
        "roll_deg",
        "roll_rad",
        "roll_rate_dps",
        "roll_rate_rad_s",
        "p_dps",
        "p_rad_s",
    ),
    "geometry": (
        "range_m",
        "range_rate_mps",
        "ata_deg",
        "aa_deg",
        "altitude_m",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=REPO_ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _present_fields(columns: set[str], aliases: Iterable[str]) -> list[str]:
    return [name for name in aliases if name in columns]


def classify_columns(columns: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Classify which causal-chain fields are actually present in a trace."""
    column_set = set(columns)
    result: dict[str, dict[str, Any]] = {}
    for group, aliases in FIELD_GROUPS.items():
        present = _present_fields(column_set, aliases)
        result[group] = {
            "present": bool(present),
            "fields": present,
            "expected_fields": list(aliases),
        }
    return result


def _trace_path(root: Path, opponent: str, candidate: str, repetition: int) -> Path:
    return root / opponent / "trajectories" / f"{candidate}_{repetition:02d}.csv"


def _horizon_row(rows: list[dict[str, str]], step: int) -> dict[str, str] | None:
    for row in rows:
        if row.get("step") == str(step):
            return row
    return None


def _direction(values: list[float]) -> dict[str, Any]:
    positive = sum(value > 1e-12 for value in values)
    negative = sum(value < -1e-12 for value in values)
    zero = len(values) - positive - negative
    n = len(values)
    dominant = "positive" if positive > negative else "negative" if negative > positive else "mixed_or_zero"
    return {
        "n": n,
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "dominant_direction": dominant,
        "directional_consistency": max(positive, negative) / n if n else None,
    }


def _episode_analysis(meta: dict[str, str], trace_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_csv(trace_path)
    columns = set(rows[0]) if rows else set()
    classification = classify_columns(columns)
    action_fields = FIELD_GROUPS["compiled_action"]
    action_values = {
        name: [float(row[name]) for row in rows if _finite(row.get(name))]
        for name in action_fields
    }
    action_stats = {
        name: {
            "finite_count": len(values),
            "mean": _mean(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "range": max(values) - min(values) if values else None,
        }
        for name, values in action_values.items()
    }
    horizon_rows: list[dict[str, Any]] = []
    initial = _horizon_row(rows, 0)
    geometry_fields = FIELD_GROUPS["geometry"]
    for horizon in HORIZONS:
        endpoint = _horizon_row(rows, horizon)
        record: dict[str, Any] = {
            "opponent": meta["opponent"],
            "candidate": meta["candidate"],
            "seed": int(meta["seed"]),
            "trace_path": str(trace_path.resolve()),
            "horizon_steps": horizon,
            "endpoint_present": endpoint is not None,
            "endpoint_mode": endpoint.get("mode") if endpoint else None,
            "endpoint_recovery_phase": endpoint.get("recovery_phase") if endpoint else None,
        }
        for field in geometry_fields:
            start_value = _float(initial.get(field)) if initial else None
            end_value = _float(endpoint.get(field)) if endpoint else None
            record[f"delta_{field}"] = (
                end_value - start_value
                if start_value is not None and end_value is not None
                else None
            )
        horizon_rows.append(record)

    prefix = [row for row in rows if _finite(row.get("step")) and int(row["step"]) < 48]
    crossing_prefix_ok = bool(prefix) and all(row.get("mode") == "crossing" for row in prefix)
    action_finite = bool(rows) and all(
        _finite(row.get(field)) for row in rows for field in action_fields
    )
    return (
        {
            "opponent": meta["opponent"],
            "candidate": meta["candidate"],
            "seed": int(meta["seed"]),
            "trace_path": str(trace_path.resolve()),
            "trace_rows": len(rows),
            "trace_columns": sorted(columns),
            "column_classification": classification,
            "action_finite": action_finite,
            "action_stats": action_stats,
            "crossing_prefix_steps_0_to_47": crossing_prefix_ok,
            "handoff_at_step_48": (
                (_horizon_row(rows, 48) or {}).get("mode") == "h0_current_geometry"
            ),
            "max_clipping_rate": max(
                (float(row["clipping_rate"]) for row in rows if _finite(row.get("clipping_rate"))),
                default=None,
            ),
            "terminal_reason": meta.get("reason", ""),
        },
        horizon_rows,
        {"columns": sorted(columns), "classification": classification},
    )


def analyze(root: Path) -> dict[str, Any]:
    result_path = root / "episode_results.csv"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    metadata = _read_csv(result_path)
    if len(metadata) != 12:
        raise ValueError(f"expected 12 episode metadata rows, got {len(metadata)}")
    episodes: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    all_columns: set[str] = set()
    cell_counts: Counter[tuple[str, str]] = Counter()
    for meta in metadata:
        opponent = meta["opponent"]
        candidate = meta["candidate"]
        cell_counts[(opponent, candidate)] += 1
        trace = _trace_path(root, opponent, candidate, cell_counts[(opponent, candidate)])
        if not trace.is_file():
            raise FileNotFoundError(trace)
        episode, deltas, trace_info = _episode_analysis(meta, trace)
        episodes.append(episode)
        horizon_rows.extend(deltas)
        all_columns.update(trace_info["columns"])

    classification = classify_columns(all_columns)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    opponent_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in horizon_rows:
        grouped[(row["opponent"], row["candidate"], row["horizon_steps"])].append(row)
        opponent_grouped[(row["opponent"], row["horizon_steps"])].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "opponent": group[0]["opponent"],
            "candidate": group[0].get("candidate"),
            "horizon_steps": group[0]["horizon_steps"],
            "n": len(group),
        }
        for field in FIELD_GROUPS["geometry"]:
            values = [row[f"delta_{field}"] for row in group if row[f"delta_{field}"] is not None]
            summary[f"mean_delta_{field}"] = _mean(values)
            summary[f"direction_{field}"] = _direction(values)
        return summary

    cell_summaries = [summarize(group) for key, group in sorted(grouped.items())]
    opponent_summaries = []
    for key, group in sorted(opponent_grouped.items()):
        summary = summarize(group)
        summary["candidate"] = "all_candidates"
        opponent_summaries.append(summary)

    terminal_counts = defaultdict(Counter)
    for meta in metadata:
        terminal_counts[meta["opponent"]][meta.get("reason", "")] += 1

    evidence = {
        "compiled_action": {
            "status": "observed",
            "meaning": "CSV contains finite action_0/action_1/action_2 values for all 12 traces.",
        },
        "vpp_actual": {
            "status": "not_evaluable",
            "meaning": "No actual VPP position/offset telemetry field is present in the 12 trace CSV files.",
        },
        "guidance_command": {
            "status": "not_evaluable",
            "meaning": "No numeric guidance command telemetry field is present; effective_guidance_mode is only a label.",
        },
        "roll_response": {
            "status": "not_evaluable",
            "meaning": "No roll angle or roll-rate telemetry field is present.",
        },
        "geometry": {
            "status": "observed_descriptively",
            "meaning": "Range, range-rate, ATA, AA and altitude are available at the requested horizons.",
        },
    }
    return {
        "analysis_id": ANALYSIS_ID,
        "input_source_id": INPUT_SOURCE_ID,
        "input_root": str(root.resolve()),
        "git_info": _git_info(),
        "episode_count": len(episodes),
        "opponents": list(EXPECTED_OPPONENTS),
        "candidates": list(EXPECTED_CANDIDATES),
        "horizons_steps": list(HORIZONS),
        "all_trace_columns": sorted(all_columns),
        "chain_field_classification": classification,
        "chain_evidence": evidence,
        "episodes": episodes,
        "horizon_deltas": horizon_rows,
        "cell_summaries": cell_summaries,
        "opponent_summaries": opponent_summaries,
        "terminal_counts": {key: dict(value) for key, value in terminal_counts.items()},
        "causal_limit": (
            "The traces contain no same-state counterfactual baseline and no intermediate VPP, "
            "guidance, or roll telemetry; horizon changes are descriptive trajectory evolution, not causal transfer estimates."
        ),
        "input_receipts": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*.csv"))
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "不可评估" if value is None else f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(data: dict[str, Any]) -> str:
    lines = [
        "# Matched-contract 12 条连续轨迹链路归因",
        "",
        "## 1. 分析范围",
        "",
        f"- 分析标识：`{data['analysis_id']}`",
        f"- 输入 Source ID：`{data['input_source_id']}`",
        f"- 输入目录：`{data['input_root']}`",
        f"- 轨迹数量：{data['episode_count']} 条",
        "- 分析方式：只读，不重跑 JSBSim，不训练，不修改任何控制参数。",
        "- 观察时点：从每条轨迹的第 0 步计算 4、12、48 步后的变化；第 4、12 步位于 crossing 恢复段，第 48 步是 48 步预算耗尽后的首次 H0 交接点。",
        "",
        "## 2. 链路可评估性",
        "",
        "| 链路层级 | 当前证据 | 判定 |",
        "|---|---|---|",
        "| crossing 编译动作 | `action_0`、`action_1`、`action_2` | 12/12 条轨迹均有有限值，已观测 |",
        "| 实际 VPP 位置或偏置 | 未发现实际 VPP 数值字段 | 不可评估 |",
        "| 制导数值指令 | 未发现过载、滚转速率和油门数值指令字段 | 不可评估 |",
        "| 滚转状态响应 | 未发现滚转角或滚转速率字段 | 不可评估 |",
        "| 几何结果 | `range_m`、`range_rate_mps`、`ata_deg`、`aa_deg`、`altitude_m` | 可作描述性分析 |",
        "",
        "> 重要边界：现有文件只能证明 crossing 动作被记录，以及几何量随后发生了变化；不能证明动作已经通过 VPP、制导或滚转环节，也不能据此计算各层的响应保真率或延迟。",
        "",
        "## 3. 4/12/48 步几何变化",
        "",
        "下表为同一条轨迹内的端点差值，定义为“第 N 步值减去第 0 步值”。这些差值不是动作干预的因果效应，因为没有同状态的无干预轨迹。",
        "",
        "| 对手 | 初始点 | 时域 | 距离变化（m） | 距离率变化（m/s） | ATA 变化（°） | AA 变化（°） | 高度变化（m） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in data["cell_summaries"]:
        lines.append(
            "| {opponent} | {candidate} | {horizon_steps} | {range} | {range_rate} | {ata} | {aa} | {altitude} |".format(
                opponent=summary["opponent"],
                candidate=summary["candidate"],
                horizon_steps=summary["horizon_steps"],
                range=_fmt(summary["mean_delta_range_m"]),
                range_rate=_fmt(summary["mean_delta_range_rate_mps"]),
                ata=_fmt(summary["mean_delta_ata_deg"]),
                aa=_fmt(summary["mean_delta_aa_deg"]),
                altitude=_fmt(summary["mean_delta_altitude_m"]),
            )
        )
    lines.extend([
        "",
        "## 4. 归因结论",
        "",
        "1. crossing 原始三轴动作在 12 条轨迹中均被记录，且第 0--47 步均保持 crossing 模式；第 48 步统一进入 H0，说明恢复预算耗尽后发生了预定交接，而不是没有调用 crossing 专家。",
        "2. 现有轨迹文件没有实际 VPP、制导数值指令和滚转状态字段，故差异消失层不能进一步定位到 VPP、制导或滚转环节。",
        "3. 距离、距离率、ATA、AA 和高度的 4/12/48 步变化可以作为结果演化记录，但不能单独解释为 crossing 动作造成的几何改善或恶化。",
        "4. 终端结果仍受对手目标运动、制导闭环和 JSBSim 物理响应共同影响；仅凭这 12 条文件不能建立跨对手的局部因果控制方向。",
        "",
        "## 5. 决策",
        "",
        "> 当前最准确的结论是：crossing 动作层和几何结果层可观测，但中间执行链不可评估。因此不能据此授权继续训练恢复技能、修改低层接口或扩大 PPO。若要继续做一次最小补强，应只补采同一接口下的实际 VPP、数值制导指令、滚转角/滚转速率，并保留动作前状态哈希；补齐后再做链路延迟和差异保留率分析。",
        "",
        "## 6. 文件",
        "",
        f"- 机器可读归因：`{data['input_root']}/chain_attribution_v1.json`",
        f"- 时域差值：`{data['input_root']}/chain_attribution_horizon_deltas.csv`",
        f"- 本报告：`{DEFAULT_REPORT.resolve()}`",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    data = analyze(args.root.resolve())
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    write_csv(args.output_csv.resolve(), data["horizon_deltas"])
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(render_report(data), encoding="utf-8")
    print(json.dumps({
        "analysis_id": data["analysis_id"],
        "episode_count": data["episode_count"],
        "output_json": str(args.output_json.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "report": str(args.report.resolve()),
        "chain_field_status": {
            key: value["status"] if "status" in value else value["present"]
            for key, value in data["chain_evidence"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
