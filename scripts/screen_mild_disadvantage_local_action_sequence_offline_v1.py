#!/usr/bin/env python3
"""Screen short handoff action sequences using verified V4 local responses.

The only identifiable temporal template in V4 is a pulse at handoff followed
by a return to the nominal action. This script uses linear superposition of the
verified single-axis sensitivities to rank that template offline. It never
runs JSBSim and never authorizes training.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-LOCAL-ACTION-SEQUENCE-OFFLINE-SCREEN-V1-20260904"
EXPECTED_LOCAL_SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V4-20260904"
OPPONENTS = ("expert", "end_to_end")
CANDIDATES = ("x625_y200", "x750_y250")
METRICS = (
    ("range_m", "range_delta_m"),
    ("range_rate_mps", "range_rate_delta_mps"),
    ("ata_deg", "ata_delta_deg"),
    ("aa_deg", "aa_delta_deg"),
    ("specific_energy_m2ps2", "energy_delta_m2ps2"),
    ("altitude_m", "altitude_delta_m"),
)
HORIZONS = (4, 12)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def load_effects(root: Path, expected_source_id: str) -> list[dict[str, Any]]:
    report = json.loads((root / "local_response_analysis_report.json").read_text(encoding="utf-8"))
    if report.get("source_id") != expected_source_id:
        raise RuntimeError("local-response analysis Source ID mismatch")
    rows = read_csv(root / "local_response_effects.csv")
    if not rows:
        raise RuntimeError("local_response_effects.csv is missing or empty")
    required = {
        "source_id", "source_trace_id", "opponent", "candidate", "axis",
        "response_offset", "safety_clean",
        *(f"{metric}_sensitivity_per_action" for metric, _ in METRICS),
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"local-response fields missing: {missing}")
    if any(row.get("source_id") != expected_source_id for row in rows):
        raise RuntimeError("effect row Source ID mismatch")
    return rows


def action_vectors(levels: list[float]) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in vector) for vector in itertools.product(levels, repeat=3) if any(vector)]


def predicted_delta(rows: list[dict[str, Any]], vector: tuple[float, float, float], metric: str, horizon: int) -> float:
    values = [
        number(row[f"{metric}_sensitivity_per_action"]) * vector[int(row["axis"])]
        for row in rows
        if int(row["response_offset"]) == horizon
    ]
    if len(values) != 3:
        raise RuntimeError(f"expected three axis sensitivities for {metric}/h{horizon}, found {len(values)}")
    return sum(values)


def source_predictions(
    effects: list[dict[str, Any]],
    vector: tuple[float, float, float],
    horizon: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in effects:
        if int(row["response_offset"]) == horizon:
            key = (row["opponent"], row["candidate"], row["source_trace_id"])
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (opponent, candidate, source_trace_id), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "candidate": candidate,
            "source_trace_id": source_trace_id,
            "response_offset": horizon,
            "pulse_delta_0": vector[0],
            "pulse_delta_1": vector[1],
            "pulse_delta_2": vector[2],
            "safety_clean": all(bool_value(row["safety_clean"]) for row in rows),
        }
        for metric, output_name in METRICS:
            record[output_name] = predicted_delta(rows, vector, metric, horizon)
        record["geometry_metrics_positive"] = int(
            record["range_delta_m"] > 0.0
        ) + int(record["range_rate_delta_mps"] > 0.0)
        record["geometry_gate"] = record["geometry_metrics_positive"] >= 2
        record["energy_gate"] = record["energy_delta_m2ps2"] >= -5.0
        record["altitude_gate"] = abs(record["altitude_delta_m"]) <= 1.0
        record["source_gate"] = bool(record["safety_clean"] and record["geometry_gate"] and record["energy_gate"] and record["altitude_gate"])
        output.append(record)
    return output


def candidate_summary(
    predictions: list[dict[str, Any]],
    vector: tuple[float, float, float],
    config: dict[str, Any],
    expected_prediction_count: int | None = None,
) -> dict[str, Any]:
    screen = config["screen"]
    expected_count = expected_prediction_count or len(OPPONENTS) * len(CANDIDATES) * 3
    expected_per_opponent = expected_count // len(OPPONENTS)
    summary: dict[str, Any] = {
        "source_id": SOURCE_ID,
        "sequence_template": config["sequence"]["template"],
        "pulse_delta_0": vector[0],
        "pulse_delta_1": vector[1],
        "pulse_delta_2": vector[2],
        "prediction_count": len(predictions),
        "expected_prediction_count": expected_count,
        "runtime_validation_required": True,
    }
    for opponent in OPPONENTS:
        own = [row for row in predictions if row["opponent"] == opponent]
        prefix = opponent
        summary[f"{prefix}_prediction_count"] = len(own)
        summary[f"{prefix}_source_gate_count"] = sum(bool(row["source_gate"]) for row in own)
        summary[f"{prefix}_range_positive_count"] = sum(row["range_delta_m"] > 0.0 for row in own)
        summary[f"{prefix}_range_rate_positive_count"] = sum(row["range_rate_delta_mps"] > 0.0 for row in own)
        summary[f"{prefix}_safety_clean_count"] = sum(bool(row["safety_clean"]) for row in own)
        summary[f"{prefix}_energy_gate_count"] = sum(bool(row["energy_gate"]) for row in own)
        summary[f"{prefix}_altitude_gate_count"] = sum(bool(row["altitude_gate"]) for row in own)
        summary[f"{prefix}_mean_range_delta_m"] = sum(row["range_delta_m"] for row in own) / len(own) if own else float("nan")
        summary[f"{prefix}_mean_range_rate_delta_mps"] = sum(row["range_rate_delta_mps"] for row in own) / len(own) if own else float("nan")
    max_abs = max(abs(value) for value in vector)
    summary["action_bounds_gate"] = max_abs <= 0.08
    summary["action_rate_gate"] = max_abs <= 0.08
    summary["all_input_safety_clean"] = all(bool(row["safety_clean"]) for row in predictions)
    summary["both_opponents_geometry_gate"] = all(
        summary[f"{opponent}_range_positive_count"] == expected_per_opponent
        and summary[f"{opponent}_range_rate_positive_count"] == expected_per_opponent
        for opponent in OPPONENTS
    )
    summary["both_opponents_energy_gate"] = all(summary[f"{opponent}_energy_gate_count"] == expected_per_opponent for opponent in OPPONENTS)
    summary["both_opponents_altitude_gate"] = all(summary[f"{opponent}_altitude_gate_count"] == expected_per_opponent for opponent in OPPONENTS)
    summary["offline_sequence_gate"] = bool(
        len(predictions) == summary["expected_prediction_count"]
        and summary["action_bounds_gate"]
        and summary["action_rate_gate"]
        and summary["all_input_safety_clean"]
        and summary["both_opponents_geometry_gate"]
        and summary["both_opponents_energy_gate"]
        and summary["both_opponents_altitude_gate"]
    )
    summary["screen_thresholds"] = json.dumps(
        {
            "minimum_range_delta_m": screen["minimum_range_delta_m"],
            "minimum_range_rate_delta_mps": screen["minimum_range_rate_delta_mps"],
            "minimum_energy_delta_m2ps2": screen["minimum_energy_delta_m2ps2"],
            "maximum_absolute_altitude_delta_m": screen["maximum_absolute_altitude_delta_m"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_yaml(args.config.resolve())
    if config["experiment"]["source_id"] != SOURCE_ID:
        raise SystemExit("Source ID mismatch")
    input_root = (args.input_root or Path(config["evidence"]["local_response_analysis_root"])).resolve()
    output_root = (args.output_root or Path(config["experiment"]["output_root"])).resolve()
    effects = load_effects(input_root, config["evidence"]["local_response_source_id"])
    levels = [float(value) for value in config["sequence"]["action_levels"]]
    vectors = action_vectors(levels)
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    horizon_summary_rows: list[dict[str, Any]] = []
    for vector in vectors:
        all_predictions: list[dict[str, Any]] = []
        for horizon in config["sequence"]["response_offsets"]:
            all_predictions.extend(source_predictions(effects, vector, int(horizon)))
        prediction_rows.extend(all_predictions)
        per_horizon = []
        for horizon in config["sequence"]["response_offsets"]:
            horizon_summary = candidate_summary(
                source_predictions(effects, vector, int(horizon)), vector, config
            )
            horizon_summary["response_offset"] = int(horizon)
            per_horizon.append(horizon_summary)
            horizon_summary_rows.append(horizon_summary)
        summary = candidate_summary(
            all_predictions,
            vector,
            config,
            expected_prediction_count=(
                len(OPPONENTS)
                * len(CANDIDATES)
                * 3
                * len(config["sequence"]["response_offsets"])
            ),
        )
        summary["horizon_gate_count"] = sum(bool(row["offline_sequence_gate"]) for row in per_horizon)
        summary["both_horizons_gate"] = summary["horizon_gate_count"] == len(config["sequence"]["response_offsets"])
        summary["offline_sequence_gate"] = bool(summary["offline_sequence_gate"] and summary["both_horizons_gate"])
        summary_rows.append(summary)

    selected = [row for row in summary_rows if bool(row["offline_sequence_gate"])]
    manifest = {
        "source_id": SOURCE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "training_started": False,
        "jsbsim_started": False,
        "runtime_probe_started": False,
        "input_root": str(input_root),
        "input_source_id": config["evidence"]["local_response_source_id"],
        "candidate_count": len(vectors),
        "selected_count": len(selected),
        "horizon_selected_counts": {
            str(horizon): sum(
                bool(row["offline_sequence_gate"])
                for row in horizon_summary_rows
                if int(row["response_offset"]) == int(horizon)
            )
            for horizon in config["sequence"]["response_offsets"]
        },
        "sequence_template": config["sequence"]["template"],
        "model": config["sequence"]["model"],
        "cross_axis_interaction_unverified": True,
        "runtime_validation_required": True,
    }
    write_csv(output_root / "sequence_candidate_summary.csv", summary_rows)
    write_csv(output_root / "sequence_horizon_summary.csv", horizon_summary_rows)
    write_csv(output_root / "sequence_source_predictions.csv", prediction_rows)
    write_csv(output_root / "selected_sequence_candidates.csv", selected)
    write_json(output_root / "screen_manifest.json", manifest)
    write_json(output_root / "screen_result.json", {
        **manifest,
        "offline_screen_gate": bool(selected),
        "selected_candidates": selected,
        "decision": "eligible_for_one_runtime_probe" if selected else "stop_no_offline_sequence_candidate",
    })
    lines = [
        "# 劣势态势短时动作序列离线筛选记录",
        "",
        f"Source ID：`{SOURCE_ID}`",
        "",
        "本次只读取 V4 局部响应结果，没有运行 JSBSim、没有训练，也没有修改奖励、PID、滤波器或控制接口。",
        "",
        f"筛选模板：`{config['sequence']['template']}`，即 handoff 时施加一次动作偏置，下一步返回 nominal 动作。",
        f"候选动作向量数：{len(vectors)}；4 步和 12 步均通过的候选数：{len(selected)}。",
        "",
        "## 判定口径",
        "",
        "候选动作向量通过三轴动作范围和变化率约束；在 expert 与 end-to-end 两个对手、两个场景和三条重复记录上，要求距离和距离率均增加，并要求比能和高度变化满足配置中的筛选阈值。",
        "",
        "该筛选采用已验证单轴脉冲灵敏度的线性叠加。V4 没有采集跨轴同时作用或延迟脉冲，因此跨轴耦合仍未验证，筛选通过不等于运行时通过。",
        "",
        "## 后续",
        "",
        "若存在候选，只允许依据 selected_sequence_candidates.csv 设计一次新的非学习 strict-JSBSim runtime probe；若运行时不能复现几何方向，应冻结为接口/闭环边界证据，不进入训练。",
        "",
        "## 输出",
        "",
        "- `sequence_candidate_summary.csv`：全部候选及门禁结果。",
        "- `sequence_horizon_summary.csv`：分别按 4 步和 12 步统计的候选结果。",
        "- `sequence_source_predictions.csv`：候选在各对手、场景和响应时刻的预测。",
        "- `selected_sequence_candidates.csv`：通过离线门的候选。",
        "- `screen_result.json`：最终决策。",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "offline_sequence_screen_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "candidate_count": len(vectors), "selected_count": len(selected), "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
