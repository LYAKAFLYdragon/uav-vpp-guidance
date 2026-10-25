#!/usr/bin/env python3
"""Screen two-node action sequences using verified V4 local responses.

The first pulse is applied at handoff (node 0). The second pulse is applied at
step 8 (node 1), with nominal action between and after both pulses. V4 has no
direct delayed-pulse measurement, so the 12-step prediction is explicitly a
linear-superposition estimate: node 0 uses the measured 12-step sensitivity
and node 1 uses the measured 4-step sensitivity. No JSBSim is run.
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

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-OFFLINE-SCREEN-V1-20260904"
LOCAL_SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V4-20260904"
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
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def action_vectors(levels: list[float]) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in vector) for vector in itertools.product(levels, repeat=3)]


def load_effects(root: Path, expected_source_id: str) -> list[dict[str, str]]:
    report_path = root / "local_response_analysis_report.json"
    effects_path = root / "local_response_effects.csv"
    if not report_path.is_file() or not effects_path.is_file():
        raise RuntimeError("verified V4 local-response analysis artifacts are missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("source_id") != expected_source_id or report.get("analysis_gate") != "pass":
        raise RuntimeError("V4 local-response analysis gate or Source ID mismatch")
    rows = read_csv(effects_path)
    required = {
        "source_id", "source_trace_id", "opponent", "candidate", "axis",
        "response_offset", "safety_clean",
        *(f"{metric}_sensitivity_per_action" for metric, _ in METRICS),
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("V4 local-response effect fields are incomplete")
    if any(row.get("source_id") != expected_source_id for row in rows):
        raise RuntimeError("V4 local-response effect Source ID mismatch")
    if set(row["opponent"] for row in rows) != set(OPPONENTS):
        raise RuntimeError("V4 local-response opponents are incomplete")
    return rows


def sensitivity(rows: list[dict[str, str]], metric: str, horizon: int, vector: tuple[float, float, float]) -> float:
    selected = [row for row in rows if int(row["response_offset"]) == horizon]
    if len(selected) != 3 or {int(row["axis"]) for row in selected} != {0, 1, 2}:
        raise RuntimeError(f"expected three axis sensitivities for {metric}/h{horizon}")
    return sum(number(row[f"{metric}_sensitivity_per_action"]) * vector[int(row["axis"])] for row in selected)


def sequence_prediction(
    effects: list[dict[str, str]],
    node0: tuple[float, float, float],
    node1: tuple[float, float, float],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in effects:
        groups.setdefault((row["opponent"], row["candidate"], row["source_trace_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (opponent, candidate, source_trace_id), rows in sorted(groups.items()):
        base: dict[str, Any] = {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "candidate": candidate,
            "source_trace_id": source_trace_id,
            "node0_delta_0": node0[0],
            "node0_delta_1": node0[1],
            "node0_delta_2": node0[2],
            "node1_delta_0": node1[0],
            "node1_delta_1": node1[1],
            "node1_delta_2": node1[2],
            "input_safety_clean": all(bool_value(row["safety_clean"]) for row in rows),
            "model": "node0_h12_plus_node1_h4",
        }
        for horizon in HORIZONS:
            record = dict(base)
            record["response_offset"] = horizon
            for metric, output_name in METRICS:
                node0_horizon = sensitivity(rows, metric, horizon, node0)
                node1_horizon = 0.0 if horizon == 4 else sensitivity(rows, metric, 4, node1)
                record[f"{output_name}"] = node0_horizon + node1_horizon
                record[f"{output_name}_node0_component"] = node0_horizon
                record[f"{output_name}_node1_component"] = node1_horizon
            record["geometry_positive_count"] = int(record["range_delta_m"] > 0.0) + int(record["range_rate_delta_mps"] > 0.0)
            record["geometry_gate"] = record["geometry_positive_count"] >= 2
            record["energy_gate"] = record["energy_delta_m2ps2"] >= -5.0
            record["altitude_gate"] = abs(record["altitude_delta_m"]) <= 1.0
            record["source_gate"] = bool(record["input_safety_clean"] and record["geometry_gate"] and record["energy_gate"] and record["altitude_gate"])
            output.append(record)
    return output


def candidate_summary(
    predictions: list[dict[str, Any]],
    node0: tuple[float, float, float],
    node1: tuple[float, float, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    screen = config["screen"]
    summary: dict[str, Any] = {
        "source_id": SOURCE_ID,
        "node0_delta_0": node0[0],
        "node0_delta_1": node0[1],
        "node0_delta_2": node0[2],
        "node1_delta_0": node1[0],
        "node1_delta_1": node1[1],
        "node1_delta_2": node1[2],
        "prediction_count": len(predictions),
        "expected_prediction_count": len(OPPONENTS) * len(CANDIDATES) * 3 * len(HORIZONS),
        "second_node_nonzero": any(abs(value) > 0.0 for value in node1),
        "action_bounds_gate": max(max(abs(value) for value in node0), max(abs(value) for value in node1)) <= 0.08,
        "action_rate_gate": max(max(abs(value) for value in node0), max(abs(value) for value in node1)) <= 0.08,
        "input_safety_gate": all(bool(row["input_safety_clean"]) for row in predictions),
        "runtime_validation_required": True,
    }
    for opponent in OPPONENTS:
        own = [row for row in predictions if row["opponent"] == opponent]
        for horizon in HORIZONS:
            scoped = [row for row in own if int(row["response_offset"]) == horizon]
            prefix = f"{opponent}_h{horizon}"
            summary[f"{prefix}_count"] = len(scoped)
            summary[f"{prefix}_range_positive_count"] = sum(row["range_delta_m"] > 0.0 for row in scoped)
            summary[f"{prefix}_range_rate_positive_count"] = sum(row["range_rate_delta_mps"] > 0.0 for row in scoped)
            summary[f"{prefix}_energy_gate_count"] = sum(bool(row["energy_gate"]) for row in scoped)
            summary[f"{prefix}_altitude_gate_count"] = sum(bool(row["altitude_gate"]) for row in scoped)
            summary[f"{prefix}_safety_clean_count"] = sum(bool(row["input_safety_clean"]) for row in scoped)
            summary[f"{prefix}_mean_range_delta_m"] = sum(row["range_delta_m"] for row in scoped) / len(scoped) if scoped else float("nan")
            summary[f"{prefix}_mean_range_rate_delta_mps"] = sum(row["range_rate_delta_mps"] for row in scoped) / len(scoped) if scoped else float("nan")
            summary[f"{prefix}_gate"] = bool(
                len(scoped) == len(CANDIDATES) * 3
                and summary[f"{prefix}_range_positive_count"] == len(scoped)
                and summary[f"{prefix}_range_rate_positive_count"] == len(scoped)
                and summary[f"{prefix}_energy_gate_count"] == len(scoped)
                and summary[f"{prefix}_altitude_gate_count"] == len(scoped)
            )
    summary["both_opponents_h4_gate"] = all(summary[f"{opponent}_h4_gate"] for opponent in OPPONENTS)
    summary["both_opponents_h12_gate"] = all(summary[f"{opponent}_h12_gate"] for opponent in OPPONENTS)
    summary["offline_sequence_gate"] = bool(
        summary["prediction_count"] == summary["expected_prediction_count"]
        and summary["second_node_nonzero"]
        and summary["action_bounds_gate"]
        and summary["action_rate_gate"]
        and summary["input_safety_gate"]
        and summary["both_opponents_h4_gate"]
        and summary["both_opponents_h12_gate"]
    )
    summary["thresholds"] = json.dumps(
        {
            "minimum_range_delta_m": screen["minimum_range_delta_m"],
            "minimum_range_rate_delta_mps": screen["minimum_range_rate_delta_mps"],
            "minimum_energy_delta_m2ps2": screen["minimum_energy_delta_m2ps2"],
            "maximum_absolute_altitude_delta_m": screen["maximum_absolute_altitude_delta_m"],
        },
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
    for node0 in vectors:
        for node1 in vectors:
            predictions = sequence_prediction(effects, node0, node1)
            prediction_rows.extend(predictions)
            summary_rows.append(candidate_summary(predictions, node0, node1, config))
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
        "candidate_count": len(summary_rows),
        "selected_count": len(selected),
        "node0_vector_count": len(vectors),
        "node1_vector_count": len(vectors),
        "horizons": list(HORIZONS),
        "node_steps": config["sequence"]["node_steps"],
        "model": config["sequence"]["model"],
        "cross_axis_interaction_unverified": True,
        "delayed_pulse_h8_response_approximated_by_h4": True,
        "selected_candidates_are_directional_only": True,
        "practical_magnitude_gate_required_before_runtime": True,
        "runtime_validation_required": True,
    }
    write_csv(output_root / "two_node_candidate_summary.csv", summary_rows)
    write_csv(output_root / "two_node_source_predictions.csv", prediction_rows)
    write_csv(output_root / "selected_two_node_candidates.csv", selected)
    write_json(output_root / "two_node_screen_manifest.json", manifest)
    write_json(output_root / "two_node_screen_result.json", {
        **manifest,
        "offline_screen_gate": bool(selected),
        "selected_candidates": selected,
        "decision": "eligible_for_one_runtime_probe" if selected else "stop_no_two_node_offline_candidate",
    })
    lines = [
        "# 劣势态势两节点短时动作序列离线设计记录",
        "",
        f"Source ID：`{SOURCE_ID}`",
        "",
        "本次只读取 V4 局部响应结果，没有运行 JSBSim、没有训练，也没有修改控制器。",
        "",
        "序列结构：第 0 步施加第一个动作节点，第 8 步施加第二个动作节点；其余步骤返回 nominal 动作。",
        f"候选数：{len(summary_rows)}；4 步和 12 步均通过的候选数：{len(selected)}。",
        "",
        "12 步结果采用第一个节点的实测 12 步灵敏度与第二个节点的实测 4 步灵敏度线性叠加。该估计没有直接观测跨节点耦合和第 8 步至第 12 步的真实响应，只能用于离线排序，不能代替运行时验证。",
        "",
        "本次阈值为零，仅表示方向一致，不表示改善幅度具有实际意义。运行时 probe 前还必须单独预注册最小距离和距离率改善幅度；本输出不会自动授权运行或训练。",
        "",
        "输出：`two_node_candidate_summary.csv`、`two_node_source_predictions.csv`、`selected_two_node_candidates.csv`、`two_node_screen_result.json`。",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "two_node_offline_screen_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "candidate_count": len(summary_rows), "selected_count": len(selected), "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
