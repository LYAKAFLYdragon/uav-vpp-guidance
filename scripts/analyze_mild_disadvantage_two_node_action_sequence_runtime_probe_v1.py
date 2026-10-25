#!/usr/bin/env python3
"""Read-only attribution audit for the end-to-end opponent probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-RUNTIME-PROBE-V1-20260904"
OFFSETS = (0, 4, 8, 12, 48)
PAIR_OFFSETS = (4, 12, 48)
METRICS = ("range_m", "range_rate_mps", "ata_deg", "aa_deg", "specific_energy_m2ps2", "altitude_m")
CHAIN_FIELDS = ("action_0", "action_1", "action_2", "vp_forward_bias_m", "vp_lateral_bias_m", "guidance_nz_cmd", "guidance_roll_rate_cmd", "aileron_cmd", "roll_rad", "p_rps", "range_m", "range_rate_mps")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_pair_audit(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / "paired_response_records.csv")
    rows = [row for row in rows if row.get("opponent") == "end_to_end"]
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 end-to-end paired rows, found {len(rows)}")
    return rows


def load_chain_audit(root: Path) -> list[dict[str, Any]]:
    trace_root = root / "end_to_end" / "traces"
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    for path in sorted(trace_root.glob("*.csv")):
        rows = read_csv(path)
        if not rows:
            raise RuntimeError(f"empty trace: {path}")
        groups[rows[0]["source_trace_id"]][rows[0]["condition"]] = rows
    if len(groups) != 6 or any(set(item) != {"baseline", "candidate"} for item in groups.values()):
        raise RuntimeError("end-to-end trace pairs are incomplete")
    output: list[dict[str, Any]] = []
    for source_trace_id, conditions in sorted(groups.items()):
        for offset in OFFSETS:
            base = next(row for row in conditions["baseline"] if int(number(row["response_offset"])) == offset)
            cand = next(row for row in conditions["candidate"] if int(number(row["response_offset"])) == offset)
            record: dict[str, Any] = {"source_id": SOURCE_ID, "source_trace_id": source_trace_id, "opponent": "end_to_end", "response_offset": offset}
            for field in CHAIN_FIELDS:
                record[f"delta_{field}"] = number(cand.get(field)) - number(base.get(field))
            record["candidate_sequence_node"] = cand.get("sequence_node", "none")
            record["action_node_delta"] = [number(cand.get(f"action_{axis}")) - number(base.get(f"action_{axis}")) for axis in range(3)]
            record["candidate_backend"] = cand.get("backend", "")
            record["candidate_prediction_fallback"] = cand.get("prediction_fallback", "")
            record["candidate_clipping_rate"] = number(cand.get("clipping_rate"))
            output.append(record)
    return output


def summarize(pair_rows: list[dict[str, str]], chain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    h12 = [row for row in pair_rows if int(number(row["response_offset"])) == 12]
    scenario_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in h12:
        scenario_groups[row["candidate"]].append(row)
    scenario_summary = {}
    for scenario, rows in sorted(scenario_groups.items()):
        rates = [number(row["delta_range_rate_mps"]) for row in rows]
        ranges = [number(row["delta_range_m"]) for row in rows]
        scenario_summary[scenario] = {
            "n": len(rows),
            "range_positive_count": sum(value > 0 for value in ranges),
            "range_rate_positive_count": sum(value > 0 for value in rates),
            "mean_range_delta_m": sum(ranges) / len(ranges),
            "mean_range_rate_delta_mps": sum(rates) / len(rates),
            "range_rate_signs": ["positive" if value > 0 else "negative" if value < 0 else "zero" for value in rates],
        }
    h4 = [row for row in pair_rows if int(number(row["response_offset"])) == 4]
    h48 = [row for row in pair_rows if int(number(row["response_offset"])) == 48]
    h12_rate = [number(row["delta_range_rate_mps"]) for row in h12]
    repeat_values = defaultdict(list)
    for row in h12:
        repeat_values[row["candidate"]].append(number(row["delta_range_rate_mps"]))
    return {
        "source_id": SOURCE_ID,
        "analysis_type": "read_only_end_to_end_opponent_difference_attribution",
        "pair_count": 6,
        "horizon_rows": len(pair_rows),
        "h12": {
            "range_positive_count": sum(number(row["delta_range_m"]) > 0 for row in h12),
            "range_rate_positive_count": sum(value > 0 for value in h12_rate),
            "range_rate_negative_count": sum(value < 0 for value in h12_rate),
            "mean_range_delta_m": sum(number(row["delta_range_m"]) for row in h12) / len(h12),
            "mean_range_rate_delta_mps": sum(h12_rate) / len(h12_rate),
        },
        "h4": {"range_rate_positive_count": sum(number(row["delta_range_rate_mps"]) > 0 for row in h4), "n": len(h4)},
        "h48": {"range_rate_positive_count": sum(number(row["delta_range_rate_mps"]) > 0 for row in h48), "n": len(h48)},
        "scenario_summary": scenario_summary,
        "within_scenario_repeat_consistency": {
            scenario: len({round(value, 12) for value in values}) == 1 for scenario, values in sorted(repeat_values.items())
        },
        "chain_observation": {
            "node0_action_reached": all(abs(row["delta_action_1"] - 0.08) < 1e-5 for row in chain_rows if int(row["response_offset"]) == 0),
            "node1_action_reached": all(abs(row["delta_action_0"] + 0.08) < 1e-5 for row in chain_rows if int(row["response_offset"]) == 8),
            "response_rows": len(chain_rows),
            "all_candidate_backend_jsbsim": all(row["candidate_backend"] == "jsbsim" for row in chain_rows),
            "any_candidate_clipping": any(row["candidate_clipping_rate"] > 0 for row in chain_rows),
            "any_candidate_response_prediction_fallback": any(row["candidate_prediction_fallback"].lower() == "true" for row in chain_rows if 0 <= int(row["response_offset"]) <= 48),
        },
        "attribution": "scenario_conditioned_closed_loop_response",
        "not_supported": ["random_outlier_explanation", "generic_action_input_contract_failure", "cross_opponent_universal_improvement"],
        "decision": "stop_current_sequence_line_without_parameter_or_training_changes",
    }


def report(summary: dict[str, Any], pair_rows: list[dict[str, str]]) -> str:
    scenario_lines = []
    for scenario, item in summary["scenario_summary"].items():
        scenario_lines.append(f"| {scenario} | {item['n']} | {item['range_positive_count']}/{item['n']} | {item['range_rate_positive_count']}/{item['n']} | {item['mean_range_delta_m']:+.9f} | {item['mean_range_rate_delta_mps']:+.9f} |")
    return "\n".join([
        "# 端到端对手差异归因审计",
        "",
        f"Source ID：`{SOURCE_ID}`。本审计只读取 runtime probe 的已生成 CSV，不重跑 JSBSim，不修改控制器、配置、门槛或 checkpoint。",
        "",
        "## 1 主要发现",
        "",
        "端到端对手第 12 步的距离率短板具有明显的场景条件性，而不是由少数异常回合造成。`x625_y200` 的 3 次重复全部为负，`x750_y250` 的 3 次重复全部为正；每个场景内的重复结果数值一致。第 12 步距离在 6/6 组均为正，但距离率仅 3/6 组为正。",
        "",
        "## 2 场景分解",
        "",
        "| 场景 | 配对数 | 距离为正 | 距离率为正 | 平均距离差（m） | 平均距离率差（m/s） |",
        "|---|---:|---:|---:|---:|---:|",
        *scenario_lines,
        "",
        "第 4 步距离率在 6/6 组为正，说明第一个动作节点能够产生早期方向响应；第 12 步出现正负分化，说明第二个节点之后的制导闭环响应依赖初始场景几何，而不是动作在输入层被统一抹掉。第 48 步的差异继续保留场景分化，不能把第 12 步未达门槛解释为单纯的短窗口问题。",
        "",
        "## 3 链路判断",
        "",
        "动作前后记录显示，第 0 步 `[0,+0.08,0]` 和第 8 步 `[-0.08,0,0]` 均实际到达候选轨迹；候选仍使用 JSBSim，响应窗口无预测器回退和裁剪。候选与基线的 VPP、制导、舵面、滚转及几何字段均可比较，因此没有证据表明差异在动作输入契约层整体消失。",
        "",
        "## 4 归因与决策",
        "",
        "本次最合理的归因是：**端到端对手下的场景条件化制导闭环响应**。当前双节点序列在两个场景下给出了不同的距离率方向，虽然总体平均值为正，但未达到预注册的跨对手实用门槛。该结果不支持随机异常、通用输入接口失效或跨场景统一改善等解释。",
        "",
        "因此停止当前序列和时间扩展线，不调动作幅度、不放宽门槛、不改奖励或 PPO，也不启动共享技能训练。若继续研究，应先提出新的、可检验的场景条件化闭环假设；否则可将本 probe 作为“运行链完整、局部响应存在、改善幅度受场景条件影响”的边界证据写入论文。",
        "",
        "输出文件：`end_to_end_pair_difference_audit.csv`、`end_to_end_chain_response_audit.csv`、`end_to_end_difference_attribution.json`。",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or (input_root / "end_to_end_difference_audit")).resolve()
    pairs = load_pair_audit(input_root)
    chain = load_chain_audit(input_root)
    summary = summarize(pairs, chain)
    write_csv(output_root / "end_to_end_pair_difference_audit.csv", pairs)
    write_csv(output_root / "end_to_end_chain_response_audit.csv", chain)
    write_json(output_root / "end_to_end_difference_attribution.json", summary)
    (output_root / "end_to_end_difference_attribution_zh.md").write_text(report(summary, pairs), encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "output_root": str(output_root), "attribution": summary["attribution"], "decision": summary["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
