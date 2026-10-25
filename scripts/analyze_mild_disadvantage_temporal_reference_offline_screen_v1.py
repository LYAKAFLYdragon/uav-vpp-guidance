#!/usr/bin/env python3
"""Audit paired local-response data and screen temporal action sequences offline.

This tool is intentionally conservative. It never runs JSBSim and never treats a
natural trajectory as a counterfactual local-response record. A sequence can be
screened only when the input explicitly identifies a handoff state, symmetric
interventions, horizon responses, and safety/provenance fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-TEMPORAL-REFERENCE-OFFLINE-SCREEN-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
REQUIRED_METRICS = (
    "range",
    "range_rate",
    "ata",
    "aa",
    "energy",
    "altitude",
)
HORIZONS = (4, 12, 48)
SAFETY_ALIASES = {
    "clipping": ("clipping_rate", "clipping", "action_clipped"),
    "backend_fallback": (
        "backend_fallback_occurred",
        "backend_fallback",
    ),
    "prediction_fallback": (
        "prediction_fallback",
        "predictor_fallback",
    ),
    "terminal": ("terminal_reason", "terminal", "reason"),
}
METRIC_ALIASES = {
    "range": ("range_m", "range", "distance_m"),
    "range_rate": ("range_rate_mps", "range_rate", "distance_rate_mps"),
    "ata": ("ata_deg", "ata", "ata_rad"),
    "aa": ("aa_deg", "aa", "aa_rad"),
    "energy": (
        "specific_energy",
        "specific_energy_m2ps2",
        "energy_mjkg",
        "energy_jkg",
        "energy",
        "energy_proxy",
    ),
    "altitude": ("altitude_m", "altitude", "own_altitude_m"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, set(reader.fieldnames or ())


def _flatten_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in ("records", "rows", "samples", "trace", "data", "episodes"):
        records = _flatten_records(value.get(key))
        if records:
            return records
    return []


def _load_json(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = _flatten_records(value)
    fields = set().union(*(row.keys() for row in rows)) if rows else set()
    return rows, fields


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first_present(fields: set[str], aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in fields), None)


def _metric_horizon_fields(fields: set[str], metric: str) -> dict[int, str]:
    found: dict[int, str] = {}
    aliases = METRIC_ALIASES[metric]
    for horizon in HORIZONS:
        patterns = []
        for alias in aliases:
            patterns.extend(
                (
                    f"{alias}_{horizon}",
                    f"{alias}_delta_{horizon}",
                    f"{alias}_h{horizon}",
                    f"delta_{alias}_{horizon}",
                    f"h{horizon}_{alias}",
                )
            )
        match = _first_present(fields, tuple(patterns))
        if match:
            found[horizon] = match
    return found


def _has_horizon_rows(fields: set[str]) -> bool:
    return any(name in fields for name in ("horizon_step", "relative_step", "response_horizon"))


def _read_file(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    return _load_json(path)


def _classify_file(path: Path, fields: set[str]) -> str:
    if "state_hash_before_action" in fields and "action_0" in fields:
        return "trajectory_with_action_hash"
    if any("pulse" in field.lower() or "delta_action" in field.lower() for field in fields):
        return "possible_local_response"
    if any("handoff" in field.lower() for field in fields):
        return "possible_handoff_record"
    return "other"


def _audit_file(path: Path) -> dict[str, Any]:
    try:
        rows, fields = _read_file(path)
    except Exception as exc:  # noqa: BLE001 - audit must report bad artifacts
        return {
            "path": str(path),
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    metric_fields = {
        metric: _metric_horizon_fields(fields, metric) for metric in REQUIRED_METRICS
    }
    handoff_fields = [
        name
        for name in (
            "handoff_id",
            "matched_handoff_id",
            "handoff_state_hash",
            "is_handoff",
            "handoff_event",
            "handoff_step",
            "phase_at_handoff",
        )
        if name in fields
    ]
    pair_fields = [
        name
        for name in (
            "pair_id",
            "matched_pair_id",
            "state_id",
            "intervention_id",
            "condition",
            "variant",
        )
        if name in fields
    ]
    pulse_fields = [
        name
        for name in fields
        if any(token in name.lower() for token in ("pulse", "delta_action", "action_delta"))
    ]
    opponent_field = _first_present(fields, ("opponent", "opponent_name"))
    safety_fields = {
        key: _first_present(fields, aliases) for key, aliases in SAFETY_ALIASES.items()
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "readable": True,
        "record_count": len(rows),
        "field_count": len(fields),
        "fields": sorted(fields),
        "classification": _classify_file(path, fields),
        "has_state_hash": "state_hash_before_action" in fields,
        "handoff_fields": handoff_fields,
        "pair_fields": pair_fields,
        "pulse_fields": sorted(pulse_fields),
        "opponent_field": opponent_field,
        "horizon_metric_fields": metric_fields,
        "has_horizon_rows": _has_horizon_rows(fields),
        "safety_fields": safety_fields,
        "has_action_3d": all(f"action_{index}" in fields for index in range(3)),
    }


def _scan(root: Path) -> dict[str, Any]:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json"}
        and path.name not in {"offline_screen_result.json", "verification.json"}
    ]
    files = [_audit_file(path) for path in sorted(candidates)]
    readable = [item for item in files if item.get("readable")]
    trace_files = [item for item in readable if item.get("classification") == "trajectory_with_action_hash"]
    response_files = [
        item
        for item in readable
        if item.get("classification") in {"possible_local_response", "possible_handoff_record"}
    ]
    all_fields = set().union(*(item.get("fields", []) for item in readable)) if readable else set()
    return {
        "root": str(root),
        "root_exists": root.is_dir(),
        "file_count": len(files),
        "readable_file_count": len(readable),
        "trace_file_count": len(trace_files),
        "possible_response_file_count": len(response_files),
        "all_fields": sorted(all_fields),
        "files": files,
    }


def _evaluate(scan: dict[str, Any]) -> dict[str, Any]:
    readable = [item for item in scan["files"] if item.get("readable")]
    trace_files = [item for item in readable if item.get("classification") == "trajectory_with_action_hash"]
    handoff_files = [item for item in readable if item.get("handoff_fields")]
    paired_files = [item for item in readable if item.get("pair_fields")]
    pulse_files = [item for item in readable if item.get("pulse_fields")]
    response_files = [
        item
        for item in readable
        if item.get("has_horizon_rows")
        or all(item.get("horizon_metric_fields", {}).get(metric) for metric in REQUIRED_METRICS)
    ]
    opponents = set()
    for item in readable:
        field = item.get("opponent_field")
        if field:
            try:
                rows, _ = _read_file(Path(item["path"]))
                opponents.update(str(row.get(field, "")) for row in rows if row.get(field))
            except Exception:
                pass
    safety_complete = all(
        _first_present(set().union(*(item.get("fields", []) for item in readable)), aliases)
        for aliases in SAFETY_ALIASES.values()
    ) if readable else False
    gates = {
        "real_handoff_state_identified": bool(handoff_files),
        "action_before_hash_present": bool(trace_files) and all(item["has_state_hash"] for item in trace_files),
        "paired_state_identity_present": bool(paired_files),
        "symmetric_three_axis_interventions_present": bool(pulse_files),
        "four_twelve_forty_eight_response_present": bool(response_files),
        "two_opponents_separately_identified": set(EXPECTED_OPPONENTS).issubset(opponents),
        "safety_and_terminal_fields_present": safety_complete,
    }
    missing = [name for name, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "missing_gates": missing,
        "observed_opponents": sorted(opponents),
        "offline_screen_authorized": not missing,
        "runtime_probe_authorized": False,
        "runtime_probe_executed": False,
        "decision": (
            "eligible_for_offline_sequence_screen"
            if not missing
            else "blocked_missing_paired_local_response_data"
        ),
    }


def _write_report(output_root: Path, scan: dict[str, Any], decision: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "source_id": SOURCE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "read_only_paired_local_response_offline_screen",
        "training_started": False,
        "jsbsim_started": False,
        "compiler_runtime_probe_started": False,
        "input_scan": scan,
        "decision": decision,
        "interpretation": (
            "Existing inputs contain natural telemetry trajectories but no explicit paired local-response contract; "
            "do not estimate a response matrix or authorize a compiler runtime probe."
            if decision["decision"] != "eligible_for_offline_sequence_screen"
            else "Input contract is complete; sequence screening may proceed offline."
        ),
    }
    (output_root / "offline_screen_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# 时间扩展参考轨迹离线动作序列筛选审计",
        "",
        f"Source ID：`{SOURCE_ID}`",
        "",
        "## 1. 本次操作",
        "",
        "本次仅读取候选数据，未运行 JSBSim、未训练、未修改控制器，也未启动编译器运行时探针。",
        f"输入目录：`{scan['root']}`",
        f"可读取文件：{scan['readable_file_count']}/{scan['file_count']}；含动作前状态哈希的轨迹文件：{scan['trace_file_count']}。",
        "",
        "## 2. 严格门禁",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]
    labels = {
        "real_handoff_state_identified": "真实 handoff 状态可识别",
        "action_before_hash_present": "动作前状态哈希存在",
        "paired_state_identity_present": "成对状态身份可识别",
        "symmetric_three_axis_interventions_present": "三轴对称动作干预可识别",
        "four_twelve_forty_eight_response_present": "4/12/48 步局部响应可识别",
        "two_opponents_separately_identified": "expert 与 end-to-end 可分开统计",
        "safety_and_terminal_fields_present": "安全与终端字段完整",
    }
    for key, passed in decision["gates"].items():
        lines.append(f"| {labels[key]} | {'通过' if passed else '未通过'} |")
    lines.extend(
        [
            "",
            "## 3. 判定",
            "",
            f"离线动作序列筛选资格：**{'通过' if decision['offline_screen_authorized'] else '不通过'}**。",
            f"运行时编译器 probe：**{'授权' if decision['runtime_probe_authorized'] else '不授权'}**。",
            "",
            "当前数据虽包含连续动作和动作前状态哈希，但没有显式的 handoff 配对标识、正负对称干预标识和 4/12/48 步反事实响应。因此不能估计局部响应矩阵，也不能从自然轨迹反推可执行动作序列。",
            "",
            "## 4. 后续唯一入口",
            "",
            "补充数据时，必须为每条记录同时提供：对手、handoff 状态哈希、配对编号、动作轴、正负符号、动作幅值、无干预基线、4/12/48 步的六类几何响应，以及裁剪、fallback、reset、terminal 字段。契约通过后，才重新运行本离线筛选器；筛选出跨两个主对手共同可行且满足动作边界/变化率/残差门的序列后，才有资格另行设计 runtime probe。",
            "",
            "## 5. 输出",
            "",
            "- `offline_screen_result.json`：输入文件哈希、字段审计和门禁判定。",
            "- 本报告：本次离线筛选的可复核结论。",
        ]
    )
    (output_root / "offline_screen_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve()
    scan = _scan(root) if root.is_dir() else {
        "root": str(root),
        "root_exists": False,
        "file_count": 0,
        "readable_file_count": 0,
        "trace_file_count": 0,
        "possible_response_file_count": 0,
        "all_fields": [],
        "files": [],
    }
    decision = _evaluate(scan)
    _write_report(args.output_root.resolve(), scan, decision)
    print(json.dumps({"source_id": SOURCE_ID, **decision}, indent=2, ensure_ascii=False))
    return 0 if decision["offline_screen_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
