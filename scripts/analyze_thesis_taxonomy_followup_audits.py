#!/usr/bin/env python3
"""Read-only follow-up audits for the frozen 30-scenario taxonomy evidence.

This tool performs two narrow analyses selected mechanically from the Step 4
diagnostic table:

1. expert/disadvantage all-method-failure candidates versus the same physical
   scenarios against the end-to-end opponent, using the two fixed specialists.
2. expert/head-on routing opportunities in which fixed crossing wins while PPO
   loses.

It neither loads a policy nor creates an environment.  All selections derive
from the frozen Step 4 artifact, and all source raw files are hashed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "THESIS-TAXONOMY-ABLATION30-V1"
PPO_METHOD = "canonical_ppo_high_level_policy"
FIXED_HEAD_ON_METHOD = "head_on_vpp_specialist_no_routing"
FIXED_CROSSING_METHOD = "crossing_vpp_specialist_no_routing"
FIXED_METHODS = (FIXED_HEAD_ON_METHOD, FIXED_CROSSING_METHOD)
EARLY_WINDOW_MAX_S = 10.0
EARLY_DIVERGENCE_THRESHOLDS = {
    "target_speed_mps": 20.0,
    "target_altitude_m": 300.0,
    "target_specific_energy_m2_s2": 3000.0,
    "target_attack_zone_step_fraction": 0.10,
    "range_rate_mps": 25.0,
}
LATE_DIVERGENCE_THRESHOLDS = {
    "mean_range_rate_mps": 25.0,
    "mean_vp_forward_bias_m": 250.0,
    "mean_vp_lateral_bias_m": 250.0,
    "mean_vp_vertical_offset_m": 150.0,
    "mean_abs_nz_tracking_error_g": 0.5,
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _mean(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return sum(finite) / len(finite) if finite else None


def _max(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return max(finite) if finite else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(value), sort_keys=True)
                    if isinstance(value, (Mapping, list, tuple))
                    else _json_safe(value)
                    for key, value in row.items()
                }
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position(frame: Mapping[str, Any], prefix: str) -> tuple[float, float, float] | None:
    values = [_finite(frame.get(f"{prefix}_pos_{axis}")) for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _velocity_between(
    first: Mapping[str, Any], second: Mapping[str, Any], prefix: str
) -> tuple[float, float, float] | None:
    first_pos = _position(first, prefix)
    second_pos = _position(second, prefix)
    first_time = _finite(first.get("time_s"))
    second_time = _finite(second.get("time_s"))
    if first_pos is None or second_pos is None or first_time is None or second_time is None:
        return None
    dt = second_time - first_time
    if dt <= 1e-9:
        return None
    return tuple((later - earlier) / dt for earlier, later in zip(first_pos, second_pos))  # type: ignore[return-value]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _phase_filter(kind: str) -> Callable[[Mapping[str, Any]], bool]:
    if kind == "early_t_le_10s":
        return lambda frame: (
            (time_s := _finite(frame.get("time_s"))) is not None
            and time_s <= EARLY_WINDOW_MAX_S
        )
    if kind == "post_merge":
        return lambda frame: _bool(frame.get("post_merge"))
    raise ValueError(f"Unknown phase kind: {kind}")


def _first_true_time(trajectory: Sequence[Mapping[str, Any]], key: str) -> float | None:
    for frame in trajectory:
        if _bool(frame.get(key)):
            return _finite(frame.get("time_s"))
    return None


def _mode_fractions(trajectory: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    counts: Counter[str] = Counter(
        str(frame[key])
        for frame in trajectory
        if frame.get(key) not in (None, "", "null")
    )
    total = sum(counts.values())
    return {mode: count / total for mode, count in sorted(counts.items())} if total else {}


def _first_mode(trajectory: Sequence[Mapping[str, Any]], key: str) -> str | None:
    for frame in trajectory:
        value = frame.get(key)
        if value not in (None, "", "null"):
            return str(value)
    return None


def _first_switch_step(trajectory: Sequence[Mapping[str, Any]]) -> int | None:
    for frame in trajectory:
        value = _finite(frame.get("commander_first_switch_step"))
        if value is not None:
            return int(value)
    for frame in trajectory:
        if _bool(frame.get("commander_mode_switched")):
            step = _finite(frame.get("step"))
            return int(step) if step is not None else None
    return None


def _first_mode_time(trajectory: Sequence[Mapping[str, Any]], mode: str) -> float | None:
    for frame in trajectory:
        if frame.get("commander_mode_name") == mode:
            return _finite(frame.get("time_s"))
    return None


def _frame_summary(trajectory: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    """Summarize a predeclared time/phase window without episode selection."""

    predicate = _phase_filter(phase)
    frames = [frame for frame in trajectory if predicate(frame)]
    target_speeds: list[float] = []
    target_energies: list[float] = []
    for first, second in zip(trajectory, trajectory[1:]):
        if not predicate(first):
            continue
        velocity = _velocity_between(first, second, "target")
        target_position = _position(first, "target")
        if velocity is not None:
            target_speeds.append(_norm(velocity))
            if target_position is not None:
                target_energies.append(9.80665 * target_position[2] + 0.5 * _norm(velocity) ** 2)
    vpp_vertical_offsets = [
        vp_z - ego_z
        for frame in frames
        if (vp_z := _finite(frame.get("vp_pos_z"))) is not None
        and (ego_z := _finite(frame.get("ego_pos_z"))) is not None
    ]
    response_errors = [
        abs(command - actual)
        for frame in frames
        if (command := _finite(frame.get("nz_cmd"))) is not None
        and (actual := _finite(frame.get("nz_g"))) is not None
    ]
    return {
        "phase": phase,
        "frame_count": len(frames),
        "time_span_s": (
            max(time_values) - min(time_values)
            if (time_values := [
                time_s
                for frame in frames
                if (time_s := _finite(frame.get("time_s"))) is not None
            ])
            else None
        ),
        "mean_target_speed_mps": _mean(target_speeds),
        "mean_target_altitude_m": _mean(frame.get("target_pos_z") for frame in frames),
        "mean_target_specific_energy_m2_s2": _mean(target_energies),
        "target_attack_zone_step_fraction": (
            sum(_bool(frame.get("target_in_attack_zone")) for frame in frames) / len(frames)
            if frames
            else None
        ),
        "mean_range_m": _mean(frame.get("range_m") for frame in frames),
        "mean_range_rate_mps": _mean(frame.get("range_rate_mps") for frame in frames),
        "mean_vp_forward_bias_m": _mean(frame.get("vp_forward_bias_m") for frame in frames),
        "mean_vp_lateral_bias_m": _mean(frame.get("vp_lateral_bias_m") for frame in frames),
        "mean_vp_vertical_offset_m": _mean(vpp_vertical_offsets),
        "mean_abs_nz_tracking_error_g": _mean(response_errors),
        "max_abs_nz_tracking_error_g": _max(response_errors),
        "nz_saturation_step_fraction": (
            sum(_bool(frame.get("nz_saturated")) for frame in frames) / len(frames)
            if frames
            else None
        ),
        "roll_rate_saturation_step_fraction": (
            sum(_bool(frame.get("roll_rate_saturated")) for frame in frames) / len(frames)
            if frames
            else None
        ),
        "throttle_saturation_step_fraction": (
            sum(_bool(frame.get("throttle_saturated")) for frame in frames) / len(frames)
            if frames
            else None
        ),
    }


def _load_raw_summary(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_bytes())
    trajectory = raw.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"No trajectory in {path}")
    if not all(isinstance(frame, Mapping) for frame in trajectory):
        raise ValueError(f"Malformed trajectory in {path}")
    constraints = [
        str(frame["commander_mode_constraint_reason"])
        for frame in trajectory
        if _bool(frame.get("commander_mode_constraint_triggered"))
        and frame.get("commander_mode_constraint_reason") not in (None, "", "null")
    ]
    return {
        "source_raw_path": str(path),
        "source_raw_sha256": _sha256_file(path),
        "outcome": "win" if _bool(raw.get("win")) else "loss" if _bool(raw.get("loss")) else "other",
        "termination_reason": raw.get("termination_reason") or raw.get("combat_reason"),
        "terminal_category": _terminal_category(raw.get("termination_reason") or raw.get("combat_reason")),
        "initial_target_attack_zone_time_s": _first_true_time(trajectory, "target_in_attack_zone"),
        "initial_ego_attack_zone_time_s": _first_true_time(trajectory, "ego_in_attack_zone"),
        "early": _frame_summary(trajectory, "early_t_le_10s"),
        "post_merge": _frame_summary(trajectory, "post_merge"),
        "first_requested_mode": _first_mode(trajectory, "commander_requested_mode_name"),
        "first_effective_mode": _first_mode(trajectory, "commander_mode_name"),
        "first_crossing_effective_time_s": _first_mode_time(trajectory, "crossing_specialist"),
        "first_switch_step": _first_switch_step(trajectory),
        "effective_mode_fractions": _mode_fractions(trajectory, "commander_mode_name"),
        "requested_mode_fractions": _mode_fractions(trajectory, "commander_requested_mode_name"),
        "guard_override_step_fraction": (
            sum(_bool(frame.get("commander_mode_constraint_triggered")) for frame in trajectory)
            / len(trajectory)
        ),
        "guard_override_reasons": sorted(set(constraints)),
    }


def _terminal_category(reason: Any) -> str:
    text = str(reason or "unknown").lower()
    if "ego" in text and ("crash" in text or "out_of_bounds" in text or "oob" in text):
        return "ego_crash_or_oob"
    if "target" in text and ("crash" in text or "out_of_bounds" in text or "oob" in text):
        return "target_crash_or_oob"
    if "ego" in text and "killed" in text:
        return "ego_killed"
    if "target" in text and "killed" in text:
        return "target_killed"
    if "timeout" in text and "advantage" in text:
        return "timeout_hp_advantage"
    if "timeout" in text and "disadvantage" in text:
        return "timeout_hp_disadvantage"
    if "timeout" in text:
        return "timeout"
    return "other_or_unknown"


def _flatten(prefix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            result.update(_flatten(f"{prefix}{key}_", value))
        else:
            result[f"{prefix}{key}"] = value
    return result


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body: list[str] = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append("-" if not math.isfinite(value) else f"{value:.3f}")
            elif isinstance(value, (Mapping, list, tuple)):
                values.append(json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True))
            elif value is None:
                values.append("-")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _load_base_artifacts(analysis_root: Path) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]], list[dict[str, str]]]:
    geometry_path = analysis_root / "geometry_taxonomy_audit.json"
    diagnostic_path = analysis_root / "diagnostic_label_by_episode.csv"
    if not geometry_path.is_file() or not diagnostic_path.is_file():
        raise ValueError("Base analysis root is missing geometry or diagnostic artifacts")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    records = geometry.get("episode_records")
    if not isinstance(records, list):
        raise ValueError("Base geometry audit has no episode records")
    record_map = {
        (str(record["opponent_stage"]), str(record["scenario"]), str(record["method"])): record
        for record in records
    }
    with diagnostic_path.open("r", encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    return record_map, diagnostics


def _selected_records(
    record_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    split: str,
    scenario: str,
    method: str,
) -> Mapping[str, Any]:
    try:
        record = record_map[(split, scenario, method)]
    except KeyError as error:
        raise ValueError(f"Missing selected raw record: {split}/{scenario}/{method}") from error
    if not record.get("source_raw_path"):
        raise ValueError(f"Selected raw record has no source path: {split}/{scenario}/{method}")
    return record


def _paired_delta(expert: Mapping[str, Any], end_to_end: Mapping[str, Any]) -> dict[str, Any]:
    deltas: dict[str, float | None] = {}
    signals: list[str] = []
    for field, threshold in EARLY_DIVERGENCE_THRESHOLDS.items():
        expert_value = _finite(expert["early"].get(f"mean_{field}"))
        end_value = _finite(end_to_end["early"].get(f"mean_{field}"))
        delta = expert_value - end_value if expert_value is not None and end_value is not None else None
        deltas[f"early_delta_{field}"] = delta
        if delta is not None and abs(delta) >= threshold:
            signals.append(field)
    early_label = (
        "early_opponent_behavior_divergence_candidate"
        if len(signals) >= 2
        else "no_predeclared_early_divergence_threshold"
    )
    late_deltas: dict[str, float | None] = {}
    late_signals: list[str] = []
    for field, threshold in LATE_DIVERGENCE_THRESHOLDS.items():
        expert_value = _finite(expert["post_merge"].get(field))
        end_value = _finite(end_to_end["post_merge"].get(field))
        delta = expert_value - end_value if expert_value is not None and end_value is not None else None
        late_deltas[f"post_merge_delta_{field}"] = delta
        if delta is not None and abs(delta) >= threshold:
            late_signals.append(field)
    terminal_outcome_differs = expert["outcome"] != end_to_end["outcome"]
    if len(signals) >= 2:
        divergence_label = "early_opponent_behavior_divergence_candidate"
    elif terminal_outcome_differs and late_signals:
        divergence_label = "late_geometry_or_execution_divergence_candidate"
    elif terminal_outcome_differs:
        divergence_label = "terminal_only_or_unresolved_divergence"
    else:
        divergence_label = "shared_or_unresolved_failure_candidate"
    return {
        **deltas,
        **late_deltas,
        "early_divergence_signal_count": len(signals),
        "early_divergence_signals": signals,
        "early_divergence_label": early_label,
        "late_divergence_signal_count": len(late_signals),
        "late_divergence_signals": late_signals,
        "divergence_label": divergence_label,
        "terminal_outcome_differs": terminal_outcome_differs,
        "terminal_category_differs": expert["terminal_category"] != end_to_end["terminal_category"],
    }


def _opponent_divergence_audit(
    record_map: Mapping[tuple[str, str, str], Mapping[str, Any]], diagnostics: Sequence[Mapping[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        row
        for row in diagnostics
        if row["opponent_stage"] == "expert"
        and row["initial_class"] == "disadvantage"
        and "library_gap_candidate" in row["labels"].split(";")
        and "all_method_failure" in row["labels"].split(";")
    ]
    rows: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for diagnostic in sorted(selected, key=lambda row: row["scenario"]):
        scenario = diagnostic["scenario"]
        for method in FIXED_METHODS:
            expert_record = _selected_records(record_map, "expert", scenario, method)
            end_record = _selected_records(record_map, "end_to_end", scenario, method)
            expert_summary = _load_raw_summary(Path(str(expert_record["source_raw_path"])))
            end_summary = _load_raw_summary(Path(str(end_record["source_raw_path"])))
            raw_hashes[expert_summary["source_raw_path"]] = expert_summary["source_raw_sha256"]
            raw_hashes[end_summary["source_raw_path"]] = end_summary["source_raw_sha256"]
            row = {
                "scenario": scenario,
                "height_condition": diagnostic["height_condition"],
                "mirror_sign": diagnostic["mirror_sign"],
                "fixed_specialist": method,
                "expert": expert_summary,
                "end_to_end": end_summary,
                "comparison": _paired_delta(expert_summary, end_summary),
            }
            rows.append(row)
    labels = Counter(row["comparison"]["divergence_label"] for row in rows)
    return {
        "scope": {
            "selection_rule": "expert/disadvantage + library_gap_candidate + all_method_failure",
            "fixed_specialists": list(FIXED_METHODS),
            "early_window": "t <= 10 s",
            "late_window": "telemetry post_merge == true",
            "early_thresholds": EARLY_DIVERGENCE_THRESHOLDS,
        },
        "selected_scenario_count": len(selected),
        "selected_pair_count": len(rows),
        "divergence_label_counts": dict(sorted(labels.items())),
        "raw_source_hashes": raw_hashes,
        "pairs": rows,
        "interpretation_boundary": "An early-divergence label is an interaction-conditioned candidate, not causal proof that opponent policy alone caused the terminal outcome.",
    }, rows


def _routing_audit(
    record_map: Mapping[tuple[str, str, str], Mapping[str, Any]], diagnostics: Sequence[Mapping[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        row
        for row in diagnostics
        if row["opponent_stage"] == "expert"
        and row["initial_class"] == "head_on"
        and "routing_opportunity" in row["labels"].split(";")
        and row["ppo_outcome"] == "loss"
        and row["fixed_crossing_outcome"] == "win"
    ]
    rows: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for diagnostic in sorted(selected, key=lambda row: row["scenario"]):
        scenario = diagnostic["scenario"]
        ppo_record = _selected_records(record_map, "expert", scenario, PPO_METHOD)
        crossing_record = _selected_records(record_map, "expert", scenario, FIXED_CROSSING_METHOD)
        ppo = _load_raw_summary(Path(str(ppo_record["source_raw_path"])))
        crossing = _load_raw_summary(Path(str(crossing_record["source_raw_path"])))
        raw_hashes[ppo["source_raw_path"]] = ppo["source_raw_sha256"]
        raw_hashes[crossing["source_raw_path"]] = crossing["source_raw_sha256"]
        initial_crossing_selected = ppo["first_effective_mode"] == "crossing_specialist"
        label = (
            "initial_crossing_not_selected_candidate"
            if not initial_crossing_selected
            else "crossing_selected_but_geometry_or_late_routing_candidate"
        )
        rows.append(
            {
                "scenario": scenario,
                "height_condition": diagnostic["height_condition"],
                "mirror_sign": diagnostic["mirror_sign"],
                "routing_audit_label": label,
                "ppo": ppo,
                "fixed_crossing": crossing,
                "vpp_delta_fixed_crossing_minus_ppo": {
                    field: (
                        _finite(crossing["early"].get(field))
                        - _finite(ppo["early"].get(field))
                        if _finite(crossing["early"].get(field)) is not None
                        and _finite(ppo["early"].get(field)) is not None
                        else None
                    )
                    for field in (
                        "mean_vp_forward_bias_m",
                        "mean_vp_lateral_bias_m",
                        "mean_vp_vertical_offset_m",
                    )
                },
            }
        )
    labels = Counter(row["routing_audit_label"] for row in rows)
    return {
        "scope": {
            "selection_rule": "expert/head_on + routing_opportunity + PPO loss + fixed-crossing win",
            "comparison": "canonical PPO versus fixed crossing specialist",
            "early_window": "t <= 10 s",
            "late_window": "telemetry post_merge == true",
        },
        "selected_scenario_count": len(selected),
        "routing_audit_label_counts": dict(sorted(labels.items())),
        "raw_source_hashes": raw_hashes,
        "scenarios": rows,
        "interpretation_boundary": "A fixed-crossing win creates a routing opportunity under this frozen scenario; it is not a retrospective deployable oracle or proof of a universal crossing-first policy.",
    }, rows


def _opponent_markdown(audit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    flat_rows = [
        {
            "scenario": row["scenario"],
            "specialist": row["fixed_specialist"],
            "height": row["height_condition"],
            "mirror": row["mirror_sign"],
            "expert_outcome": row["expert"]["outcome"],
            "end_to_end_outcome": row["end_to_end"]["outcome"],
            "early_signal_count": row["comparison"]["early_divergence_signal_count"],
            "early_signal_count": row["comparison"]["early_divergence_signal_count"],
            "late_signal_count": row["comparison"]["late_divergence_signal_count"],
            "label": row["comparison"]["divergence_label"],
            "early_signals": row["comparison"]["early_divergence_signals"],
            "late_signals": row["comparison"]["late_divergence_signals"],
        }
        for row in rows
    ]
    return "\n".join(
        [
            "# 五个 Expert/Disadvantage 候选的对手分歧审计",
            "",
            "## 冻结规则",
            "",
            "- 场景由 Step 4 的 `expert/disadvantage + library_gap_candidate + all_method_failure` 自动选取。",
            "- 每个场景仅比较相同的 always-head-on 与 always-crossing specialist，expert 与 end-to-end 使用同一物理初始状态。",
            "- 早期窗口固定为 `t <= 10 s`；晚期窗口为 telemetry `post_merge=true`。早期分歧需要预先声明的五类信号中至少命中两项。",
            "",
            "## 配对摘要",
            "",
            _markdown_table(
                flat_rows,
                (
                    ("scenario", "场景"),
                    ("specialist", "固定 specialist"),
                    ("height", "高度"),
                    ("mirror", "镜像"),
                    ("expert_outcome", "expert"),
                    ("end_to_end_outcome", "end-to-end"),
                    ("early_signal_count", "早期信号数"),
                    ("late_signal_count", "后期信号数"),
                    ("label", "判定"),
                    ("early_signals", "早期信号"),
                    ("late_signals", "后期信号"),
                ),
            ),
            "",
            "## 解释边界",
            "",
            "早期分歧标签仅说明在相同 specialist 与初始几何下，交互早期已有多个对手相关行为指标不同。若早期未达阈值、但终端不同且 post-merge 的 VPP/range-rate/过载响应至少一项达阈值，则标为后期几何或执行链分歧候选。两者都不是单独的因果证明。",
            "",
            f"候选物理场景：{audit['selected_scenario_count']}；specialist 配对：{audit['selected_pair_count']}；综合分歧标签计数：`{json.dumps(audit['divergence_label_counts'], sort_keys=True)}`。",
        ]
    ) + "\n"


def _routing_markdown(audit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    flat_rows = [
        {
            "scenario": row["scenario"],
            "height": row["height_condition"],
            "mirror": row["mirror_sign"],
            "label": row["routing_audit_label"],
            "ppo_first_requested": row["ppo"]["first_requested_mode"],
            "ppo_first_effective": row["ppo"]["first_effective_mode"],
            "ppo_first_switch": row["ppo"]["first_switch_step"],
            "ppo_guard_fraction": row["ppo"]["guard_override_step_fraction"],
            "ppo_guard_reasons": row["ppo"]["guard_override_reasons"],
            "ppo_outcome": row["ppo"]["outcome"],
            "fixed_crossing_outcome": row["fixed_crossing"]["outcome"],
        }
        for row in rows
    ]
    return "\n".join(
        [
            "# Expert/Head-on Routing Opportunity 审计",
            "",
            "## 冻结规则",
            "",
            "- 场景由 Step 4 的 `expert/head_on + routing_opportunity + PPO loss + fixed-crossing win` 自动选取。",
            "- 对每个场景记录 PPO 首次 requested/effective mode、首次切换、guard 覆盖和 mode fraction，并与同场景 fixed crossing 的早期和 post-merge VPP 三维几何进行比较。",
            "",
            "## 场景摘要",
            "",
            _markdown_table(
                flat_rows,
                (
                    ("scenario", "场景"),
                    ("height", "高度"),
                    ("mirror", "镜像"),
                    ("label", "路由标签"),
                    ("ppo_first_requested", "PPO 首次请求"),
                    ("ppo_first_effective", "PPO 首次生效"),
                    ("ppo_first_switch", "首次切换 step"),
                    ("ppo_guard_fraction", "guard 占比"),
                    ("ppo_outcome", "PPO"),
                    ("fixed_crossing_outcome", "fixed crossing"),
                ),
            ),
            "",
            "## 解释边界",
            "",
            "fixed crossing 在该冻结小场景中获胜只构成 routing opportunity；它不是可部署 Oracle，也不能推出所有 head-on 都应以 crossing 起始。若 PPO 已在首拍选择 crossing，剩余问题应被描述为跨模式持续性、后期几何或低层执行候选，而非简单的首拍误选。",
            "",
            f"自动选中场景数：{audit['selected_scenario_count']}；路由标签计数：`{json.dumps(audit['routing_audit_label_counts'], sort_keys=True)}`。",
        ]
    ) + "\n"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def analyze(*, analysis_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    record_map, diagnostics = _load_base_artifacts(analysis_root)
    opponent_audit, opponent_rows = _opponent_divergence_audit(record_map, diagnostics)
    routing_audit, routing_rows = _routing_audit(record_map, diagnostics)
    if opponent_audit["selected_scenario_count"] != 5:
        raise ValueError(
            "Expected exactly five expert/disadvantage candidates; found "
            f"{opponent_audit['selected_scenario_count']}"
        )
    if routing_audit["selected_scenario_count"] != 3:
        raise ValueError(
            "Expected exactly three expert/head-on routing scenarios; found "
            f"{routing_audit['selected_scenario_count']}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "expert_disadvantage_opponent_divergence_audit.json", opponent_audit)
    _write_csv(
        output_dir / "expert_disadvantage_opponent_divergence_audit.csv",
        [
            {
                "scenario": row["scenario"],
                "height_condition": row["height_condition"],
                "mirror_sign": row["mirror_sign"],
                "fixed_specialist": row["fixed_specialist"],
                **_flatten("expert_", row["expert"]),
                **_flatten("end_to_end_", row["end_to_end"]),
                **_flatten("comparison_", row["comparison"]),
            }
            for row in opponent_rows
        ],
    )
    (output_dir / "expert_disadvantage_opponent_divergence_audit_zh.md").write_text(
        _opponent_markdown(opponent_audit, opponent_rows), encoding="utf-8"
    )
    _write_json(output_dir / "expert_headon_routing_audit.json", routing_audit)
    _write_csv(
        output_dir / "expert_headon_routing_audit.csv",
        [
            {
                "scenario": row["scenario"],
                "height_condition": row["height_condition"],
                "mirror_sign": row["mirror_sign"],
                "routing_audit_label": row["routing_audit_label"],
                **_flatten("ppo_", row["ppo"]),
                **_flatten("fixed_crossing_", row["fixed_crossing"]),
                **_flatten("vpp_delta_", row["vpp_delta_fixed_crossing_minus_ppo"]),
            }
            for row in routing_rows
        ],
    )
    (output_dir / "expert_headon_routing_audit_zh.md").write_text(
        _routing_markdown(routing_audit, routing_rows), encoding="utf-8"
    )
    source_files = {
        name: _sha256_file(analysis_root / name)
        for name in (
            "geometry_taxonomy_audit.json",
            "diagnostic_label_by_episode.csv",
            "artifact_source_hash_manifest.json",
        )
    }
    source_manifest = {
        "source_id": SOURCE_ID,
        "analysis_kind": "followup_opponent_divergence_and_headon_routing_audits",
        "analysis_git_sha": _git_sha(),
        "analysis_code_sha256": _sha256_file(Path(__file__)),
        "base_analysis_root": str(analysis_root),
        "base_analysis_source_files": source_files,
        "selected_raw_hashes": {
            "opponent_divergence": opponent_audit["raw_source_hashes"],
            "headon_routing": routing_audit["raw_source_hashes"],
        },
        "generated_output_sha256": {
            path.name: _sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    _write_json(output_dir / "artifact_source_hash_manifest.json", source_manifest)
    return {
        "output_dir": str(output_dir),
        "opponent_divergence_scenarios": opponent_audit["selected_scenario_count"],
        "opponent_divergence_pairs": opponent_audit["selected_pair_count"],
        "headon_routing_scenarios": routing_audit["selected_scenario_count"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        analysis_root=args.analysis_root.resolve(), output_dir=args.output_dir.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
