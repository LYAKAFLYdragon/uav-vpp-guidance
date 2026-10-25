#!/usr/bin/env python3
"""Analyze paired local impulse responses without running JSBSim.

The analyzer consumes the verified baseline/plus/minus records and reports
signed local effects, symmetry residuals, repetition-level direction
consistency, and cross-opponent agreement. It never changes the source data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V1-20260903"
CONDITIONS = ("baseline", "axis0_plus", "axis0_minus", "axis1_plus", "axis1_minus", "axis2_plus", "axis2_minus")
AXES = (0, 1, 2)
OFFSETS = (4, 12, 48)
PULSE_AMPLITUDE = 0.08
METRICS = (
    ("range_m", "距离", "m"),
    ("range_rate_mps", "距离率", "m/s"),
    ("ata_deg", "攻击区角", "deg"),
    ("aa_deg", "进入角", "deg"),
    ("specific_energy_m2ps2", "比能", "m²/s²"),
    ("altitude_m", "高度", "m"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_row_at_step(path: Path, step: str) -> dict[str, str] | None:
    """Read only the requested CSV row to keep the audit lightweight."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("step") == step:
                return row
    return None


def locate_trace_file(root: Path, source_trace_id: str, condition: str) -> Path:
    """Locate a trace in either the legacy or opponent-scoped layout."""
    filename = source_trace_id.replace("/", "__").replace(".csv", "") + f"__{condition}.csv"
    candidates = [
        root / "paired_traces" / filename,
        root / "expert" / "traces" / filename,
        root / "end_to_end" / "traces" / filename,
    ]
    candidates.extend(root.glob(f"**/{filename}"))
    for path in candidates:
        if path.is_file():
            return path
    return root / "paired_traces" / filename


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


def signed_effect(plus: float, minus: float, baseline: float) -> tuple[float, float, float, float]:
    """Return plus delta, minus delta, antisymmetric effect, and residual."""
    plus_delta = plus - baseline
    minus_delta = minus - baseline
    antisymmetric = 0.5 * (plus_delta - minus_delta)
    symmetry_residual = 0.5 * (plus_delta + minus_delta)
    return plus_delta, minus_delta, antisymmetric, symmetry_residual


def direction_consistency(values: list[float]) -> tuple[int, int, float, bool]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0, 0, float("nan"), False
    med = median(values)
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    dominant = max(positive, negative)
    consistent = dominant >= 2 and abs(med) > 1e-12
    return positive, negative, med, consistent


def safety_clean(row: dict[str, str]) -> bool:
    return (
        row.get("backend_effective") == "jsbsim"
        and not bool_value(row.get("backend_fallback_occurred"))
        and not bool_value(row.get("response_window_prediction_fallback"))
        and not bool_value(row.get("pulse_clipped"))
        and not bool_value(row.get("is_crash"))
        and not bool_value(row.get("is_out_of_bounds"))
        and bool_value(row.get("handoff_hash_match"))
    )


def load_verified_records(root: Path, expected_source_id: str = SOURCE_ID) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    condition_rows = read_csv(root / "condition_results.csv")
    response_rows = read_csv(root / "local_response_records.csv")
    if not condition_rows or not response_rows:
        raise RuntimeError("verified local-response CSV files are missing or empty")
    if any(row.get("source_id") != expected_source_id for row in condition_rows + response_rows):
        raise RuntimeError("source_id mismatch in local-response inputs")
    return condition_rows, response_rows


def audit_pulse_contract(root: Path, condition_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Check that recorded minus pulses actually have the requested sign."""
    output: list[dict[str, Any]] = []
    for condition in condition_rows:
        label = condition.get("condition", "")
        if label == "baseline":
            continue
        source_trace_id = condition["source_trace_id"]
        path = locate_trace_file(root, source_trace_id, label)
        row: dict[str, str] | None = None
        if path.is_file():
            row = read_row_at_step(path, condition.get("handoff_step", ""))
        axis = int(float(condition.get("pulse_axis", "nan")))
        sign = int(float(condition.get("pulse_sign", "nan")))
        expected = [0.0, 0.0, 0.0]
        expected[axis] = sign * PULSE_AMPLITUDE
        actual = [number(row.get(f"pulse_delta_{i}")) for i in AXES] if row else [float("nan")] * 3
        action_delta_ok = bool(row) and all(abs(actual[i] - expected[i]) <= 1e-10 for i in AXES)
        output.append({
            "source_id": SOURCE_ID,
            "source_trace_id": source_trace_id,
            "condition": label,
            "axis": axis,
            "recorded_sign": sign,
            "expected_delta_0": expected[0],
            "expected_delta_1": expected[1],
            "expected_delta_2": expected[2],
            "actual_delta_0": actual[0],
            "actual_delta_1": actual[1],
            "actual_delta_2": actual[2],
            "action_delta_contract_ok": action_delta_ok,
            "trace_file": str(path),
        })
    return output


def build_effect_rows(condition_rows: list[dict[str, str]], response_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    conditions = {(row["source_trace_id"], row["condition"]): row for row in condition_rows}
    responses = {
        (row["source_trace_id"], row["condition"], int(row["response_offset"])): row
        for row in response_rows
    }
    source_ids = sorted({row["source_trace_id"] for row in condition_rows})
    output: list[dict[str, Any]] = []
    for source_trace_id in source_ids:
        base_condition = conditions.get((source_trace_id, "baseline"))
        if base_condition is None:
            raise RuntimeError(f"missing baseline for {source_trace_id}")
        for axis in AXES:
            plus_label = f"axis{axis}_plus"
            minus_label = f"axis{axis}_minus"
            plus_condition = conditions.get((source_trace_id, plus_label))
            minus_condition = conditions.get((source_trace_id, minus_label))
            if plus_condition is None or minus_condition is None:
                raise RuntimeError(f"missing symmetric pulse for {source_trace_id}/axis{axis}")
            for offset in OFFSETS:
                base = responses.get((source_trace_id, "baseline", offset))
                plus = responses.get((source_trace_id, plus_label, offset))
                minus = responses.get((source_trace_id, minus_label, offset))
                if not base or not plus or not minus:
                    raise RuntimeError(f"missing response window for {source_trace_id}/axis{axis}/h{offset}")
                common = {
                    "source_id": SOURCE_ID,
                    "source_trace_id": source_trace_id,
                    "opponent": base["opponent"],
                    "candidate": base["candidate"],
                    "seed": base["seed"],
                    "axis": axis,
                    "response_offset": offset,
                    "pulse_amplitude": PULSE_AMPLITUDE,
                    "safety_clean": safety_clean(base_condition) and safety_clean(plus_condition) and safety_clean(minus_condition),
                }
                for metric, _label, _unit in METRICS:
                    baseline = number(base[metric])
                    plus_value = number(plus[metric])
                    minus_value = number(minus[metric])
                    if not all(math.isfinite(value) for value in (baseline, plus_value, minus_value)):
                        raise RuntimeError(f"non-finite {metric} for {source_trace_id}/axis{axis}/h{offset}")
                    plus_delta, minus_delta, antisymmetric, residual = signed_effect(plus_value, minus_value, baseline)
                    common[f"{metric}_baseline"] = baseline
                    common[f"{metric}_plus_delta"] = plus_delta
                    common[f"{metric}_minus_delta"] = minus_delta
                    common[f"{metric}_antisymmetric"] = antisymmetric
                    common[f"{metric}_sensitivity_per_action"] = antisymmetric / PULSE_AMPLITUDE
                    common[f"{metric}_symmetry_residual"] = residual
                output.append(common)
    return output


def summarise(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in effect_rows:
        key = (row["opponent"], row["candidate"], int(row["axis"]), int(row["response_offset"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (opponent, candidate, axis, offset), rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "candidate": candidate,
            "axis": axis,
            "response_offset": offset,
            "n_handoff_states": len(rows),
            "safety_clean_count": sum(bool(row["safety_clean"]) for row in rows),
        }
        for metric, _label, _unit in METRICS:
            values = [float(row[f"{metric}_sensitivity_per_action"]) for row in rows]
            residuals = [float(row[f"{metric}_symmetry_residual"]) for row in rows]
            positive, negative, med, consistent = direction_consistency(values)
            summary[f"{metric}_sensitivity_mean"] = mean(values)
            summary[f"{metric}_sensitivity_median"] = med
            summary[f"{metric}_sensitivity_min"] = min(values)
            summary[f"{metric}_sensitivity_max"] = max(values)
            summary[f"{metric}_positive_count"] = positive
            summary[f"{metric}_negative_count"] = negative
            summary[f"{metric}_direction_consistent_2of3"] = consistent
            summary[f"{metric}_symmetry_residual_median"] = median(residuals)
        output.append(summary)
    return output


def cross_opponent_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        groups.setdefault((row["candidate"], int(row["axis"]), int(row["response_offset"])), {})[row["opponent"]] = row
    output: list[dict[str, Any]] = []
    for (candidate, axis, offset), opponents in sorted(groups.items()):
        if not all(name in opponents for name in ("expert", "end_to_end")):
            continue
        row: dict[str, Any] = {
            "source_id": SOURCE_ID,
            "candidate": candidate,
            "axis": axis,
            "response_offset": offset,
        }
        for metric, _label, _unit in METRICS:
            expert = float(opponents["expert"][f"{metric}_sensitivity_median"])
            end_to_end = float(opponents["end_to_end"][f"{metric}_sensitivity_median"])
            row[f"{metric}_expert_median"] = expert
            row[f"{metric}_end_to_end_median"] = end_to_end
            row[f"{metric}_same_direction"] = expert * end_to_end > 0.0
            row[f"{metric}_both_direction_consistent"] = bool(
                opponents["expert"][f"{metric}_direction_consistent_2of3"]
                and opponents["end_to_end"][f"{metric}_direction_consistent_2of3"]
            )
        output.append(row)
    return output


def plot_heatmap(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["expert/x625", "expert/x750", "end_to_end/x625", "end_to_end/x750"]
    columns = [f"a{axis}@{offset}" for axis in AXES for offset in OFFSETS]
    lookup = {(row["opponent"], row["candidate"], int(row["axis"]), int(row["response_offset"])): row for row in summary_rows}
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for panel, (metric, label, unit) in zip(axes.flat, METRICS):
        matrix = []
        for opponent, candidate_prefix in (("expert", "x625_y200"), ("expert", "x750_y250"), ("end_to_end", "x625_y200"), ("end_to_end", "x750_y250")):
            values = []
            for axis in AXES:
                for offset in OFFSETS:
                    row = lookup.get((opponent, candidate_prefix, axis, offset))
                    values.append(float(row[f"{metric}_sensitivity_median"]) if row else np.nan)
            matrix.append(values)
        matrix_array = np.asarray(matrix, dtype=float)
        vmax = float(np.nanmax(np.abs(matrix_array))) if np.isfinite(matrix_array).any() else 1.0
        vmax = max(vmax, 1e-12)
        image = panel.imshow(matrix_array, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        panel.set_title(f"{label}（{unit}/动作单位）")
        panel.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        panel.set_yticks(range(len(labels)), labels)
        for i in range(matrix_array.shape[0]):
            for j in range(matrix_array.shape[1]):
                if np.isfinite(matrix_array[i, j]):
                    panel.text(j, i, f"{matrix_array[i, j]:.3g}", ha="center", va="center", fontsize=7)
        figure.colorbar(image, ax=panel, shrink=0.8)
    figure.suptitle("真实 handoff 三轴局部响应中位数敏感度")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def main() -> int:
    global SOURCE_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/mild_disadvantage_handoff_local_response_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mild_disadvantage_handoff_local_response_v1/offline_analysis"))
    parser.add_argument("--source-id", default=SOURCE_ID)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = args.output_root.resolve()
    SOURCE_ID = args.source_id
    condition_rows, response_rows = load_verified_records(root, SOURCE_ID)
    pulse_audit = audit_pulse_contract(root, condition_rows)
    write_csv(output / "pulse_contract_audit.csv", pulse_audit)
    if not pulse_audit or not all(bool(row["action_delta_contract_ok"]) for row in pulse_audit):
        report = {
            "source_id": SOURCE_ID,
            "analysis_type": "read_only_paired_local_impulse_response",
            "analysis_gate": "fail",
            "failure": "pulse_action_contract_mismatch",
            "input_root": str(root),
            "output_root": str(output),
            "condition_count": len(condition_rows),
            "response_count": len(response_rows),
            "pulse_contract_count": len(pulse_audit),
            "pulse_contract_pass_count": sum(bool(row["action_delta_contract_ok"]) for row in pulse_audit),
            "interpretation_boundary": "No local-response effect or cross-opponent conclusion is valid until recorded positive and negative action deltas satisfy the requested symmetric contract.",
            "training_started": False,
            "runtime_probe_started": False,
        }
        write_json(output / "local_response_analysis_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 2
    effect_rows = build_effect_rows(condition_rows, response_rows)
    summary_rows = summarise(effect_rows)
    cross_rows = cross_opponent_rows(summary_rows)
    write_csv(output / "local_response_effects.csv", effect_rows)
    write_csv(output / "local_response_summary.csv", summary_rows)
    write_csv(output / "cross_opponent_direction_consistency.csv", cross_rows)
    if not args.no_plot:
        plot_heatmap(summary_rows, output / "local_response_sensitivity_heatmap.svg")
    report = {
        "source_id": SOURCE_ID,
        "analysis_type": "read_only_paired_local_impulse_response",
        "analysis_gate": "pass",
        "input_root": str(root),
        "output_root": str(output),
        "condition_count": len(condition_rows),
        "response_count": len(response_rows),
        "effect_count": len(effect_rows),
        "summary_count": len(summary_rows),
        "cross_opponent_count": len(cross_rows),
        "pulse_amplitude": PULSE_AMPLITUDE,
        "response_offsets": list(OFFSETS),
        "opponents": ["expert", "end_to_end"],
        "metrics": [{"name": name, "label": label, "unit": unit} for name, label, unit in METRICS],
        "method": {
            "plus_delta": "plus - baseline",
            "minus_delta": "minus - baseline",
            "antisymmetric_effect": "0.5 * (plus_delta - minus_delta)",
            "sensitivity_per_action": "antisymmetric_effect / 0.08",
            "symmetry_residual": "0.5 * (plus_delta + minus_delta)",
            "direction_consistent_2of3": "at least two of three handoff states share a non-zero sign",
        },
        "interpretation_boundary": "This is a local response map, not a skill-performance or win-rate claim. No post-hoc practical threshold is declared by this analyzer.",
        "training_started": False,
        "runtime_probe_started": False,
    }
    write_json(output / "local_response_analysis_report.json", report)
    print(json.dumps({"source_id": SOURCE_ID, "effect_count": len(effect_rows), "summary_count": len(summary_rows), "output_root": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
