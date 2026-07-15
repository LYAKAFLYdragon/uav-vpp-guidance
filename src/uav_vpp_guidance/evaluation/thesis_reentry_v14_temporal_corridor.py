"""Fixed escape-then-reentry corridor schedule and its paired gate."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "frozen_fixed_head_on"
METHODS = (
    REFERENCE_METHOD,
    "static_range_extension",
    "escape_only_corridor",
    "escape_then_reentry_corridor",
)
FULL_AXES = {"range_extension": True, "signed_turn": True, "vertical_energy": True}
NO_AXES = {"range_extension": False, "signed_turn": False, "vertical_energy": False}


class TemporalCorridorError(ValueError):
    """Raised when a temporal-corridor contract is malformed."""


def is_ego_failure(reason: Any) -> bool:
    normalized = str(reason or "").lower()
    return not normalized.startswith("target_") and ("crash" in normalized or "out_of_bounds" in normalized)


def objective_profile(local_step: int, *, escape_steps: int) -> str:
    """The shared time-indexed objective, independent of controller method."""

    return "range_extension" if int(local_step) < int(escape_steps) else "reentry_preparation"


def scheduled_action(method: str, local_step: int, *, escape_steps: int, corridor_steps: int) -> tuple[str, dict[str, bool]]:
    """Return the frozen action profile and enabled axes for one corridor tick."""

    if method not in METHODS:
        raise TemporalCorridorError(f"unknown corridor method: {method}")
    tick = int(local_step)
    if method == REFERENCE_METHOD:
        return objective_profile(tick, escape_steps=escape_steps), dict(NO_AXES)
    if method == "static_range_extension":
        return "range_extension", dict(FULL_AXES) if tick < corridor_steps else dict(NO_AXES)
    if method == "escape_only_corridor":
        return "range_extension", dict(FULL_AXES) if tick < escape_steps else dict(NO_AXES)
    if tick < escape_steps:
        return "range_extension", dict(FULL_AXES)
    if tick < corridor_steps:
        return "reentry_preparation", dict(FULL_AXES)
    return "reentry_preparation", dict(NO_AXES)


def _mean(values: Iterable[float]) -> float | None:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else None


def _qualifying(records: Iterable[Mapping[str, Any]], min_window_steps: int) -> list[Mapping[str, Any]]:
    return [row for row in records if bool(row.get("handoff_reached")) and bool(row.get("contract_passed")) and int(row.get("window_steps", 0)) >= min_window_steps]


def _paired_stage_mean(record: Mapping[str, Any], reference: Mapping[str, Any], *, stage: str, key: str, index: int | None = None) -> float | None:
    values = []
    for actual, baseline in zip(record.get("window", []), reference.get("window", [])):
        if actual.get("objective_stage") != stage or baseline.get("objective_stage") != stage:
            continue
        actual_value = actual.get(key)
        baseline_value = baseline.get(key)
        if index is not None:
            actual_value = np.asarray(actual_value, dtype=np.float64).reshape(-1)
            baseline_value = np.asarray(baseline_value, dtype=np.float64).reshape(-1)
            if actual_value.shape != baseline_value.shape or actual_value.shape[0] <= index:
                continue
            delta = float(actual_value[index] - baseline_value[index])
        else:
            try:
                delta = float(actual_value) - float(baseline_value)
            except (TypeError, ValueError):
                continue
        if np.isfinite(delta):
            values.append(delta)
    return _mean(values)


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    min_window_steps: int,
    min_pairs: int,
    min_total_weighted_improvement: float,
    min_reentry_weighted_improvement: float,
    max_escape_range_rate_worsening: float,
    safety_delta_max: float,
) -> dict[str, Any]:
    """Summarize the corridor matrix without pooling opponents."""

    by_method: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHODS}
    for row in records:
        method = str(row.get("method"))
        if method not in by_method:
            raise TemporalCorridorError(f"unexpected method: {method}")
        by_method[method].append(row)
    if len(by_method[REFERENCE_METHOD]) != len(OPPONENTS) * scenarios_per_opponent:
        raise TemporalCorridorError("reference universe is incomplete")
    reference_by_key = {(str(row.get("opponent")), str(row.get("pair_key"))): row for row in by_method[REFERENCE_METHOD]}
    matrix: dict[str, dict[str, Any]] = {}
    candidates: list[str] = []
    for method in METHODS:
        method_summary: dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in by_method[method] if str(row.get("opponent")) == opponent]
            qualifying = _qualifying(group, min_window_steps)
            reference_group = [row for row in by_method[REFERENCE_METHOD] if str(row.get("opponent")) == opponent]
            pairs = []
            hashes_match = True
            for row in qualifying:
                reference = reference_by_key.get((opponent, str(row.get("pair_key"))))
                if reference is None or reference not in _qualifying([reference], min_window_steps):
                    continue
                hashes_match = hashes_match and row.get("handoff_state_hash") == reference.get("handoff_state_hash")
                pairs.append((row, reference))
            total = [_paired_stage_mean(row, ref, stage="escape", key="after_weighted_error") for row, ref in pairs]
            total += [_paired_stage_mean(row, ref, stage="reentry", key="after_weighted_error") for row, ref in pairs]
            reentry = [_paired_stage_mean(row, ref, stage="reentry", key="after_weighted_error") for row, ref in pairs]
            escape_rate = [_paired_stage_mean(row, ref, stage="escape", key="after_component_errors", index=3) for row, ref in pairs]
            ego_rate = sum(bool(row.get("ego_failure")) for row in group) / max(len(group), 1)
            reference_ego_rate = sum(bool(row.get("ego_failure")) for row in reference_group) / max(len(reference_group), 1)
            method_summary[opponent] = {
                "records": len(group),
                "qualifying_episodes": len(qualifying),
                "paired_count": len(pairs),
                "handoff_hashes_match": hashes_match,
                "mean_total_weighted_error_delta_vs_reference": _mean(value for value in total if value is not None),
                "mean_reentry_weighted_error_delta_vs_reference": _mean(value for value in reentry if value is not None),
                "mean_escape_range_rate_error_delta_vs_reference": _mean(value for value in escape_rate if value is not None),
                "ego_crash_oob_delta_vs_reference": ego_rate - reference_ego_rate,
                "safety_pass": ego_rate - reference_ego_rate <= safety_delta_max,
            }
        matrix[method] = method_summary
        if method != REFERENCE_METHOD:
            passed = [
                opponent
                for opponent, cell in method_summary.items()
                if cell["records"] == scenarios_per_opponent
                and cell["paired_count"] >= min_pairs
                and cell["handoff_hashes_match"]
                and cell["safety_pass"]
                and cell["mean_total_weighted_error_delta_vs_reference"] is not None
                and cell["mean_total_weighted_error_delta_vs_reference"] <= -min_total_weighted_improvement
                and cell["mean_reentry_weighted_error_delta_vs_reference"] is not None
                and cell["mean_reentry_weighted_error_delta_vs_reference"] <= -min_reentry_weighted_improvement
                and cell["mean_escape_range_rate_error_delta_vs_reference"] is not None
                and cell["mean_escape_range_rate_error_delta_vs_reference"] <= max_escape_range_rate_worsening
            ]
            if len(passed) >= 2:
                candidates.append(method)
    expected_records = len(METHODS) * len(OPPONENTS) * scenarios_per_opponent
    return {"records": len(records), "expected_records": expected_records, "matrix": matrix, "corridor_candidates": candidates, "passes": len(records) == expected_records and "escape_then_reentry_corridor" in candidates}
