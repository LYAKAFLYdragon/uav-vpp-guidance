"""Pure six-objective local-controllability contracts for the V12 probe."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.hierarchy.shared_intent_profiles import INTENT_FIELDS, compile_profile


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "frozen_fixed_head_on"
METHODS = (
    REFERENCE_METHOD,
    "range_only",
    "turn_only",
    "vertical_only",
    "range_turn",
    "range_vertical",
    "turn_vertical",
    "range_turn_vertical",
)
SCALES = np.asarray([180.0, 180.0, 5000.0, 200.0, 1000.0, 1000.0], dtype=np.float64)
HIGH_PRIORITY_INDICES = (2, 3)  # range and range rate for range_extension.


class IntentControllabilityError(ValueError):
    """Raised when a V12 six-objective record is malformed."""


def is_ego_failure(reason: Any) -> bool:
    normalized = str(reason or "").lower()
    return not normalized.startswith("target_") and ("crash" in normalized or "out_of_bounds" in normalized)


def component_errors(intent_vector: Any, profile_name: str) -> np.ndarray:
    """Return six clipped, normalized absolute profile errors in fixed order."""

    vector = np.asarray(intent_vector, dtype=np.float64).reshape(-1)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise IntentControllabilityError("intent_vector must be finite 6-D")
    profile = compile_profile(profile_name)
    target = np.asarray(profile["physical_targets"], dtype=np.float64)
    return np.minimum(np.abs(vector - target) / SCALES, 2.0)


def _mean(values: Iterable[float]) -> float | None:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else None


def _qualifying(records: Iterable[Mapping[str, Any]], min_window_steps: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in records
        if bool(row.get("handoff_reached"))
        and bool(row.get("contract_passed"))
        and int(row.get("window_steps", 0)) >= min_window_steps
    ]


def _paired_component_delta(record: Mapping[str, Any], reference: Mapping[str, Any]) -> np.ndarray | None:
    deltas: list[np.ndarray] = []
    for actual, baseline in zip(record.get("window", []), reference.get("window", [])):
        actual_errors = np.asarray(actual.get("after_component_errors"), dtype=np.float64).reshape(-1)
        baseline_errors = np.asarray(baseline.get("after_component_errors"), dtype=np.float64).reshape(-1)
        if actual_errors.shape == baseline_errors.shape == (6,) and np.isfinite(actual_errors).all() and np.isfinite(baseline_errors).all():
            deltas.append(actual_errors - baseline_errors)
    return np.mean(np.stack(deltas, axis=0), axis=0) if deltas else None


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    min_window_steps: int,
    min_pairs: int,
    max_high_priority_worsening: float,
    max_any_component_worsening: float,
    min_component_improvement: float,
    min_improved_components: int,
    min_weighted_error_improvement: float,
    safety_delta_max: float,
    profile_name: str,
) -> dict[str, Any]:
    """Summarize one full-factorial V12 matrix without pooling opponents."""

    by_method: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHODS}
    for row in records:
        method = str(row.get("method"))
        if method not in by_method:
            raise IntentControllabilityError(f"unexpected method: {method}")
        by_method[method].append(row)
    expected_reference_count = len(OPPONENTS) * scenarios_per_opponent
    if len(by_method[REFERENCE_METHOD]) != expected_reference_count:
        raise IntentControllabilityError("reference universe is incomplete")
    reference_by_key = {
        (str(row.get("opponent")), str(row.get("pair_key"))): row
        for row in by_method[REFERENCE_METHOD]
    }
    weights = np.asarray(compile_profile(profile_name)["weight_vector"], dtype=np.float64)
    matrix: dict[str, dict[str, Any]] = {}
    candidates: list[str] = []
    for method in METHODS:
        method_summary: dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in by_method[method] if str(row.get("opponent")) == opponent]
            qualifying = _qualifying(group, min_window_steps)
            reference_group = [row for row in by_method[REFERENCE_METHOD] if str(row.get("opponent")) == opponent]
            pairs = []
            handoff_hashes_match = True
            for row in qualifying:
                reference = reference_by_key.get((opponent, str(row.get("pair_key"))))
                if reference is None or reference not in _qualifying([reference], min_window_steps):
                    continue
                handoff_hashes_match = handoff_hashes_match and row.get("handoff_state_hash") == reference.get("handoff_state_hash")
                pairs.append((row, reference))
            component_deltas = [value for row, reference in pairs for value in [_paired_component_delta(row, reference)] if value is not None]
            mean_delta = np.mean(np.stack(component_deltas, axis=0), axis=0) if component_deltas else None
            weighted_delta = None if mean_delta is None else float(np.dot(weights, mean_delta) / max(float(np.sum(weights)), 1e-12))
            ego_rate = sum(bool(row.get("ego_failure")) for row in group) / max(len(group), 1)
            reference_ego_rate = sum(bool(row.get("ego_failure")) for row in reference_group) / max(len(reference_group), 1)
            components = None if mean_delta is None else {name: float(mean_delta[index]) for index, name in enumerate(INTENT_FIELDS)}
            method_summary[opponent] = {
                "records": len(group),
                "qualifying_episodes": len(qualifying),
                "paired_count": len(pairs),
                "handoff_hashes_match": handoff_hashes_match,
                "mean_component_error_delta_vs_reference": components,
                "weighted_component_error_delta_vs_reference": weighted_delta,
                "ego_crash_oob_delta_vs_reference": ego_rate - reference_ego_rate,
                "safety_pass": ego_rate - reference_ego_rate <= safety_delta_max,
            }
        matrix[method] = method_summary
        if method != REFERENCE_METHOD:
            passed_opponents = []
            for opponent, cell in method_summary.items():
                components = cell["mean_component_error_delta_vs_reference"]
                if components is None:
                    continue
                delta = np.asarray([components[field] for field in INTENT_FIELDS], dtype=np.float64)
                if (
                    cell["records"] == scenarios_per_opponent
                    and cell["paired_count"] >= min_pairs
                    and cell["handoff_hashes_match"]
                    and cell["safety_pass"]
                    and np.all(delta[list(HIGH_PRIORITY_INDICES)] <= max_high_priority_worsening)
                    and np.all(delta <= max_any_component_worsening)
                    and int(np.sum(delta <= -min_component_improvement)) >= min_improved_components
                    and cell["weighted_component_error_delta_vs_reference"] is not None
                    and cell["weighted_component_error_delta_vs_reference"] <= -min_weighted_error_improvement
                ):
                    passed_opponents.append(opponent)
            if len(passed_opponents) >= 2:
                candidates.append(method)
    expected_records = len(METHODS) * len(OPPONENTS) * scenarios_per_opponent
    return {
        "records": len(records),
        "expected_records": expected_records,
        "matrix": matrix,
        "local_multiobjective_candidates": candidates,
        "passes": len(records) == expected_records and bool(candidates),
    }
