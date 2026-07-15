"""V16 dynamic-reference-path method contract and mechanism-only gate."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "frozen_fixed_head_on"
METHODS = (
    REFERENCE_METHOD,
    "static_range_extension",
    "progress_triggered_reference_path",
)
FULL_AXES = {"range_extension": True, "signed_turn": True, "vertical_energy": True}
NO_AXES = {"range_extension": False, "signed_turn": False, "vertical_energy": False}
SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-DYNAMIC-REFERENCE-PATH-V16"


class DynamicReferencePathGateError(ValueError):
    """Raised when a V16 record or gate universe violates its contract."""


def is_ego_failure(reason: Any) -> bool:
    """Return whether a terminal reason represents an ego crash/OOB event."""

    normalized = str(reason or "").lower()
    return not normalized.startswith("target_") and (
        "crash" in normalized or "out_of_bounds" in normalized
    )


def semantic_profile_for_method(method: str, selected_path_profile: str) -> str:
    """Keep the static comparator distinct from the path-selected controller."""

    if method not in METHODS:
        raise DynamicReferencePathGateError(f"unknown V16 method: {method}")
    if method == "static_range_extension":
        return "range_extension"
    return str(selected_path_profile)


def semantic_axes_for_method(method: str) -> dict[str, bool]:
    """Return the frozen semantic-axis setting for one V16 controller."""

    if method not in METHODS:
        raise DynamicReferencePathGateError(f"unknown V16 method: {method}")
    return dict(NO_AXES if method == REFERENCE_METHOD else FULL_AXES)


def _mean(values: Iterable[float]) -> float | None:
    numbers = [float(value) for value in values if np.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else None


def _qualifying(records: Iterable[Mapping[str, Any]], min_window_steps: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in records
        if bool(row.get("handoff_reached"))
        and bool(row.get("contract_passed"))
        and int(row.get("window_steps", 0)) >= min_window_steps
    ]


def _vector_norm(value: Any) -> float | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    return float(np.linalg.norm(vector))


def _paired_vector_delta(
    record: Mapping[str, Any], reference: Mapping[str, Any], key: str
) -> float | None:
    values: list[float] = []
    for actual, baseline in zip(record.get("window", []), reference.get("window", [])):
        try:
            delta = np.asarray(actual[key], dtype=np.float64).reshape(-1) - np.asarray(
                baseline[key], dtype=np.float64
            ).reshape(-1)
        except (KeyError, TypeError, ValueError):
            continue
        if delta.shape == (3,) and np.isfinite(delta).all():
            values.append(float(np.linalg.norm(delta)))
    return _mean(values)


def _paired_escape_range_rate_error(
    record: Mapping[str, Any], reference: Mapping[str, Any]
) -> float | None:
    values: list[float] = []
    for actual, baseline in zip(record.get("window", []), reference.get("window", [])):
        if actual.get("objective_stage") != "escape" or baseline.get("objective_stage") != "escape":
            continue
        try:
            actual_error = float(np.asarray(actual["after_component_errors"], dtype=np.float64).reshape(-1)[3])
            baseline_error = float(np.asarray(baseline["after_component_errors"], dtype=np.float64).reshape(-1)[3])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if np.isfinite(actual_error) and np.isfinite(baseline_error):
            values.append(actual_error - baseline_error)
    return _mean(values)


def _valid_path_step(step: Mapping[str, Any]) -> bool:
    receipt = step.get("path_action_receipt")
    if not isinstance(receipt, Mapping):
        return False
    required = (
        "generator_anchor_neu",
        "selected_anchor_neu",
        "semantic_world_offset_m",
        "requested_path_world_delta_m",
        "actual_path_world_delta_m",
        "requested_vpp_world_offset_m",
        "applied_vpp_world_offset_m",
        "applied_action",
    )
    if any(_vector_norm(receipt.get(key)) is None for key in required if key != "applied_action"):
        return False
    return _vector_norm(receipt.get("applied_action")) is not None and bool(step.get("prediction_valid")) and not bool(step.get("prediction_fallback"))


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    min_window_steps: int,
    min_pairs: int,
    min_anchor_path_delta_norm_m: float,
    min_paired_vpp_delta_norm_m: float,
    min_paired_guidance_delta_norm: float,
    max_escape_range_rate_worsening: float,
    safety_delta_max: float,
) -> dict[str, Any]:
    """Compute a per-opponent mechanism gate without pooling any opponent."""

    by_method: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHODS}
    seen: set[tuple[str, str, str]] = set()
    for row in records:
        method = str(row.get("method"))
        opponent = str(row.get("opponent"))
        key = str(row.get("pair_key"))
        if method not in by_method or opponent not in OPPONENTS:
            raise DynamicReferencePathGateError("record is outside the V16 frozen universe")
        identity = (method, opponent, key)
        if identity in seen:
            raise DynamicReferencePathGateError("V16 record universe contains a duplicate")
        seen.add(identity)
        by_method[method].append(row)

    expected_records = len(METHODS) * len(OPPONENTS) * scenarios_per_opponent
    reference_by_key = {
        (str(row.get("opponent")), str(row.get("pair_key"))): row
        for row in by_method[REFERENCE_METHOD]
    }
    matrix: dict[str, dict[str, Any]] = {}
    candidate_opponents: list[str] = []

    for method in METHODS:
        method_summary: dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in by_method[method] if str(row.get("opponent")) == opponent]
            qualifying = _qualifying(group, min_window_steps)
            reference_group = [
                row for row in by_method[REFERENCE_METHOD] if str(row.get("opponent")) == opponent
            ]
            pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            hashes_match = True
            for row in qualifying:
                reference = reference_by_key.get((opponent, str(row.get("pair_key"))))
                if reference is None or reference not in _qualifying([reference], min_window_steps):
                    continue
                hashes_match = hashes_match and row.get("handoff_state_hash") == reference.get(
                    "handoff_state_hash"
                )
                pairs.append((row, reference))

            ego_rate = sum(bool(row.get("ego_failure")) for row in group) / max(len(group), 1)
            reference_ego_rate = sum(bool(row.get("ego_failure")) for row in reference_group) / max(
                len(reference_group), 1
            )
            path_steps = [
                step
                for row in qualifying
                for step in row.get("window", [])
                if _valid_path_step(step)
            ]
            anchor_signal_steps = [
                step
                for step in path_steps
                if float(step["path_action_receipt"]["actual_path_delta_norm_m"])
                >= min_anchor_path_delta_norm_m
            ]
            vpp_deltas = [
                _paired_vector_delta(row, ref, "env_vpp_world_offset_m") for row, ref in pairs
            ]
            guidance_deltas = [
                _paired_vector_delta(row, ref, "guidance_command_normalized") for row, ref in pairs
            ]
            escape_rate_errors = [
                _paired_escape_range_rate_error(row, ref) for row, ref in pairs
            ]
            cell = {
                "records": len(group),
                "qualifying_episodes": len(qualifying),
                "paired_count": len(pairs),
                "handoff_hashes_match": hashes_match,
                "path_steps_valid": len(path_steps),
                "path_steps_expected": len(qualifying) * min_window_steps,
                "anchor_signal_steps": len(anchor_signal_steps),
                "mean_paired_vpp_delta_norm_m": _mean(value for value in vpp_deltas if value is not None),
                "mean_paired_guidance_delta_norm": _mean(
                    value for value in guidance_deltas if value is not None
                ),
                "mean_escape_range_rate_error_delta_vs_reference": _mean(
                    value for value in escape_rate_errors if value is not None
                ),
                "ego_crash_oob_delta_vs_reference": ego_rate - reference_ego_rate,
                "safety_pass": ego_rate - reference_ego_rate <= safety_delta_max,
            }
            cell["receipt_pass"] = (
                cell["path_steps_valid"] == cell["path_steps_expected"]
                and cell["anchor_signal_steps"] > 0
            )
            cell["mechanism_pass"] = (
                cell["records"] == scenarios_per_opponent
                and cell["paired_count"] >= min_pairs
                and cell["handoff_hashes_match"]
                and cell["safety_pass"]
                and cell["receipt_pass"]
                and cell["mean_paired_vpp_delta_norm_m"] is not None
                and cell["mean_paired_vpp_delta_norm_m"] >= min_paired_vpp_delta_norm_m
                and cell["mean_paired_guidance_delta_norm"] is not None
                and cell["mean_paired_guidance_delta_norm"] >= min_paired_guidance_delta_norm
                and cell["mean_escape_range_rate_error_delta_vs_reference"] is not None
                and cell["mean_escape_range_rate_error_delta_vs_reference"]
                <= max_escape_range_rate_worsening
            )
            method_summary[opponent] = cell
        matrix[method] = method_summary
        if method == "progress_triggered_reference_path":
            candidate_opponents = [
                opponent
                for opponent, cell in method_summary.items()
                if bool(cell["mechanism_pass"])
            ]

    return {
        "records": len(records),
        "expected_records": expected_records,
        "matrix": matrix,
        "mechanism_candidate_opponents": candidate_opponents,
        "passes": len(records) == expected_records and len(candidate_opponents) >= 2,
    }
