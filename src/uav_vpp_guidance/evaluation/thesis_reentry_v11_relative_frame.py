"""Pure contracts and summaries for the V11 semantic VPP transfer probe."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "frozen_fixed_head_on"
METHODS = (
    REFERENCE_METHOD,
    "range_extension_los",
    "range_extension_los_turn",
    "range_extension_full",
)


class RelativeFrameProbeError(ValueError):
    """Raised when a V11 record matrix is structurally invalid."""


def is_ego_failure(reason: Any) -> bool:
    """Target crash/OOB remains a legal terminal outcome, not an ego failure."""

    normalized = str(reason or "").lower()
    return not normalized.startswith("target_") and ("crash" in normalized or "out_of_bounds" in normalized)


def _mean(values: Iterable[float]) -> float | None:
    collected = [float(value) for value in values]
    return sum(collected) / len(collected) if collected else None


def _qualifying(records: Iterable[Mapping[str, Any]], min_window_steps: int) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if bool(record.get("handoff_reached"))
        and bool(record.get("contract_passed"))
        and int(record.get("window_steps", 0)) >= min_window_steps
    ]


def _window_mean(record: Mapping[str, Any], key: str) -> float | None:
    values = []
    for step in record.get("window", []):
        value = step.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
    return _mean(values)


def _paired_mean_norm(record: Mapping[str, Any], reference: Mapping[str, Any], key: str) -> float | None:
    values = []
    for actual, baseline in zip(record.get("window", []), reference.get("window", [])):
        actual_vector = np.asarray(actual.get(key), dtype=np.float64).reshape(-1)
        baseline_vector = np.asarray(baseline.get(key), dtype=np.float64).reshape(-1)
        if actual_vector.shape == baseline_vector.shape == (3,) and np.isfinite(actual_vector).all() and np.isfinite(baseline_vector).all():
            values.append(float(np.linalg.norm(actual_vector - baseline_vector)))
    return _mean(values)


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    min_window_steps: int,
    min_pairs: int,
    min_semantic_delta_norm_m: float,
    min_vpp_delta_norm_m: float,
    min_guidance_delta_norm: float,
    min_range_rate_delta_mps: float,
    practical_intent_loss_delta: float,
    safety_delta_max: float,
) -> dict[str, Any]:
    """Summarize V11 without pooling opponent evidence."""

    by_method: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHODS}
    for record in records:
        method = str(record.get("method"))
        if method not in by_method:
            raise RelativeFrameProbeError(f"unexpected method: {method}")
        by_method[method].append(record)
    if len(by_method[REFERENCE_METHOD]) != len(OPPONENTS) * scenarios_per_opponent:
        raise RelativeFrameProbeError("reference record universe is incomplete")
    reference_by_key = {
        (str(row.get("opponent")), str(row.get("pair_key"))): row
        for row in by_method[REFERENCE_METHOD]
    }
    matrix: dict[str, dict[str, Any]] = {}
    training_candidates: list[str] = []
    for method in METHODS:
        method_summary: dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in by_method[method] if str(row.get("opponent")) == opponent]
            qualifying = _qualifying(group, min_window_steps)
            reference_group = [row for row in by_method[REFERENCE_METHOD] if str(row.get("opponent")) == opponent]
            paired = []
            handoff_hashes_match = True
            for row in qualifying:
                reference = reference_by_key.get((opponent, str(row.get("pair_key"))))
                if reference is None or reference not in _qualifying([reference], min_window_steps):
                    continue
                handoff_hashes_match = handoff_hashes_match and row.get("handoff_state_hash") == reference.get("handoff_state_hash")
                paired.append((row, reference))
            intent_deltas = [
                float(row["intent_loss_auc20"]) - float(reference["intent_loss_auc20"])
                for row, reference in paired
                if row.get("intent_loss_auc20") is not None and reference.get("intent_loss_auc20") is not None
            ]
            range_rate_deltas = [
                _window_mean(row, "after_range_rate_mps") - _window_mean(reference, "after_range_rate_mps")
                for row, reference in paired
                if _window_mean(row, "after_range_rate_mps") is not None and _window_mean(reference, "after_range_rate_mps") is not None
            ]
            semantic_norm = _mean(
                _window_mean(row, "actual_semantic_delta_norm_m") or 0.0 for row in qualifying
            )
            vpp_norm = _mean(
                value
                for row, reference in paired
                for value in [_paired_mean_norm(row, reference, "env_vpp_world_offset_m")]
                if value is not None
            )
            guidance_norm = _mean(
                value
                for row, reference in paired
                for value in [_paired_mean_norm(row, reference, "guidance_command_normalized")]
                if value is not None
            )
            ego_rate = sum(bool(row.get("ego_failure")) for row in group) / max(len(group), 1)
            reference_ego_rate = sum(bool(row.get("ego_failure")) for row in reference_group) / max(len(reference_group), 1)
            method_summary[opponent] = {
                "records": len(group),
                "qualifying_episodes": len(qualifying),
                "paired_count": len(paired),
                "handoff_hashes_match": handoff_hashes_match,
                "mean_actual_semantic_delta_norm_m": semantic_norm,
                "mean_env_vpp_delta_norm_m_vs_reference": vpp_norm,
                "mean_guidance_delta_norm_vs_reference": guidance_norm,
                "mean_range_rate_delta_mps_vs_reference": _mean(range_rate_deltas),
                "mean_intent_loss_delta_vs_reference": _mean(intent_deltas),
                "ego_crash_oob_delta_vs_reference": ego_rate - reference_ego_rate,
                "safety_pass": ego_rate - reference_ego_rate <= safety_delta_max,
            }
        matrix[method] = method_summary
        if method != REFERENCE_METHOD:
            passed_opponents = [
                opponent
                for opponent, cell in method_summary.items()
                if cell["records"] == scenarios_per_opponent
                and cell["paired_count"] >= min_pairs
                and cell["handoff_hashes_match"]
                and cell["safety_pass"]
                and cell["mean_actual_semantic_delta_norm_m"] is not None
                and cell["mean_actual_semantic_delta_norm_m"] >= min_semantic_delta_norm_m
                and cell["mean_env_vpp_delta_norm_m_vs_reference"] is not None
                and cell["mean_env_vpp_delta_norm_m_vs_reference"] >= min_vpp_delta_norm_m
                and cell["mean_guidance_delta_norm_vs_reference"] is not None
                and cell["mean_guidance_delta_norm_vs_reference"] >= min_guidance_delta_norm
                and cell["mean_range_rate_delta_mps_vs_reference"] is not None
                and cell["mean_range_rate_delta_mps_vs_reference"] >= min_range_rate_delta_mps
                and cell["mean_intent_loss_delta_vs_reference"] is not None
                and cell["mean_intent_loss_delta_vs_reference"] <= practical_intent_loss_delta
            ]
            if len(passed_opponents) >= 2:
                training_candidates.append(method)
    expected_records = len(METHODS) * len(OPPONENTS) * scenarios_per_opponent
    return {
        "records": len(records),
        "expected_records": expected_records,
        "matrix": matrix,
        "semantic_transfer_candidates": training_candidates,
        "passes": len(records) == expected_records and "range_extension_full" in training_candidates,
    }
