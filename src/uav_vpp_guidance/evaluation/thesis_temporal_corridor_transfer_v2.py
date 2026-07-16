"""Unpooled evidence gate for the repaired V2 temporal-corridor transfer probe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


SOURCE_ID = "THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V2"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = ("frozen_fixed_head_on", "moderate_left", "moderate_right")
REFERENCE_METHOD = "frozen_fixed_head_on"


def is_ego_failure(reason: Any) -> bool:
    value = str(reason or "").lower()
    return not value.startswith("target_") and ("crash" in value or "out_of_bounds" in value)


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _qualifying(rows: Sequence[Mapping[str, Any]], steps: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("handoff_reached"))
        and bool(row.get("contract_passed"))
        and int(row.get("window_steps", 0)) == steps
    ]


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    window_steps: int,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute all paired gates independently for each candidate and opponent."""

    grouped = {
        (method, opponent): [
            row for row in records if row.get("method") == method and row.get("opponent") == opponent
        ]
        for method in METHODS
        for opponent in OPPONENTS
    }
    expected = len(METHODS) * len(OPPONENTS) * scenarios_per_opponent
    matrix: dict[str, dict[str, Any]] = {}
    candidates: list[str] = []
    for method in METHODS:
        per_opponent: dict[str, Any] = {}
        for opponent in OPPONENTS:
            rows = grouped[(method, opponent)]
            reference_rows = grouped[(REFERENCE_METHOD, opponent)]
            references = {
                str(row.get("pair_key")): row for row in _qualifying(reference_rows, window_steps)
            }
            pairs = [
                (row, references[str(row.get("pair_key"))])
                for row in _qualifying(rows, window_steps)
                if str(row.get("pair_key")) in references
            ]
            hashes_match = all(
                candidate.get("handoff_state_hash") == reference.get("handoff_state_hash")
                for candidate, reference in pairs
            )
            component_deltas: list[np.ndarray] = []
            weighted_deltas: list[float] = []
            residuals: list[float] = []
            hard_bound_steps = 0
            for candidate, reference in pairs:
                for step, baseline in zip(candidate.get("window", []), reference.get("window", [])):
                    error = np.asarray(step.get("after_component_errors"), dtype=np.float64)
                    baseline_error = np.asarray(baseline.get("after_component_errors"), dtype=np.float64)
                    if error.shape == (6,) and baseline_error.shape == (6,) and np.isfinite(error).all() and np.isfinite(baseline_error).all():
                        component_deltas.append(error - baseline_error)
                    weighted_deltas.append(
                        float(step.get("after_weighted_error")) - float(baseline.get("after_weighted_error"))
                    )
                    receipt = step.get("temporal_corridor_receipt")
                    if isinstance(receipt, Mapping):
                        residuals.append(float(receipt.get("projection_residual_norm_m")))
                        hard_bound_steps += int(bool(receipt.get("hard_bound_active")))

            ego_rate = sum(bool(row.get("ego_failure")) for row in rows) / max(len(rows), 1)
            reference_ego_rate = sum(bool(row.get("ego_failure")) for row in reference_rows) / max(len(reference_rows), 1)
            component_mean = np.mean(component_deltas, axis=0).tolist() if component_deltas else [float("nan")] * 6
            cell = {
                "records": len(rows),
                "qualifying_pairs": len(pairs),
                "handoff_hashes_match": hashes_match,
                "mean_component_delta": component_mean,
                "mean_weighted_error_delta": _mean(weighted_deltas),
                "mean_projection_residual_m": _mean(residuals),
                "hard_bound_steps": hard_bound_steps,
                "ego_crash_oob_delta": ego_rate - reference_ego_rate,
            }
            if method == REFERENCE_METHOD:
                cell["passes"] = False
            else:
                improved = sum(
                    value <= float(gates["improved_component_delta_max"])
                    for value in component_mean
                )
                cell["improved_components"] = improved
                cell["passes"] = bool(
                    cell["qualifying_pairs"] >= int(gates["min_qualifying_pairs_per_opponent"])
                    and cell["handoff_hashes_match"]
                    and cell["mean_projection_residual_m"] is not None
                    and cell["mean_projection_residual_m"] <= float(gates["max_mean_projection_residual_m"])
                    and cell["hard_bound_steps"] <= int(gates["hard_bound_steps_max"])
                    and cell["ego_crash_oob_delta"] <= float(gates["ego_crash_oob_delta_vs_reference_max"])
                    and cell["mean_weighted_error_delta"] is not None
                    and cell["mean_weighted_error_delta"] <= float(gates["weighted_error_delta_vs_reference_max"])
                    and improved >= int(gates["min_improved_components"])
                    and max(component_mean) <= float(gates["component_worsening_delta_max"])
                )
            per_opponent[opponent] = cell
        matrix[method] = per_opponent
        if method != REFERENCE_METHOD and sum(bool(cell["passes"]) for cell in per_opponent.values()) >= int(gates["candidate_opponents_required"]):
            candidates.append(method)
    return {
        "records": len(records),
        "expected_records": expected,
        "matrix": matrix,
        "mechanism_candidates": candidates,
        "passes": len(records) == expected and bool(candidates),
    }
