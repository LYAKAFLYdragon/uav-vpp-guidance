"""Read-only gates for the bounded temporal-corridor compiler."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from uav_vpp_guidance.evaluation.thesis_three_knot_envelope import OPPONENTS


class TemporalCorridorFeasibilityError(RuntimeError):
    """Raised for a malformed temporal-corridor feasibility matrix."""


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_episodes: int,
    max_mean_residual_m: float,
    min_peak_action_norm: float,
) -> dict[str, Any]:
    """Require two independent opponents to have bounded, non-trivial corridors."""
    candidates = sorted({str(row["candidate_id"]) for row in rows})
    matrix: dict[str, dict[str, Any]] = {}
    passing_candidates: list[str] = []
    for candidate_id in candidates:
        per_opponent: dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in rows if row["candidate_id"] == candidate_id and row["opponent"] == opponent]
            episodes = {str(row["pair_key"]) for row in group}
            residuals = [float(row["projection_residual_norm_m"]) for row in group]
            peaks = [float(row["applied_action_norm"]) for row in group]
            hard_bound_steps = sum(bool(row["hard_bound_active"]) for row in group)
            cell = {
                "qualifying_episodes": len(episodes),
                "steps": len(group),
                "mean_projection_residual_m": sum(residuals) / max(len(residuals), 1),
                "max_applied_action_norm": max(peaks) if peaks else 0.0,
                "hard_bound_steps": hard_bound_steps,
            }
            cell["passes"] = (
                cell["qualifying_episodes"] >= min_episodes
                and cell["mean_projection_residual_m"] <= max_mean_residual_m
                and cell["max_applied_action_norm"] >= min_peak_action_norm
                and cell["hard_bound_steps"] == 0
            )
            per_opponent[opponent] = cell
        matrix[candidate_id] = per_opponent
        if sum(bool(cell["passes"]) for cell in per_opponent.values()) >= 2:
            passing_candidates.append(candidate_id)
    return {"matrix": matrix, "passing_candidates": passing_candidates, "passes": bool(passing_candidates)}
