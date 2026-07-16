"""Read-only feasibility analysis for three-knot reference-path candidates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.virtual_point.dynamic_reference_path import (
    DynamicReferencePathConfig,
    build_three_anchor_path,
    compile_path_anchor_action,
    path_progress_from_range_rate,
    select_path_anchor,
)


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


class ThreeKnotEnvelopeError(RuntimeError):
    """Raised for malformed frozen V16 telemetry."""


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ThreeKnotEnvelopeError(f"missing {name}") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ThreeKnotEnvelopeError(f"invalid {name}")
    return result


def evaluate_step(step: Mapping[str, Any], candidate: Mapping[str, Any], raw_action_ranges: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Replay a candidate path request against the frozen one-point VPP envelope."""
    own = _vector(step.get("own_position_neu"), "own_position_neu")
    target = _vector(step.get("target_position_neu"), "target_position_neu")
    predicted = _vector(step.get("predicted_target_position_neu"), "predicted_target_position_neu")
    semantic = step.get("semantic_receipt")
    if not isinstance(semantic, Mapping):
        raise ThreeKnotEnvelopeError("missing semantic action receipt")
    config = DynamicReferencePathConfig(
        escape_extent_m=float(candidate["escape_extent_m"]),
        connector_fraction=float(candidate["connector_fraction"]),
        opening_rate_start_mps=float(candidate["opening_rate_start_mps"]),
        opening_rate_full_mps=float(candidate["opening_rate_full_mps"]),
    )
    path = build_three_anchor_path(
        own_position_neu=own,
        target_position_neu=target,
        predicted_target_position_neu=predicted,
        config=config,
    )
    progress = path_progress_from_range_rate(float(step["range_rate_mps"]), config)
    selected_index, selected_anchor = select_path_anchor(path, progress)
    receipt = compile_path_anchor_action(
        semantic_action=semantic["applied_action"],
        selected_anchor_neu=selected_anchor,
        generator_anchor_neu=predicted,
        raw_action_ranges=raw_action_ranges,
        raw_action_offset_frame="world_neu",
        own_state={"position_neu": own},
        target_state={"position_neu": target},
    )
    unclipped = _vector(receipt["unclipped_action"], "unclipped_action")
    return {
        "selected_anchor_index": int(selected_index),
        "path_progress": float(progress),
        "requested_path_delta_norm_m": float(receipt["requested_path_delta_norm_m"]),
        "actual_path_delta_norm_m": float(receipt["actual_path_delta_norm_m"]),
        "clipped": bool(receipt["clipped"]),
        "clipped_axes": [bool(abs(value) > 1.0 + 1e-8) for value in unclipped],
        "unclipped_action": unclipped.tolist(),
        "applied_action": _vector(receipt["applied_action"], "applied_action").tolist(),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, min_fully_unclipped_episodes: int, min_unclipped_step_fraction: float) -> dict[str, Any]:
    """Apply the preregistration-readiness gate separately for each opponent."""
    candidates = sorted({str(row["candidate_id"]) for row in rows})
    matrix: dict[str, dict[str, Any]] = {}
    passing_candidates: list[str] = []
    for candidate_id in candidates:
        candidate_matrix: dict[str, Any] = {}
        for opponent in OPPONENTS:
            group = [row for row in rows if row["candidate_id"] == candidate_id and row["opponent"] == opponent]
            by_episode: dict[str, list[Mapping[str, Any]]] = {}
            for row in group:
                by_episode.setdefault(str(row["pair_key"]), []).append(row)
            clipped_axis_counts = [sum(bool(row["clipped_axes"][axis]) for row in group) for axis in range(3)]
            unclipped_steps = sum(not bool(row["clipped"]) for row in group)
            fully_unclipped = sum(not any(bool(row["clipped"]) for row in episode_rows) for episode_rows in by_episode.values())
            fraction = unclipped_steps / max(len(group), 1)
            candidate_matrix[opponent] = {
                "qualifying_episodes": len(by_episode),
                "steps": len(group),
                "unclipped_step_fraction": fraction,
                "fully_unclipped_episodes": fully_unclipped,
                "clipped_axis_counts_xyz": clipped_axis_counts,
                "passes": fully_unclipped >= min_fully_unclipped_episodes and fraction >= min_unclipped_step_fraction,
            }
        matrix[candidate_id] = candidate_matrix
        if sum(bool(cell["passes"]) for cell in candidate_matrix.values()) >= 2:
            passing_candidates.append(candidate_id)
    return {"matrix": matrix, "passing_candidates": passing_candidates, "passes": bool(passing_candidates)}
