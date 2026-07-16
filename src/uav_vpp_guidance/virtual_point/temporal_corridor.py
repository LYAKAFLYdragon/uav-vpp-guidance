"""Bounded receding-horizon VPP corridor compiler.

This module changes the *temporal parameterization* of a VPP command.  It does
not expand the aircraft, guidance, PID, or instantaneous raw-action envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class TemporalCorridorError(ValueError):
    """Raised when a corridor request cannot form a finite bounded action."""


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise TemporalCorridorError(f"{name} must be a finite 3-D vector") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise TemporalCorridorError(f"{name} must be a finite 3-D vector")
    return result


@dataclass(frozen=True)
class TemporalCorridorConfig:
    """A fixed four-second escape-to-reentry corridor in physical SI units."""

    horizon_steps: int = 20
    longitudinal_extent_m: float = 450.0
    lateral_extent_m: float = 300.0
    vertical_extent_m: float = 100.0
    lateral_sign: int = 1
    max_action_delta: float = 0.18

    def __post_init__(self) -> None:
        if self.horizon_steps < 3:
            raise TemporalCorridorError("horizon_steps must be at least three")
        if min(self.longitudinal_extent_m, self.lateral_extent_m, self.vertical_extent_m) < 0.0:
            raise TemporalCorridorError("corridor extents must be non-negative")
        if self.lateral_sign not in (-1, 1):
            raise TemporalCorridorError("lateral_sign must be -1 or 1")
        if not 0.0 < self.max_action_delta <= 2.0:
            raise TemporalCorridorError("max_action_delta must lie in (0, 2]")


def _unit_horizontal(vector: np.ndarray, name: str) -> np.ndarray:
    horizontal = vector.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-8:
        raise TemporalCorridorError(f"{name} horizontal direction is undefined")
    return horizontal / norm


def desired_world_offset(
    *,
    own_position_neu: Any,
    predicted_target_position_neu: Any,
    phase_step: int,
    config: TemporalCorridorConfig,
) -> np.ndarray:
    """Build one current-information corridor knot.

    The sine envelope starts and ends at zero, producing a bounded escape arc
    followed by a re-entry handoff.  The basis is rebuilt from the *current*
    own/predicted-target state, so no future simulator state is consumed.
    """
    own = _vector(own_position_neu, "own_position_neu")
    predicted = _vector(predicted_target_position_neu, "predicted_target_position_neu")
    progress = float(np.clip(int(phase_step), 0, config.horizon_steps - 1)) / float(config.horizon_steps - 1)
    envelope = float(np.sin(np.pi * progress))
    away = _unit_horizontal(own - predicted, "away")
    lateral = np.asarray([-away[1], away[0], 0.0], dtype=np.float64) * float(config.lateral_sign)
    desired = envelope * (
        config.longitudinal_extent_m * away
        + config.lateral_extent_m * lateral
        + np.asarray([0.0, 0.0, config.vertical_extent_m], dtype=np.float64)
    )
    if not np.isfinite(desired).all():
        raise TemporalCorridorError("corridor desired offset is non-finite")
    return desired


def compile_temporal_corridor_action(
    *,
    own_position_neu: Any,
    predicted_target_position_neu: Any,
    phase_step: int,
    previous_action: Any,
    raw_action_ranges: Sequence[Sequence[float]],
    config: TemporalCorridorConfig,
) -> dict[str, Any]:
    """Project a current corridor knot into the existing bounded VPP action head."""
    if len(raw_action_ranges) != 3:
        raise TemporalCorridorError("raw_action_ranges must contain three axes")
    limits = np.asarray(raw_action_ranges, dtype=np.float64)
    if limits.shape != (3, 2) or not np.isfinite(limits).all() or np.any(limits[:, 1] <= limits[:, 0]):
        raise TemporalCorridorError("raw_action_ranges are invalid")
    desired = desired_world_offset(
        own_position_neu=own_position_neu,
        predicted_target_position_neu=predicted_target_position_neu,
        phase_step=phase_step,
        config=config,
    )
    center = (limits[:, 0] + limits[:, 1]) / 2.0
    half_range = (limits[:, 1] - limits[:, 0]) / 2.0
    desired_action = (desired - center) / half_range
    previous = _vector(previous_action, "previous_action")
    slew_lower = previous - config.max_action_delta
    slew_upper = previous + config.max_action_delta
    applied = np.minimum(np.maximum(desired_action, slew_lower), slew_upper)
    applied = np.clip(applied, -1.0, 1.0)
    applied_offset = center + half_range * applied
    residual = desired - applied_offset
    return {
        "phase_step": int(phase_step),
        "desired_world_offset_m": desired.tolist(),
        "desired_action": desired_action.tolist(),
        "applied_action": applied.tolist(),
        "applied_world_offset_m": applied_offset.tolist(),
        "projection_residual_world_m": residual.tolist(),
        "projection_residual_norm_m": float(np.linalg.norm(residual)),
        "hard_bound_active": bool(np.any(np.abs(desired_action) > 1.0 + 1e-12)),
        "slew_bound_active": bool(np.any(np.abs(applied - previous) >= config.max_action_delta - 1e-12)),
    }
