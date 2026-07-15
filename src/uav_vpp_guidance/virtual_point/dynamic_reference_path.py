"""A non-learning, current-information-only dynamic reference-path core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.virtual_point.coordinate_transform import (
    offset_to_world,
    world_to_offset_frame,
)


class DynamicReferencePathError(ValueError):
    """Raised when a V16 reference path violates its geometry contract."""


def _vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise DynamicReferencePathError(f"{name} must be a finite 3-D vector")
    return result


@dataclass(frozen=True)
class DynamicReferencePathConfig:
    """Frozen V16 path geometry; all values are physical SI units."""

    escape_extent_m: float = 1200.0
    connector_fraction: float = 0.5
    opening_rate_start_mps: float = 0.0
    opening_rate_full_mps: float = 80.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.escape_extent_m) or self.escape_extent_m <= 0.0:
            raise DynamicReferencePathError("escape_extent_m must be finite and positive")
        if not 0.0 < self.connector_fraction < 1.0:
            raise DynamicReferencePathError("connector_fraction must lie in (0, 1)")
        if not self.opening_rate_full_mps > self.opening_rate_start_mps:
            raise DynamicReferencePathError("opening-rate bounds must be increasing")


def build_three_anchor_path(
    *,
    own_position_neu: Any,
    target_position_neu: Any,
    predicted_target_position_neu: Any,
    config: DynamicReferencePathConfig,
) -> np.ndarray:
    """Build escape, connector, and predicted-reentry anchors without future state.

    The only future-looking quantity is the already-frozen predictor output.
    The path is not a synthetic target and does not change terminal semantics.
    """

    own = _vector(own_position_neu, name="own_position_neu")
    target = _vector(target_position_neu, name="target_position_neu")
    predicted = _vector(predicted_target_position_neu, name="predicted_target_position_neu")
    los_to_ego = own - target
    distance = float(np.linalg.norm(los_to_ego))
    if distance <= 1e-6:
        raise DynamicReferencePathError("own and target positions must be distinct")
    escape = target + config.escape_extent_m * los_to_ego / distance
    connector = (1.0 - config.connector_fraction) * escape + config.connector_fraction * predicted
    path = np.vstack((escape, connector, predicted))
    if not np.isfinite(path).all():
        raise DynamicReferencePathError("reference path contains non-finite values")
    return path


def path_progress_from_range_rate(range_rate_mps: float, config: DynamicReferencePathConfig) -> float:
    """Map observed opening rate to a deterministic [0, 1] path progress."""

    value = float(range_rate_mps)
    if not np.isfinite(value):
        raise DynamicReferencePathError("range_rate_mps must be finite")
    return float(
        np.clip(
            (value - config.opening_rate_start_mps)
            / (config.opening_rate_full_mps - config.opening_rate_start_mps),
            0.0,
            1.0,
        )
    )


def select_path_anchor(path_neu: Any, progress: float) -> tuple[int, np.ndarray]:
    """Select escape, connector, or re-entry anchor from observed progress."""

    path = np.asarray(path_neu, dtype=np.float64)
    if path.shape != (3, 3) or not np.isfinite(path).all():
        raise DynamicReferencePathError("path_neu must be a finite [3, 3] matrix")
    if not np.isfinite(progress) or not 0.0 <= float(progress) <= 1.0:
        raise DynamicReferencePathError("progress must lie in [0, 1]")
    index = 0 if progress < 1.0 / 3.0 else 1 if progress < 2.0 / 3.0 else 2
    return index, path[index].copy()


def semantic_profile_for_anchor(index: int) -> str:
    """Map observable path progress to an existing, frozen VPP intent profile."""

    if int(index) in (0, 1):
        return "range_extension"
    if int(index) == 2:
        return "reentry_preparation"
    raise DynamicReferencePathError("anchor index must be 0, 1, or 2")


def _action_to_world_offset(
    action: Any,
    raw_action_ranges: Sequence[Sequence[float]],
    raw_action_offset_frame: str,
    own_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> np.ndarray:
    normalized = _vector(action, name="semantic_action")
    if np.any(np.abs(normalized) > 1.0 + 1e-6):
        raise DynamicReferencePathError("semantic_action must lie in [-1, 1]")
    if len(raw_action_ranges) != 3:
        raise DynamicReferencePathError("raw_action_ranges must contain three ranges")
    local_values: list[float] = []
    for limits, value in zip(raw_action_ranges, normalized):
        if len(limits) != 2:
            raise DynamicReferencePathError("each raw action range must contain [min, max]")
        lower, upper = float(limits[0]), float(limits[1])
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise DynamicReferencePathError("raw action ranges must be finite and increasing")
        local_values.append(lower + (float(value) + 1.0) * (upper - lower) / 2.0)
    return offset_to_world(
        np.asarray(local_values, dtype=np.float64),
        raw_action_offset_frame,
        own_state=dict(own_state),
        target_state=dict(target_state),
    )


def _world_offset_to_action(
    world_offset_neu: Any,
    raw_action_ranges: Sequence[Sequence[float]],
    raw_action_offset_frame: str,
    own_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> np.ndarray:
    world = _vector(world_offset_neu, name="world_offset_neu")
    if len(raw_action_ranges) != 3:
        raise DynamicReferencePathError("raw_action_ranges must contain three ranges")
    local = world_to_offset_frame(
        world,
        raw_action_offset_frame,
        own_state=dict(own_state),
        target_state=dict(target_state),
    )
    values: list[float] = []
    for limits, value in zip(raw_action_ranges, local):
        if len(limits) != 2:
            raise DynamicReferencePathError("each raw action range must contain [min, max]")
        lower, upper = float(limits[0]), float(limits[1])
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise DynamicReferencePathError("raw action ranges must be finite and increasing")
        values.append(2.0 * (float(value) - lower) / (upper - lower) - 1.0)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise DynamicReferencePathError("unclipped path action must be finite")
    return result


def compile_path_anchor_action(
    *,
    semantic_action: Any,
    selected_anchor_neu: Any,
    generator_anchor_neu: Any,
    raw_action_ranges: Sequence[Sequence[float]],
    raw_action_offset_frame: str,
    own_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay one selected path anchor on the existing bounded VPP action head.

    The existing generator still consumes one normalized 3-D action.  This
    adapter turns a selected absolute path anchor into an additional world-frame
    VPP offset relative to the generator's current predicted-target anchor.
    It neither replaces the target dynamics nor modifies guidance/PID.
    """

    selected = _vector(selected_anchor_neu, name="selected_anchor_neu")
    generator_anchor = _vector(generator_anchor_neu, name="generator_anchor_neu")
    semantic_world = _action_to_world_offset(
        semantic_action,
        raw_action_ranges,
        raw_action_offset_frame,
        own_state,
        target_state,
    )
    requested_path_delta = selected - generator_anchor
    requested_world_offset = semantic_world + requested_path_delta
    unclipped_action = _world_offset_to_action(
        requested_world_offset,
        raw_action_ranges,
        raw_action_offset_frame,
        own_state,
        target_state,
    )
    applied_action = np.clip(unclipped_action, -1.0, 1.0)
    applied_world_offset = _action_to_world_offset(
        applied_action,
        raw_action_ranges,
        raw_action_offset_frame,
        own_state,
        target_state,
    )
    actual_path_delta = applied_world_offset - semantic_world
    return {
        "generator_anchor_neu": generator_anchor.tolist(),
        "selected_anchor_neu": selected.tolist(),
        "semantic_world_offset_m": semantic_world.tolist(),
        "requested_path_world_delta_m": requested_path_delta.tolist(),
        "actual_path_world_delta_m": actual_path_delta.tolist(),
        "requested_path_delta_norm_m": float(np.linalg.norm(requested_path_delta)),
        "actual_path_delta_norm_m": float(np.linalg.norm(actual_path_delta)),
        "requested_vpp_world_offset_m": requested_world_offset.tolist(),
        "applied_vpp_world_offset_m": applied_world_offset.tolist(),
        "requested_vpp_position_neu": (generator_anchor + requested_world_offset).tolist(),
        "applied_vpp_position_neu": (generator_anchor + applied_world_offset).tolist(),
        "unclipped_action": unclipped_action.tolist(),
        "applied_action": applied_action.tolist(),
        "clipped": bool(not np.allclose(unclipped_action, applied_action, rtol=0.0, atol=1e-8)),
    }
