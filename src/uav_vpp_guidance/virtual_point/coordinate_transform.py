"""
Coordinate transforms for virtual point generation.

Converts offsets in tactical local frames to NEU world coordinates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


VALID_OFFSET_FRAMES = {
    "world_neu",
    "target_velocity",
    "los_relative",
    "encounter",
    "encounter_stable",
}
ENCOUNTER_STABLE_MAX_HEADING_DELTA_DEG = 45.0


def _as_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    arr = arr[:3]
    if not np.isfinite(arr).all():
        return None
    return arr


def _get_position(state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(state, dict):
        return None
    for key in ("position_neu", "position_m", "position"):
        value = _as_vector(state.get(key))
        if value is not None:
            return value
    return None


def _get_velocity_neu(state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(state, dict):
        return None
    for key in ("velocity_vector_mps", "velocity"):
        value = _as_vector(state.get(key))
        if value is not None:
            return value
    value = _as_vector(state.get("velocity_ned"))
    if value is not None:
        return np.array([value[0], value[1], -value[2]], dtype=np.float64)
    return None


def _heading_vector_from_state(state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(state, dict):
        return None
    for key in ("heading_rad", "yaw_rad", "psi_rad"):
        if key in state:
            heading = float(state[key])
            if np.isfinite(heading):
                return np.array(
                    [np.cos(heading), np.sin(heading), 0.0],
                    dtype=np.float64,
                )
    for key in ("heading_deg", "yaw_deg", "psi_deg"):
        if key in state:
            heading = np.deg2rad(float(state[key]))
            if np.isfinite(heading):
                return np.array(
                    [np.cos(heading), np.sin(heading), 0.0],
                    dtype=np.float64,
                )
    return None


def _horizontal_unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    value = _as_vector(vector)
    if value is None:
        return None
    horiz = np.array([value[0], value[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(horiz[:2]))
    if norm <= 1e-8:
        return None
    return horiz / norm


def _right_normal(forward: np.ndarray) -> np.ndarray:
    """Return the horizontal right-hand normal for a NEU forward vector."""
    return np.array([-forward[1], forward[0], 0.0], dtype=np.float64)


def _basis_from_forward(forward: np.ndarray) -> np.ndarray:
    forward_unit = _horizontal_unit(forward)
    if forward_unit is None:
        forward_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    lateral_unit = _right_normal(forward_unit)
    up_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.column_stack([forward_unit, lateral_unit, up_unit])


def _target_velocity_forward(target_state: Optional[Dict[str, Any]]) -> np.ndarray:
    velocity_forward = _horizontal_unit(_get_velocity_neu(target_state))
    if velocity_forward is not None:
        return velocity_forward
    heading_forward = _horizontal_unit(_heading_vector_from_state(target_state))
    if heading_forward is not None:
        return heading_forward
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _los_forward(
    own_state: Optional[Dict[str, Any]],
    target_state: Optional[Dict[str, Any]],
) -> np.ndarray:
    own_pos = _get_position(own_state)
    target_pos = _get_position(target_state)
    if own_pos is not None and target_pos is not None:
        los_forward = _horizontal_unit(target_pos - own_pos)
        if los_forward is not None:
            return los_forward
    return _target_velocity_forward(target_state)


def _encounter_forward(
    own_state: Optional[Dict[str, Any]],
    target_state: Optional[Dict[str, Any]],
) -> np.ndarray:
    target_forward = _target_velocity_forward(target_state)
    los_forward = _los_forward(own_state, target_state)
    bisector = _horizontal_unit(target_forward + los_forward)
    if bisector is not None:
        return bisector
    own_vel = _get_velocity_neu(own_state)
    target_vel = _get_velocity_neu(target_state)
    if own_vel is not None and target_vel is not None:
        relative_forward = _horizontal_unit(target_vel - own_vel)
        if relative_forward is not None:
            return relative_forward
    return target_forward


def _signed_horizontal_angle(from_forward: np.ndarray, to_forward: np.ndarray) -> float:
    source = _horizontal_unit(from_forward)
    target = _horizontal_unit(to_forward)
    if source is None or target is None:
        return 0.0
    cross_z = source[0] * target[1] - source[1] * target[0]
    dot = float(np.clip(np.dot(source[:2], target[:2]), -1.0, 1.0))
    return float(np.arctan2(cross_z, dot))


def _rotate_horizontal(forward: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = _horizontal_unit(forward)
    if unit is None:
        unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return np.array(
        [
            unit[0] * cos_a - unit[1] * sin_a,
            unit[0] * sin_a + unit[1] * cos_a,
            0.0,
        ],
        dtype=np.float64,
    )


def _encounter_stable_forward(
    own_state: Optional[Dict[str, Any]],
    target_state: Optional[Dict[str, Any]],
    *,
    max_heading_delta_deg: float = ENCOUNTER_STABLE_MAX_HEADING_DELTA_DEG,
) -> np.ndarray:
    """Clamp encounter forward toward target motion for stable rear-quarter geometry."""
    target_forward = _target_velocity_forward(target_state)
    encounter_forward = _encounter_forward(own_state, target_state)
    delta_rad = _signed_horizontal_angle(target_forward, encounter_forward)
    max_delta_rad = np.deg2rad(float(max_heading_delta_deg))
    if abs(delta_rad) <= max_delta_rad:
        return encounter_forward
    return _rotate_horizontal(
        target_forward,
        float(np.clip(delta_rad, -max_delta_rad, max_delta_rad)),
    )


def offset_frame_basis(
    offset_frame: str,
    *,
    own_state: Optional[Dict[str, Any]] = None,
    target_state: Optional[Dict[str, Any]] = None,
    encounter_stable_max_heading_delta_deg: float = ENCOUNTER_STABLE_MAX_HEADING_DELTA_DEG,
) -> np.ndarray:
    """Return a 3x3 local-to-world basis for a VPP offset frame."""
    frame = str(offset_frame or "world_neu")
    if frame not in VALID_OFFSET_FRAMES:
        raise ValueError(
            f"Unknown virtual_point.offset_frame={frame!r}; "
            f"expected one of {sorted(VALID_OFFSET_FRAMES)}"
        )
    if frame == "world_neu":
        return np.eye(3, dtype=np.float64)
    if frame == "target_velocity":
        return _basis_from_forward(_target_velocity_forward(target_state))
    if frame == "los_relative":
        return _basis_from_forward(_los_forward(own_state, target_state))
    if frame == "encounter_stable":
        return _basis_from_forward(
            _encounter_stable_forward(
                own_state,
                target_state,
                max_heading_delta_deg=encounter_stable_max_heading_delta_deg,
            )
        )
    return _basis_from_forward(_encounter_forward(own_state, target_state))


def offset_to_world(
    offset,
    offset_frame: str = "world_neu",
    *,
    own_state: Optional[Dict[str, Any]] = None,
    target_state: Optional[Dict[str, Any]] = None,
    encounter_stable_max_heading_delta_deg: float = ENCOUNTER_STABLE_MAX_HEADING_DELTA_DEG,
) -> np.ndarray:
    """Convert a local VPP offset to a NEU world offset."""
    local = np.asarray(offset, dtype=np.float64).reshape(3)
    basis = offset_frame_basis(
        offset_frame,
        own_state=own_state,
        target_state=target_state,
        encounter_stable_max_heading_delta_deg=encounter_stable_max_heading_delta_deg,
    )
    return basis @ local


def world_to_offset_frame(
    world_offset,
    offset_frame: str = "world_neu",
    *,
    own_state: Optional[Dict[str, Any]] = None,
    target_state: Optional[Dict[str, Any]] = None,
    encounter_stable_max_heading_delta_deg: float = ENCOUNTER_STABLE_MAX_HEADING_DELTA_DEG,
) -> np.ndarray:
    """Project a NEU world offset into a VPP local frame."""
    world = np.asarray(world_offset, dtype=np.float64).reshape(3)
    basis = offset_frame_basis(
        offset_frame,
        own_state=own_state,
        target_state=target_state,
        encounter_stable_max_heading_delta_deg=encounter_stable_max_heading_delta_deg,
    )
    return basis.T @ world


def heading_to_rotation_matrix(heading_deg, pitch_deg=0.0, roll_deg=0.0):
    """
    Construct a rotation matrix from heading-pitch-roll angles (Z-Y-X convention).

    Args:
        heading_deg (float): Heading angle in degrees.
        pitch_deg (float): Pitch angle in degrees.
        roll_deg (float): Roll angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    heading = np.deg2rad(float(heading_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))

    ch, sh = np.cos(heading), np.sin(heading)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rz = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def target_relative_to_world(offset, target_state):
    """
    Convert a target-velocity-relative offset to a world NEU offset.

    Args:
        offset (np.ndarray): [dx, dy, dz] in target-velocity frame.
        target_state (dict): Target aircraft state with velocity or heading.

    Returns:
        np.ndarray: World NEU offset [x, y, z].
    """
    return offset_to_world(
        offset,
        "target_velocity",
        target_state=target_state,
    )
