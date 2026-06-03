"""
Unified coordinate conversion utilities for trajectory prediction.

Convention:
- NEU: North-East-Up  (z-axis points up)
- NED: North-East-Down (z-axis points down)

All predictors and feature builders should use NEU internally.
When reading state dicts that may contain NED fields, use the helpers
below to normalize to NEU before any physics calculation.
"""

import numpy as np


def get_position_neu(state: dict) -> np.ndarray:
    """Extract position in NEU coordinates.

    Priority: position_neu > position_m > position

    Raises:
        ValueError: if no position field is found.
    """
    pos = state.get("position_neu")
    if pos is None:
        pos = state.get("position_m")
    if pos is None:
        pos = state.get("position")
    if pos is None:
        raise ValueError(
            "State missing position field (position_neu, position_m, or position)"
        )
    arr = np.asarray(pos, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            f"Position must be a 3-element vector, got shape {arr.shape}"
        )
    return arr


def get_velocity_neu(state: dict) -> np.ndarray:
    """Extract velocity in NEU coordinates.

    Priority:
      1. velocity_vector_mps (assumed NEU: [vn, ve, vu])
      2. velocity_ned (converted to NEU: [vn, ve, -vd])
      3. velocity (assumed NEU)
      4. Fallback: scalar velocity_mps / speed_mps / vt_mps + heading_rad

    Raises:
        ValueError: if no usable velocity information is found.
    """
    # 1. velocity_vector_mps (already NEU)
    vel = state.get("velocity_vector_mps")
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"velocity_vector_mps must be a 3-element vector, got shape {arr.shape}"
            )
        return arr

    # 2. velocity_ned -> convert to NEU
    vel_ned = state.get("velocity_ned")
    if vel_ned is not None:
        v = np.asarray(vel_ned, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(
                f"velocity_ned must be a 3-element vector, got shape {v.shape}"
            )
        return np.array([v[0], v[1], -v[2]], dtype=np.float64)

    # 3. velocity (assumed NEU)
    vel = state.get("velocity")
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"velocity must be a 3-element vector, got shape {arr.shape}"
            )
        return arr

    # 4. Scalar fallback
    speed = (
        state.get("velocity_mps")
        or state.get("speed_mps")
        or state.get("vt_mps")
    )
    heading = state.get("heading_rad")
    if heading is None and "attitude_rpy" in state:
        heading = np.asarray(state["attitude_rpy"], dtype=np.float64)[2]

    if speed is not None and heading is not None:
        return np.array([
            speed * np.cos(heading),
            speed * np.sin(heading),
            0.0,
        ], dtype=np.float64)

    raise ValueError(
        "State missing velocity field (velocity_vector_mps, velocity_ned, velocity, "
        "or velocity_mps + heading_rad)"
    )


def get_acceleration_neu(state: dict) -> np.ndarray:
    """Extract acceleration in NEU coordinates if available.

    Priority: acceleration_vector_mps2 > acceleration_ned (converted)

    Returns:
        np.ndarray of shape (3,) or None if not available.
    """
    acc = state.get("acceleration_vector_mps2")
    if acc is not None:
        return np.asarray(acc, dtype=np.float64)

    acc_ned = state.get("acceleration_ned")
    if acc_ned is not None:
        a = np.asarray(acc_ned, dtype=np.float64)
        return np.array([a[0], a[1], -a[2]], dtype=np.float64)

    return None


def ned_to_neu(v_ned: np.ndarray) -> np.ndarray:
    """Convert a vector from NED to NEU: [n, e, -d]."""
    v = np.asarray(v_ned, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"Expected 3-element vector, got shape {v.shape}")
    return np.array([v[0], v[1], -v[2]], dtype=np.float64)


def neu_to_ned(v_neu: np.ndarray) -> np.ndarray:
    """Convert a vector from NEU to NED: [n, e, -u]."""
    v = np.asarray(v_neu, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"Expected 3-element vector, got shape {v.shape}")
    return np.array([v[0], v[1], -v[2]], dtype=np.float64)
