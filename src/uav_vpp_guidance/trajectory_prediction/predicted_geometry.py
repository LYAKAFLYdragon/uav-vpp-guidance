"""Predicted relative-geometry features for the high-level policy.

Only the current own-aircraft state, current target state, and the
predictor-produced target position are used. No future simulator state is
read by this module.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def _position(state: Dict[str, Any]) -> np.ndarray:
    for key in ("position_neu", "position_m", "position"):
        value = state.get(key)
        if value is not None:
            result = np.asarray(value, dtype=np.float64).reshape(-1)
            if result.shape == (3,) and np.isfinite(result).all():
                return result
    raise ValueError("state has no finite 3-D position")


def _velocity(state: Dict[str, Any]) -> np.ndarray:
    value = state.get("velocity_vector_mps")
    if value is not None:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        value = state.get("velocity_ned")
        if value is not None:
            ned = np.asarray(value, dtype=np.float64).reshape(-1)
            result = np.array([ned[0], ned[1], -ned[2]], dtype=np.float64)
        else:
            value = state.get("velocity")
            result = (
                np.asarray(value, dtype=np.float64).reshape(-1)
                if value is not None
                else None
            )
    if result is None or result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("state has no finite 3-D velocity")
    return result


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def compute_predicted_geometry(
    own_state: Dict[str, Any],
    target_state: Dict[str, Any],
    predicted_target_position: Any,
    lookahead_time_s: float,
) -> Dict[str, float]:
    """Compute predicted distance, rate, ATA and AA from a predicted position.

    The predicted target velocity is an estimate from predicted displacement
    divided by the configured lookahead. It is not a future target velocity
    read from the simulator.
    """
    own_pos = _position(own_state)
    target_pos = _position(target_state)
    predicted_pos = np.asarray(predicted_target_position, dtype=np.float64).reshape(-1)
    if predicted_pos.shape != (3,) or not np.isfinite(predicted_pos).all():
        raise ValueError("predicted target position must be a finite 3-D vector")

    horizon = max(float(lookahead_time_s), 1e-6)
    predicted_relative = predicted_pos - own_pos
    predicted_displacement = predicted_pos - target_pos
    predicted_velocity = predicted_displacement / horizon
    predicted_distance = float(np.linalg.norm(predicted_relative))
    current_distance = float(np.linalg.norm(target_pos - own_pos))
    predicted_range_rate = (predicted_distance - current_distance) / horizon

    los = predicted_relative / max(predicted_distance, 1e-8)
    predicted_ata_rad = _angle_between(-los, predicted_velocity)
    predicted_aa_rad = _angle_between(los, _velocity(own_state))

    return {
        "predicted_range_m": predicted_distance,
        "predicted_range_rate_mps": float(predicted_range_rate),
        "predicted_ata_rad": predicted_ata_rad,
        "predicted_aa_rad": predicted_aa_rad,
        "predicted_ata_deg": float(np.degrees(predicted_ata_rad)),
        "predicted_aa_deg": float(np.degrees(predicted_aa_rad)),
        "predicted_target_velocity_mps": float(np.linalg.norm(predicted_velocity)),
    }


def normalize_for_base_observation(geometry: Dict[str, float]) -> Dict[str, float]:
    """Map predicted geometry to the existing 16-D base-observation scale."""
    return {
        "range_m": float(geometry["predicted_range_m"] / 5000.0),
        "range_rate_mps": float(geometry["predicted_range_rate_mps"] / 200.0),
        "ata_sin": float(np.sin(geometry["predicted_ata_rad"])),
        "ata_cos": float(np.cos(geometry["predicted_ata_rad"])),
        "aa_sin": float(np.sin(geometry["predicted_aa_rad"])),
        "aa_cos": float(np.cos(geometry["predicted_aa_rad"])),
    }
