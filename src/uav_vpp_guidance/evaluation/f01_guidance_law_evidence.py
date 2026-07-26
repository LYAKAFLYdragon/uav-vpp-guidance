"""Pure, offline F01 guidance-law evidence models; not runtime control code."""
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

F04_NZ_LIMITS = (-2.0, 7.0)


@dataclass(frozen=True)
class F01CandidateParameters:
    """Proposed offline-only vertical-plane LOS-rate/PN parameters."""

    base_nz: float = 1.0
    geometric_gain_nz: float = 0.5
    navigation_constant: float = 1.5
    gravity_mps2: float = 9.80665
    max_closing_speed_mps: float = 300.0
    max_abs_los_rate_radps: float = 0.1
    epsilon_m: float = 1.0e-6
    capture_radius_m: float = 50.0


def _vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("relative position and velocity must be finite 3-vectors")
    return vector


def elevation_rad(relative_position_neu: Iterable[float], epsilon_m: float = 1.0e-6) -> float:
    """Elevation λ = atan2(up, horizontal-range) in the NEU frame."""
    position = _vector(relative_position_neu)
    horizontal_range = float(np.hypot(position[0], position[1]))
    return float(np.arctan2(position[2], max(horizontal_range, epsilon_m)))


def vertical_los_rate_radps(
    relative_position_neu: Iterable[float], relative_velocity_neu_mps: Iterable[float], epsilon_m: float = 1.0e-6
) -> float:
    """Time derivative of NEU elevation LOS angle; zero at singular range."""
    position, velocity = _vector(relative_position_neu), _vector(relative_velocity_neu_mps)
    horizontal_range = float(np.hypot(position[0], position[1]))
    total_range_sq = float(np.dot(position, position))
    if horizontal_range <= epsilon_m or total_range_sq <= epsilon_m**2:
        return 0.0
    horizontal_rate = float(np.dot(position[:2], velocity[:2]) / horizontal_range)
    return float((horizontal_range * velocity[2] - position[2] * horizontal_rate) / total_range_sq)


def range_rate_mps(relative_position_neu: Iterable[float], relative_velocity_neu_mps: Iterable[float], epsilon_m: float = 1.0e-6) -> float:
    position, velocity = _vector(relative_position_neu), _vector(relative_velocity_neu_mps)
    distance = float(np.linalg.norm(position))
    return 0.0 if distance <= epsilon_m else float(np.dot(position, velocity) / distance)


def frozen_baseline_nz(
    relative_position_neu: Iterable[float], *, base_nz: float = 1.0, k_los: float = 1.0, k_pos: float = 0.5, distance_scale_m: float = 2000.0
) -> Dict[str, float]:
    """Exact audited F01 normal-load composition, reproduced offline only."""
    position = _vector(relative_position_neu)
    distance = float(np.linalg.norm(position))
    geometry = k_pos * distance / distance_scale_m
    return {"distance_m": distance, "elevation_rad": elevation_rad(position), "los_rate_contribution_nz": 0.0, "geometric_contribution_nz": geometry, "raw_nz": base_nz + k_los * elevation_rad(position) + geometry}


def candidate_nz(
    relative_position_neu: Iterable[float],
    relative_velocity_neu_mps: Iterable[float],
    parameters: F01CandidateParameters = F01CandidateParameters(),
) -> Dict[str, float]:
    """Evaluate the specified candidate without importing runtime guidance code."""
    position, velocity = _vector(relative_position_neu), _vector(relative_velocity_neu_mps)
    distance = float(np.linalg.norm(position))
    if distance < parameters.capture_radius_m:
        return {
            "distance_m": distance,
            "elevation_rad": 0.0,
            "lambda_dot_radps": 0.0,
            "closing_speed_mps": 0.0,
            "geometric_contribution_nz": 0.0,
            "los_rate_contribution_nz": 0.0,
            "raw_nz": parameters.base_nz,
            "nz_cmd": parameters.base_nz,
            "capture_hold": True,
        }
    elevation = elevation_rad(position, parameters.epsilon_m)
    lambda_dot = vertical_los_rate_radps(position, velocity, parameters.epsilon_m)
    closing_speed = float(np.clip(-range_rate_mps(position, velocity, parameters.epsilon_m), 0.0, parameters.max_closing_speed_mps))
    geometric = parameters.geometric_gain_nz * float(np.sin(elevation))
    los_rate = parameters.navigation_constant * closing_speed * float(np.clip(lambda_dot, -parameters.max_abs_los_rate_radps, parameters.max_abs_los_rate_radps)) / parameters.gravity_mps2
    raw = parameters.base_nz + geometric + los_rate
    return {
        "distance_m": distance,
        "elevation_rad": elevation,
        "lambda_dot_radps": lambda_dot,
        "closing_speed_mps": closing_speed,
        "geometric_contribution_nz": geometric,
        "los_rate_contribution_nz": los_rate,
        "raw_nz": raw,
        "nz_cmd": float(np.clip(raw, *F04_NZ_LIMITS)),
        "capture_hold": False,
    }
