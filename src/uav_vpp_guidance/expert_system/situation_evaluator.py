"""
SituationEvaluator

Computes tactical situation scores and classifies the engagement state.

Metrics used:
  - range_m
  - aspect_deg (Antenna Train Angle / AA): smaller = better for ego
  - ata_deg (Aspect Target Angle): larger ≈ target tail-on to ego
  - range_rate_mps (closure rate, positive = closing)
  - altitude_diff_m
  - ego_speed, target_speed
  - ego_altitude
"""

import numpy as np


class TacticalState:
    UNSAFE = "UNSAFE"
    STRONG_OFFENSIVE = "STRONG_OFFENSIVE"
    OFFENSIVE = "OFFENSIVE"
    NEUTRAL = "NEUTRAL"
    DEFENSIVE = "DEFENSIVE"


class SituationEvaluator:
    """
    Evaluate air-combat tactical situation based on relative geometry.
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.strong_offensive_range_m = cfg.get("strong_offensive_range_m", 1200.0)
        self.offensive_range_m = cfg.get("offensive_range_m", 2500.0)
        self.strong_offensive_aspect_deg = cfg.get("strong_offensive_aspect_deg", 30.0)
        self.offensive_aspect_deg = cfg.get("offensive_aspect_deg", 60.0)
        self.defensive_aspect_deg = cfg.get("defensive_aspect_deg", 120.0)
        self.min_safe_altitude_m = cfg.get("min_safe_altitude_m", 300.0)
        self.min_safe_speed_mps = cfg.get("min_safe_speed_mps", 80.0)
        self.close_range_m = cfg.get("close_range_m", 700.0)
        self.high_closure_rate_mps = cfg.get("high_closure_rate_mps", 120.0)

        # Scoring weights
        weights = cfg.get("weights", {})
        self.w_angle = weights.get("angle", 0.4)
        self.w_range = weights.get("range", 0.25)
        self.w_energy = weights.get("energy", 0.2)
        self.w_altitude = weights.get("altitude", 0.1)
        self.w_closure = weights.get("closure", 0.05)

        # Scoring scales
        self.r_desired = cfg.get("r_desired_m", 900.0)
        self.r_scale = cfg.get("r_scale_m", 3000.0)
        self.energy_scale = cfg.get("energy_scale", 1e6)
        self.altitude_scale = cfg.get("altitude_scale_m", 1000.0)
        self.closure_scale = cfg.get("closure_scale_mps", 200.0)

    def evaluate(self, own_state, target_state, rel_state):
        """
        Evaluate situation and return tactical state + scores.

        Returns:
            dict with keys:
                tactical_state: str
                scores: dict of raw score components
                situation_score: float aggregate score
                diagnostics: dict for logging
        """
        range_m = float(rel_state.get("range_m", 5000.0))
        aspect_deg = float(np.rad2deg(rel_state.get("aa_rad", np.pi)))   # AA -> aspect_deg in info
        ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))       # Aspect Target Angle
        closure_rate = -float(rel_state.get("range_rate_mps", 0.0))      # positive = closing
        altitude_diff_m = float(rel_state.get("altitude_diff_m", 0.0))   # target - own

        ego_speed = float(np.linalg.norm(_get_velocity(own_state)))
        target_speed = float(np.linalg.norm(_get_velocity(target_state)))
        ego_altitude = _get_altitude(own_state)
        target_altitude = _get_altitude(target_state)

        # --- Score components ---
        angle_score = 1.0 - np.clip(aspect_deg / 180.0, 0.0, 1.0) * 2.0

        range_score = 1.0 - abs(range_m - self.r_desired) / self.r_scale
        range_score = np.clip(range_score, -1.0, 1.0)

        # Energy proxy (simplified)
        g = 9.81
        ego_energy = 0.5 * ego_speed ** 2 + g * ego_altitude
        target_energy = 0.5 * target_speed ** 2 + g * target_altitude
        energy_advantage = ego_energy - target_energy
        energy_score = float(np.tanh(energy_advantage / self.energy_scale))

        # Altitude score
        if ego_altitude < self.min_safe_altitude_m:
            altitude_score = -1.0
        else:
            altitude_score = float(np.tanh(altitude_diff_m / self.altitude_scale))

        closure_score = float(np.tanh(closure_rate / self.closure_scale))

        situation_score = (
            self.w_angle * angle_score
            + self.w_range * range_score
            + self.w_energy * energy_score
            + self.w_altitude * altitude_score
            + self.w_closure * closure_score
        )

        # --- Tactical state classification ---
        altitude_safe = ego_altitude >= self.min_safe_altitude_m
        speed_safe = ego_speed >= self.min_safe_speed_mps

        if not altitude_safe or not speed_safe:
            tactical_state = TacticalState.UNSAFE
        elif aspect_deg >= self.defensive_aspect_deg:
            tactical_state = TacticalState.DEFENSIVE
        elif range_m <= self.strong_offensive_range_m and aspect_deg <= self.strong_offensive_aspect_deg and altitude_safe:
            tactical_state = TacticalState.STRONG_OFFENSIVE
        elif range_m <= self.offensive_range_m and aspect_deg <= self.offensive_aspect_deg:
            tactical_state = TacticalState.OFFENSIVE
        else:
            tactical_state = TacticalState.NEUTRAL

        diagnostics = {
            "range_m": range_m,
            "aspect_deg": aspect_deg,
            "ata_deg": ata_deg,
            "closure_rate": closure_rate,
            "altitude_diff_m": altitude_diff_m,
            "ego_speed": ego_speed,
            "target_speed": target_speed,
            "ego_altitude": ego_altitude,
            "target_altitude": target_altitude,
            "altitude_safe": altitude_safe,
            "speed_safe": speed_safe,
            "angle_score": float(angle_score),
            "range_score": float(range_score),
            "energy_score": float(energy_score),
            "altitude_score": float(altitude_score),
            "closure_score": float(closure_score),
            "situation_score": float(situation_score),
            "tactical_state": tactical_state,
        }

        return {
            "tactical_state": tactical_state,
            "scores": {
                "angle": float(angle_score),
                "range": float(range_score),
                "energy": float(energy_score),
                "altitude": float(altitude_score),
                "closure": float(closure_score),
            },
            "situation_score": float(situation_score),
            "diagnostics": diagnostics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_velocity(state):
    vel = state.get("velocity_ned")
    if vel is None:
        vel = state.get("velocity_vector_mps")
    if vel is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vel, dtype=np.float64)


def _get_altitude(state):
    alt = state.get("altitude_m")
    if alt is not None:
        return float(alt)
    pos = state.get("position_m")
    if pos is None:
        pos = state.get("position_neu")
    if pos is not None:
        return float(np.asarray(pos, dtype=np.float64)[2])
    return 0.0
