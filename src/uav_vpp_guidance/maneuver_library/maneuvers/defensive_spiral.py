"""Defensive spiral maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class DefensiveSpiral(Maneuver):
    """Execute a defensive spiral: a sustained steep coordinated turn combined
    with a controlled descent.

    Used as a last-ditch defensive maneuver when the bandit is tightly
    pursued.  The descent trades altitude for energy and turn rate, while the
    continuous turn makes it hard for the opponent to stabilize a gun solution.

    Parameters
    ----------
    bank_angle_deg : float
        Steep bank angle (deg).  Default 65 deg.
    dive_angle_deg : float
        Descent flight-path angle (deg).  Default 15 deg.
    throttle : float
        Throttle setting.  Default 0.8.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1000 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 160 m/s.
    max_duration_s : float
        Maximum maneuver duration (s).  Default 8.0 s.
    """

    name = "defensive_spiral"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 65.0))
        self.dive_angle = math.radians(self.params.get("dive_angle_deg", 15.0))
        self.throttle = self.params.get("throttle", 0.8)
        self.min_altitude = self.params.get("min_altitude_m", 1000.0)
        self.min_speed = self.params.get("min_speed_mps", 160.0)
        self.max_duration = self.params.get("max_duration_s", 8.0)
        self._sign = 1.0

    def can_enter(self, state: FlightState) -> bool:
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        min_speed = 80.0 * math.sqrt(n_req)
        return (
            state.velocity_mps >= min_speed
            and state.altitude_m >= self.min_altitude + 1000.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._sign = 1.0

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        n_req = 1.0 / math.cos(abs(self.bank_angle)) + 0.2
        return ManeuverSetpoint(
            phi_ref=self._sign * self.bank_angle,
            theta_ref=-self.dive_angle,
            nz_ref=n_req,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        return self._elapsed >= self.max_duration
