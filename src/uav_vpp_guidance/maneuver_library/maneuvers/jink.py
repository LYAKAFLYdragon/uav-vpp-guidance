"""Defensive jink maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Jink(Maneuver):
    """Execute a rapid roll/yaw jink to spoil a gun or missile tracking solution.

    The maneuver alternates left/right bank commands in a short doublet-like
    pattern.  It is most useful in the terminal phase of an engagement where
    the bandit is inside the opponent's weapon envelope.

    Parameters
    ----------
    bank_angle_deg : float
        Peak bank angle magnitude (deg).  Default 45 deg.
    roll_rate_dps : float
        Roll rate during the jink (deg/s).  Default 120 deg/s.
    cycles : int
        Number of left-right doublets.  Default 3.
    nz_ref : float
        Load factor to maintain during the jink (g).  Default 1.3 g.
    throttle : float
        Throttle setting.  Default 0.9.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1500 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 150 m/s.
    """

    name = "jink"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 45.0))
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 120.0))
        self.cycles = self.params.get("cycles", 3)
        self.nz_ref = self.params.get("nz_ref", 1.3)
        self.throttle = self.params.get("throttle", 0.9)
        self.min_altitude = self.params.get("min_altitude_m", 1500.0)
        self.min_speed = self.params.get("min_speed_mps", 150.0)
        self._half_period = abs(self.bank_angle) / max(abs(self.roll_rate), 1e-6)

    def can_enter(self, state: FlightState) -> bool:
        return (
            state.velocity_mps >= 180.0
            and state.altitude_m >= self.min_altitude + 500.0
        )

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        # Use a triangular/sawtooth phi reference: +bank -> 0 -> -bank -> 0 ...
        t_mod = self._elapsed % (2.0 * self._half_period)
        if t_mod < self._half_period:
            phi_ref = self.bank_angle * (1.0 - 2.0 * t_mod / self._half_period)
        else:
            t2 = t_mod - self._half_period
            phi_ref = -self.bank_angle * (1.0 - 2.0 * t2 / self._half_period)
        return ManeuverSetpoint(
            phi_ref=phi_ref,
            theta_ref=0.0,
            nz_ref=self.nz_ref,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        total_time = 2.0 * self._half_period * self.cycles
        return self._elapsed >= total_time
