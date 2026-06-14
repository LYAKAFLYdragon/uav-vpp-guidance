"""Coordinated level turn maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class CoordinatedTurn(Maneuver):
    """Execute a coordinated, constant-altitude turn through a heading change.

    Parameters
    ----------
    heading_change_deg : float
        Signed heading change in degrees.  Positive = right turn.
        Default 90 deg.
    bank_angle_deg : float
        Desired bank angle magnitude (deg).  Default 30 deg.
    velocity_ref_mps : float
        Airspeed to hold during the turn (m/s).  Default 250 m/s.
    """

    name = "coordinated_turn"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.heading_change = math.radians(self.params.get("heading_change_deg", 90.0))
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 30.0))
        self.velocity_ref = self.params.get("velocity_ref_mps", 250.0)
        self._target_psi: float | None = None
        self._sign = 1.0 if self.heading_change >= 0 else -1.0

    def can_enter(self, state: FlightState) -> bool:
        # Need enough speed to sustain the commanded bank angle.
        # Required load factor: n = 1 / cos(phi).  At 30 deg, n = 1.15.
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        min_speed = 80.0 * math.sqrt(n_req)
        return state.velocity_mps >= min_speed and state.altitude_m > 500.0

    def enter(self, state: FlightState):
        super().enter(state)
        self._target_psi = state.psi_rad + self.heading_change

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        return ManeuverSetpoint(
            phi_ref=self._sign * self.bank_angle,
            theta_ref=0.0,
            nz_ref=n_req,
            velocity_ref=self.velocity_ref,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._target_psi is None:
            return False
        err = (self._target_psi - state.psi_rad + math.pi) % (2.0 * math.pi) - math.pi
        return abs(err) < math.radians(5.0)
