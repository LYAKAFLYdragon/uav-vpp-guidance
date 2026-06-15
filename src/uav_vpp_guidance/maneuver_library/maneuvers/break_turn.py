"""High-G defensive break-turn maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class BreakTurn(Maneuver):
    """Execute a sustained high-G turn to break away from a threatening opponent.

    This is the classic defensive maneuver used when the bandit is being
    pursued: roll to a steep bank angle and pull maximum sustained load factor
    to rapidly change heading and bleed energy, forcing the pursuer to
    overshoot or lose the shot.

    Parameters
    ----------
    bank_angle_deg : float
        Steep bank angle magnitude (deg).  Default 70 deg.
    heading_change_deg : float
        Total heading change to achieve before completion (deg).
        Default 135 deg.  Positive = right turn.
    velocity_ref_mps : float
        Target airspeed during the turn (m/s).  Default 240 m/s.
    throttle : float
        Throttle setting.  Default 1.0.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1500 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 150 m/s.
    """

    name = "break_turn"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 70.0))
        self.heading_change = math.radians(self.params.get("heading_change_deg", 135.0))
        self.velocity_ref = self.params.get("velocity_ref_mps", 240.0)
        self.throttle = self.params.get("throttle", 1.0)
        self.min_altitude = self.params.get("min_altitude_m", 1500.0)
        self.min_speed = self.params.get("min_speed_mps", 150.0)
        self._sign = 1.0 if self.heading_change >= 0 else -1.0
        self._target_psi: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        # Required load factor for the steep bank.
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        # Conservative minimum speed: level-turn stall margin scaled by sqrt(n).
        min_speed = 80.0 * math.sqrt(n_req)
        return (
            state.velocity_mps >= min_speed
            and state.altitude_m >= self.min_altitude + 500.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._target_psi = state.psi_rad + self.heading_change

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        # Add a small margin to ensure the aircraft actually pulls, not just
        # holds the bank angle.
        nz_ref = n_req + 0.3
        return ManeuverSetpoint(
            phi_ref=self._sign * self.bank_angle,
            theta_ref=0.0,
            nz_ref=nz_ref,
            velocity_ref=self.velocity_ref,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._target_psi is None:
            return False
        err = (self._target_psi - state.psi_rad + math.pi) % (2.0 * math.pi) - math.pi
        return abs(err) < math.radians(10.0)
