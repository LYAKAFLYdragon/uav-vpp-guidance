"""Scissors maneuver (ownship pattern)."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Scissors(Maneuver):
    """Execute a defensive scissors pattern: alternating hard turns with a
    slight climb, used to force an overshooting opponent to fly ahead.

    This is a self-contained pattern; it does not receive opponent state.
    In a full combat AI it should be triggered when the opponent is closing
    from behind and inside the turn circle.

    Parameters
    ----------
    bank_angle_deg : float
        Bank angle magnitude for each reversal (deg).  Default 60 deg.
    turn_angle_deg : float
        Heading change per reversal (deg).  Default 90 deg.
    cycles : int
        Number of left-right reversals.  Default 3.
    throttle : float
        Reduced throttle to bleed energy and tighten turns.  Default 0.6.
    climb_margin_g : float
        Extra load factor above level-turn requirement to gain a little
        altitude during each reversal.  Default 0.3 g.
    """

    name = "scissors"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 60.0))
        self.turn_angle = math.radians(self.params.get("turn_angle_deg", 90.0))
        self.cycles = self.params.get("cycles", 3)
        self.throttle = self.params.get("throttle", 0.6)
        self.climb_margin = self.params.get("climb_margin_g", 0.3)
        self._sign = 1.0
        self._target_psi: float | None = None
        self._completed_cycles = 0

    def can_enter(self, state: FlightState) -> bool:
        n_req = 1.0 / math.cos(abs(self.bank_angle)) + self.climb_margin
        min_speed = 80.0 * math.sqrt(n_req)
        return state.velocity_mps >= min_speed and state.altitude_m > 1000.0

    def enter(self, state: FlightState):
        super().enter(state)
        self._sign = 1.0
        self._completed_cycles = 0
        self._target_psi = state.psi_rad + self._sign * self.turn_angle

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        n_req = 1.0 / math.cos(abs(self.bank_angle)) + self.climb_margin
        return ManeuverSetpoint(
            phi_ref=self._sign * self.bank_angle,
            nz_ref=n_req,
            throttle_ref=self.throttle,
            min_altitude_m=1000.0,
            min_speed_mps=120.0,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._target_psi is None:
            return False
        err = (self._target_psi - state.psi_rad + math.pi) % (2.0 * math.pi) - math.pi
        if abs(err) < math.radians(10.0):
            self._completed_cycles += 1
            if self._completed_cycles >= 2 * self.cycles:
                return True
            self._sign *= -1.0
            self._target_psi = state.psi_rad + self._sign * self.turn_angle
        return False
