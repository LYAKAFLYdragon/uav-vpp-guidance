"""Vertical loop maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Loop(Maneuver):
    """Fly a vertical circle (loop) in the pitch plane.

    Parameters
    ----------
    entry_speed_mps : float
        True airspeed at loop entry (m/s).  Default 350 m/s.
    nz_target : float
        Target load factor during the pull (g).  Default 5.0 g.
    throttle : float
        Throttle setting during the maneuver.  Default 0.9.
    """

    name = "loop"

    GRAVITY = 9.80665

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.entry_speed = self.params.get("entry_speed_mps", 350.0)
        self.nz_target = self.params.get("nz_target", 5.0)
        self.throttle = self.params.get("throttle", 0.9)
        self._duration: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        # Need enough energy and altitude to complete a loop.
        # Conservative: entry speed >= 300 m/s and altitude >= 2500 m.
        return (
            state.velocity_mps >= self.entry_speed * 0.9
            and state.altitude_m >= 2500.0
            and self.nz_target <= 7.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        # Radius of the loop: r = V^2 / ((nz - 1) * g)
        r = self.entry_speed**2 / ((self.nz_target - 1.0) * self.GRAVITY)
        # Time to fly one circle: t = 2 * pi * r / V
        self._duration = 2.0 * math.pi * r / self.entry_speed

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        # Desired pitch rate for a constant-radius loop:
        # q = V / r = ((nz - 1) * g) / V
        q_ref = ((self.nz_target - 1.0) * self.GRAVITY) / max(state.velocity_mps, 50.0)
        return ManeuverSetpoint(
            phi_ref=0.0,
            q_ref=q_ref,
            nz_ref=self.nz_target,
            throttle_ref=self.throttle,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._duration is None:
            return False
        return self._elapsed >= self._duration
