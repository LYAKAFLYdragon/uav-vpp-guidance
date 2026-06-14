"""Barrel roll maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class BarrelRoll(Maneuver):
    """Execute a 360-degree roll around the velocity vector.

    Parameters
    ----------
    roll_rate_dps : float
        Desired roll rate (deg/s).  Default 60 deg/s.
    duration_pad_s : float
        Extra time to allow for settling (s).  Default 1.0 s.
    throttle : float
        Throttle setting during the maneuver.  Default 0.85.
    """

    name = "barrel_roll"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 60.0))
        self.duration_pad = self.params.get("duration_pad_s", 1.0)
        self.throttle = self.params.get("throttle", 0.85)
        self._start_phi: float | None = None
        self._duration: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        # Need sufficient speed and altitude; barrel roll bleeds energy.
        return state.velocity_mps >= 200.0 and state.altitude_m >= 1500.0

    def enter(self, state: FlightState):
        super().enter(state)
        self._start_phi = state.phi_rad
        self._duration = 2.0 * math.pi / abs(self.roll_rate) + self.duration_pad

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        if self._start_phi is None:
            self._start_phi = state.phi_rad
        # Advance roll angle linearly.
        phi_ref = self._start_phi + self.roll_rate * self._elapsed
        # Maintain pitch attitude and light positive load factor to keep
        # the nose from dropping through the roll.
        return ManeuverSetpoint(
            phi_ref=phi_ref,
            theta_ref=0.0,
            nz_ref=1.2,
            throttle_ref=self.throttle,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._duration is None or self._start_phi is None:
            return False
        return self._elapsed >= self._duration
