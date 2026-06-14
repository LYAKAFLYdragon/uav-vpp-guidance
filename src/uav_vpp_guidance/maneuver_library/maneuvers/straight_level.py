"""Straight-and-level flight maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint, ManeuverState


class StraightLevel(Maneuver):
    """Maintain wings-level flight at a reference speed and altitude band.

    Parameters
    ----------
    velocity_ref_mps : float
        Desired true airspeed (m/s).  Default 250 m/s.
    duration_s : float
        How long to hold the maneuver (s).  Default 10 s.
    """

    name = "straight_level"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.velocity_ref = self.params.get("velocity_ref_mps", 250.0)
        self.duration = self.params.get("duration_s", 10.0)

    def can_enter(self, state: FlightState) -> bool:
        # Wings-level flight is always possible above stall margin and above
        # minimum safe altitude.
        return state.altitude_m > 500.0 and state.velocity_mps > 80.0

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        return ManeuverSetpoint(
            phi_ref=0.0,
            theta_ref=0.0,
            nz_ref=1.0,
            velocity_ref=self.velocity_ref,
        )

    def is_complete(self, state: FlightState) -> bool:
        return self._elapsed >= self.duration
