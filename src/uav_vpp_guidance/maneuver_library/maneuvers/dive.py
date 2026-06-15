"""Controlled dive maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Dive(Maneuver):
    """Descend at a specified flight-path angle until a target altitude.

    Parameters
    ----------
    gamma_deg : float
        Desired dive angle below horizon (deg).  Default 15 deg.
    target_altitude_m : float
        Altitude to recover at (m).  Default 2000 m.
    velocity_ref_mps : float
        Target speed during the dive (m/s).  Default 300 m/s.
    """

    name = "dive"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.gamma = math.radians(self.params.get("gamma_deg", 15.0))
        self.target_altitude = self.params.get("target_altitude_m", 2000.0)
        self.velocity_ref = self.params.get("velocity_ref_mps", 300.0)

    def can_enter(self, state: FlightState) -> bool:
        return state.altitude_m > self.target_altitude + 200.0 and state.velocity_mps > 80.0

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        # Desired pitch angle for a wings-level dive.
        theta_ref = -self.gamma
        # Push-over to dive: nz < 1 g.
        nz_ref = 0.8
        return ManeuverSetpoint(
            phi_ref=0.0,
            theta_ref=theta_ref,
            nz_ref=nz_ref,
            velocity_ref=self.velocity_ref,
            min_altitude_m=self.target_altitude - 100.0,
        )

    def is_complete(self, state: FlightState) -> bool:
        return state.altitude_m <= self.target_altitude
