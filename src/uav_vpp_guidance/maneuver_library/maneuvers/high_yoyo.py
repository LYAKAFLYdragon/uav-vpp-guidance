"""High Yo-Yo maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint, ManeuverState


class HighYoYo(Maneuver):
    """Out-of-plane energy-management turn: climb to slow down and tighten the
    turn, then dive to regain energy.

    Parameters
    ----------
    bank_angle_deg : float
        Bank angle during the maneuver (deg).  Default 45 deg.
    pitch_up_deg : float
        Climb angle above horizon (deg).  Default 25 deg.
    dive_angle_deg : float
        Dive angle below horizon for recovery (deg).  Default 20 deg.
    min_speed_mps : float
        Speed at which to transition from climb to dive (m/s).  Default 200 m/s.
    max_speed_mps : float
        Speed at which to recover to level flight (m/s).  Default 280 m/s.
    throttle : float
        Throttle setting.  Default 0.9.
    """

    name = "high_yoyo"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 45.0))
        self.pitch_up = math.radians(self.params.get("pitch_up_deg", 25.0))
        self.dive_angle = math.radians(self.params.get("dive_angle_deg", 20.0))
        self.min_speed = self.params.get("min_speed_mps", 200.0)
        self.max_speed = self.params.get("max_speed_mps", 280.0)
        self.throttle = self.params.get("throttle", 0.9)
        self._phase = "climb"

    def can_enter(self, state: FlightState) -> bool:
        return state.velocity_mps >= self.min_speed + 50.0 and state.altitude_m >= 2000.0

    def enter(self, state: FlightState):
        super().enter(state)
        self._phase = "climb"

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        if self._phase == "climb":
            if state.velocity_mps <= self.min_speed:
                self._phase = "dive"
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=self.pitch_up,
                throttle_ref=self.throttle,
            )
        else:
            if state.velocity_mps >= self.max_speed:
                self.state = ManeuverState.EXIT
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=-self.dive_angle,
                throttle_ref=self.throttle,
            )

    def is_complete(self, state: FlightState) -> bool:
        return self.state == ManeuverState.EXIT and state.theta_rad >= -math.radians(5.0)
