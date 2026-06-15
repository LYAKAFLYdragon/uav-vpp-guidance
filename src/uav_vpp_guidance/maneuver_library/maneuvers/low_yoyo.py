"""Low Yo-Yo maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint, ManeuverState


class LowYoYo(Maneuver):
    """In-plane energy-management turn: dive to accelerate and reduce turn
    radius, then climb to recover altitude.

    Parameters
    ----------
    bank_angle_deg : float
        Bank angle during the turn (deg).  Default 45 deg.
    dive_angle_deg : float
        Dive angle below horizon (deg).  Default 20 deg.
    climb_angle_deg : float
        Climb angle for recovery (deg).  Default 15 deg.
    max_speed_mps : float
        Speed at which to transition from dive to climb (m/s).  Default 280 m/s.
    min_speed_mps : float
        Speed at which to resume level flight (m/s).  Default 220 m/s.
    min_altitude_m : float
        Hard altitude floor; forces climb recovery if approached (m).
        Default 2000 m.
    throttle : float
        Throttle setting.  Default 0.9.
    """

    name = "low_yoyo"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 45.0))
        self.dive_angle = math.radians(self.params.get("dive_angle_deg", 20.0))
        self.climb_angle = math.radians(self.params.get("climb_angle_deg", 15.0))
        self.max_speed = self.params.get("max_speed_mps", 280.0)
        self.min_speed = self.params.get("min_speed_mps", 220.0)
        self.min_altitude = self.params.get("min_altitude_m", 2000.0)
        self.throttle = self.params.get("throttle", 0.9)
        self._phase = "dive"

    def can_enter(self, state: FlightState) -> bool:
        return state.altitude_m >= self.min_altitude + 500.0 and state.velocity_mps >= 180.0

    def enter(self, state: FlightState):
        super().enter(state)
        self._phase = "dive"

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        floor_buffer = 200.0

        if self._phase == "dive":
            if state.velocity_mps >= self.max_speed:
                self._phase = "climb"
            elif state.altitude_m <= self.min_altitude + floor_buffer:
                self._phase = "climb"
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=-self.dive_angle,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
            )
        else:
            if state.velocity_mps <= self.min_speed:
                self.state = ManeuverState.EXIT
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=self.climb_angle,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
            )

    def is_complete(self, state: FlightState) -> bool:
        return self.state == ManeuverState.EXIT and state.theta_rad <= math.radians(5.0)
