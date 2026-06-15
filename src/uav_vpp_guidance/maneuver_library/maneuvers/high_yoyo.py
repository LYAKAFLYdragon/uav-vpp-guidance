"""High Yo-Yo maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint, ManeuverState


class HighYoYo(Maneuver):
    """Out-of-plane energy-management turn: climb to slow down and tighten the
    turn, then dive to regain energy and recover to level.

    Parameters
    ----------
    bank_angle_deg : float
        Bank angle during the maneuver (deg).  Default 45 deg.
    pitch_up_deg : float
        Climb angle above horizon (deg).  Default 25 deg.
    min_speed_mps : float
        Speed at which to transition from climb to dive (m/s).  Default 200 m/s.
    min_altitude_m : float
        Hard altitude floor (m).  Default 3000 m.
    throttle : float
        Throttle setting.  Default 0.9.
    """

    name = "high_yoyo"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 45.0))
        self.pitch_up = math.radians(self.params.get("pitch_up_deg", 25.0))
        self.min_speed = self.params.get("min_speed_mps", 200.0)
        self.min_altitude = self.params.get("min_altitude_m", 3000.0)
        self.throttle = self.params.get("throttle", 0.9)
        self._phase = "climb"
        self._entry_altitude: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        return state.velocity_mps >= self.min_speed + 50.0 and state.altitude_m >= self.min_altitude + 500.0

    def enter(self, state: FlightState):
        super().enter(state)
        self._phase = "climb"
        self._entry_altitude = state.altitude_m

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        entry_alt = self._entry_altitude if self._entry_altitude is not None else state.altitude_m

        if self._phase == "climb":
            if state.velocity_mps <= self.min_speed:
                self._phase = "dive"
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=self.pitch_up,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
            )
        elif self._phase == "dive":
            # Recover when we have dived back to roughly the entry altitude.
            if state.altitude_m <= entry_alt + 100.0:
                self._phase = "recover"
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=-math.radians(20.0),
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
            )
        else:
            if state.theta_rad >= -math.radians(5.0):
                self.state = ManeuverState.EXIT
            return ManeuverSetpoint(
                phi_ref=self.bank_angle,
                theta_ref=0.0,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
            )

    def is_complete(self, state: FlightState) -> bool:
        return self.state == ManeuverState.EXIT and state.theta_rad >= -math.radians(5.0)
