"""Split-S maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class SplitS(Maneuver):
    """Execute a Split-S: roll inverted, then pull through a half loop to
    reverse heading and lose altitude.

    Parameters
    ----------
    entry_speed_mps : float
        Desired entry speed (m/s).  Default 350 m/s.
    nz_pull : float
        Load factor during the pull (g).  Default 5.0 g.
    roll_rate_dps : float
        Roll rate to invert the aircraft (deg/s).  Default 90 deg/s.
    throttle : float
        Throttle setting.  Default 0.9.
    min_altitude_m : float
        Hard altitude floor (m).  Default 500 m.
    """

    name = "split_s"

    GRAVITY = 9.80665

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.entry_speed = self.params.get("entry_speed_mps", 350.0)
        self.nz_pull = self.params.get("nz_pull", 5.0)
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 90.0))
        self.throttle = self.params.get("throttle", 0.9)
        self.min_altitude = self.params.get("min_altitude_m", 500.0)
        self._phase = "roll"
        self._phase_end_t = 0.0
        self._pitch_integrated = 0.0

    def can_enter(self, state: FlightState) -> bool:
        r = self.entry_speed**2 / ((self.nz_pull - 1.0) * self.GRAVITY)
        alt_loss = 2.0 * r
        return (
            state.velocity_mps >= self.entry_speed * 0.9
            and state.altitude_m >= alt_loss + 1000.0
            and self.nz_pull <= 7.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._phase = "roll"
        roll_time = math.pi / abs(self.roll_rate)
        self._phase_end_t = self._elapsed + roll_time
        self._pitch_integrated = 0.0

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)

        if self._phase == "roll":
            if self._elapsed >= self._phase_end_t:
                self._phase = "pull"
                self._pitch_integrated = 0.0
            return ManeuverSetpoint(
                phi_ref=math.pi,
                nz_ref=1.0,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
                min_speed_mps=120.0,
            )
        elif self._phase == "pull":
            self._pitch_integrated += state.q_rps * dt
            if self._pitch_integrated >= math.radians(120.0):
                self._phase = "recover"
            q_ref = ((self.nz_pull - 1.0) * self.GRAVITY) / max(state.velocity_mps, 50.0)
            q_ref = math.copysign(min(abs(q_ref), 0.8), q_ref)
            return ManeuverSetpoint(
                phi_ref=math.pi,
                q_ref=q_ref,
                nz_ref=self.nz_pull,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
                min_speed_mps=120.0,
                max_alpha_rad=math.radians(30.0),
            )
        else:
            return ManeuverSetpoint(
                phi_ref=0.0,
                theta_ref=0.0,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
                min_speed_mps=120.0,
            )

    def is_complete(self, state: FlightState) -> bool:
        return self._phase == "recover" and abs(state.phi_rad) < math.radians(15.0) and abs(state.theta_rad) < math.radians(10.0)
