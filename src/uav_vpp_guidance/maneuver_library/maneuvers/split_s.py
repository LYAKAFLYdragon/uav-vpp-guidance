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
    """

    name = "split_s"

    GRAVITY = 9.80665

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.entry_speed = self.params.get("entry_speed_mps", 350.0)
        self.nz_pull = self.params.get("nz_pull", 5.0)
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 90.0))
        self.throttle = self.params.get("throttle", 0.9)
        self._phase = "roll"
        self._phase_end_t = 0.0

    def can_enter(self, state: FlightState) -> bool:
        # Need energy and altitude margin.
        r = self.entry_speed**2 / ((self.nz_pull - 1.0) * self.GRAVITY)
        alt_loss = 2.0 * r  # diameter of the half loop
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

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)

        if self._phase == "roll":
            if self._elapsed >= self._phase_end_t:
                self._phase = "pull"
                r = self.entry_speed**2 / ((self.nz_pull - 1.0) * self.GRAVITY)
                half_loop_time = math.pi * r / self.entry_speed
                self._phase_end_t = self._elapsed + half_loop_time
            return ManeuverSetpoint(
                phi_ref=math.pi,
                nz_ref=1.0,
                throttle_ref=self.throttle,
            )
        elif self._phase == "pull":
            if self._elapsed >= self._phase_end_t:
                self._phase = "recover"
            q_ref = ((self.nz_pull - 1.0) * self.GRAVITY) / max(state.velocity_mps, 50.0)
            return ManeuverSetpoint(
                phi_ref=math.pi,
                q_ref=q_ref,
                nz_ref=self.nz_pull,
                throttle_ref=self.throttle,
            )
        else:
            return ManeuverSetpoint(
                phi_ref=0.0,
                nz_ref=1.0,
                throttle_ref=self.throttle,
            )

    def is_complete(self, state: FlightState) -> bool:
        return self._phase == "recover" and abs(state.phi_rad) < math.radians(15.0)
