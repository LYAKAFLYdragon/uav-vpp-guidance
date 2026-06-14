"""Immelmann maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Immelmann(Maneuver):
    """Execute an Immelmann: pull through a half loop, then roll upright at
    the top to reverse heading while gaining altitude.

    Parameters
    ----------
    entry_speed_mps : float
        Desired entry speed (m/s).  Default 350 m/s.
    nz_pull : float
        Load factor during the pull (g).  Default 5.0 g.
    roll_rate_dps : float
        Roll rate at the top of the loop (deg/s).  Default 90 deg/s.
    throttle : float
        Throttle setting.  Default 0.95.
    """

    name = "immelmann"

    GRAVITY = 9.80665

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.entry_speed = self.params.get("entry_speed_mps", 350.0)
        self.nz_pull = self.params.get("nz_pull", 5.0)
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 90.0))
        self.throttle = self.params.get("throttle", 0.95)
        self._phase = "pull"
        self._phase_end_t = 0.0

    def can_enter(self, state: FlightState) -> bool:
        r = self.entry_speed**2 / ((self.nz_pull - 1.0) * self.GRAVITY)
        alt_gain = 2.0 * r
        return (
            state.velocity_mps >= self.entry_speed * 0.9
            and state.altitude_m >= 1500.0
            and state.altitude_m + alt_gain <= 14000.0
            and self.nz_pull <= 7.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._phase = "pull"
        r = self.entry_speed**2 / ((self.nz_pull - 1.0) * self.GRAVITY)
        half_loop_time = math.pi * r / self.entry_speed
        self._phase_end_t = self._elapsed + half_loop_time

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)

        if self._phase == "pull":
            if self._elapsed >= self._phase_end_t:
                self._phase = "roll"
                roll_time = math.pi / abs(self.roll_rate)
                self._phase_end_t = self._elapsed + roll_time
            q_ref = ((self.nz_pull - 1.0) * self.GRAVITY) / max(state.velocity_mps, 50.0)
            return ManeuverSetpoint(
                phi_ref=0.0,
                q_ref=q_ref,
                nz_ref=self.nz_pull,
                throttle_ref=self.throttle,
            )
        elif self._phase == "roll":
            if self._elapsed >= self._phase_end_t:
                self._phase = "recover"
            return ManeuverSetpoint(
                phi_ref=math.pi,
                nz_ref=1.0,
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
