"""Immelmann maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint, ManeuverState


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
        Throttle setting.  Default 1.0.
    min_altitude_m : float
        Hard altitude floor (m).  Default 2000 m.
    """

    name = "immelmann"

    GRAVITY = 9.80665

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.entry_speed = self.params.get("entry_speed_mps", 350.0)
        self.nz_pull = self.params.get("nz_pull", 5.0)
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 90.0))
        self.throttle = self.params.get("throttle", 1.0)
        self.min_altitude = self.params.get("min_altitude_m", 2000.0)
        self._phase = "pull"
        self._phase_end_t = 0.0
        self._theta_start: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        r = self.entry_speed**2 / ((self.nz_pull - 1.0) * self.GRAVITY)
        alt_gain = 2.0 * r
        return (
            state.velocity_mps >= self.entry_speed * 0.9
            and state.altitude_m >= 1500.0
            and state.altitude_m + alt_gain <= 30000.0
            and self.nz_pull <= 7.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._phase = "pull"
        self._theta_start = state.theta_rad

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        return (target - current + math.pi) % (2.0 * math.pi) - math.pi

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)

        # Abort to a level recovery if energy is too low to continue safely.
        if state.velocity_mps < 120.0 and self._phase != "recover":
            self._phase = "recover"

        if self._phase == "pull":
            theta_progress = 0.0
            if self._theta_start is not None:
                theta_progress = abs(self._angle_diff(state.theta_rad, self._theta_start))
            if theta_progress >= math.radians(160.0):
                self._phase = "roll"
                roll_time = math.pi / abs(self.roll_rate)
                self._phase_end_t = self._elapsed + roll_time
            return ManeuverSetpoint(
                phi_ref=0.0,
                nz_ref=self.nz_pull,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
                min_speed_mps=130.0,
                max_alpha_rad=math.radians(25.0),
            )
        elif self._phase == "roll":
            if self._elapsed >= self._phase_end_t:
                self._phase = "recover"
            return ManeuverSetpoint(
                phi_ref=math.pi,
                nz_ref=1.0,
                throttle_ref=self.throttle,
                min_altitude_m=self.min_altitude,
                min_speed_mps=120.0,
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
