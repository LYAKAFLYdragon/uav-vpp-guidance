"""Vertical scissors maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class VerticalScissors(Maneuver):
    """Execute a vertical scissors pattern: alternating steep turns with
    pitch reversals, used in very close-range dogfights to force an opponent
    to overshoot while trading altitude for turn rate.

    The internal phase sequence is ``entry`` -> ``climb``/``dive`` reversals
    -> ``exit``.  The entry/exit phases give the inner-loop controller time
    to establish and recover wings-level attitude, respectively.

    Parameters
    ----------
    bank_angle_deg : float
        Bank angle magnitude for each reversal (deg).  Default 60 deg.
    turn_angle_deg : float
        Heading change per reversal (deg).  Default 90 deg.
    climb_angle_deg : float
        Climb angle during the first half of each reversal (deg).  Default 20 deg.
    dive_angle_deg : float
        Dive angle during the second half (deg).  Default 20 deg.
    cycles : int
        Number of left-right reversals.  Default 3.
    throttle : float
        Throttle setting.  Default 0.7.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1500 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 150 m/s.
    """

    name = "vertical_scissors"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 60.0))
        self.turn_angle = math.radians(self.params.get("turn_angle_deg", 90.0))
        self.climb_angle = math.radians(self.params.get("climb_angle_deg", 20.0))
        self.dive_angle = math.radians(self.params.get("dive_angle_deg", 20.0))
        self.cycles = self.params.get("cycles", 3)
        self.throttle = self.params.get("throttle", 0.7)
        self.min_altitude = self.params.get("min_altitude_m", 1500.0)
        self.min_speed = self.params.get("min_speed_mps", 150.0)
        self._sign = 1.0
        self._target_psi: float | None = None
        self._completed_cycles = 0
        self._phase = "entry"
        self._entry_duration_s = 0.2
        self._exit_duration_s = 0.5
        self._exit_start_t: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        min_speed = 80.0 * math.sqrt(n_req)
        return (
            state.velocity_mps >= min_speed
            and state.altitude_m >= self.min_altitude + 500.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._sign = 1.0
        self._completed_cycles = 0
        self._phase = "entry"
        self._exit_start_t = None
        self._target_psi = state.psi_rad + self._sign * self.turn_angle

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        n_req = 1.0 / math.cos(abs(self.bank_angle)) + 0.2

        if self._phase == "entry":
            if self._elapsed >= self._entry_duration_s:
                self._phase = "climb"
            theta_ref = self.climb_angle
        elif self._phase == "exit":
            theta_ref = 0.0
        else:
            theta_ref = self.climb_angle if self._phase == "climb" else -self.dive_angle

        return ManeuverSetpoint(
            phi_ref=self._sign * self.bank_angle,
            theta_ref=theta_ref,
            nz_ref=n_req,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._phase == "exit":
            if self._exit_start_t is None:
                self._exit_start_t = self._elapsed
            return self._elapsed - self._exit_start_t >= self._exit_duration_s
        if self._target_psi is None:
            return False
        err = (self._target_psi - state.psi_rad + math.pi) % (2.0 * math.pi) - math.pi
        if abs(err) < math.radians(15.0):
            self._completed_cycles += 1
            if self._completed_cycles >= 2 * self.cycles:
                self._phase = "exit"
                return False
            self._sign *= -1.0
            self._phase = "climb" if self._phase == "dive" else "dive"
            self._target_psi = state.psi_rad + self._sign * self.turn_angle
        return False
