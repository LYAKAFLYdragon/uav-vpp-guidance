"""Displacement roll (lag roll) maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class DisplacementRoll(Maneuver):
    """Execute a displacement roll to shift the aircraft laterally/vertically
    relative to a pursuer while preserving energy.

    Unlike a full barrel roll, this maneuver rolls to a target bank angle
    (typically 90°–180°), holds it briefly to generate sideslip/lateral
    displacement, and then rolls back to wings-level.  It is used to avoid
    being tracked in a straight line and to force an overshooting opponent to
    fly ahead.

    Parameters
    ----------
    roll_target_deg : float
        Maximum bank angle to roll to (deg).  Default 135 deg.
    roll_rate_dps : float
        Roll rate during the in/out phases (deg/s).  Default 90 deg/s.
    hold_time_s : float
        Time to hold the displaced attitude (s).  Default 1.5 s.
    nz_ref : float
        Load factor during the roll to keep the nose from dropping (g).
        Default 1.3 g.
    throttle : float
        Throttle setting.  Default 0.85.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1500 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 150 m/s.
    """

    name = "displacement_roll"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.roll_target = math.radians(self.params.get("roll_target_deg", 135.0))
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 90.0))
        self.hold_time = self.params.get("hold_time_s", 1.5)
        self.nz_ref = self.params.get("nz_ref", 1.3)
        self.throttle = self.params.get("throttle", 0.85)
        self.min_altitude = self.params.get("min_altitude_m", 1500.0)
        self.min_speed = self.params.get("min_speed_mps", 150.0)
        self._sign = 1.0 if self.roll_target >= 0 else -1.0
        self._roll_in_end_t: float | None = None
        self._hold_end_t: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        return (
            state.velocity_mps >= 180.0
            and state.altitude_m >= self.min_altitude + 500.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        roll_in_time = abs(self.roll_target) / max(abs(self.roll_rate), 1e-6)
        self._roll_in_end_t = self._elapsed + roll_in_time
        self._hold_end_t = self._roll_in_end_t + self.hold_time

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)

        if self._roll_in_end_t is None or self._hold_end_t is None:
            # Defensive: should have been set in enter().
            return ManeuverSetpoint(
                theta_ref=0.0,
                nz_ref=self.nz_ref,
                throttle_ref=self.throttle,
            )

        if self._elapsed < self._roll_in_end_t:
            # Roll in to the target bank angle.
            phi_ref = self._sign * self.roll_rate * self._elapsed
            phi_ref = self._sign * min(abs(phi_ref), abs(self.roll_target))
        elif self._elapsed < self._hold_end_t:
            # Hold the displaced attitude.
            phi_ref = self._sign * self.roll_target
        else:
            # Roll back to wings level.
            roll_out_elapsed = self._elapsed - self._hold_end_t
            phi_ref = self._sign * (
                abs(self.roll_target) - self.roll_rate * roll_out_elapsed
            )
            phi_ref = self._sign * max(0.0, abs(phi_ref))

        return ManeuverSetpoint(
            phi_ref=phi_ref,
            theta_ref=0.0,
            nz_ref=self.nz_ref,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._hold_end_t is None:
            return False
        roll_out_time = abs(self.roll_target) / max(abs(self.roll_rate), 1e-6)
        complete_time = self._hold_end_t + roll_out_time
        return self._elapsed >= complete_time and abs(state.phi_rad) < math.radians(10.0)
