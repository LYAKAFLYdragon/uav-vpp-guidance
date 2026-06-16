"""High-G defensive break-turn maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class BreakTurn(Maneuver):
    """Execute a sustained high-G turn to break away from a threatening opponent.

    This is the classic defensive maneuver used when the bandit is being
    pursued: roll to a steep bank angle and pull maximum sustained load factor
    to rapidly change heading and bleed energy, forcing the pursuer to
    overshoot or lose the shot.

    The maneuver now exposes a small phase FSM for telemetry and safer
    transitions: ``entry`` (roll in), ``execute`` (sustained turn), and
    ``exit`` (roll out).

    Parameters
    ----------
    bank_angle_deg : float
        Steep bank angle magnitude (deg).  Default 70 deg.
    heading_change_deg : float
        Total heading change to achieve before completion (deg).
        Default 135 deg.  Positive = right turn.
    velocity_ref_mps : float
        Target airspeed during the turn (m/s).  Default 240 m/s.
    throttle : float
        Throttle setting.  Default 1.0.
    roll_rate_dps : float
        Roll-in/roll-out rate (deg/s).  Default 90 deg/s.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1500 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 150 m/s.
    """

    name = "break_turn"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bank_angle = math.radians(self.params.get("bank_angle_deg", 70.0))
        self.heading_change = math.radians(self.params.get("heading_change_deg", 135.0))
        self.velocity_ref = self.params.get("velocity_ref_mps", 240.0)
        self.throttle = self.params.get("throttle", 1.0)
        self.roll_rate = math.radians(self.params.get("roll_rate_dps", 90.0))
        self.min_altitude = self.params.get("min_altitude_m", 1500.0)
        self.min_speed = self.params.get("min_speed_mps", 150.0)
        self._sign = 1.0 if self.heading_change >= 0 else -1.0
        self._target_psi: float | None = None
        self._phase = "entry"
        self._entry_duration_s = abs(self.bank_angle) / max(abs(self.roll_rate), 1e-6)
        self._exit_duration_s = self._entry_duration_s
        self._exit_start_t: float | None = None

    def can_enter(self, state: FlightState) -> bool:
        # Required load factor for the steep bank.
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        # Conservative minimum speed: level-turn stall margin scaled by sqrt(n).
        min_speed = 80.0 * math.sqrt(n_req)
        return (
            state.velocity_mps >= min_speed
            and state.altitude_m >= self.min_altitude + 500.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._target_psi = state.psi_rad + self.heading_change
        self._phase = "entry"
        self._exit_start_t = None

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        n_req = 1.0 / math.cos(abs(self.bank_angle))
        # Add a small margin to ensure the aircraft actually pulls, not just
        # holds the bank angle.
        nz_ref = n_req + 0.3

        if self._phase == "entry":
            if self._elapsed >= self._entry_duration_s:
                self._phase = "execute"
            progress = min(1.0, self._elapsed / max(self._entry_duration_s, 1e-6))
            phi_ref = self._sign * progress * self.bank_angle
        elif self._phase == "execute":
            if self._target_psi is not None:
                err = (self._target_psi - state.psi_rad + math.pi) % (2.0 * math.pi) - math.pi
                if abs(err) < math.radians(15.0):
                    self._phase = "exit"
                    self._exit_start_t = self._elapsed
            phi_ref = self._sign * self.bank_angle
        else:  # exit
            if self._exit_start_t is None:
                self._exit_start_t = self._elapsed
            progress = min(1.0, (self._elapsed - self._exit_start_t) / max(self._exit_duration_s, 1e-6))
            phi_ref = self._sign * (1.0 - progress) * self.bank_angle

        return ManeuverSetpoint(
            phi_ref=phi_ref,
            theta_ref=0.0,
            nz_ref=nz_ref,
            velocity_ref=self.velocity_ref,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._phase != "exit" or self._exit_start_t is None:
            return False
        if self._elapsed - self._exit_start_t >= self._exit_duration_s:
            return True
        return abs(state.phi_rad) < math.radians(5.0)
