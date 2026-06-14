"""Inner-loop attitude controller for maneuver setpoints."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ControlCommand, FlightState, ManeuverSetpoint


@dataclass
class ControllerGains:
    """PID gains for the inner-loop controller."""

    # Roll channel
    Kp_phi: float = 2.0
    Kd_p: float = 0.5
    # Pitch rate channel
    Kp_q: float = 1.5
    Ki_q: float = 0.05
    Kd_q: float = 0.1
    # Yaw coordination
    Kp_beta: float = 1.0
    # Speed/throttle channel
    Kp_v: float = 0.005
    Ki_v: float = 0.0001


class InnerLoopController:
    """Convert maneuver setpoints to normalized JSBSim surface commands.

    The controller assumes a conventional aircraft response:
      - aileron controls roll angle / roll rate
      - elevator controls pitch rate / load factor
      - rudder coordinates turns (keeps sideslip near zero)
      - throttle controls airspeed
    """

    def __init__(self, gains: Optional[ControllerGains] = None):
        self.gains = gains or ControllerGains()
        self._q_int = 0.0
        self._v_int = 0.0

    def reset(self):
        self._q_int = 0.0
        self._v_int = 0.0

    def update(self, state: FlightState, setpoint: ManeuverSetpoint, dt: float) -> ControlCommand:
        # --- Aileron: track roll angle or roll rate ---
        if setpoint.phi_ref is not None:
            phi_err = self._shortest_angle(setpoint.phi_ref, state.phi_rad)
            aileron = self.gains.Kp_phi * phi_err + self.gains.Kd_p * (setpoint.p_ref - state.p_rps if setpoint.p_ref is not None else 0.0)
        elif setpoint.p_ref is not None:
            aileron = self.gains.Kp_phi * (setpoint.p_ref - state.p_rps)
        else:
            aileron = -self.gains.Kd_p * state.p_rps

        # --- Elevator: track pitch rate or load factor ---
        if setpoint.q_ref is not None:
            q_err = setpoint.q_ref - state.q_rps
            self._q_int += q_err * dt
            self._q_int = float(np.clip(self._q_int, -1.0, 1.0))
            elevator = self.gains.Kp_q * q_err + self.gains.Ki_q * self._q_int - self.gains.Kd_q * state.q_rps
        elif setpoint.nz_ref is not None:
            # nz_ref is in g; current nz from state.  Positive elevator -> higher load factor.
            nz_err = setpoint.nz_ref - state.nz
            elevator = 0.15 * nz_err
        else:
            elevator = 0.0

        # --- Rudder: coordinated turn ---
        if setpoint.phi_ref is not None and abs(setpoint.phi_ref) > 0.05:
            # Rudder opposite to adverse yaw / sideslip; keep beta near zero.
            rudder = -self.gains.Kp_beta * state.beta_rad
        else:
            rudder = -self.gains.Kp_beta * state.beta_rad

        # --- Throttle: track velocity or direct throttle setpoint ---
        if setpoint.throttle_ref is not None:
            throttle = setpoint.throttle_ref
        elif setpoint.velocity_ref is not None:
            v_err = setpoint.velocity_ref - state.velocity_mps
            self._v_int += v_err * dt
            self._v_int = float(np.clip(self._v_int, -100.0, 100.0))
            throttle = 0.5 + self.gains.Kp_v * v_err + self.gains.Ki_v * self._v_int
        else:
            throttle = 0.5

        return ControlCommand(
            elevator=float(np.clip(elevator, -1.0, 1.0)),
            aileron=float(np.clip(aileron, -1.0, 1.0)),
            rudder=float(np.clip(rudder, -1.0, 1.0)),
            throttle=float(np.clip(throttle, 0.0, 1.0)),
        )

    @staticmethod
    def _shortest_angle(target: float, current: float) -> float:
        err = (target - current + math.pi) % (2.0 * math.pi) - math.pi
        return err
