"""High-level controller that drives the bandit (red) aircraft via the maneuver library."""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from ..maneuver_library import InnerLoopController, ManeuverExecutor, ManeuverLibrary, ManeuverTelemetry
from ..maneuver_library.base import ControlCommand, FlightState, Maneuver
from .bandit_selector import BanditManeuverSelector


FT2M = 0.3048


class BanditManeuverController:
    """Wraps the maneuver library to produce actuator commands for the bandit.

    Responsibilities:
      - maintain a ManeuverExecutor + InnerLoopController for the bandit;
      - use BanditManeuverSelector to choose maneuvers based on situation;
      - convert the JSBSim state dict into a FlightState each sub-step;
      - return a normalized ControlCommand for JSBSim.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.selector = BanditManeuverSelector(self.config.get("selector", {}))
        self.fallback_name = str(self.config.get("fallback_maneuver", "straight_level"))
        self.fallback = ManeuverLibrary.create(self.fallback_name, {})
        self._executor: ManeuverExecutor | None = None
        self._telemetry = ManeuverTelemetry()
        self._current_maneuver_name: str | None = None
        self._maneuver_start_t: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        bandit_flight_state: FlightState,
        sim_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Reset internal state and select the first maneuver.

        Returns:
            dict: Info dict with keys "bandit_maneuver", "situation".
        """
        self._telemetry.reset()
        self._executor = ManeuverExecutor(
            InnerLoopController(), telemetry=self._telemetry
        )
        self._executor.set_fallback(self.fallback)

        maneuver = self.selector.select_initial(own_state, bandit_state, sim_time)
        if maneuver is None:
            maneuver = self.fallback
        self._enter_maneuver(maneuver, bandit_flight_state, sim_time, reason="initial")

        situation = self.selector.evaluator.evaluate(own_state, bandit_state)
        return {
            "bandit_maneuver": self._current_maneuver_name,
            "situation": situation,
            "maneuver_telemetry": self._telemetry.summary(),
        }

    def update(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        bandit_flight_state: FlightState,
        sim_time: float,
        dt: float,
    ) -> Tuple[ControlCommand, Dict[str, Any]]:
        """Step the selector and executor and return actuator commands.

        Returns:
            (ControlCommand, info dict)
        """
        if self._executor is None:
            raise RuntimeError("BanditManeuverController.reset() must be called before update().")

        situation = self.selector.evaluator.evaluate(own_state, bandit_state)

        # Emergency interruption: bypass reaction time and current maneuver when
        # the opponent enters the emergency envelope.
        if (
            self.selector.is_emergency(situation)
            and self._current_maneuver_name not in self.selector.emergency_maneuvers
        ):
            emergency_maneuver = self.selector.select_emergency(
                own_state, bandit_state, sim_time
            )
            if emergency_maneuver is not None:
                self._enter_maneuver(
                    emergency_maneuver, bandit_flight_state, sim_time, reason="emergency"
                )
        else:
            # Decide whether to switch maneuvers.
            elapsed = sim_time - self._maneuver_start_t
            new_maneuver = self.selector.select(
                own_state,
                bandit_state,
                self._current_maneuver_name,
                elapsed,
                sim_time,
            )
            if new_maneuver is not None:
                self._enter_maneuver(
                    new_maneuver, bandit_flight_state, sim_time, reason="switch"
                )

        # Step executor: if current maneuver completed, it transitions to fallback.
        command = self._executor.update(bandit_flight_state, dt)

        info = {
            "bandit_maneuver": self._current_maneuver_name,
            "situation": situation,
            "bandit_command": {
                "elevator": command.elevator,
                "aileron": command.aileron,
                "rudder": command.rudder,
                "throttle": command.throttle,
            },
            "maneuver_telemetry": self._telemetry.summary(),
        }
        return command, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enter_maneuver(
        self,
        maneuver: Maneuver,
        bandit_flight_state: FlightState,
        sim_time: float,
        reason: str = "switch",
    ):
        if maneuver.can_enter(bandit_flight_state):
            self._executor.select(maneuver, bandit_flight_state, reason=reason)
            self._current_maneuver_name = maneuver.name
        else:
            # Fallback to safe maneuver if entry conditions not met.
            self._executor.select(self.fallback, bandit_flight_state, reason="fallback")
            self._current_maneuver_name = self.fallback.name
        self._maneuver_start_t = sim_time


def build_bandit_flight_state(
    bandit_state: Dict[str, Any],
    alpha_rad: float,
    beta_rad: float,
    mach: float,
) -> FlightState:
    """Convert the JSBSim state dict used by the env into a maneuver-library FlightState."""
    pos = bandit_state.get("position_m")
    if pos is None:
        pos = bandit_state.get("position_neu")
    if pos is None:
        pos = np.zeros(3)
    vel_vec = bandit_state.get("velocity_vector_mps")
    if vel_vec is None:
        vel_vec = np.zeros(3)
    speed = float(np.linalg.norm(np.asarray(vel_vec)))

    attitude = bandit_state.get("attitude_rpy")
    if attitude is not None:
        phi, theta, psi = float(attitude[0]), float(attitude[1]), float(attitude[2])
    else:
        phi = float(bandit_state.get("roll_rad", 0.0))
        theta = float(bandit_state.get("pitch_rad", 0.0))
        psi = float(bandit_state.get("yaw_rad", 0.0))

    rates = bandit_state.get("body_rates_rps")
    if rates is None:
        rates = np.zeros(3)
    p, q, r = float(rates[0]), float(rates[1]), float(rates[2])

    return FlightState(
        t=float(bandit_state.get("sim_time", 0.0)),
        position_m=np.asarray(pos, dtype=np.float64),
        velocity_mps=speed if speed > 0.0 else float(bandit_state.get("speed_mps", 0.0)),
        altitude_m=float(bandit_state.get("altitude_m", 0.0)),
        phi_rad=phi,
        theta_rad=theta,
        psi_rad=psi,
        p_rps=p,
        q_rps=q,
        r_rps=r,
        nz=float(bandit_state.get("nz_g", 1.0)),
        alpha_rad=float(alpha_rad),
        beta_rad=float(beta_rad),
        mach=float(mach),
    )
