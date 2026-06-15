"""Wrapper that swaps own/bandit roles so a bandit controller can drive ownship."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..maneuver_library.base import ControlCommand, FlightState


class SideSwapController:
    """Swap own/bandit arguments for a controller designed to drive the bandit.

    When used as the ownship controller, the underlying bandit controller sees
    the opponent as "own" and the own aircraft as "bandit", so it produces
    maneuvers for ownship to counter the opponent.
    """

    def __init__(self, base_controller: Any):
        self.base = base_controller

    def reset(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        bandit_flight_state: FlightState,
        sim_time: float = 0.0,
    ) -> Dict[str, Any]:
        return self.base.reset(
            bandit_state, own_state, bandit_flight_state, sim_time=sim_time
        )

    def update(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        bandit_flight_state: FlightState,
        sim_time: float,
        dt: float,
    ) -> Tuple[ControlCommand, Dict[str, Any]]:
        return self.base.update(
            bandit_state, own_state, bandit_flight_state, sim_time=sim_time, dt=dt
        )
