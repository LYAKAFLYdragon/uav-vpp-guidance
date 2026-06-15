"""Situation evaluation for bandit (red) aircraft tactical decision-making."""
from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np

from .observation import compute_relative_geometry


GRAVITY = 9.80665


def _specific_energy(state: Dict[str, Any]) -> float:
    """Approximate specific energy = gh + V^2/2 [J/kg]."""
    alt = float(state.get("altitude_m", 0.0))
    vel = state.get("velocity_vector_mps", None)
    if vel is None:
        vel = state.get("velocity_ned", np.zeros(3))
        vel = np.array([vel[0], vel[1], -vel[2]])
    speed = float(np.linalg.norm(np.asarray(vel)))
    return GRAVITY * alt + 0.5 * speed * speed


class SituationEvaluator:
    """Evaluate relative geometry and tactical situation between own and bandit.

    The evaluator is own-centric: it computes how the bandit sees the ownship
    and labels the engagement zone / advantage.  Outputs are used by the rule
    based `BanditManeuverSelector`.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.max_weapon_range_m = float(self.config.get("max_weapon_range_m", 8000.0))
        self.min_weapon_range_m = float(self.config.get("min_weapon_range_m", 500.0))
        self.tail_angle_thresh_deg = float(self.config.get("tail_angle_thresh_deg", 30.0))
        self.energy_advantage_thresh_j_kg = float(
            self.config.get("energy_advantage_thresh_j_kg", 5000.0)
        )

    def evaluate(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a dict describing the tactical situation.

        Keys:
            - rel: raw relative geometry from compute_relative_geometry
            - range_m, range_rate_mps, ata_deg, aa_deg, altitude_diff_m, speed_diff_mps
            - own_on_bandit_tail (bool): ownship is pointed at bandit tail
            - bandit_on_own_tail (bool): bandit is pointed at ownship tail
            - weapon_zone (str): outside / max_range / ideal / merge
            - bandit_energy_advantage (bool): bandit has more specific energy
            - energy_diff_j_kg: bandit - own specific energy
            - closure_rate_mps: positive = closing
            - advantage (str): "bandit", "own", or "neutral"
        """
        rel = compute_relative_geometry(own_state, bandit_state)

        range_m = float(rel["range_m"])
        range_rate_mps = float(rel["range_rate_mps"])
        ata_deg = float(np.rad2deg(rel["ata_rad"]))
        aa_deg = float(np.rad2deg(rel["aa_rad"]))

        own_on_bandit_tail = abs(aa_deg) <= self.tail_angle_thresh_deg
        bandit_on_own_tail = abs(ata_deg) <= self.tail_angle_thresh_deg

        # Weapon zone based on range and tail aspect.
        if range_m > self.max_weapon_range_m:
            weapon_zone = "outside"
        elif range_m > self.max_weapon_range_m * 0.6:
            weapon_zone = "max_range"
        elif range_m > self.min_weapon_range_m * 2.0:
            weapon_zone = "ideal" if bandit_on_own_tail else "max_range"
        else:
            weapon_zone = "merge"

        own_e = _specific_energy(own_state)
        bandit_e = _specific_energy(bandit_state)
        energy_diff_j_kg = bandit_e - own_e
        bandit_energy_advantage = energy_diff_j_kg > self.energy_advantage_thresh_j_kg

        if bandit_on_own_tail and bandit_energy_advantage:
            advantage = "bandit"
        elif own_on_bandit_tail and not bandit_energy_advantage:
            advantage = "own"
        else:
            advantage = "neutral"

        return {
            "rel": rel,
            "range_m": range_m,
            "range_rate_mps": range_rate_mps,
            "closure_rate_mps": -range_rate_mps,
            "ata_deg": ata_deg,
            "aa_deg": aa_deg,
            "altitude_diff_m": float(rel["altitude_diff_m"]),
            "speed_diff_mps": float(rel["speed_diff_mps"]),
            "own_on_bandit_tail": own_on_bandit_tail,
            "bandit_on_own_tail": bandit_on_own_tail,
            "weapon_zone": weapon_zone,
            "bandit_energy_advantage": bandit_energy_advantage,
            "energy_diff_j_kg": energy_diff_j_kg,
            "advantage": advantage,
        }
