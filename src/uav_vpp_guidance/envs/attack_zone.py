"""Bidirectional HP and attack-zone mechanics for close air combat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


def get_range_reward(range_km: float, aoa_rad: float) -> float:
    """Legacy artillery attack-zone score from ``singlecombat_task.py``."""
    d_f_min = 7.0
    d_f_max = 12.0
    d_n_min = 3.0
    d_n_max = 5.0
    range_km = float(range_km)
    aoa_rad = float(aoa_rad)

    if d_n_min < range_km < d_n_max:
        s_d = 1.0 - np.exp((d_n_min - range_km) / (d_n_max - d_n_min))
    elif d_n_max < range_km < d_f_min:
        s_d = 1.0
    elif d_f_min < range_km < d_f_max:
        s_d = 1.0 - np.exp((range_km - d_f_max) / (d_f_max - d_f_min))
    else:
        s_d = 0.0

    if 0.0 < aoa_rad < 2.0 * np.pi / 9.0:
        s_a = 1.0 - (81.0 * aoa_rad**2) / (4.0 * np.pi**2)
    else:
        s_a = 0.0
    return float(max(0.0, s_d * s_a))


def compute_attack_zone_score(
    range_km: float,
    aoa_rad: float,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Return the configured attack-zone score.

    The legacy JSBSim artillery reward is a 3-12 km annulus.  The current
    close-range tasks often start inside 3 km, so a configurable near gun-zone
    is layered on top while keeping ``get_range_reward`` unchanged.
    """
    config = config or {}
    scores = []
    if bool(config.get("legacy_range_enabled", True)):
        scores.append(get_range_reward(range_km, aoa_rad))
    if bool(config.get("close_range_enabled", True)):
        scores.append(_close_range_reward(range_km, aoa_rad, config))
    return float(max(scores) if scores else 0.0)


def _close_range_reward(
    range_km: float,
    aoa_rad: float,
    config: Dict[str, Any],
) -> float:
    min_km = float(config.get("close_range_min_km", 0.3))
    full_km = float(config.get("close_range_full_score_km", 1.0))
    max_km = float(config.get("close_range_max_km", 3.0))
    max_aoa = _close_range_max_aoa_rad(config)
    scale = float(config.get("close_range_score_scale", 1.0))
    if range_km < min_km or range_km > max_km or not (0.0 < aoa_rad < max_aoa):
        return 0.0
    if range_km <= full_km:
        s_d = 1.0
    else:
        s_d = (max_km - range_km) / max(max_km - full_km, 1e-8)
    s_a = 1.0 - (aoa_rad / max(max_aoa, 1e-8)) ** 2
    return float(max(0.0, scale * s_d * s_a))


def _close_range_max_aoa_rad(config: Dict[str, Any]) -> float:
    if "close_range_max_aoa_deg" in config:
        return float(np.deg2rad(float(config["close_range_max_aoa_deg"])))
    return float(config.get("close_range_max_aoa_rad", 2.0 * np.pi / 9.0))


def compute_attack_angle_rad(attacker_state: Dict[str, Any], defender_state: Dict[str, Any]) -> float:
    """Return angle between attacker velocity and LOS to defender."""
    attacker_pos = _get_position(attacker_state)
    defender_pos = _get_position(defender_state)
    attacker_vel = _get_velocity(attacker_state)
    rel_pos = defender_pos - attacker_pos
    range_m = float(np.linalg.norm(rel_pos))
    speed = float(np.linalg.norm(attacker_vel))
    if range_m <= 1e-8 or speed <= 1e-8:
        return float(np.pi)
    cos_aoa = float(np.dot(rel_pos, attacker_vel) / (range_m * speed + 1e-8))
    return float(np.arccos(np.clip(cos_aoa, -1.0, 1.0)))


def evaluate_attack_zone(
    attacker_state: Dict[str, Any],
    defender_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one directional attack-zone relation."""
    attacker_pos = _get_position(attacker_state)
    defender_pos = _get_position(defender_state)
    range_m = float(np.linalg.norm(defender_pos - attacker_pos))
    aoa_rad = compute_attack_angle_rad(attacker_state, defender_state)
    score = compute_attack_zone_score(range_m / 1000.0, aoa_rad, config=config)
    return {
        "range_m": range_m,
        "range_km": range_m / 1000.0,
        "aoa_rad": aoa_rad,
        "aoa_deg": float(np.rad2deg(aoa_rad)),
        "score": score,
        "in_attack_zone": bool(score > 0.0),
    }


@dataclass
class CombatHPManager:
    """Small stateful HP manager for bidirectional gun-zone damage."""

    initial_hp: float = 100.0
    damage_per_step: float = 0.5
    enabled: bool = False
    attack_zone_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reset()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombatHPManager":
        cfg = config.get("attack_zone", config.get("combat", {}))
        return cls(
            initial_hp=float(cfg.get("initial_hp", 100.0)),
            damage_per_step=float(cfg.get("damage_per_step", 0.5)),
            enabled=bool(cfg.get("enabled", False)),
            attack_zone_config=dict(cfg),
        )

    def reset(self) -> None:
        self.ego_hp = float(self.initial_hp)
        self.target_hp = float(self.initial_hp)
        self.last_info: Dict[str, Any] = self._base_info()

    def _base_info(self) -> Dict[str, Any]:
        return {
            "attack_zone_enabled": self.enabled,
            "ego_hp": float(getattr(self, "ego_hp", self.initial_hp)),
            "target_hp": float(getattr(self, "target_hp", self.initial_hp)),
            "ego_attack_score": 0.0,
            "target_attack_score": 0.0,
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
            "combat_outcome": None,
            "combat_success": False,
            "combat_time_to_kill": np.nan,
        }

    def step(
        self,
        ego_state: Dict[str, Any],
        target_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one bidirectional damage update."""
        if not self.enabled:
            self.last_info = self._base_info()
            return dict(self.last_info)

        ego_attack = evaluate_attack_zone(
            ego_state,
            target_state,
            config=self.attack_zone_config,
        )
        target_attack = evaluate_attack_zone(
            target_state,
            ego_state,
            config=self.attack_zone_config,
        )
        if ego_attack["in_attack_zone"]:
            self.target_hp -= self.damage_per_step
        if target_attack["in_attack_zone"]:
            self.ego_hp -= self.damage_per_step
        self.ego_hp = max(0.0, float(self.ego_hp))
        self.target_hp = max(0.0, float(self.target_hp))

        self.last_info = {
            "attack_zone_enabled": True,
            "ego_hp": self.ego_hp,
            "target_hp": self.target_hp,
            "ego_attack": ego_attack,
            "target_attack": target_attack,
            "ego_attack_score": float(ego_attack["score"]),
            "target_attack_score": float(target_attack["score"]),
            "ego_in_attack_zone": bool(ego_attack["in_attack_zone"]),
            "target_in_attack_zone": bool(target_attack["in_attack_zone"]),
            "combat_outcome": None,
            "combat_success": False,
            "combat_time_to_kill": np.nan,
        }
        return dict(self.last_info)

    def resolve_terminal(
        self,
        term_info: Dict[str, Any],
        current_time_s: float,
        ego_failed: bool = False,
        target_failed: bool = False,
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Overlay combat win/loss/draw semantics on an existing termination."""
        info = dict(self.last_info)
        if not self.enabled:
            return False, False, info

        outcome = None
        reason = None
        terminated = False
        truncated = False

        if target_failed and not ego_failed:
            outcome = "win"
            reason = "target_crash_or_out_of_bounds"
            terminated = True
        elif ego_failed and not target_failed:
            outcome = "loss"
            reason = "ego_crash_or_out_of_bounds"
            terminated = True
        elif ego_failed and target_failed:
            outcome = "draw"
            reason = "mutual_aircraft_failure"
            terminated = True
        elif self.target_hp <= 0.0 and self.ego_hp <= 0.0:
            outcome = "draw"
            reason = "mutual_kill"
            terminated = True
        elif self.target_hp <= 0.0:
            outcome = "win"
            reason = "target_killed"
            terminated = True
        elif self.ego_hp <= 0.0:
            outcome = "loss"
            reason = "ego_killed"
            terminated = True
        elif term_info.get("is_timeout"):
            truncated = True
            if self.ego_hp > self.target_hp:
                outcome = "win"
                reason = "timeout_hp_advantage"
            elif self.ego_hp < self.target_hp:
                outcome = "loss"
                reason = "timeout_hp_disadvantage"
            else:
                outcome = "draw"
                reason = "timeout_draw"

        if outcome is None:
            return False, False, info

        info.update(
            {
                "combat_outcome": outcome,
                "combat_success": outcome == "win",
                "win": outcome == "win",
                "loss": outcome == "loss",
                "draw": outcome == "draw",
                "combat_reason": reason,
                "ego_hp": self.ego_hp,
                "target_hp": self.target_hp,
                "hp_advantage": self.ego_hp - self.target_hp,
                "combat_time_to_kill": (
                    float(current_time_s)
                    if reason == "target_killed"
                    else np.nan
                ),
            }
        )
        return terminated, truncated, info


def _get_position(state: Dict[str, Any]) -> np.ndarray:
    for key in ("position_m", "position_neu", "position"):
        value = state.get(key)
        if value is not None:
            return np.asarray(value, dtype=np.float64)
    raise ValueError("state missing position field")


def _get_velocity(state: Dict[str, Any]) -> np.ndarray:
    for key in ("velocity_vector_mps", "velocity", "velocity_ned"):
        value = state.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=np.float64)
            if key == "velocity_ned":
                return np.array([arr[0], arr[1], -arr[2]], dtype=np.float64)
            return arr
    return np.zeros(3, dtype=np.float64)
