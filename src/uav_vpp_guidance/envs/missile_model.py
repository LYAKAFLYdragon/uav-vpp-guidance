"""Simplified missile engagement model for air-to-air combat baseline."""
from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np


GRAVITY = 9.80665


class MissileModel:
    """Point-mass missile guided by proportional navigation (PN).

    The missile is intentionally simple: it has no boost/sustain phase, no seeker
    dynamics, and a fixed speed magnitude.  It is used only for win/loss
    determination, not for high-fidelity missile physics.
    """

    def __init__(
        self,
        launch_position_m: np.ndarray,
        launch_velocity_mps: np.ndarray,
        target_uid: str,
        side: str,
        config: Dict[str, Any],
    ):
        """
        Args:
            launch_position_m: 3-D NEU launch position [m].
            launch_velocity_mps: 3-D launch velocity [m/s].
            target_uid: UID of the target aircraft.
            side: "own" or "bandit" (who fired).
            config: missile parameters.
        """
        self.position_m = np.asarray(launch_position_m, dtype=np.float64)
        self.side = side
        self.target_uid = target_uid
        self.config = config

        self.speed_mps = float(config.get("speed_mps", 600.0))
        self.max_g = float(config.get("max_g", 30.0))
        self.navigation_constant = float(config.get("navigation_constant", 3.0))
        self.fov_rad = float(np.deg2rad(config.get("fov_deg", 30.0)))
        self.kill_radius_m = float(config.get("kill_radius_m", 15.0))
        self.max_range_m = float(config.get("max_range_m", 10000.0))
        self.max_flight_time_s = float(config.get("max_flight_time_s", 30.0))

        vel = np.asarray(launch_velocity_mps, dtype=np.float64)
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 1e-6:
            self.velocity_mps = vel / vel_norm * self.speed_mps
        else:
            self.velocity_mps = np.array([self.speed_mps, 0.0, 0.0])

        self.age_s = 0.0
        self.alive = True
        self.hit = False
        self.missed = False
        self.miss_reason: str | None = None

    def step(self, dt: float, target_position_m: np.ndarray, target_velocity_mps: np.ndarray):
        """Advance the missile one integration step."""
        if not self.alive:
            return

        self.age_s += dt
        target_pos = np.asarray(target_position_m, dtype=np.float64)
        target_vel = np.asarray(target_velocity_mps, dtype=np.float64)

        rel_pos = target_pos - self.position_m
        range_m = float(np.linalg.norm(rel_pos))

        # Maximum range / time expiry.
        if range_m > self.max_range_m or self.age_s > self.max_flight_time_s:
            self._mark_miss("expired")
            return

        # LOS and LOS rate.
        if range_m < 1e-6:
            self._mark_hit()
            return

        los_unit = rel_pos / range_m
        closing_velocity = target_vel - self.velocity_mps
        los_rate = np.cross(closing_velocity, los_unit) / (range_m + 1e-8)

        # PN acceleration perpendicular to LOS.
        closing_speed = -float(np.dot(closing_velocity, los_unit))
        accel_cmd = self.navigation_constant * closing_speed * np.cross(los_unit, los_rate)

        # Gravity bias (keep the missile from nosing over unrealistically).
        accel_cmd[2] += GRAVITY

        # Saturate by max_g.
        accel_norm = np.linalg.norm(accel_cmd)
        max_accel = self.max_g * GRAVITY
        if accel_norm > max_accel:
            accel_cmd = accel_cmd / accel_norm * max_accel

        # Update velocity direction while keeping speed constant.
        dv = accel_cmd * dt
        new_vel = self.velocity_mps + dv
        vel_norm = np.linalg.norm(new_vel)
        if vel_norm > 1e-6:
            self.velocity_mps = new_vel / vel_norm * self.speed_mps

        # FOV / seeker limit.
        vel_norm = np.linalg.norm(self.velocity_mps)
        if vel_norm > 1e-6:
            boresight_error = np.arccos(np.clip(np.dot(self.velocity_mps / vel_norm, los_unit), -1.0, 1.0))
            if boresight_error > self.fov_rad:
                self._mark_miss("seeker_fov")
                return

        # Integrate position.
        self.position_m += self.velocity_mps * dt

        # Hit check.
        if range_m <= self.kill_radius_m:
            self._mark_hit()

    def _mark_hit(self):
        self.hit = True
        self.alive = False

    def _mark_miss(self, reason: str):
        self.missed = True
        self.alive = False
        self.miss_reason = reason


class EngagementTracker:
    """Track all in-flight missiles and handle launch logic."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.missiles: List[MissileModel] = []
        self._last_launch_time: Dict[str, float] = {"bandit": -1e9, "own": -1e9}
        self._rng = np.random.default_rng(self.config.get("seed", None))
        self._hits: List[Dict[str, Any]] = []
        self._misses: List[Dict[str, Any]] = []

    def reset(self):
        self.missiles.clear()
        self._last_launch_time = {"bandit": -1e9, "own": -1e9}
        self._hits.clear()
        self._misses.clear()

    def step(self, dt: float, states: Dict[str, Dict[str, Any]]):
        """Advance missiles and remove dead ones."""
        for missile in self.missiles:
            target_state = states.get(missile.target_uid)
            if target_state is None:
                missile._mark_miss("target_lost")
                continue
            target_pos = target_state.get("position_m")
            if target_pos is None:
                target_pos = target_state.get("position_neu")
            target_vel = target_state.get("velocity_vector_mps")
            if target_vel is None:
                target_vel = np.zeros(3)
            missile.step(dt, target_pos, target_vel)
            if not missile.alive:
                record = {
                    "side": missile.side,
                    "target_uid": missile.target_uid,
                    "age_s": missile.age_s,
                }
                if missile.hit:
                    self._hits.append(record)
                elif missile.missed:
                    record["miss_reason"] = missile.miss_reason
                    self._misses.append(record)

        self.missiles = [m for m in self.missiles if m.alive]

    def check_launch(
        self,
        shooter_state: Dict[str, Any],
        target_state: Dict[str, Any],
        side: str,
        sim_time: float,
        difficulty: str = "medium",
    ) -> Optional[MissileModel]:
        """Launch a missile if the shooter is within envelope.

        Args:
            shooter_state: state dict of the firing aircraft.
            target_state: state dict of the intended target.
            side: "own" or "bandit".
            sim_time: current simulation time [s].
            difficulty: easy / medium / hard (affects launch probability).

        Returns:
            MissileModel if launched, else None.
        """
        if not self.config.get("enabled", False):
            return None

        cooldown_s = float(self.config.get("launch_cooldown_s", 5.0))
        if sim_time - self._last_launch_time.get(side, -1e9) < cooldown_s:
            return None

        shooter_pos = shooter_state.get("position_m")
        if shooter_pos is None:
            shooter_pos = shooter_state.get("position_neu")
        shooter_pos = np.asarray(shooter_pos)
        target_pos = target_state.get("position_m")
        if target_pos is None:
            target_pos = target_state.get("position_neu")
        target_pos = np.asarray(target_pos)
        shooter_vel = shooter_state.get("velocity_vector_mps")
        if shooter_vel is None:
            shooter_vel = np.zeros(3)
        shooter_vel = np.asarray(shooter_vel)

        rel_pos = target_pos - shooter_pos
        range_m = float(np.linalg.norm(rel_pos))
        min_range = float(self.config.get("min_range_m", 500.0))
        max_range = float(self.config.get("max_range_m", 8000.0))

        if range_m < min_range or range_m > max_range:
            return None

        # Simple tail-chase launch constraint: shooter velocity roughly aligned
        # with line-of-sight to target.
        los_unit = rel_pos / (range_m + 1e-8)
        shooter_speed = float(np.linalg.norm(shooter_vel))
        if shooter_speed > 1e-6:
            cos_angle = float(np.dot(shooter_vel / shooter_speed, los_unit))
        else:
            cos_angle = 1.0
        if cos_angle < 0.5:  # > 60 deg off-boresight
            return None

        # Difficulty-dependent launch probability.
        base_prob = float(self.config.get("launch_probability", 0.05))
        difficulty_scale = {"easy": 0.3, "medium": 1.0, "hard": 2.0}.get(difficulty, 1.0)
        launch_prob = min(1.0, base_prob * difficulty_scale)

        if self._rng.random() >= launch_prob:
            return None

        missile = MissileModel(
            launch_position_m=shooter_pos,
            launch_velocity_mps=shooter_vel,
            target_uid=("own" if side == "bandit" else "target"),
            side=side,
            config=self.config,
        )
        self.missiles.append(missile)
        self._last_launch_time[side] = sim_time
        return missile

    def check_hits(self) -> Dict[str, Any]:
        """Check whether any missile has hit or missed.

        Returns:
            dict with keys:
                - "own_hit" / "bandit_hit" (bool)
                - "hit_by" (side of the missile that hit)
                - "missiles" (list of dict summaries)
        """
        result = {
            "own_hit": False,
            "bandit_hit": False,
            "hit_by": None,
            "missiles": [],
        }
        for hit in self._hits:
            result["missiles"].append({**hit, "hit": True})
            if hit["side"] == "bandit":
                result["own_hit"] = True
                result["hit_by"] = "bandit"
            else:
                result["bandit_hit"] = True
                result["hit_by"] = "own"
        for miss in self._misses:
            result["missiles"].append({**miss, "missed": True})
        for m in self.missiles:
            result["missiles"].append({
                "side": m.side,
                "target_uid": m.target_uid,
                "hit": m.hit,
                "missed": m.missed,
                "miss_reason": m.miss_reason,
                "age_s": m.age_s,
            })
        return result
