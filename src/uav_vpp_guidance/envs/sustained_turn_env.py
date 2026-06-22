"""
Sustained turn environment.

The aircraft must orbit a fixed target at a constant radius. The episode runs
for a fixed duration (90 s by default). Success is measured by the number of
completed orbits, turn rate stability, and energy maintenance.
"""

import copy
import numpy as np
from typing import Optional, Tuple

from .tracking_env import CloseRangeTrackingEnv


class SustainedTurnEnv(CloseRangeTrackingEnv):
    """
    Sustained turn task: orbit a fixed target at a commanded radius.

    A unified virtual point generator creates a tangential pursuit reference
    shared by all controllers. The default success condition is disabled;
    episodes terminate only on timeout, stall, or absolute bounds violation.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.task_cfg = config.get("task", {}).get("sustained_turn", {})
        self.target_pos = np.array(
            self.task_cfg.get("target_pos_m", [2000.0, 0.0, 5000.0]), dtype=float
        )
        self.orbit_direction = 1.0
        self.theta_unwrapped = []
        self.radius_hist = []
        self.speed_hist = []
        self.nz_hist = []
        self.agg_hist = []

    # ------------------------------------------------------------------
    # Gym-like interface overrides
    # ------------------------------------------------------------------

    def reset(self, scenario=None, seed=None) -> dict:
        """Reset with a task-specific default scenario if none is provided."""
        if scenario is None:
            scenario = self._build_default_scenario(seed)
        return super().reset(scenario=scenario, seed=seed)

    # ------------------------------------------------------------------
    # Task hooks
    # ------------------------------------------------------------------

    def _task_reset(self, seed: Optional[int] = None) -> None:
        """Initialize orbit direction and per-episode history."""
        rng = np.random.default_rng(seed)
        self.orbit_direction = 1.0 if rng.random() < 0.5 else -1.0
        self.theta_unwrapped = []
        self.radius_hist = []
        self.speed_hist = []
        self.nz_hist = []
        self.agg_hist = []
        self._prev_yaw_rad = None
        self._orbit_bonus_count = 0

    def _task_get_target_state(self, backend_target_state: dict, own_state: dict) -> dict:
        """Return the fixed orbit center as the target state."""
        return {
            "position_m": self.target_pos.copy(),
            "position_neu": self.target_pos.copy(),
            "velocity_vector_mps": np.zeros(3, dtype=float),
            "heading_deg": 0.0,
            "speed_mps": 0.0,
        }

    def _task_pre_step(self, own_state: dict, target_state: dict) -> Tuple[dict, dict]:
        """Replace target state with a virtual point on the desired orbit."""
        own_pos = np.asarray(own_state.get("position_m", own_state.get("position_neu", [0.0, 0.0, 5000.0])))
        virtual_point_pos = self._compute_orbit_virtual_point(own_pos)
        synthetic_target = {
            "position_m": virtual_point_pos,
            "position_neu": virtual_point_pos,
            "velocity_vector_mps": np.zeros(3, dtype=float),
            "heading_deg": 0.0,
            "speed_mps": 0.0,
        }
        return own_state, synthetic_target

    def _task_post_step(
        self,
        own_state_post: dict,
        target_state_post: dict,
        info: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[float, bool, bool, dict]:
        """Track orbit history, disable success termination, add stall protection."""
        own_pos = np.asarray(
            own_state_post.get("position_m", own_state_post.get("position_neu", [0.0, 0.0, 5000.0]))
        )
        rel = own_pos[:2] - self.target_pos[:2]
        theta = float(np.arctan2(rel[1], rel[0]))

        if not self.theta_unwrapped:
            self.theta_unwrapped.append(theta)
        else:
            self.theta_unwrapped.append(
                float(np.unwrap([self.theta_unwrapped[-1], theta])[-1])
            )

        radius = float(np.linalg.norm(rel))
        speed = float(
            own_state_post.get("speed_mps", 250.0)
            or np.linalg.norm(own_state_post.get("velocity_vector_mps", [0.0, 0.0, 0.0]))
        )
        self.radius_hist.append(radius)
        self.speed_hist.append(speed)
        self.nz_hist.append(float(info.get("nz_cmd", np.nan)))
        self.agg_hist.append(
            float(info.get("aggressiveness"))
            if info.get("aggressiveness") is not None
            else np.nan
        )

        if len(self.theta_unwrapped) >= 2:
            laps = abs(self.theta_unwrapped[-1] - self.theta_unwrapped[0]) / (2.0 * np.pi)
        else:
            laps = 0.0
        info["completed_orbits"] = laps
        info["turn_radius_m"] = radius
        info["orbit_direction"] = self.orbit_direction

        # Disable the default "success" termination (proximity to target).
        if info.get("is_success"):
            terminated = False
            truncated = False
            info["is_success"] = False
            info["reason"] = None

        # Orbit-tracking reward shaping.
        orbit_radius = float(self.task_cfg.get("orbit_radius_m", 1200.0))
        shaping_cfg = self.task_cfg.get("reward_shaping", {})
        w_radius = float(shaping_cfg.get("w_radius_error", 0.5))
        w_turn = float(shaping_cfg.get("w_turn_rate_error", 0.5))
        orbit_bonus = float(shaping_cfg.get("orbit_bonus", 50.0))
        dt = float(self.env_config.get("high_level_dt", 0.2))

        yaw = float(own_state_post.get("yaw_rad", np.nan))
        if not np.isfinite(yaw):
            vel = own_state_post.get("velocity_vector_mps", own_state_post.get("velocity_ned", [1.0, 0.0, 0.0]))
            vel = np.asarray(vel, dtype=float)
            yaw = float(np.arctan2(vel[1], vel[0]))

        radius_error = abs(radius - orbit_radius)
        desired_turn_rate = self.orbit_direction * speed / max(orbit_radius, 1.0)

        if self._prev_yaw_rad is not None:
            yaw_diff = self._stable_angle_diff(yaw, self._prev_yaw_rad)
            actual_turn_rate = yaw_diff / max(dt, 1e-6)
        else:
            actual_turn_rate = desired_turn_rate
        self._prev_yaw_rad = yaw

        shaping = -w_radius * (radius_error / max(orbit_radius, 1.0))
        turn_err = actual_turn_rate - desired_turn_rate
        shaping -= w_turn * (turn_err ** 2) / (desired_turn_rate ** 2 + 0.01)

        # Sparse bonus for each completed full orbit.
        completed_full = int(np.floor(laps))
        if completed_full > self._orbit_bonus_count:
            shaping += orbit_bonus * (completed_full - self._orbit_bonus_count)
            self._orbit_bonus_count = completed_full

        reward += float(shaping)

        # Stall protection.
        min_safe = float(self.task_cfg.get("min_safe_speed_mps", 150.0))
        if speed < min_safe:
            terminated = True
            truncated = False
            info["termination_reason"] = "stall"
            info["reason"] = "stall"
            info["is_crash"] = True

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Termination override: use absolute bounds, not range-to-target
    # ------------------------------------------------------------------

    def _check_done(self, own_state, target_state, rel_state):
        """Check only altitude, absolute XY bounds, and timeout."""
        term_info = {
            "reason": None,
            "is_success": False,
            "is_crash": False,
            "is_timeout": False,
            "is_out_of_bounds": False,
        }

        altitude_m = float(own_state.get("altitude_m", 5000.0))
        pos = np.asarray(own_state.get("position_m", own_state.get("position_neu", [0.0, 0.0, 5000.0])))
        xy_limit = float(self.env_config.get("xy_limit_m", 20000.0))
        min_alt = float(self.env_config.get("min_altitude_m", 500.0))
        max_alt = float(self.env_config.get("max_altitude_m", 15000.0))
        max_steps = int(self.env_config.get("max_high_level_steps", 450))

        if altitude_m < min_alt or altitude_m > max_alt:
            term_info["reason"] = "crash"
            term_info["is_crash"] = True
            return True, False, term_info

        if len(pos) >= 2 and (abs(pos[0]) > xy_limit or abs(pos[1]) > xy_limit):
            term_info["reason"] = "out_of_bounds"
            term_info["is_out_of_bounds"] = True
            return True, False, term_info

        if self.current_step >= max_steps:
            term_info["reason"] = "timeout"
            term_info["is_timeout"] = True
            return False, True, term_info

        return False, False, term_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_states(self):
        """Override to return a fixed target state (orbit center)."""
        own, target = super()._get_current_states()
        # Force target to remain fixed at the orbit center.
        # This prevents JSBSim target aircraft dynamics from drifting
        # (e.g., descending under gravity) and corrupting range/reward computations.
        target = {
            "position_m": self.target_pos.copy(),
            "position_neu": self.target_pos.copy(),
            "velocity_vector_mps": np.zeros(3, dtype=float),
            "heading_deg": 0.0,
            "speed_mps": 0.0,
        }
        return own, target

    def _step_jsbsim(self, command, aggressiveness=None, pid_gain_deltas=None):
        """Step JSBSim after resetting the target aircraft to fixed position."""
        # Re-apply the target aircraft initial condition BEFORE stepping JSBSim
        # to prevent any target_dynamics movement from affecting the step.
        target_init = self._scenario_to_jsbsim_init({
            "position_m": self.target_pos,
            "velocity_mps": 0.0,
            "heading_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        })
        self.jsbsim_env.apply_aircraft_ic_state(self.target_uid, target_init)
        result = super()._step_jsbsim(command, aggressiveness, pid_gain_deltas)
        return result

    def _build_default_scenario(self, seed):
        """Build a default scenario matching the task definition."""
        return {
            "name": "sustained_turn_default",
            "own_init": {
                "position_m": [0.0, 0.0, float(self.target_pos[2])],
                "velocity_mps": 250.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": self.target_pos.tolist(),
                "velocity_mps": 0.0,
                "heading_deg": 0.0,
            },
        }

    @staticmethod
    def _stable_angle_diff(a: float, b: float) -> float:
        """Return the smallest signed difference between two angles (rad)."""
        diff = float(a - b)
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        return diff

    def _compute_orbit_virtual_point(self, own_pos: np.ndarray) -> np.ndarray:
        """Compute a tangential virtual pursuit point on the desired orbit."""
        rel = own_pos[:2] - self.target_pos[:2]
        theta = float(np.arctan2(rel[1], rel[0]))
        tangent_lead = float(np.deg2rad(self.task_cfg.get("tangent_lead_deg", 90.0)))
        orbit_r = float(self.task_cfg.get("orbit_radius_m", 1200.0))
        tangent = theta + self.orbit_direction * tangent_lead
        vp_xy = self.target_pos[:2] + orbit_r * np.array(
            [np.cos(tangent), np.sin(tangent)], dtype=float
        )
        return np.array([vp_xy[0], vp_xy[1], self.target_pos[2]], dtype=float)
