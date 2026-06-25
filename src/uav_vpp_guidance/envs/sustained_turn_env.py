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

    def __init__(self, config: dict, opponent_policy=None, opponent_config=None):
        super().__init__(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
        self.task_cfg = config.get("task", {}).get("sustained_turn", {})
        self.max_planned_altitude_m = float(
            self.task_cfg.get("max_planned_altitude_m", 5000.0)
        )
        self.supervisor_cfg = copy.deepcopy(self.task_cfg.get("supervisor", {}))
        self.supervisor_enabled = bool(self.supervisor_cfg.get("enabled", True))
        self.target_pos = np.array(
            self.task_cfg.get("target_pos_m", [2000.0, 0.0, 5000.0]), dtype=float
        )
        self.target_pos, self._planned_altitude_limited = self._limit_target_altitude(
            self.target_pos
        )
        self._planned_orbit_altitude_m = float(self.target_pos[2])
        self.orbit_direction = 1.0
        self.theta_unwrapped = []
        self.radius_hist = []
        self.speed_hist = []
        self.nz_hist = []
        self.agg_hist = []
        self._task_supervisor_state = "nominal"
        self._task_supervisor_pause_orbit_tracking = False

    # ------------------------------------------------------------------
    # Gym-like interface overrides
    # ------------------------------------------------------------------

    def reset(self, scenario=None, seed=None) -> dict:
        """Reset with a task-specific default scenario if none is provided."""
        if scenario is None:
            scenario = self._build_default_scenario(seed)
        scenario = self._apply_planned_altitude_guard(scenario)
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
        self._task_supervisor_state = "nominal"
        self._task_supervisor_pause_orbit_tracking = False

    def _task_get_target_state(self, backend_target_state: dict, own_state: dict) -> dict:
        """Return the fixed orbit center as the target state."""
        return self._build_reference_target_state(self.target_pos)

    def _task_pre_step(self, own_state: dict, target_state: dict) -> Tuple[dict, dict]:
        """Replace target state with a virtual point on the desired orbit."""
        own_pos = np.asarray(
            own_state.get("position_m", own_state.get("position_neu", [0.0, 0.0, 5000.0]))
        )
        if self.supervisor_enabled:
            snapshot = self._get_supervisor_snapshot(own_state)
            self._task_supervisor_pause_orbit_tracking = snapshot["pause_orbit_tracking"]

        if self._task_supervisor_pause_orbit_tracking:
            reference_pos = np.asarray(
                target_state.get(
                    "position_m",
                    target_state.get("position_neu", self.target_pos),
                ),
                dtype=float,
            )
        else:
            reference_pos = self._compute_orbit_virtual_point(own_pos)

        synthetic_target = self._build_reference_target_state(reference_pos)
        return own_state, synthetic_target

    def _task_adjust_command(
        self,
        raw_command: dict,
        own_state: dict,
        target_state: dict,
        rel_state: dict,
        *,
        use_command_override: bool = False,
    ) -> Tuple[dict, dict]:
        if use_command_override:
            return dict(raw_command), {
                "task_supervisor_active": False,
                "task_supervisor_bypassed": True,
                "task_supervisor_state": "bypassed",
                "task_supervisor_pause_orbit_tracking": False,
                "high_altitude_protection_active": False,
            }

        adjusted = dict(raw_command)
        if not self.supervisor_enabled:
            return adjusted, {
                "task_supervisor_active": False,
                "task_supervisor_bypassed": False,
                "task_supervisor_state": "disabled",
                "task_supervisor_pause_orbit_tracking": False,
                "high_altitude_protection_active": False,
            }

        snapshot = self._get_supervisor_snapshot(own_state)
        speed = snapshot["speed_mps"]
        roll_rad = snapshot["roll_rad"]
        altitude = snapshot["altitude_m"]
        roll_rate_max = snapshot["roll_rate_max"]
        recovery_nz_max = snapshot["recovery_nz_max"]
        recovery_nz_target = snapshot["recovery_nz_target"]
        caution_nz_max = snapshot["caution_nz_max"]
        recovery_roll_scale = snapshot["recovery_roll_rate_scale"]
        caution_roll_scale = snapshot["caution_roll_rate_scale"]
        recovery_roll_rate_cmd = snapshot["recovery_roll_rate_cmd"]
        recovery_roll_deadband_rad = snapshot["recovery_roll_deadband_rad"]
        recovery_throttle_cmd = snapshot["recovery_throttle_cmd"]
        caution_throttle_cmd = snapshot["caution_throttle_cmd"]
        high_altitude_threshold = snapshot["high_altitude_threshold_m"]
        high_altitude_nz_cap = snapshot["high_altitude_nz_cap"]
        high_altitude_roll_rate_scale = snapshot["high_altitude_roll_rate_scale"]
        high_altitude_throttle_cmd = snapshot["high_altitude_throttle_cmd"]
        high_altitude_nz_attenuation = snapshot["high_altitude_nz_attenuation"]
        specific_energy_m = snapshot["specific_energy_m"]
        specific_energy_floor_m = snapshot["specific_energy_floor_m"]
        specific_energy_caution_m = snapshot["specific_energy_caution_m"]
        state = snapshot["state"]
        self._task_supervisor_pause_orbit_tracking = snapshot["pause_orbit_tracking"]

        nz_cap = None
        roll_scale = 1.0
        throttle_floor = float(adjusted.get("throttle_cmd", 0.7))

        if state == "caution":
            nz_cap = caution_nz_max
            roll_scale = min(roll_scale, caution_roll_scale)
            throttle_floor = max(throttle_floor, caution_throttle_cmd)
        elif state == "recovery":
            nz_cap = recovery_nz_max
            roll_scale = min(roll_scale, recovery_roll_scale)
            throttle_floor = max(throttle_floor, recovery_throttle_cmd)

        high_altitude_active = altitude > high_altitude_threshold + 1e-6
        if high_altitude_active:
            if state == "nominal":
                state = "high_altitude"
            nz_cap = (
                high_altitude_nz_cap
                if nz_cap is None
                else min(nz_cap, high_altitude_nz_cap)
            )
            roll_scale = min(roll_scale, high_altitude_roll_rate_scale)
            throttle_floor = max(throttle_floor, high_altitude_throttle_cmd)
            adjusted["nz_cmd"] = 1.0 + (
                float(adjusted.get("nz_cmd", 1.0)) - 1.0
            ) * max(0.0, 1.0 - high_altitude_nz_attenuation)

        if nz_cap is not None:
            adjusted["nz_cmd"] = min(float(adjusted.get("nz_cmd", 1.0)), nz_cap)

        if state == "recovery":
            roll_rate_cmd = 0.0
            if abs(roll_rad) > recovery_roll_deadband_rad:
                roll_rate_cmd = -np.sign(roll_rad) * recovery_roll_rate_cmd
            adjusted["nz_cmd"] = recovery_nz_target
            adjusted["roll_rate_cmd"] = float(
                np.clip(roll_rate_cmd, -roll_rate_max, roll_rate_max)
            )
            adjusted["throttle_cmd"] = float(
                np.clip(
                    max(float(adjusted.get("throttle_cmd", 0.7)), throttle_floor),
                    0.0,
                    1.0,
                )
            )
        else:
            adjusted["roll_rate_cmd"] = (
                float(adjusted.get("roll_rate_cmd", 0.0)) * roll_scale
            )
            adjusted["throttle_cmd"] = float(
                np.clip(
                    max(float(adjusted.get("throttle_cmd", 0.7)), throttle_floor),
                    0.0,
                    1.0,
                )
            )

        self._task_supervisor_state = state
        supervisor_active = (
            state != "nominal"
            or high_altitude_active
            or self._task_supervisor_pause_orbit_tracking
        )
        return adjusted, {
            "task_supervisor_active": supervisor_active,
            "task_supervisor_bypassed": False,
            "task_supervisor_state": state,
            "task_supervisor_pause_orbit_tracking": self._task_supervisor_pause_orbit_tracking,
            "task_supervisor_recovery_active": state == "recovery",
            "high_altitude_protection_active": high_altitude_active,
            "specific_energy_m": specific_energy_m,
            "specific_energy_floor_m": specific_energy_floor_m,
            "specific_energy_caution_m": specific_energy_caution_m,
            "planned_orbit_altitude_m": self._planned_orbit_altitude_m,
            "planned_altitude_limited": self._planned_altitude_limited,
        }

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
        planned_orbit_altitude_m = float(
            getattr(self, "_planned_orbit_altitude_m", float(self.target_pos[2]))
        )
        planned_altitude_limited = bool(
            getattr(self, "_planned_altitude_limited", False)
        )
        supervisor_state = getattr(self, "_task_supervisor_state", "nominal")
        pause_orbit_tracking = bool(
            getattr(self, "_task_supervisor_pause_orbit_tracking", False)
        )
        info["completed_orbits"] = laps
        info["turn_radius_m"] = radius
        info["orbit_direction"] = self.orbit_direction
        info["planned_orbit_altitude_m"] = planned_orbit_altitude_m
        info["planned_altitude_limited"] = planned_altitude_limited
        info["task_supervisor_state"] = supervisor_state
        info["task_supervisor_pause_orbit_tracking"] = pause_orbit_tracking
        info["task_supervisor_recovery_active"] = supervisor_state == "recovery"

        # Disable only the default proximity success termination. Combat wins
        # also set is_success=True and must keep their terminal state.
        if info.get("is_success") and not info.get("combat_outcome"):
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

        completed_full = int(np.floor(laps))
        if pause_orbit_tracking:
            self._orbit_bonus_count = max(self._orbit_bonus_count, completed_full)
            shaping = 0.0
        else:
            shaping = -w_radius * (radius_error / max(orbit_radius, 1.0))
            turn_err = actual_turn_rate - desired_turn_rate
            shaping -= w_turn * (turn_err ** 2) / (desired_turn_rate ** 2 + 0.01)

            # Sparse bonus for each completed full orbit.
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

    def _get_current_states(self, noisy: bool = False):
        """Override to return a fixed target state (orbit center)."""
        own, target = super()._get_current_states(noisy=noisy)
        # Force target to remain fixed at the orbit center.
        # This prevents JSBSim target aircraft dynamics from drifting
        # (e.g., descending under gravity) and corrupting range/reward computations.
        target = self._build_reference_target_state(self.target_pos)
        return own, target

    def _step_jsbsim(
        self,
        command,
        aggressiveness=None,
        pid_gain_deltas=None,
        target_command=None,
    ):
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
        self.jsbsim_env.reload_aircraft(self.target_uid, target_init)
        result = super()._step_jsbsim(
            command,
            aggressiveness,
            pid_gain_deltas,
            target_command=target_command,
        )
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

    def _limit_target_altitude(self, target_pos: np.ndarray) -> Tuple[np.ndarray, bool]:
        limited = np.asarray(target_pos, dtype=float).copy()
        limited_altitude = min(float(limited[2]), self.max_planned_altitude_m)
        was_limited = abs(limited_altitude - float(limited[2])) > 1e-6
        limited[2] = limited_altitude
        return limited, was_limited

    def _apply_planned_altitude_guard(self, scenario: dict) -> dict:
        guarded = copy.deepcopy(scenario)
        self.target_pos, target_limited = self._limit_target_altitude(self.target_pos)

        own_init = guarded.get("own_init", {})
        target_init = guarded.get("target_init", {})
        own_limited = False
        target_init_limited = False

        own_pos = own_init.get("position_m")
        if own_pos is not None:
            own_pos_arr = np.asarray(own_pos, dtype=float).copy()
            limited_alt = min(float(own_pos_arr[2]), self.max_planned_altitude_m)
            own_limited = abs(limited_alt - float(own_pos_arr[2])) > 1e-6
            own_pos_arr[2] = limited_alt
            own_init["position_m"] = own_pos_arr.tolist()

        target_pos = target_init.get("position_m")
        if target_pos is not None:
            target_pos_arr, target_init_limited = self._limit_target_altitude(
                np.asarray(target_pos, dtype=float)
            )
            target_init["position_m"] = target_pos_arr.tolist()

        self._planned_altitude_limited = bool(
            target_limited or own_limited or target_init_limited
        )
        self._planned_orbit_altitude_m = float(self.target_pos[2])
        return guarded

    @staticmethod
    def _stable_angle_diff(a: float, b: float) -> float:
        """Return the smallest signed difference between two angles (rad)."""
        diff = float(a - b)
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        return diff

    def _build_reference_target_state(self, position_m: np.ndarray) -> dict:
        pos = np.asarray(position_m, dtype=float).copy()
        return {
            "position_m": pos,
            "position_neu": pos.copy(),
            "velocity_vector_mps": np.zeros(3, dtype=float),
            "heading_deg": 0.0,
            "speed_mps": 0.0,
        }

    def _get_supervisor_snapshot(self, own_state: dict) -> dict:
        speed = float(
            own_state.get(
                "speed_mps",
                np.linalg.norm(own_state.get("velocity_vector_mps", [0.0, 0.0, 0.0])),
            )
        )
        roll_rad = float(own_state.get("roll_rad", 0.0))
        if not np.isfinite(roll_rad):
            roll_rad = 0.0
        altitude = float(
            own_state.get(
                "altitude_m",
                np.asarray(
                    own_state.get(
                        "position_m",
                        own_state.get("position_neu", [0.0, 0.0, 5000.0]),
                    ),
                    dtype=float,
                )[2],
            )
        )
        roll_rate_max = abs(float(self.config.get("limits", {}).get("roll_rate_max", 1.5)))
        speed_floor = float(
            self.supervisor_cfg.get(
                "speed_floor_mps",
                max(float(self.task_cfg.get("min_safe_speed_mps", 150.0)), 175.0),
            )
        )
        recovery_speed = float(
            self.supervisor_cfg.get("recovery_speed_mps", max(speed_floor + 35.0, 210.0))
        )
        caution_speed = float(
            self.supervisor_cfg.get(
                "caution_speed_mps",
                max(speed_floor + 20.0, recovery_speed - 15.0),
            )
        )
        recovery_nz_min = float(self.supervisor_cfg.get("recovery_nz_min", 1.2))
        recovery_nz_max = float(self.supervisor_cfg.get("recovery_nz_max", 2.5))
        recovery_nz_soft_max = float(
            self.supervisor_cfg.get("recovery_nz_soft_max", min(recovery_nz_max, 1.5))
        )
        recovery_nz_soft_max = max(recovery_nz_min, min(recovery_nz_soft_max, recovery_nz_max))
        recovery_nz_target = float(
            np.clip(
                self.supervisor_cfg.get("recovery_nz_target", recovery_nz_min + 0.2),
                recovery_nz_min,
                recovery_nz_soft_max,
            )
        )
        caution_nz_max = float(self.supervisor_cfg.get("caution_nz_max", 4.0))
        recovery_roll_scale = float(
            self.supervisor_cfg.get("recovery_roll_rate_scale", 0.5)
        )
        caution_roll_scale = float(
            self.supervisor_cfg.get("caution_roll_rate_scale", 0.75)
        )
        recovery_roll_rate_cmd = float(
            self.supervisor_cfg.get(
                "recovery_roll_rate_cmd", max(0.8 * roll_rate_max, 0.6)
            )
        )
        recovery_roll_rate_cmd = float(np.clip(recovery_roll_rate_cmd, 0.0, roll_rate_max))
        recovery_roll_deadband_rad = float(
            np.deg2rad(self.supervisor_cfg.get("recovery_roll_deadband_deg", 1.0))
        )
        recovery_throttle_cmd = float(
            self.supervisor_cfg.get("recovery_throttle_cmd", 0.95)
        )
        caution_throttle_cmd = float(
            self.supervisor_cfg.get("caution_throttle_cmd", 0.8)
        )
        high_altitude_threshold = float(
            self.supervisor_cfg.get(
                "high_altitude_threshold_m", self.max_planned_altitude_m
            )
        )
        high_altitude_nz_cap = float(
            self.supervisor_cfg.get("high_altitude_nz_cap", 3.0)
        )
        high_altitude_roll_rate_scale = float(
            self.supervisor_cfg.get("high_altitude_roll_rate_scale", 0.7)
        )
        high_altitude_throttle_cmd = float(
            self.supervisor_cfg.get(
                "high_altitude_throttle_cmd", caution_throttle_cmd
            )
        )
        high_altitude_nz_attenuation = float(
            self.supervisor_cfg.get("high_altitude_nz_attenuation", 0.5)
        )

        g = 9.80665
        specific_energy_m = altitude + speed ** 2 / (2.0 * g)
        specific_energy_floor_m = float(
            self.supervisor_cfg.get(
                "specific_energy_floor_m",
                self._planned_orbit_altitude_m + speed_floor ** 2 / (2.0 * g),
            )
        )
        specific_energy_caution_m = float(
            self.supervisor_cfg.get(
                "specific_energy_caution_m",
                self._planned_orbit_altitude_m + caution_speed ** 2 / (2.0 * g),
            )
        )
        resume_speed_mps = float(
            self.supervisor_cfg.get("resume_speed_mps", recovery_speed)
        )
        resume_specific_energy_m = float(
            self.supervisor_cfg.get(
                "resume_specific_energy_m", specific_energy_caution_m
            )
        )

        state = "nominal"
        pause_orbit_tracking = self._task_supervisor_pause_orbit_tracking
        recovery_triggered = (
            speed < speed_floor or specific_energy_m < specific_energy_floor_m
        )
        caution_triggered = (
            speed < recovery_speed or specific_energy_m < specific_energy_caution_m
        )

        if pause_orbit_tracking:
            if speed >= resume_speed_mps and specific_energy_m >= resume_specific_energy_m:
                pause_orbit_tracking = False
            else:
                state = "recovery"

        if state != "recovery":
            if recovery_triggered:
                state = "recovery"
                pause_orbit_tracking = True
            elif caution_triggered:
                state = "caution"

        return {
            "state": state,
            "pause_orbit_tracking": pause_orbit_tracking,
            "speed_mps": speed,
            "roll_rad": roll_rad,
            "altitude_m": altitude,
            "roll_rate_max": roll_rate_max,
            "recovery_nz_max": recovery_nz_max,
            "recovery_nz_target": recovery_nz_target,
            "caution_nz_max": caution_nz_max,
            "recovery_roll_rate_scale": recovery_roll_scale,
            "caution_roll_rate_scale": caution_roll_scale,
            "recovery_roll_rate_cmd": recovery_roll_rate_cmd,
            "recovery_roll_deadband_rad": recovery_roll_deadband_rad,
            "recovery_throttle_cmd": recovery_throttle_cmd,
            "caution_throttle_cmd": caution_throttle_cmd,
            "high_altitude_threshold_m": high_altitude_threshold,
            "high_altitude_nz_cap": high_altitude_nz_cap,
            "high_altitude_roll_rate_scale": high_altitude_roll_rate_scale,
            "high_altitude_throttle_cmd": high_altitude_throttle_cmd,
            "high_altitude_nz_attenuation": high_altitude_nz_attenuation,
            "specific_energy_m": specific_energy_m,
            "specific_energy_floor_m": specific_energy_floor_m,
            "specific_energy_caution_m": specific_energy_caution_m,
        }

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
