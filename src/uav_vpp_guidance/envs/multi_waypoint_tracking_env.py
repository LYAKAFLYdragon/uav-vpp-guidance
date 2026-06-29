"""
Multi-waypoint tracking environment.

Extends CloseRangeTrackingEnv with a chain of 5 waypoints. The agent must
sequentially reach each waypoint. Reaching one waypoint switches the active
target to the next; the episode succeeds only when all waypoints are completed.
"""

import copy
import numpy as np
from typing import Optional, Tuple

from .tracking_env import CloseRangeTrackingEnv


class MultiWaypointTrackingEnv(CloseRangeTrackingEnv):
    """
    Sequential multi-waypoint tracking task.

    Each episode samples a chain of 5 waypoints relative to the previous one.
    The policy must drive the aircraft through each waypoint in order. The
    default success condition (range/ATA hold) is overridden by the waypoint
    switching logic in `_task_post_step`.
    """

    def __init__(self, config: dict, opponent_policy=None, opponent_config=None):
        super().__init__(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
        self.task_cfg = config.get("task", {}).get("multi_waypoint", {})
        self.rng = None
        self.waypoints = []
        self.active_idx = 0
        self.completed_waypoints = 0
        self.segment_elapsed_s = 0.0
        self.segment_min_waypoint_range_m = float("inf")
        self.segment_min_waypoint_range_step = None
        self.switch_events = []
        self.synthetic_target_state = None
        self.guidance_target_state = None
        self._last_approach_guidance_info = {
            "multi_waypoint_approach_guidance_enabled": False,
            "multi_waypoint_static_waypoint_blend": 0.0,
        }

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
        """Generate waypoint sequence and initialize synthetic target."""
        self.rng = np.random.default_rng(seed)
        self.waypoints = self._generate_waypoint_sequence(self.rng)
        self.active_idx = 0
        self.completed_waypoints = 0
        self.segment_elapsed_s = 0.0
        self.segment_min_waypoint_range_m = float("inf")
        self.segment_min_waypoint_range_step = None
        self.switch_events = []
        self.synthetic_target_state = self._make_target_state(self.waypoints[0])
        self.guidance_target_state = self.synthetic_target_state
        self._last_approach_guidance_info = {
            "multi_waypoint_approach_guidance_enabled": False,
            "multi_waypoint_static_waypoint_blend": 0.0,
        }

    def _task_get_target_state(self, backend_target_state: dict, own_state: dict) -> dict:
        """Return the synthetic waypoint target instead of the backend target."""
        return self._select_guidance_target_state(own_state)

    def _task_uses_backend_target(self) -> bool:
        """The waypoint target is synthetic; the backend target is a placeholder."""
        return False

    def _task_pre_step(self, own_state: dict, target_state: dict) -> Tuple[dict, dict]:
        """Propagate the synthetic target forward at constant velocity."""
        dt = self.env_config.get("high_level_dt", 0.2)
        self.segment_elapsed_s += dt
        self.synthetic_target_state = self._propagate_target(
            self.synthetic_target_state, dt
        )
        return own_state, self._select_guidance_target_state(own_state)

    def _task_adjust_command(
        self,
        raw_command: dict,
        own_state: dict,
        target_state: dict,
        rel_state: dict,
        *,
        use_command_override: bool = False,
    ) -> Tuple[dict, dict]:
        """Apply task-local roll moderation without changing low-level protection."""
        if use_command_override:
            return dict(raw_command), {
                "multi_waypoint_command_shaping_bypassed": True,
            }

        shaping_cfg = self.task_cfg.get("command_shaping", {})
        enabled = bool(shaping_cfg.get("enabled", False))
        adjusted = dict(raw_command)
        waypoint_range_m = self._distance_to_active_waypoint(own_state)
        heading_error_rad = self._heading_error_to_active_waypoint(own_state)
        heading_error_deg = (
            float(np.degrees(heading_error_rad))
            if np.isfinite(heading_error_rad)
            else float("nan")
        )
        roll_rad = self._safe_float(own_state.get("roll_rad", 0.0), 0.0)
        roll_deg = float(np.degrees(abs(roll_rad))) if np.isfinite(roll_rad) else float("nan")

        meta = {
            "multi_waypoint_command_shaping_enabled": enabled,
            "multi_waypoint_waypoint_range_m": waypoint_range_m,
            "multi_waypoint_heading_error_to_waypoint_deg": heading_error_deg,
            "multi_waypoint_roll_deg": roll_deg,
            "multi_waypoint_roll_moderation_active": False,
            "multi_waypoint_roll_recovery_active": False,
            "multi_waypoint_same_bank_guard_active": False,
            "multi_waypoint_same_bank_guard_soft_cap_active": False,
            "multi_waypoint_same_bank_guard_action": "none",
            "multi_waypoint_roll_rate_scale": 1.0,
            "reset_command_filter": False,
        }
        meta.update(self._last_approach_guidance_info)
        if not enabled:
            return adjusted, meta

        reach_radius = float(self.task_cfg.get("reach_radius_m", 600.0))
        roll_rate = self._safe_float(adjusted.get("roll_rate_cmd", 0.0), 0.0)

        if bool(shaping_cfg.get("near_waypoint_roll_moderation_enabled", True)):
            moderation_range = float(
                shaping_cfg.get(
                    "roll_moderation_range_m",
                    max(2.0 * reach_radius, reach_radius + 1.0),
                )
            )
            min_scale = float(
                np.clip(shaping_cfg.get("roll_moderation_min_scale", 0.45), 0.0, 1.0)
            )
            if (
                moderation_range > 1.0
                and np.isfinite(waypoint_range_m)
                and waypoint_range_m <= moderation_range
            ):
                ratio = float(np.clip(waypoint_range_m / moderation_range, 0.0, 1.0))
                roll_scale = min_scale + (1.0 - min_scale) * ratio
                roll_rate *= roll_scale
                adjusted["roll_rate_cmd"] = roll_rate
                meta["multi_waypoint_roll_moderation_active"] = True
                meta["multi_waypoint_roll_rate_scale"] = roll_scale

        if bool(shaping_cfg.get("same_bank_guard_enabled", False)):
            guard_range = float(shaping_cfg.get("same_bank_guard_range_m", 3000.0))
            guard_start_deg = float(shaping_cfg.get("same_bank_guard_start_deg", 70.0))
            soft_cap_enabled = bool(
                shaping_cfg.get("same_bank_guard_soft_cap_enabled", False)
            )
            soft_cap_heading_error_deg = float(
                shaping_cfg.get("same_bank_guard_soft_cap_heading_error_deg", 45.0)
            )
            soft_cap_rate_cmd = abs(
                float(shaping_cfg.get("same_bank_guard_soft_cap_rate_cmd", 0.25))
            )
            soft_cap_max_recede_m = float(
                shaping_cfg.get("same_bank_guard_soft_cap_max_recede_m", 200.0)
            )
            if (
                guard_range > 1.0
                and np.isfinite(waypoint_range_m)
                and waypoint_range_m <= guard_range
                and np.isfinite(roll_deg)
                and roll_deg >= guard_start_deg
                and abs(roll_rad) > 1e-6
                and np.sign(roll_rate) == np.sign(roll_rad)
            ):
                segment_min = self.segment_min_waypoint_range_m
                range_receded_m = (
                    waypoint_range_m - segment_min
                    if np.isfinite(segment_min)
                    else 0.0
                )
                heading_supports_same_bank = (
                    np.isfinite(heading_error_rad)
                    and abs(heading_error_deg) >= soft_cap_heading_error_deg
                    and np.sign(heading_error_rad) == np.sign(roll_rad)
                )
                recently_approaching = (
                    not np.isfinite(range_receded_m)
                    or range_receded_m <= soft_cap_max_recede_m
                )
                if (
                    soft_cap_enabled
                    and heading_supports_same_bank
                    and recently_approaching
                ):
                    limits = self.config.get("limits", {})
                    roll_rate_max = abs(float(limits.get("roll_rate_max", 1.5)))
                    cap = float(np.clip(soft_cap_rate_cmd, 0.0, roll_rate_max))
                    roll_rate = float(np.sign(roll_rate)) * min(abs(roll_rate), cap)
                    adjusted["roll_rate_cmd"] = roll_rate
                    meta["multi_waypoint_same_bank_guard_soft_cap_active"] = True
                    meta["multi_waypoint_same_bank_guard_action"] = "soft_cap"
                else:
                    adjusted["roll_rate_cmd"] = 0.0
                    roll_rate = 0.0
                    meta["multi_waypoint_same_bank_guard_action"] = "zero"
                meta["multi_waypoint_same_bank_guard_active"] = True
                meta["reset_command_filter"] = True

        if bool(shaping_cfg.get("overbank_recovery_enabled", True)):
            recovery_start_deg = float(shaping_cfg.get("roll_recovery_start_deg", 85.0))
            if np.isfinite(roll_deg) and roll_deg >= recovery_start_deg and abs(roll_rad) > 1e-6:
                limits = self.config.get("limits", {})
                roll_rate_max = abs(float(limits.get("roll_rate_max", 1.5)))
                recovery_cmd = abs(
                    float(
                        shaping_cfg.get(
                            "roll_recovery_rate_cmd",
                            min(1.2, roll_rate_max),
                        )
                    )
                )
                recovery_cmd = float(np.clip(recovery_cmd, 0.0, roll_rate_max))
                adjusted["roll_rate_cmd"] = -float(np.sign(roll_rad)) * recovery_cmd
                meta["multi_waypoint_roll_recovery_active"] = True
                meta["reset_command_filter"] = True

        return adjusted, meta

    def _task_post_step(
        self,
        own_state_post: dict,
        target_state_post: dict,
        info: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[float, bool, bool, dict]:
        """Handle waypoint switching and override the default success semantics."""
        segment_idx = int(self.active_idx)
        backend_range_m = float(info.get("range_m", np.nan))
        waypoint_range_m = self._distance_to_active_waypoint(own_state_post)
        synthetic_target_range_m = self._distance_to_state(
            own_state_post, self.synthetic_target_state
        )
        if np.isfinite(waypoint_range_m):
            if waypoint_range_m < self.segment_min_waypoint_range_m:
                self.segment_min_waypoint_range_m = waypoint_range_m
                self.segment_min_waypoint_range_step = int(self.current_step)

        capture_reference = str(
            self.task_cfg.get("capture_reference", "waypoint")
        ).lower()
        if capture_reference in ("synthetic_target", "target", "moving_target"):
            range_m = synthetic_target_range_m
            min_capture_range_m = synthetic_target_range_m
        else:
            range_m = waypoint_range_m
            min_capture_range_m = self.segment_min_waypoint_range_m

        reach_radius = float(self.task_cfg.get("reach_radius_m", 600.0))
        capture_radius = float(self.task_cfg.get("capture_radius_m", reach_radius))
        segment_timeout = float(self.task_cfg.get("segment_timeout_s", 20.0))
        near_miss_cfg = self.task_cfg.get("near_miss_capture", {})
        near_miss_enabled = bool(near_miss_cfg.get("enabled", False))
        near_miss_radius = float(
            near_miss_cfg.get(
                "capture_radius_m",
                capture_radius * float(near_miss_cfg.get("radius_scale", 1.2)),
            )
        )
        near_miss_min_recede_m = float(
            near_miss_cfg.get("min_recede_m", max(50.0, 0.05 * capture_radius))
        )
        near_miss_min_elapsed_s = float(near_miss_cfg.get("min_elapsed_s", 0.0))
        range_receded_m = (
            range_m - min_capture_range_m
            if np.isfinite(range_m) and np.isfinite(min_capture_range_m)
            else float("nan")
        )

        inside_capture_radius = (
            np.isfinite(range_m)
            and (
                range_m <= capture_radius
                or (
                    np.isfinite(min_capture_range_m)
                    and min_capture_range_m <= capture_radius
                )
            )
        )
        near_miss_capture = (
            not inside_capture_radius
            and near_miss_enabled
            and capture_reference not in ("synthetic_target", "target", "moving_target")
            and np.isfinite(min_capture_range_m)
            and np.isfinite(range_m)
            and min_capture_range_m <= near_miss_radius
            and range_receded_m >= near_miss_min_recede_m
            and self.segment_elapsed_s >= near_miss_min_elapsed_s
        )
        reached = inside_capture_radius or near_miss_capture
        capture_reason = None
        if inside_capture_radius:
            capture_reason = "inside_capture_radius"
        elif near_miss_capture:
            capture_reason = "near_miss_passed_waypoint"
        timeout = self.segment_elapsed_s >= segment_timeout

        info["backend_range_m"] = backend_range_m
        info["waypoint_range_m"] = waypoint_range_m
        info["synthetic_target_range_m"] = synthetic_target_range_m
        info["segment_min_waypoint_range_m"] = self.segment_min_waypoint_range_m
        info["segment_min_waypoint_range_step"] = self.segment_min_waypoint_range_step
        info["segment_waypoint_index"] = segment_idx
        info["segment_elapsed_s"] = float(self.segment_elapsed_s)
        info["capture_reference"] = capture_reference
        info["capture_radius_m"] = capture_radius
        info["capture_reason"] = capture_reason
        info["near_miss_capture_enabled"] = near_miss_enabled
        info["near_miss_capture_active"] = near_miss_capture
        info["near_miss_capture_radius_m"] = near_miss_radius
        info["near_miss_min_recede_m"] = near_miss_min_recede_m
        info["segment_range_receded_m"] = range_receded_m
        info.update(self._last_approach_guidance_info)
        info["range_m"] = range_m

        if reached or timeout:
            self.switch_events.append(
                {
                    "time_s": float(self._sim_time_s),
                    "from_idx": segment_idx,
                    "reached": bool(reached),
                    "timeout": bool(timeout),
                    "capture_reason": capture_reason,
                    "range_m": float(range_m),
                    "capture_radius_m": float(capture_radius),
                    "waypoint_range_m": float(waypoint_range_m),
                    "synthetic_target_range_m": float(synthetic_target_range_m),
                    "backend_range_m": float(backend_range_m),
                    "segment_min_waypoint_range_m": float(self.segment_min_waypoint_range_m),
                    "segment_min_waypoint_range_step": self.segment_min_waypoint_range_step,
                    "segment_range_receded_m": float(range_receded_m),
                    "near_miss_capture_active": bool(near_miss_capture),
                    "near_miss_capture_radius_m": float(near_miss_radius),
                    "static_waypoint_blend": float(
                        self._last_approach_guidance_info.get(
                            "multi_waypoint_static_waypoint_blend", 0.0
                        )
                    ),
                }
            )
            if reached:
                self.completed_waypoints += 1

            self.active_idx += 1
            self.segment_elapsed_s = 0.0
            self.segment_min_waypoint_range_m = float("inf")
            self.segment_min_waypoint_range_step = None

            if self.active_idx >= len(self.waypoints):
                # Reached the end of the waypoint chain.
                if self.completed_waypoints >= len(self.waypoints):
                    # All waypoints actually reached -> real task success.
                    terminated = True
                    truncated = False
                    info["task_success"] = True
                    info["termination_reason"] = "all_waypoints_completed"
                    info["reason"] = "all_waypoints_completed"
                    info["is_success"] = True
                else:
                    # Waypoint chain exhausted but not all were actually reached
                    # (e.g. every segment timed out). Treat as incomplete/timeout.
                    terminated = False
                    truncated = True
                    info["task_success"] = False
                    info["termination_reason"] = "waypoints_incomplete"
                    info["reason"] = "waypoints_incomplete"
                    info["is_success"] = False
                info["completed_waypoints"] = self.completed_waypoints
                info["switch_events"] = copy.deepcopy(self.switch_events)
                info["waypoints"] = [self._serialize_waypoint(wp) for wp in self.waypoints]
                return reward, terminated, truncated, info

            self.synthetic_target_state = self._make_target_state(
                self.waypoints[self.active_idx]
            )
            # Reset the termination checker success counter for the new target.
            if hasattr(self, "termination_checker") and self.termination_checker is not None:
                self.termination_checker.reset()

        # Override default success/termination: do not end episode on single-waypoint success.
        if info.get("is_success"):
            terminated = False
            truncated = False
            info["is_success"] = False
            info["reason"] = None

        info["completed_waypoints"] = self.completed_waypoints
        info["active_waypoint_index"] = self.active_idx
        info["switch_events"] = copy.deepcopy(self.switch_events)
        info["waypoints"] = [self._serialize_waypoint(wp) for wp in self.waypoints]
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
        max_steps = int(self.env_config.get("max_high_level_steps", 512))

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

    def _select_guidance_target_state(self, own_state: dict) -> dict:
        """Return a target state for guidance, optionally biased to the static waypoint."""
        if self.synthetic_target_state is None:
            self._last_approach_guidance_info = {
                "multi_waypoint_approach_guidance_enabled": False,
                "multi_waypoint_static_waypoint_blend": 0.0,
            }
            return self.synthetic_target_state

        approach_cfg = self.task_cfg.get("approach_guidance", {})
        enabled = bool(approach_cfg.get("enabled", False))
        waypoint_range_m = self._distance_to_active_waypoint(own_state)
        info = {
            "multi_waypoint_approach_guidance_enabled": enabled,
            "multi_waypoint_static_waypoint_blend": 0.0,
            "multi_waypoint_approach_target_range_m": waypoint_range_m,
        }
        if (
            not enabled
            or self.active_idx < 0
            or self.active_idx >= len(self.waypoints)
            or not np.isfinite(waypoint_range_m)
        ):
            self.guidance_target_state = self.synthetic_target_state
            self._last_approach_guidance_info = info
            return self.guidance_target_state

        static_range = float(approach_cfg.get("static_range_m", 1800.0))
        blend_start = float(approach_cfg.get("blend_start_range_m", 3200.0))
        if blend_start <= static_range:
            blend_start = static_range + 1.0

        if waypoint_range_m <= static_range:
            blend = 1.0
        elif waypoint_range_m >= blend_start:
            blend = 0.0
        else:
            blend = (blend_start - waypoint_range_m) / (blend_start - static_range)
        blend = float(np.clip(blend, 0.0, 1.0))

        static_target = self._make_target_state(self.waypoints[self.active_idx])
        synthetic_pos = self._extract_position(self.synthetic_target_state)
        static_pos = self._extract_position(static_target)
        if synthetic_pos is None or static_pos is None:
            self.guidance_target_state = self.synthetic_target_state
            self._last_approach_guidance_info = info
            return self.guidance_target_state

        target_pos = (1.0 - blend) * synthetic_pos + blend * static_pos
        target_state = {
            **self.synthetic_target_state,
            "position_m": target_pos,
            "position_neu": target_pos,
        }
        self.guidance_target_state = target_state
        info["multi_waypoint_static_waypoint_blend"] = blend
        self._last_approach_guidance_info = info
        return self.guidance_target_state

    def _distance_to_active_waypoint(self, own_state: dict) -> float:
        """Return distance from own aircraft to the active static waypoint."""
        if self.active_idx < 0 or self.active_idx >= len(self.waypoints):
            return float("nan")
        own_pos = self._extract_position(own_state)
        if own_pos is None:
            return float("nan")
        wp_pos = np.asarray(self.waypoints[self.active_idx]["pos"], dtype=float)
        return float(np.linalg.norm(own_pos - wp_pos))

    def _distance_to_state(self, own_state: dict, target_state: Optional[dict]) -> float:
        """Return distance from own aircraft to a target state."""
        if target_state is None:
            return float("nan")
        own_pos = self._extract_position(own_state)
        target_pos = self._extract_position(target_state)
        if own_pos is None or target_pos is None:
            return float("nan")
        return float(np.linalg.norm(own_pos - target_pos))

    def _heading_error_to_active_waypoint(self, own_state: dict) -> float:
        """Return signed heading error from ownship heading to active waypoint."""
        if self.active_idx < 0 or self.active_idx >= len(self.waypoints):
            return float("nan")
        own_pos = self._extract_position(own_state)
        if own_pos is None:
            return float("nan")
        wp_pos = np.asarray(self.waypoints[self.active_idx]["pos"], dtype=float)
        rel = wp_pos - own_pos
        if not np.isfinite(rel[:2]).all() or np.linalg.norm(rel[:2]) <= 1e-6:
            return 0.0
        bearing = float(np.arctan2(rel[1], rel[0]))
        heading = self._extract_heading_rad(own_state)
        if not np.isfinite(heading):
            return float("nan")
        return self._stable_angle_diff(bearing, heading)

    @staticmethod
    def _extract_heading_rad(state: dict) -> float:
        for key in ("yaw_rad", "heading_rad"):
            value = state.get(key) if isinstance(state, dict) else None
            if value is not None:
                try:
                    out = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(out):
                    return out

        if isinstance(state, dict):
            vel = state.get("velocity_vector_mps")
            if vel is None:
                vel = state.get("velocity")
            if vel is None:
                vel = state.get("velocity_ned")
            if vel is not None:
                arr = np.asarray(vel, dtype=float)
                if arr.shape == (3,) and np.isfinite(arr[:2]).all():
                    speed_xy = float(np.linalg.norm(arr[:2]))
                    if speed_xy > 1e-6:
                        return float(np.arctan2(arr[1], arr[0]))

        return 0.0

    @staticmethod
    def _stable_angle_diff(a: float, b: float) -> float:
        delta = a - b
        if not np.isfinite(delta):
            return float(delta)
        return float(np.arctan2(np.sin(delta), np.cos(delta)))

    @staticmethod
    def _extract_position(state: Optional[dict]) -> Optional[np.ndarray]:
        if not isinstance(state, dict):
            return None
        pos = state.get("position_m")
        if pos is None:
            pos = state.get("position_neu")
        if pos is None:
            return None
        arr = np.asarray(pos, dtype=float)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            return None
        return arr

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if np.isfinite(out) else float(default)

    def _build_default_scenario(self, seed):
        """Build a default scenario matching the task definition."""
        # If seed is not provided yet, waypoints are not generated here but in
        # _task_reset. We only need a placeholder target init that puts the
        # JSBSim target somewhere valid; the actual target is synthetic.
        alt = float(self.task_cfg.get("target_altitude_m", 5000.0))
        rng = np.random.default_rng(seed)
        dist_range = self.task_cfg.get("waypoint_distance_range_m", [800.0, 3000.0])
        speed_range = self.task_cfg.get("target_speed_range_mps", [100.0, 120.0])
        heading_limit = float(self.task_cfg.get("heading_delta_limit_deg", 30.0))

        first_dist = float(rng.uniform(*dist_range))
        first_heading = float(rng.uniform(-heading_limit, heading_limit) * np.pi / 180.0)
        first_speed = float(rng.uniform(*speed_range))
        first_pos = np.array(
            [first_dist * np.cos(first_heading), first_dist * np.sin(first_heading), alt]
        )

        return {
            "name": "multi_waypoint_default",
            "own_init": {
                "position_m": [0.0, 0.0, alt],
                "velocity_mps": 250.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": first_pos.tolist(),
                "velocity_mps": first_speed,
                "heading_deg": np.degrees(first_heading),
            },
        }

    def _generate_waypoint_sequence(self, rng):
        """Chain-generate the waypoint sequence relative to previous waypoint."""
        n = int(self.task_cfg.get("n_waypoints", 5))
        dist_range = self.task_cfg.get("waypoint_distance_range_m", [800.0, 3000.0])
        speed_range = self.task_cfg.get("target_speed_range_mps", [100.0, 120.0])
        heading_limit = float(self.task_cfg.get("heading_delta_limit_deg", 30.0))
        alt = float(self.task_cfg.get("target_altitude_m", 5000.0))

        waypoints = []
        prev_pos = np.array([0.0, 0.0, alt], dtype=float)

        for i in range(n):
            dist = float(rng.uniform(*dist_range))
            if i == 0:
                base_heading = 0.0
            else:
                # Continue from the previous segment's heading for a smooth chain.
                base_heading = float(waypoints[-1]["heading"])
            heading_delta = float(rng.uniform(-heading_limit, heading_limit) * np.pi / 180.0)
            heading = base_heading + heading_delta
            dx = dist * np.cos(heading)
            dy = dist * np.sin(heading)
            new_pos = prev_pos + np.array([dx, dy, 0.0], dtype=float)
            new_pos[2] = alt
            speed = float(rng.uniform(*speed_range))
            waypoints.append({"pos": new_pos, "speed": speed, "heading": heading})
            prev_pos = new_pos

        return waypoints

    @staticmethod
    def _make_target_state(waypoint: dict) -> dict:
        """Create a synthetic target state from a waypoint at time t=0."""
        pos = np.asarray(waypoint["pos"], dtype=float).copy()
        heading = float(waypoint["heading"])
        speed = float(waypoint["speed"])
        vel = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0], dtype=float)
        return {
            "position_m": pos,
            "position_neu": pos,
            "velocity_vector_mps": vel,
            "heading_deg": float(np.degrees(heading) % 360.0),
            "speed_mps": speed,
        }

    @staticmethod
    def _propagate_target(state: dict, dt: float) -> dict:
        """Propagate synthetic target at constant velocity for dt seconds."""
        pos = np.asarray(state["position_m"], dtype=float)
        vel = np.asarray(state["velocity_vector_mps"], dtype=float)
        new_pos = pos + vel * dt
        return {
            **state,
            "position_m": new_pos,
            "position_neu": new_pos,
        }

    @staticmethod
    def _serialize_waypoint(wp: dict) -> dict:
        """Serialize waypoint for JSON output."""
        return {
            "pos": wp["pos"].tolist(),
            "speed": float(wp["speed"]),
            "heading": float(wp["heading"]),
        }
