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
        self.switch_events = []
        self.synthetic_target_state = None

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
        self.switch_events = []
        self.synthetic_target_state = self._make_target_state(self.waypoints[0])

    def _task_get_target_state(self, backend_target_state: dict, own_state: dict) -> dict:
        """Return the synthetic waypoint target instead of the backend target."""
        return self.synthetic_target_state

    def _task_pre_step(self, own_state: dict, target_state: dict) -> Tuple[dict, dict]:
        """Propagate the synthetic target forward at constant velocity."""
        dt = self.env_config.get("high_level_dt", 0.2)
        self.segment_elapsed_s += dt
        self.synthetic_target_state = self._propagate_target(
            self.synthetic_target_state, dt
        )
        return own_state, self.synthetic_target_state

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
        range_m = float(info.get("range_m", np.nan))
        reach_radius = float(self.task_cfg.get("reach_radius_m", 600.0))
        segment_timeout = float(self.task_cfg.get("segment_timeout_s", 20.0))

        reached = not np.isnan(range_m) and range_m <= reach_radius
        timeout = self.segment_elapsed_s >= segment_timeout

        if reached or timeout:
            self.switch_events.append(
                {
                    "time_s": float(self._sim_time_s),
                    "from_idx": int(self.active_idx),
                    "reached": bool(reached),
                    "timeout": bool(timeout),
                    "range_m": float(range_m),
                }
            )
            if reached:
                self.completed_waypoints += 1

            self.active_idx += 1
            self.segment_elapsed_s = 0.0

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
