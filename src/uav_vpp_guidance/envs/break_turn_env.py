"""
Break-turn / yo-yo maneuver tracking environment (Tier 2 prototype).

The target aircraft follows a preset aggressive maneuver:

- ``break_turn``: target starts ahead of the own aircraft, flies straight for a
  short delay, then executes a constant-rate heading reversal (default 180 deg)
  at constant speed and altitude.
- ``yo_yo``: target flies a straight track while oscillating in altitude and
  speed sinusoidally.

The own aircraft must track the maneuver.  Episodes terminate only on timeout,
stall, or absolute bounds violation; the primary metric is range tracking error.
"""

import copy
import math
from typing import Optional, Tuple

import numpy as np

from .tracking_env import CloseRangeTrackingEnv


class BreakTurnEnv(CloseRangeTrackingEnv):
    """
    Preset maneuver tracking task.

    The target trajectory is generated deterministically from the episode seed
    and replayed during the episode.  The controller must keep the own aircraft
    close to the maneuvering target.
    """

    def __init__(self, config: dict, opponent_policy=None, opponent_config=None):
        super().__init__(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
        self.task_cfg = config.get("task", {}).get("break_turn", {})
        self.maneuver_type = str(self.task_cfg.get("maneuver_type", "break_turn")).lower()
        self.rng = None
        self.trajectory: list = []
        self._current_target_state: Optional[dict] = None
        self.range_hist = []
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
        """Generate the preset target trajectory and per-episode history."""
        self.rng = np.random.default_rng(seed)
        self.trajectory = self._generate_trajectory()
        self._current_target_state = None
        self.range_hist = []
        self.speed_hist = []
        self.nz_hist = []
        self.agg_hist = []

    def _task_get_target_state(self, backend_target_state: dict, own_state: dict) -> dict:
        """Return the preset target state for the current simulation time."""
        return self._sample_trajectory(self._sim_time_s)

    def _task_pre_step(self, own_state: dict, target_state: dict) -> Tuple[dict, dict]:
        """Store the current target state for guidance computation."""
        self._current_target_state = target_state
        return own_state, target_state

    def _task_post_step(
        self,
        own_state_post: dict,
        target_state_post: dict,
        info: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[float, bool, bool, dict]:
        """Track maneuver history and add tracking-error reward shaping."""
        range_m = float(info.get("range_m", np.nan))
        speed = float(
            own_state_post.get("speed_mps", 250.0)
            or np.linalg.norm(own_state_post.get("velocity_vector_mps", [0.0, 0.0, 0.0]))
        )
        self.range_hist.append(range_m)
        self.speed_hist.append(speed)
        self.nz_hist.append(float(info.get("nz_cmd", np.nan)))
        self.agg_hist.append(
            float(info.get("aggressiveness"))
            if info.get("aggressiveness") is not None
            else np.nan
        )

        info["maneuver_type"] = self.maneuver_type
        info["maneuver_progress"] = self._maneuver_progress()

        # Disable only the default proximity success termination. Combat wins
        # also set is_success=True and must keep their terminal state.
        if info.get("is_success") and not info.get("combat_outcome"):
            terminated = False
            truncated = False
            info["is_success"] = False
            info["reason"] = None

        # Reward shaping: penalize range error ( encourage close tracking ).
        shaping_cfg = self.task_cfg.get("reward_shaping", {})
        w_range = float(shaping_cfg.get("w_range_error", 0.01))
        if np.isfinite(range_m):
            reward -= w_range * max(0.0, range_m - 200.0)

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
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_trajectory(self) -> list:
        """Generate a preset target trajectory from the task config."""
        if self.maneuver_type == "yo_yo":
            return self._generate_yo_yo_trajectory()
        return self._generate_break_turn_trajectory()

    def _generate_break_turn_trajectory(self) -> list:
        """
        Build a multi-segment break-turn trajectory with heading AND altitude reversals.

        The trajectory consists of: straight → climb+right-turn → dive+left-turn → climb+right-turn.
        Heading reversals change roll direction; altitude reversals (climb↔dive) force nz_cmd
        to cross the 1g line, enabling _recovery_delay_steps and _overshoot_pct metrics.
        """
        cfg = self.task_cfg
        dt = float(self.env_config.get("high_level_dt", 0.2))
        max_steps = int(self.env_config.get("max_high_level_steps", 512))
        duration = max_steps * dt

        initial_range = float(cfg.get("initial_range_m", 1500.0))
        base_alt = float(cfg.get("base_altitude_m", 5000.0))
        speed = float(cfg.get("target_speed_mps", 250.0))
        delay = float(cfg.get("straight_delay_s", 2.0))
        turn_rate_dps = float(cfg.get("turn_rate_deg_s", 6.0))
        climb_rate_mps = float(cfg.get("climb_rate_mps", 30.0))

        # Multi-segment maneuver with heading + altitude reversals.
        # seg_direction: +1=right-turn, -1=left-turn
        # alt_sign: +1=climb, -1=dive, 0=level
        turn_rate_rps = math.radians(turn_rate_dps)
        segments: list[tuple[str, float, float, float]] = [
            # (type, duration_s, heading_dir, alt_sign)
            ("straight", delay, 0.0, 0.0),                         # 0-2s: straight level
            ("turn",    90.0 / turn_rate_dps, 1.0, 1.0),           # 2-17s: right turn CLIMB
            ("turn",    180.0 / turn_rate_dps, -1.0, -1.0),        # 17-47s: left turn DIVE (nz reversal #1)
            ("turn",    90.0 / turn_rate_dps, 1.0, 1.0),           # 47-62s: right turn CLIMB (nz reversal #2)
        ]

        pos = np.array([initial_range, 0.0, base_alt], dtype=float)
        heading = 0.0
        trajectory = []
        segment_idx = 0
        segment_elapsed = 0.0

        t = 0.0
        while t <= duration + 1e-9:
            # Advance segment if needed
            while segment_idx < len(segments) and segment_elapsed >= segments[segment_idx][1]:
                segment_elapsed -= segments[segment_idx][1]
                segment_idx += 1

            if segment_idx >= len(segments):
                # All segments done; continue with last heading and altitude trend
                pass
            else:
                seg_type, seg_duration, seg_direction, alt_sign = segments[segment_idx]
                if seg_type == "turn":
                    heading += turn_rate_rps * seg_direction * dt

            vz = climb_rate_mps * alt_sign if segment_idx < len(segments) else 0.0
            vel = np.array([
                speed * math.cos(heading),
                speed * math.sin(heading),
                vz,
            ])
            trajectory.append(
                {
                    "time_s": float(t),
                    "position_m": pos.copy(),
                    "velocity_vector_mps": vel.copy(),
                    "heading_deg": float(math.degrees(heading) % 360.0),
                    "speed_mps": float(speed),
                }
            )
            t += dt
            pos += vel * dt
            segment_elapsed += dt

        return trajectory

    def _generate_yo_yo_trajectory(self) -> list:
        """
        Build a yo-yo trajectory.

        The target flies straight while oscillating in altitude and speed.
        """
        cfg = self.task_cfg
        dt = float(self.env_config.get("high_level_dt", 0.2))
        max_steps = int(self.env_config.get("max_high_level_steps", 512))
        duration = max_steps * dt

        initial_range = float(cfg.get("initial_range_m", 1500.0))
        base_alt = float(cfg.get("base_altitude_m", 5000.0))
        base_speed = float(cfg.get("target_speed_mps", 250.0))
        alt_amp = float(cfg.get("altitude_amplitude_m", 500.0))
        speed_amp = float(cfg.get("speed_amplitude_mps", 30.0))
        freq = float(cfg.get("oscillation_freq_hz", 0.05))
        heading = 0.0

        pos = np.array([initial_range, 0.0, base_alt], dtype=float)
        trajectory = []

        t = 0.0
        while t <= duration + 1e-9:
            z_offset = alt_amp * math.sin(2.0 * math.pi * freq * t)
            speed = base_speed + speed_amp * math.cos(2.0 * math.pi * freq * t)
            speed = max(100.0, speed)
            vel = np.array([speed, 0.0, 0.0])

            sample_pos = pos.copy()
            sample_pos[2] = base_alt + z_offset
            trajectory.append(
                {
                    "time_s": float(t),
                    "position_m": sample_pos,
                    "velocity_vector_mps": vel.copy(),
                    "heading_deg": float(math.degrees(heading) % 360.0),
                    "speed_mps": float(speed),
                }
            )
            t += dt
            # Integrate horizontal motion with the current speed; altitude
            # oscillation is treated as a disturbance on a straight track.
            pos[0] += speed * dt

        return trajectory

    def _sample_trajectory(self, time_s: float) -> dict:
        """Return the target state at the given simulation time."""
        if not self.trajectory:
            return self._build_default_target_state()

        dt = float(self.env_config.get("high_level_dt", 0.2))
        idx = min(int(round(float(time_s) / dt)), len(self.trajectory) - 1)
        return copy.deepcopy(self.trajectory[idx])

    def _maneuver_progress(self) -> float:
        """Return maneuver progress as a fraction of the total trajectory."""
        if not self.trajectory:
            return 0.0
        dt = float(self.env_config.get("high_level_dt", 0.2))
        idx = min(int(round(self._sim_time_s / dt)), len(self.trajectory) - 1)
        return idx / max(1, len(self.trajectory) - 1)

    def _get_current_states(self, noisy: bool = False):
        """Override to inject the preset maneuver reference as the target state.

        The backend aircraft dynamics still run, but the guidance, observation,
        and metrics all see the synthetic reference trajectory so the task is
        truly a maneuver-tracking problem.
        """
        own_state, _ = super()._get_current_states(noisy=noisy)
        target_state = self._sample_trajectory(self._sim_time_s)
        return own_state, target_state

    def _build_default_target_state(self) -> dict:
        """Fallback target state when no trajectory is available."""
        base_alt = float(self.task_cfg.get("base_altitude_m", 5000.0))
        speed = float(self.task_cfg.get("target_speed_mps", 250.0))
        pos = np.array([1500.0, 0.0, base_alt], dtype=float)
        return {
            "position_m": pos,
            "position_neu": pos,
            "velocity_vector_mps": np.array([speed, 0.0, 0.0]),
            "heading_deg": 0.0,
            "speed_mps": speed,
        }

    def _build_default_scenario(self, seed):
        """Build a default scenario matching the task definition."""
        base_alt = float(self.task_cfg.get("base_altitude_m", 5000.0))
        own_speed = float(self.task_cfg.get("own_speed_mps", 280.0))
        return {
            "name": f"break_turn_default_{self.maneuver_type}",
            "own_init": {
                "position_m": [0.0, 0.0, base_alt],
                "velocity_mps": own_speed,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, base_alt],
                "velocity_mps": 250.0,
                "heading_deg": 0.0,
            },
        }
