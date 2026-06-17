"""
Perturbed JSBSim environment for robustness testing.

Wraps CloseRangeTrackingEnv to add:
- Wind gusts (Gaussian random velocity perturbations via JSBSim atmosphere)
- Sensor noise (Gaussian noise on observation components)
- Actuator delay (FIFO action buffer)

The perturbation magnitude is controlled by a ``scale`` parameter (0.0 = off,
1.0 = full magnitude as configured).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerturbedJSBSimEnv:
    """
    Wrapper that adds perturbations to an existing tracking environment.

    Designed to wrap either ``CloseRangeTrackingEnv`` (single-agent) or
    ``AdversarialJSBSimEnv`` (two-agent). Detects which type is wrapped and
    applies perturbations accordingly.

    Perturbations:
    1. **Wind**: Sets JSBSim atmosphere wind properties per step with
       Gaussian-sampled gust components.
    2. **Sensor noise**: Adds Gaussian noise to the observation vector.
    3. **Actuator delay**: Buffers actions for a configurable number of
       decision steps before execution.
    """

    def __init__(
        self,
        base_env: Any,
        perturbation_config: Optional[Dict[str, Any]] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Args:
            base_env: The underlying environment (CloseRangeTrackingEnv or
                AdversarialJSBSimEnv).
            perturbation_config: Configuration dict with ``wind``, ``sensor_noise``,
                and ``actuator_delay`` sections.
            scale: Multiplier for all perturbation magnitudes (0.0 = off).
        """
        self._env = base_env
        self._config = perturbation_config or {}
        self.scale = float(scale)

        # Detect if this is an adversarial (two-agent) environment
        self._is_adversarial = hasattr(base_env, "_target_uid")

        # ---- Wind ----
        wind_cfg = self._config.get("wind", {})
        self._wind_enabled = bool(wind_cfg.get("enabled", True))
        self._wind_speed_std = float(wind_cfg.get("speed_std_mps", 5.0))
        self._wind_dir_std = float(wind_cfg.get("direction_std_deg", 30.0))

        # ---- Sensor noise ----
        noise_cfg = self._config.get("sensor_noise", {})
        self._noise_enabled = bool(noise_cfg.get("enabled", True))
        self._pos_noise_std = float(noise_cfg.get("position_noise_std_m", 10.0))
        self._vel_noise_std = float(noise_cfg.get("velocity_noise_std_mps", 5.0))
        self._range_noise_std = float(noise_cfg.get("range_noise_std_m", 20.0))
        self._angle_noise_std = float(np.deg2rad(noise_cfg.get("angle_noise_std_deg", 1.0)))

        # ---- Actuator delay ----
        delay_cfg = self._config.get("actuator_delay", {})
        self._delay_enabled = bool(delay_cfg.get("enabled", True))
        self._min_delay = int(delay_cfg.get("min_delay_steps", 0))
        self._max_delay = int(delay_cfg.get("max_delay_steps", 2))
        self._delay_buffer: deque = deque()
        self._current_delay: int = 0

        # ---- RNG ----
        self._rng = np.random.default_rng(42)

        logger.info(
            "PerturbedJSBSimEnv initialized: scale=%.2f wind=%s noise=%s delay=%s adversarial=%s",
            self.scale,
            self._wind_enabled,
            self._noise_enabled,
            self._delay_enabled,
            self._is_adversarial,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_steps(self) -> int:
        return getattr(self._env, "max_steps", 512)

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, scenario=None, seed=None) -> Any:
        """Reset environment with fresh perturbation state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset delay buffer
        self._delay_buffer.clear()
        self._current_delay = self._rng.integers(
            self._min_delay, self._max_delay + 1
        ) if self._delay_enabled else 0

        # Fill delay buffer with neutral actions.
        # ``_current_delay`` neutral entries guarantees that the first
        # ``_current_delay`` executed steps are neutral, after which the
        # originally submitted actions begin to emerge from the FIFO.
        neutral = np.zeros(3) if not self._is_adversarial else (np.zeros(3), np.zeros(3))
        for _ in range(self._current_delay):
            self._delay_buffer.append(neutral)

        obs = self._env.reset(scenario=scenario, seed=seed)
        if self._is_adversarial:
            p_obs, t_obs = obs
            return (
                self._apply_sensor_noise(p_obs),
                self._apply_sensor_noise(t_obs),
            )
        return self._apply_sensor_noise(obs)

    def step(self, action):
        """
        Execute one step with perturbations.

        For single-agent: ``action`` is np.ndarray [3].
        For adversarial: ``action`` is tuple (pursuer_action, target_action).
        """
        # Apply wind before stepping
        if self._wind_enabled and self.scale > 0.0:
            self._apply_wind()

        # Apply actuator delay
        if self._delay_enabled and self.scale > 0.0:
            self._delay_buffer.append(action)
            delayed_action = self._delay_buffer.popleft()
        else:
            delayed_action = action

        # Step the base environment
        if self._is_adversarial:
            result = self._env.step(*delayed_action)
        else:
            result = self._env.step(delayed_action)

        # Unpack result based on environment type
        if self._is_adversarial:
            p_obs, t_obs, p_rew, t_rew, terminated, truncated, info = result
            p_obs = self._apply_sensor_noise(p_obs)
            t_obs = self._apply_sensor_noise(t_obs)
            return (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info)
        else:
            obs, reward, terminated, truncated, info = result
            obs = self._apply_sensor_noise(obs)
            return (obs, reward, terminated, truncated, info)

    def close(self) -> None:
        """Release resources."""
        if hasattr(self._env, "close"):
            self._env.close()

    # ------------------------------------------------------------------
    # Perturbation methods
    # ------------------------------------------------------------------

    def _apply_wind(self) -> None:
        """Apply random wind gust to JSBSim atmosphere."""
        # Target the base JSBSim env through the environment hierarchy
        jsbsim_env = self._find_jsbsim_env()
        if jsbsim_env is None:
            return

        # Sample wind components (N, E, D) in m/s
        sigma = self._wind_speed_std * self.scale
        wind_n = self._rng.normal(0.0, sigma)
        wind_e = self._rng.normal(0.0, sigma)
        wind_d = self._rng.normal(0.0, sigma * 0.3)  # vertical wind is weaker

        fps = 1.0 / 0.3048  # m/s to ft/s
        wind_n_fps = wind_n * fps
        wind_e_fps = wind_e * fps
        wind_d_fps = -wind_d * fps  # JSBSim down is negative

        # Set on all aircraft
        for uid, ac in jsbsim_env._aircraft.items():
            try:
                ac.set_property_value(
                    "atmosphere/total-wind-north-fps", wind_n_fps
                )
                ac.set_property_value(
                    "atmosphere/total-wind-east-fps", wind_e_fps
                )
                ac.set_property_value(
                    "atmosphere/total-wind-down-fps", wind_d_fps
                )
            except Exception as exc:
                logger.warning(
                    "Failed to set wind property for %s: %s", uid, exc
                )

    def _apply_sensor_noise(self, obs: Any) -> Any:
        """Add component-wise Gaussian noise scaled to physical standard deviations."""
        if not self._noise_enabled or self.scale <= 0.0:
            return obs

        if isinstance(obs, dict) and "observation_vector" in obs:
            vec = obs["observation_vector"].copy()
            rng = self._rng
            s = self.scale

            # Reference normalisation values (same order as the observation vector)
            ref_range = 3000.0
            ref_range_rate = 200.0
            ref_alt = 10000.0
            ref_speed = 400.0

            # 0: range
            vec[0] += rng.normal(0.0, s * self._range_noise_std / ref_range)
            # 1: range_rate
            vec[1] += rng.normal(0.0, s * self._vel_noise_std / ref_range_rate)
            # 2: altitude_diff
            vec[2] += rng.normal(0.0, s * self._pos_noise_std / ref_alt)
            # 3: speed_diff
            vec[3] += rng.normal(0.0, s * self._vel_noise_std / ref_speed)

            # 4-5: los_azimuth sin/cos
            az = math.atan2(float(vec[4]), float(vec[5]))
            az += rng.normal(0.0, s * self._angle_noise_std)
            vec[4] = math.sin(az)
            vec[5] = math.cos(az)

            # 6-7: los_elevation sin/cos
            el = math.atan2(float(vec[6]), float(vec[7]))
            el += rng.normal(0.0, s * self._angle_noise_std)
            vec[6] = math.sin(el)
            vec[7] = math.cos(el)

            # 8-9: ata sin/cos
            ata = math.atan2(float(vec[8]), float(vec[9]))
            ata += rng.normal(0.0, s * self._angle_noise_std)
            vec[8] = math.sin(ata)
            vec[9] = math.cos(ata)

            # 10-11: aa sin/cos
            aa = math.atan2(float(vec[10]), float(vec[11]))
            aa += rng.normal(0.0, s * self._angle_noise_std)
            vec[10] = math.sin(aa)
            vec[11] = math.cos(aa)

            # 12: own_speed
            vec[12] += rng.normal(0.0, s * self._vel_noise_std / ref_speed)
            # 13: target_speed
            vec[13] += rng.normal(0.0, s * self._vel_noise_std / ref_speed)
            # 14: own_altitude
            vec[14] += rng.normal(0.0, s * self._pos_noise_std / ref_alt)
            # 15: target_altitude
            vec[15] += rng.normal(0.0, s * self._pos_noise_std / ref_alt)

            return {**obs, "observation_vector": vec, "noise_applied": True}

        return obs

    def _find_jsbsim_env(self):
        """Walk the environment hierarchy to find the JSBSimEnv instance."""
        env = self._env
        for _ in range(5):
            if hasattr(env, "jsbsim_env"):
                return env.jsbsim_env
            if hasattr(env, "_env"):
                env = env._env
            else:
                break
        return None


def set_perturbation_scale(env: PerturbedJSBSimEnv, scale: float) -> None:
    """Convenience function to update perturbation scale."""
    env.scale = float(scale)
