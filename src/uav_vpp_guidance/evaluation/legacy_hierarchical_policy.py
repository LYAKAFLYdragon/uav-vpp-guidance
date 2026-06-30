"""Bridge wrapper for the legacy hierarchical pursuit-strategy PPO baseline.

This adapter loads the previously published SB3 PPO checkpoint that outputs one
of three discrete pursuit strategies:

- ``0``: lag
- ``1``: lead
- ``2``: pure

At evaluation time, the adapter reconstructs the legacy 16-D high-level
observation directly from the current simulator state, runs the frozen discrete
policy, and then translates the selected strategy into a reference point that
is compatible with the current comparison loop.

The bridge intentionally reuses the current environment's guidance and
low-level controller stack after strategy selection. This keeps the evaluation
inside the current paper protocol while preserving the old high-level action
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from uav_vpp_guidance.envs.observation import compute_relative_geometry


def _safe_norm(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(x) + eps)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _extract_position(state: Dict[str, Any]) -> np.ndarray:
    for key in ("position_neu", "position_m", "position"):
        pos = state.get(key)
        if pos is not None:
            arr = np.asarray(pos, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(f"{key} must be a 3-vector, got {arr.shape}")
            return arr
    raise ValueError("State is missing position_neu / position_m / position")


def _extract_velocity(state: Dict[str, Any]) -> np.ndarray:
    vel = state.get("velocity_vector_mps")
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"velocity_vector_mps must be a 3-vector, got {arr.shape}"
            )
        return arr

    vel_ned = state.get("velocity_ned")
    if vel_ned is not None:
        arr = np.asarray(vel_ned, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"velocity_ned must be a 3-vector, got {arr.shape}")
        return np.array([arr[0], arr[1], -arr[2]], dtype=np.float64)

    vel = state.get("velocity")
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"velocity must be a 3-vector, got {arr.shape}")
        return arr

    raise ValueError(
        "State is missing velocity_vector_mps / velocity_ned / velocity"
    )


def _extract_speed(state: Dict[str, Any], velocity_neu: np.ndarray) -> float:
    for key in ("vt_mps", "speed_mps"):
        value = state.get(key)
        if value is not None:
            value_f = float(value)
            if np.isfinite(value_f):
                return value_f
    return float(np.linalg.norm(velocity_neu))


def _extract_angle(state: Dict[str, Any], key: str) -> float:
    value = state.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def body_to_inertial_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Legacy body-to-inertial rotation order used by the Aerospace baseline."""
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def build_legacy_high_level_obs(
    own_state: Dict[str, Any],
    target_state: Dict[str, Any],
) -> np.ndarray:
    """Reconstruct the legacy 16-D high-level observation from simulator state."""
    own_pos = _extract_position(own_state)
    target_pos = _extract_position(target_state)
    own_vel = _extract_velocity(own_state)
    target_vel = _extract_velocity(target_state)
    rel = target_pos - own_pos

    range_m = _safe_norm(rel)
    own_xy = own_vel[:2]
    rel_xy = rel[:2]

    phi = np.arccos(
        np.clip(
            np.dot(own_xy, rel_xy) / (_safe_norm(own_xy) * _safe_norm(rel_xy)),
            -1.0,
            1.0,
        )
    )
    if np.cross(own_xy, rel_xy) < 0.0:
        phi = -phi

    q_vec = -rel[:2]
    q = np.arccos(
        np.clip(
            np.dot(target_vel[:2], q_vec)
            / (_safe_norm(target_vel[:2]) * _safe_norm(q_vec)),
            -1.0,
            1.0,
        )
    )

    own_speed = _extract_speed(own_state, own_vel)
    target_speed = _extract_speed(target_state, target_vel)

    own_roll = _extract_angle(own_state, "roll_rad")
    own_pitch = _extract_angle(own_state, "pitch_rad")
    own_yaw = _extract_angle(own_state, "yaw_rad")
    target_roll = _extract_angle(target_state, "roll_rad")
    target_pitch = _extract_angle(target_state, "pitch_rad")
    target_yaw = _extract_angle(target_state, "yaw_rad")

    obs = np.array(
        [
            np.clip(range_m / 10000.0, 0.0, 5.0),
            _wrap_angle(phi),
            _wrap_angle(q),
            np.clip((target_pos[2] - own_pos[2]) / 5000.0, -5.0, 5.0),
            np.clip(own_speed / 340.0, 0.0, 5.0),
            np.clip(target_speed / 340.0, 0.0, 5.0),
            np.sin(own_yaw),
            np.cos(own_yaw),
            np.sin(target_yaw),
            np.cos(target_yaw),
            np.sin(own_pitch),
            np.cos(own_pitch),
            np.sin(target_pitch),
            np.cos(target_pitch),
            np.sin(own_roll),
            np.sin(target_roll),
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, 10.0)


@dataclass
class LegacyStrategyBridgeConfig:
    """Minimal legacy guidance geometry needed by the bridge."""

    bullet_speed: float = 1000.0
    lag_distance: float = 500.0
    max_lead_time_s: float = 5.0


def build_legacy_strategy_reference_point(
    strategy: int,
    own_state: Dict[str, Any],
    target_state: Dict[str, Any],
    cfg: LegacyStrategyBridgeConfig,
) -> np.ndarray:
    """Map a legacy discrete strategy into a reference point in NEU coordinates."""
    own_pos = _extract_position(own_state)
    target_pos = _extract_position(target_state)
    own_vel = _extract_velocity(own_state)
    target_vel = _extract_velocity(target_state)
    target_speed = _extract_speed(target_state, target_vel)

    if strategy == 0:
        target_roll = _extract_angle(target_state, "roll_rad")
        target_pitch = _extract_angle(target_state, "pitch_rad")
        target_yaw = _extract_angle(target_state, "yaw_rad")
        rotation = body_to_inertial_rotation(target_roll, target_pitch, target_yaw)
        return target_pos + rotation @ np.array(
            [-cfg.lag_distance, 0.0, 0.0],
            dtype=np.float64,
        )

    if strategy == 1:
        rel = target_pos - own_pos
        range_3d = _safe_norm(rel)
        phi = np.arccos(
            np.clip(
                np.dot(own_vel[:2], rel[:2])
                / (_safe_norm(own_vel[:2]) * _safe_norm(rel[:2])),
                -1.0,
                1.0,
            )
        )
        denom = cfg.bullet_speed + max(target_speed * np.cos(phi), 1.0)
        lead_time_s = float(np.clip(range_3d / denom, 0.0, cfg.max_lead_time_s))
        cmd_pos = target_pos + target_vel * lead_time_s
        cmd_pos = np.asarray(cmd_pos, dtype=np.float64).copy()
        cmd_pos[2] -= 0.5 * 9.81 * lead_time_s * lead_time_s
        return cmd_pos

    if strategy == 2:
        return target_pos.copy()

    raise ValueError(f"Unknown legacy strategy index: {strategy}")


class LegacyHierarchicalPolicy:
    """SB3 PPO policy wrapper for the legacy hierarchical discrete baseline."""

    STRATEGY_NAMES = {
        0: "lag",
        1: "lead",
        2: "pure",
    }

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        guidance_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.current_task_name: Optional[str] = None
        self.env = None
        self.bridge_config = LegacyStrategyBridgeConfig(
            **(guidance_config or {})
        )
        self._last_step_metadata: Dict[str, Any] = {}
        self._model = self._load_model(Path(self.checkpoint_path), device)

    @staticmethod
    def _load_model(checkpoint_path: Path, device: str):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "LegacyHierarchicalPolicy requires stable_baselines3 to load "
                f"the SB3 PPO checkpoint at {checkpoint_path}. "
                "Install it before running the legacy bridge."
            ) from exc
        return PPO.load(str(checkpoint_path), device=device)

    def set_task_name(self, task_name: str) -> None:
        self.current_task_name = str(task_name)

    def set_env(self, env: Any) -> None:
        self.env = env

    def _get_current_states(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(
                "LegacyHierarchicalPolicy.env is not set. Call set_env() first."
            )
        if hasattr(self.env, "_get_current_states"):
            return self.env._get_current_states(noisy=False)
        raise RuntimeError(
            "Environment does not expose _get_current_states(), "
            "so the legacy bridge cannot reconstruct its observation."
        )

    def _predict_strategy(self) -> tuple[int, np.ndarray]:
        own_state, target_state = self._get_current_states()
        legacy_obs = build_legacy_high_level_obs(own_state, target_state)
        action, _ = self._model.predict(legacy_obs, deterministic=True)
        strategy = int(np.asarray(action).reshape(-1)[0])
        return strategy, legacy_obs

    def _compute_command_override(
        self,
        strategy: int,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
    ) -> Dict[str, float]:
        if self.env is None:
            raise RuntimeError(
                "LegacyHierarchicalPolicy.env is not set. Call set_env() first."
            )
        if not hasattr(self.env, "guidance"):
            raise RuntimeError("Environment does not expose a guidance object.")

        reference_point = build_legacy_strategy_reference_point(
            strategy,
            own_state,
            target_state,
            self.bridge_config,
        )
        virtual_point = {
            "position_neu": np.asarray(reference_point, dtype=np.float64),
        }
        raw_command = self.env.guidance.compute_command(
            own_state,
            target_state,
            virtual_point,
            self.env.current_gains,
        )
        if getattr(self.env, "command_post_processor", None) is not None:
            rel_geom = compute_relative_geometry(own_state, target_state)
            raw_command = self.env.command_post_processor.process(
                raw_command,
                own_state=own_state,
                target_state=target_state,
                relative_state=rel_geom,
            )
        self._last_step_metadata = {
            "legacy_strategy_index": int(strategy),
            "legacy_strategy_name": self.STRATEGY_NAMES.get(strategy, "unknown"),
            "legacy_bridge_reference_point_x": float(reference_point[0]),
            "legacy_bridge_reference_point_y": float(reference_point[1]),
            "legacy_bridge_reference_point_z": float(reference_point[2]),
            "legacy_bridge_command_override_active": True,
        }
        return {
            "nz_cmd": float(raw_command["nz_cmd"]),
            "roll_rate_cmd": float(raw_command["roll_rate_cmd"]),
            "throttle_cmd": float(raw_command["throttle_cmd"]),
        }

    def get_env_step_kwargs(self, obs: np.ndarray) -> Dict[str, Any]:
        del obs
        strategy, legacy_obs = self._predict_strategy()
        own_state, target_state = self._get_current_states()
        command_override = self._compute_command_override(
            strategy,
            own_state,
            target_state,
        )
        self._last_step_metadata["legacy_bridge_obs_dim"] = int(legacy_obs.shape[0])
        self._last_step_metadata["legacy_bridge_task_name"] = self.current_task_name
        return {
            "command_override": command_override,
        }

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Compatibility method for call sites that still expect an action."""
        _ = self.get_env_step_kwargs(obs)
        return np.zeros(3, dtype=np.float32)

    def load(self, path: str) -> None:
        """Compatibility no-op: the SB3 model is already loaded in __init__."""
        del path
