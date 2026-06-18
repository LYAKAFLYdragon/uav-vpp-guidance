"""
Numerical Jacobian estimators for the CBF safety filter.

The default estimator builds a local :class:`SimplePointMassEnv`, drives it
with the same VPP -> LOS-rate-guidance pipeline used by the high-level
environment, and measures how the pursuer velocity changes when each action
dimension is perturbed.  This keeps the CBF filter stateless with respect to
the real JSBSim backend while still capturing the control influence of the
VPP offset.
"""

from typing import Any, Dict

import numpy as np


class PointMassJacobianEstimator:
    """
    Estimate ``K = ∂v_o / ∂a`` by finite differences on a local point-mass
    copy of the current state.

    The estimator reuses the host environment's virtual-point generator and
    guidance law (if available) so that the Jacobian reflects the actual
    policy-to-command mapping, then propagates the command through a
    :class:`SimplePointMassEnv` to obtain velocity changes.
    """

    def __call__(
        self,
        env: Any,
        state: Dict[str, Any],
        base_action: np.ndarray,
        eps: float,
        dt: float,
    ) -> np.ndarray:
        """
        Estimate the 3x3 Jacobian ``K``.

        Args:
            env: The host environment (``CloseRangeTrackingEnv`` or
                ``AdversarialJSBSimEnv``).
            state: CBF state dict with ``pursuer``/``target`` or
                ``own``/``target`` containing ``position`` and ``velocity``.
            base_action: Nominal action in [-1, 1]^3.
            eps: Perturbation size for finite differences.
            dt: Decision interval (s).

        Returns:
            ``K`` as a (3, 3) ndarray.
        """
        own_state, target_state = self._extract_env_states(state)
        own_pos = np.asarray(own_state["position_m"], dtype=np.float64)
        own_vel = np.asarray(own_state["velocity_vector_mps"], dtype=np.float64)
        target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
        target_vel = np.asarray(target_state["velocity_vector_mps"], dtype=np.float64)

        # Build a local point-mass environment seeded with the current geometry.
        pm_env = self._build_point_mass_env(env, dt)
        self._init_point_mass(pm_env, own_pos, own_vel, target_pos, target_vel)

        # Base velocity.
        v_base = self._simulate_action(pm_env, env, own_state, target_state, base_action, dt)

        K = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            self._init_point_mass(pm_env, own_pos, own_vel, target_pos, target_vel)
            a_perturbed = np.asarray(base_action, dtype=np.float64).copy()
            a_perturbed[j] += eps
            v_perturbed = self._simulate_action(
                pm_env, env, own_state, target_state, a_perturbed, dt
            )
            # K[:, j] = Δv / (eps * dt)  -> acceleration sensitivity.
            K[:, j] = (v_perturbed - v_base) / (eps * dt)

        return K

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_env_states(state: Dict[str, Any]) -> tuple:
        """Convert CBF state dict into own/target state dicts."""
        if "pursuer" in state and "target" in state:
            own = state["pursuer"]
            target = state["target"]
        elif "own" in state and "target" in state:
            own = state["own"]
            target = state["target"]
        else:
            own = state
            target = state

        def _to_dict(obj, kind: str) -> Dict[str, Any]:
            pos = obj.get("position") if isinstance(obj, dict) else getattr(obj, "position", None)
            vel = obj.get("velocity") if isinstance(obj, dict) else getattr(obj, "velocity", None)
            pos = np.asarray(pos, dtype=np.float64)
            vel = np.asarray(vel, dtype=np.float64)
            speed = float(np.linalg.norm(vel))
            heading = float(np.arctan2(vel[1], vel[0]))
            horizontal = float(np.hypot(vel[0], vel[1]))
            pitch = float(np.arctan2(vel[2], horizontal + 1e-12))
            return {
                "position_m": pos,
                "velocity_vector_mps": vel,
                "speed_mps": speed,
                "heading_rad": heading,
                "pitch_rad": pitch,
                "roll_rad": 0.0,
                "yaw_rad": heading,
                "altitude_m": float(pos[2]),
                "nz": 1.0,
                "uid": kind,
            }

        return _to_dict(own, "own"), _to_dict(target, "target")

    def _build_point_mass_env(self, env: Any, dt: float) -> Any:
        """Create a fresh SimplePointMassEnv with the host environment's config."""
        # Import here to avoid circular dependencies at module load time.
        from ..envs.simple_point_mass_env import SimplePointMassEnv

        config = getattr(env, "env_config", {}) if hasattr(env, "env_config") else {}
        # Make a shallow copy and force the decision interval.
        cfg = dict(config)
        cfg["decision_freq"] = int(round(1.0 / dt)) if dt > 0 else 5
        return SimplePointMassEnv(cfg)

    @staticmethod
    def _init_point_mass(
        pm_env: Any,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> None:
        """Reset the local point-mass env to the current geometry."""
        own_init = {
            "position_m": own_pos.copy(),
            "velocity_vector_mps": own_vel.copy(),
            "heading_rad": float(np.arctan2(own_vel[1], own_vel[0])),
            "altitude_m": float(own_pos[2]),
            "roll_rad": 0.0,
            "pitch_rad": 0.0,
            "yaw_rad": float(np.arctan2(own_vel[1], own_vel[0])),
            "nz": 1.0,
        }
        target_init = {
            "position_m": target_pos.copy(),
            "velocity_vector_mps": target_vel.copy(),
            "heading_rad": float(np.arctan2(target_vel[1], target_vel[0])),
            "altitude_m": float(target_pos[2]),
        }
        pm_env.reset(own_init=own_init, target_init=target_init)

    def _simulate_action(
        self,
        pm_env: Any,
        env: Any,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
        action: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute the physical guidance command for ``action`` using the host
        environment's VPP/guidance pipeline and advance the point-mass env
        by one decision step.
        """
        command = self._compute_guidance_command(env, own_state, target_state, action)
        pm_env.step(own_command=command, target_command=None)
        return np.asarray(pm_env.own_state["velocity_vector_mps"], dtype=np.float64).copy()

    @staticmethod
    def _compute_guidance_command(
        env: Any,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
        action: np.ndarray,
    ) -> Dict[str, float]:
        """Reproduce the VPP -> LOS-rate-guidance command for an action."""
        from ..flight_control.command_limiter import clip_command

        action = np.asarray(action, dtype=np.float64)
        own_pos = np.asarray(own_state["position_m"], dtype=np.float64)
        own_vel = np.asarray(own_state["velocity_vector_mps"], dtype=np.float64)
        target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
        target_vel = np.asarray(target_state["velocity_vector_mps"], dtype=np.float64)

        # Build state dicts compatible with the guidance pipeline.
        own = {
            "position_m": own_pos,
            "position_neu": own_pos,
            "velocity_vector_mps": own_vel,
            "velocity_ned": np.array([own_vel[0], own_vel[1], -own_vel[2]]),
            "altitude_m": float(own_pos[2]),
            "speed_mps": float(np.linalg.norm(own_vel)),
        }
        target = {
            "position_m": target_pos,
            "position_neu": target_pos,
            "velocity_vector_mps": target_vel,
            "velocity_ned": np.array([target_vel[0], target_vel[1], -target_vel[2]]),
            "altitude_m": float(target_pos[2]),
            "speed_mps": float(np.linalg.norm(target_vel)),
        }

        cfg = env.config if hasattr(env, "config") else {}

        # AdversarialJSBSimEnv branch.
        if hasattr(env, "_compute_virtual_point"):
            vp = env._compute_virtual_point(action, own, target)
            raw_cmd = env._pursuer_guidance.compute_command(
                own, target, vp, env._pursuer_gains
            )
        else:
            # CloseRangeTrackingEnv branch.
            vp_config = cfg.get(
                "virtual_point", cfg.get("guidance", {}).get("virtual_point", {})
            )
            anchor_mode = vp_config.get("anchor_mode", "current_target")
            target_for_vp = {
                "position_neu": target_pos,
                "velocity_vector_mps": target_vel,
            }
            initial_range_m = getattr(env, "_initial_range_m", None)
            try:
                vp_result = env.virtual_point_generator.action_to_virtual_point(
                    action,
                    own,
                    target_for_vp,
                    anchor_mode=anchor_mode,
                    return_info=True,
                    initial_range_m=initial_range_m,
                )
            except TypeError:
                # Older generator signatures may not accept return_info.
                vp_result = (
                    env.virtual_point_generator.action_to_virtual_point(
                        action, own, target_for_vp, anchor_mode=anchor_mode
                    ),
                    {},
                )
            if isinstance(vp_result, tuple):
                vp, _ = vp_result
            else:
                vp = vp_result
            raw_cmd = env.guidance.compute_command(
                own, target, vp, env.current_gains
            )

        limits = cfg.get("limits", {}) if hasattr(cfg, "get") else {}
        return clip_command(raw_cmd, limits)
