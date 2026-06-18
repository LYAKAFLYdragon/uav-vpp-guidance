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
    Estimate ``K = ∂a_o / ∂a`` by finite differences on a local point-mass
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
    def _compute_raw_guidance_command(
        env: Any,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
        action: np.ndarray,
    ) -> Dict[str, float]:
        """Reproduce the VPP -> LOS-rate-guidance *raw* command (unclipped)."""
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

        return raw_cmd

    @staticmethod
    def _compute_guidance_command(
        env: Any,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
        action: np.ndarray,
    ) -> Dict[str, float]:
        """Reproduce the VPP -> LOS-rate-guidance command for an action."""
        from ..flight_control.command_limiter import clip_command

        raw_cmd = PointMassJacobianEstimator._compute_raw_guidance_command(
            env, own_state, target_state, action
        )
        cfg = env.config if hasattr(env, "config") else {}
        limits = cfg.get("limits", {}) if hasattr(cfg, "get") else {}
        return clip_command(raw_cmd, limits)


class JSBSimFiniteDifferenceJacobianEstimator:
    """
    Estimate ``K = ∂a_o / ∂a`` by finite differences directly inside JSBSim.

    This estimator rebuilds a sandbox JSBSim environment seeded to the current
dynamic state of the host environment, computes the guidance command for the
    nominal (and perturbed) VPP action, and steps the sandbox for one decision
    interval.  It is much slower than the point-mass estimator but captures the
    real F-16 response, eliminating the model-mismatch problem.

    Notes:
        - A fresh JSBSim instance is used for every evaluation.  This gives
          deterministic restarts and avoids the save/restore problem.
        - Engine / throttle states are restored after ``run_ic`` so that the
          sandbox starts from a realistic power state.
        - The target aircraft is replayed with its current control surface
          commands held constant, so its short-horizon trajectory is identical
          across the base and perturbed roll-outs.
    """

    def __call__(
        self,
        env: Any,
        state: Dict[str, Any],
        base_action: np.ndarray,
        eps: float,
        dt: float,
    ) -> np.ndarray:
        """Estimate the 3x3 Jacobian ``K``."""
        own_state, target_state = PointMassJacobianEstimator._extract_env_states(state)

        # Build sandbox once and reuse it for base + perturbed roll-outs.
        sandbox = self._build_sandbox(env)
        snapshot = self._capture_host_jsbsim(env)

        v_base = self._simulate_action(
            sandbox, snapshot, env, own_state, target_state, base_action, dt
        )

        K = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            a_perturbed = np.asarray(base_action, dtype=np.float64).copy()
            a_perturbed[j] += eps
            v_perturbed = self._simulate_action(
                sandbox, snapshot, env, own_state, target_state, a_perturbed, dt
            )
            # Acceleration sensitivity.
            K[:, j] = (v_perturbed - v_base) / (eps * dt)

        sandbox.close()
        return K

    # ------------------------------------------------------------------
    # Sandbox helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sandbox(env: Any) -> Any:
        """Create a fresh JSBSimEnv mirroring the host configuration."""
        from ..envs.jsbsim_env import JSBSimEnv

        host_env_config = getattr(env, "env_config", {}) if hasattr(env, "env_config") else {}
        # Make a shallow copy so we do not mutate the host config.
        cfg = dict(host_env_config)
        cfg.setdefault("jsbsim_data_dir", None)
        cfg.setdefault("sim_freq", getattr(env, "sim_freq", 60))

        sandbox = JSBSimEnv(cfg)

        # Use the same UIDs as the host so the snapshot maps directly.
        own_uid = getattr(env, "own_uid", None) or getattr(env, "_pursuer_uid", "own")
        target_uid = getattr(env, "target_uid", None) or getattr(env, "_target_uid", "target")
        own_model = getattr(env, "aircraft_model", "f16")
        sandbox.add_aircraft(own_uid, {"model": own_model})
        sandbox.add_aircraft(target_uid, {"model": "f16"})
        return sandbox

    @staticmethod
    def _capture_host_jsbsim(env: Any) -> Dict[str, Dict[str, float]]:
        """Capture the current JSBSim state from the host environment."""
        from .jsbsim_snapshot import capture_jsbsim_env

        return capture_jsbsim_env(env.jsbsim_env)

    def _simulate_action(
        self,
        sandbox: Any,
        snapshot: Dict[str, Dict[str, float]],
        env: Any,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
        action: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Reset sandbox to snapshot, apply action, step, and return own velocity."""
        from .jsbsim_snapshot import apply_jsbsim_env

        # Restore all aircraft to the captured dynamic state.
        apply_jsbsim_env(sandbox, snapshot)

        own_uid = getattr(env, "own_uid", None) or getattr(env, "_pursuer_uid", "own")
        target_uid = getattr(env, "target_uid", None) or getattr(env, "_target_uid", "target")

        # Compute the guidance command for this action using the host pipeline.
        command = PointMassJacobianEstimator._compute_guidance_command(
            env, own_state, target_state, action
        )

        # Compute JSBSim actuator properties from the command.
        own_full_state = sandbox.get_state()[own_uid]
        own_props = self._command_to_jsbsim_props(env, command, own_full_state, pursuer=True)

        # Hold target control surfaces at their captured values so the target
        # trajectory is identical across roll-outs.
        target_props = self._capture_target_props(env)

        # Determine how many JSBSim integration steps correspond to one decision.
        sim_steps = self._sim_steps_per_decision(env)

        control_inputs = {own_uid: own_props, target_uid: target_props}
        for _ in range(sim_steps):
            sandbox.step(control_inputs)
            control_inputs = None

        return np.asarray(sandbox.get_state()[own_uid]["velocity_vector_mps"], dtype=np.float64).copy()

    @staticmethod
    def _command_to_jsbsim_props(
        env: Any,
        command: Dict[str, float],
        own_state: Dict[str, Any],
        pursuer: bool = True,
    ) -> Dict[str, float]:
        """Map a high-level command to JSBSim fcs properties."""
        if pursuer:
            llc = getattr(env, "_low_level_controller", None) or getattr(env, "_pursuer_llc")
        else:
            llc = getattr(env, "_target_llc", None)
        actuator_output = llc.compute_actuator(command, own_state)
        return {k: v for k, v in actuator_output.items() if k.startswith("fcs/")}

    @staticmethod
    def _capture_target_props(env: Any) -> Dict[str, float]:
        """Read the target's current fcs commands from the host JSBSim instance."""
        target_uid = getattr(env, "target_uid", None) or getattr(env, "_target_uid", "target")
        exec_ = env.jsbsim_env._aircraft[target_uid].jsbsim_exec
        props = {}
        for name in [
            "fcs/elevator-cmd-norm",
            "fcs/aileron-cmd-norm",
            "fcs/rudder-cmd-norm",
            "fcs/throttle-cmd-norm",
        ]:
            try:
                props[name] = float(exec_.get_property_value(name))
            except Exception:
                props[name] = 0.0
        return props

    @staticmethod
    def _sim_steps_per_decision(env: Any) -> int:
        """Resolve the number of JSBSim steps per high-level decision."""
        if hasattr(env, "_sim_steps_per_decision"):
            return int(env._sim_steps_per_decision)
        sim_freq = float(getattr(env, "sim_freq", 60))
        decision_freq = float(getattr(env, "decision_freq", 5))
        return max(1, int(round(sim_freq / decision_freq)))
