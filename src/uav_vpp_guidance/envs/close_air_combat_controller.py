"""Adapter that turns a CloseAirCombat_control hierarchical RL agent into a bandit controller.

The wrapped agent has three layers:
    1. High-level SB3 PPO -> discrete pursuit intent {lag, lead, pure}.
    2. TacticalGuidance -> [delta_altitude, delta_heading, delta_velocity].
    3. LowLevelPPOController (recurrent actor) -> 4-D actuator rates.

The adapter integrates those rates each JSBSim sub-step exactly like the original
CloseAirCombat_control environment and outputs absolute normalized surface commands.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

from ..maneuver_library.base import ControlCommand


class _AgentView:
    """Minimal aircraft view that satisfies CloseAirCombat_control's env.agents[uid] API."""

    def __init__(self, aircraft: Any, state: Dict[str, Any]):
        self._aircraft = aircraft
        self._state = state
        self._roll = float(state.get("roll_rad", state.get("attitude_rpy", [0.0, 0.0, 0.0])[0]))
        self._pitch = float(
            state.get("pitch_rad", state.get("attitude_rpy", [0.0, 0.0, 0.0])[1])
        )
        self._yaw = float(state.get("yaw_rad", state.get("attitude_rpy", [0.0, 0.0, 0.0])[2]))
        self._vel_body = np.asarray(
            state.get("velocity_body", np.zeros(3)), dtype=np.float64
        )

    def get_position(self) -> np.ndarray:
        # CloseAirCombat_control uses NEU internally; our state already stores NEU.
        return np.asarray(self._state.get("position_neu", np.zeros(3)), dtype=np.float64)

    def get_velocity(self) -> np.ndarray:
        # Horizontal components are used for aspect/target-angle calculations.
        return np.asarray(self._state.get("velocity_vector_mps", np.zeros(3)), dtype=np.float64)

    def get_rpy(self) -> Tuple[float, float, float]:
        return (self._roll, self._pitch, self._yaw)

    def get_property_value(self, prop: Any) -> float:
        """Resolve a CloseAirCombat_control Catalog property from our JSBSim state."""
        name = prop.name_jsbsim if hasattr(prop, "name_jsbsim") else str(prop)

        # ExtraCatalog properties computed from our unified state dict.
        if name == "position/h-sl-m":
            return float(self._state.get("altitude_m", 0.0))
        if name == "velocities/vt-mps":
            return float(self._state.get("speed_mps", 0.0))
        if name == "velocities/u-mps":
            return float(self._vel_body[0])
        if name == "velocities/v-mps":
            return float(self._vel_body[1])
        if name == "velocities/w-mps":
            return float(self._vel_body[2])
        if name == "velocities/vc-mps":
            try:
                return float(self._aircraft.get_property_value("velocities/vc-fps")) * 0.3048
            except Exception:
                return float(self._state.get("speed_mps", 0.0))
        if name == "attitude/roll-rad":
            return self._roll
        if name == "attitude/pitch-rad":
            return self._pitch
        if name == "attitude/heading-true-rad":
            return self._yaw

        # These ExtraCatalog delta properties are only placeholders in the
        # low-level observation vector (their slots are overwritten by the
        # guidance setpoint), so returning zero is safe.
        if name in (
            "position/delta-altitude-to-target-m",
            "position/delta-heading-to-target-deg",
            "position/delta-velocities_u-to-target-mps",
        ):
            return 0.0

        # Standard JSBSim properties (controls, etc.).
        try:
            return float(self._aircraft.get_property_value(name))
        except Exception as exc:
            raise RuntimeError(
                f"CloseAirCombat controller cannot read JSBSim property: {name}"
            ) from exc

    def get_property_values(self, props):
        return [self.get_property_value(p) for p in props]


class _EnvView:
    """Lightweight env view with the `agents` dict expected by CAC helpers."""

    def __init__(self, bandit_aircraft: Any, bandit_state: Dict[str, Any],
                 own_aircraft: Any, own_state: Dict[str, Any]):
        # Insert bandit first so ego_id=0 always refers to the bandit/target.
        self.agents = {
            "bandit": _AgentView(bandit_aircraft, bandit_state),
            "own": _AgentView(own_aircraft, own_state),
        }


class CloseAirCombatBanditController:
    """Bandit controller powered by the hierarchical CloseAirCombat_control agent.

    Config fields (under ``bandit.close_air_combat``):
        project_root (str): Path to CloseAirCombat_control root.
        high_level_model (str): Relative or absolute path to ``proposed_ppo.zip``.
        low_level_actor (str): Relative or absolute path to ``actor_latest.pt``.
        device (str): "cpu" or "cuda".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        jsbsim_env: Any,
        target_uid: str,
        own_uid: str,
    ):
        self.config = config or {}
        self.project_root = os.path.abspath(
            self.config.get("project_root", r"E:\CloseAirCombat_control")
        )
        if not os.path.isdir(self.project_root):
            raise FileNotFoundError(
                f"CloseAirCombat_control project root not found: {self.project_root}"
            )

        self.high_level_model_path = self._resolve_path(
            self.config.get("high_level_model", "baseline_results/checkpoints/proposed_ppo.zip")
        )
        self.low_level_actor_path = self._resolve_path(
            self.config.get(
                "low_level_actor",
                "results/SingleControl/1/heading/ppo/v1/wandb/run-20240530_084912-70pgl4xt/files/actor_latest.pt",
            )
        )
        self.device = str(self.config.get("device", "cpu"))

        self.jsbsim_env = jsbsim_env
        self.target_uid = target_uid
        self.own_uid = own_uid

        # Ensure CAC modules are importable.
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

        try:
            import run_table5_r3_7_complete as cac
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import CloseAirCombat_control modules from {self.project_root}. "
                "Please verify the project root and required dependencies."
            ) from exc

        # Load high-level discrete PPO (lag/lead/pure).
        from stable_baselines3 import PPO

        if not os.path.isfile(self.high_level_model_path):
            raise FileNotFoundError(f"High-level PPO model not found: {self.high_level_model_path}")
        self._high_level_ppo = PPO.load(self.high_level_model_path, device=self.device)

        # Load low-level recurrent PPO flight controller.
        if not os.path.isfile(self.low_level_actor_path):
            raise FileNotFoundError(f"Low-level actor not found: {self.low_level_actor_path}")
        self._low_level_controller = cac.LowLevelPPOController(
            agent_id=0,
            actor_path=self.low_level_actor_path,
            project_root=self.project_root,
            device=self.device,
        )

        # Tactical guidance layer (geometry, no learned parameters).
        guidance_overrides = self.config.get("guidance_config", {}) or {}
        guidance_cfg = cac.GuidanceConfig(**guidance_overrides)
        self._guidance = cac.TacticalGuidance(self.project_root, guidance_cfg)

        self._cac_module = cac
        self._last_action_rate = np.zeros(4, dtype=np.float32)
        self._base_cmd = np.zeros(4, dtype=np.float32)
        self._sub_step_counter = 0
        # 5 Hz high-level decisions; infer number of JSBSim sub-steps.
        self._decision_dt = float(getattr(self.jsbsim_env, "dt", 1.0 / 60.0))
        self._steps_per_decision = max(1, int(round(0.2 / self._decision_dt)))
        self._last_sim_time = -1.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        bandit_flight_state: Any,
        sim_time: float = 0.0,
    ) -> Dict[str, Any]:
        self._low_level_controller.reset()
        self._sub_step_counter = 0
        self._last_sim_time = float(sim_time)
        self._recompute_action_rate(own_state, bandit_state)
        return {"bandit_maneuver": "close_air_combat", "situation": {}}

    def update(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        bandit_flight_state: Any,
        sim_time: float,
        dt: float,
    ) -> Tuple[ControlCommand, Dict[str, Any]]:
        # Recompute the high-level intent + low-level rates at the first sub-step
        # of each 0.2 s decision interval.
        if self._sub_step_counter % self._steps_per_decision == 0:
            self._recompute_action_rate(own_state, bandit_state)
            self._sub_step_counter = 0

        # Integrate rates exactly like CAC: cmd(t) = base_cmd + action_rate * (i+1)/60.
        elapsed = (self._sub_step_counter + 1) * self._decision_dt
        cmd = self._base_cmd + self._last_action_rate * elapsed
        cmd = np.clip(cmd, [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0])

        self._sub_step_counter += 1
        self._last_sim_time = float(sim_time)

        control = ControlCommand(
            aileron=float(cmd[0]),
            elevator=float(cmd[1]),
            rudder=float(cmd[2]),
            throttle=float(cmd[3]),
        )
        info = {
            "bandit_maneuver": "close_air_combat",
            "bandit_command": {
                "aileron": control.aileron,
                "elevator": control.elevator,
                "rudder": control.rudder,
                "throttle": control.throttle,
            },
            "cac_action_rate": self._last_action_rate.tolist(),
            "cac_base_cmd": self._base_cmd.tolist(),
        }
        return control, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, rel_or_abs: str) -> str:
        path = os.path.expanduser(rel_or_abs)
        if os.path.isabs(path):
            return path
        return os.path.join(self.project_root, path)

    def _recompute_action_rate(self, own_state: Dict[str, Any], bandit_state: Dict[str, Any]):
        bandit_aircraft = self.jsbsim_env._aircraft[self.target_uid]
        own_aircraft = self.jsbsim_env._aircraft[self.own_uid]

        env_view = _EnvView(bandit_aircraft, bandit_state, own_aircraft, own_state)

        # High-level tactical observation (bandit is ego_id=0 because it is first in _EnvView).
        obs = self._cac_module.extract_high_level_obs(env_view, ego_id=0, project_root=self.project_root)
        strategy, _ = self._high_level_ppo.predict(obs, deterministic=True)
        strategy = int(strategy)

        # Mid-level guidance: discrete intent -> [delta_alt, delta_heading, delta_vel].
        delta = self._guidance.get_delta(env_view, ego_id=0, strategy=strategy)

        # Low-level controller: guidance setpoint -> 4-D actuator rates.
        action_rate = self._low_level_controller.act(env_view, delta)
        self._last_action_rate = np.asarray(action_rate, dtype=np.float32).reshape(4)

        # Snapshot current FCS position to start the ramp from.
        self._base_cmd = np.array([
            float(bandit_aircraft.get_property_value("fcs/aileron-cmd-norm")),
            float(bandit_aircraft.get_property_value("fcs/elevator-cmd-norm")),
            float(bandit_aircraft.get_property_value("fcs/rudder-cmd-norm")),
            float(bandit_aircraft.get_property_value("fcs/throttle-cmd-norm")),
        ], dtype=np.float32)
