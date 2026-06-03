"""
High-level close-range tracking environment.

Connects policy action -> virtual pursuit point -> LOS-rate guidance
-> low-level controller -> JSBSim dynamics -> reward/termination.

Migrated from legacy project:
  - E:/CloseAirCombat_control/envs/JSBSim/envs/env_base.py
  - E:/CloseAirCombat_control/envs/JSBSim/envs/singlecombat_env.py

P1 scope: minimal closed loop with JSBSimEnv.
P3 scope: no-prediction VPP baseline with SimplePointMassEnv fallback.
P4 scope: JSBSim high-fidelity bridge with unified backend interface.
"""

import numpy as np
from typing import Optional, Tuple

from .jsbsim_env import JSBSimEnv
from .simple_point_mass_env import SimplePointMassEnv
from .observation import compute_relative_geometry, build_observation
from .reward import RewardCalculator
from .termination import TerminationChecker
from ..virtual_point.generator import VirtualPointGenerator
from ..guidance.los_rate_guidance import LOSRateGuidance
from ..guidance.proportional_navigation import ProportionalNavigationGuidance
from ..guidance.hybrid_guidance import HybridGuidance
from ..guidance.overload_rollrate import CommandPostProcessor
from ..guidance.gain_config import GuidanceGains
from ..flight_control.command_limiter import clip_command
from ..flight_control.command_filter import MultiChannelCommandFilter
from ..flight_control.low_level_controller import LowLevelController
from ..trajectory_prediction import (
    TrajectoryPredictorAdapter,
    create_predictor_from_config,
    create_state_buffer_from_config,
)


class CloseRangeTrackingEnv:
    """
    High-level close-range tracking environment.

    This environment connects:
    policy action -> virtual pursuit point -> LOS-rate guidance
    -> low-level controller -> JSBSim dynamics -> reward/termination.

    Supports two backends:
    - JSBSimEnv: high-fidelity JSBSim flight dynamics (requires legacy data).
    - SimplePointMassEnv: simplified kinematics for smoke testing.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Full environment and guidance configuration.
        """
        self.config = config
        self.env_config = config.get("env", {})
        self.sim_freq = self.env_config.get("sim_freq", 60)
        self.decision_freq = self.env_config.get("decision_freq", 5)
        self.max_steps = self.env_config.get("max_high_level_steps", 512)
        self.aircraft_model = self.env_config.get("aircraft_model", "f16")

        # Backend selection: explicit "backend" field takes priority,
        # then fall back to legacy "use_jsbsim" boolean.
        explicit_backend = config.get("backend")
        if explicit_backend is not None:
            requested_backend = explicit_backend
        else:
            requested_backend = (
                "jsbsim" if self.env_config.get("use_jsbsim", True) else "simple"
            )

        # Number of JSBSim integration steps per high-level decision step
        self._sim_steps_per_decision = max(1, self.sim_freq // self.decision_freq)

        # Backend initialization
        if requested_backend == "jsbsim":
            try:
                self.jsbsim_env = JSBSimEnv(self.env_config)
                self.own_uid = "own"
                self.target_uid = "target"
                self.jsbsim_env.add_aircraft(
                    self.own_uid, {"model": self.aircraft_model}
                )
                self.jsbsim_env.add_aircraft(
                    self.target_uid, {"model": self.aircraft_model}
                )
                self._backend = "jsbsim"
                self._low_level_controller = LowLevelController(
                    config.get("guidance", {}).get("gains", {})
                )
            except Exception:
                # Fallback to simple env if JSBSim init fails
                self.jsbsim_env = None
                self._low_level_controller = None
                self._simple_env = SimplePointMassEnv(self.env_config)
                self._backend = "simple"
        else:
            self.jsbsim_env = None
            self._low_level_controller = None
            self._simple_env = SimplePointMassEnv(self.env_config)
            self._backend = "simple"

        # Submodules
        self.virtual_point_generator = VirtualPointGenerator(
            config.get(
                "virtual_point", config.get("guidance", {}).get("virtual_point", {})
            )
        )

        # Guidance law selection based on mode
        guidance_config = config.get("guidance", {})
        guidance_mode = str(guidance_config.get("mode", "los_rate")).lower()
        if guidance_mode == "los_rate":
            self.guidance = LOSRateGuidance(guidance_config)
        elif guidance_mode == "proportional_navigation":
            self.guidance = ProportionalNavigationGuidance(guidance_config)
        elif guidance_mode == "hybrid":
            self.guidance = HybridGuidance(guidance_config)
        else:
            raise ValueError(f"Unknown guidance mode: {guidance_mode}")

        # Optional command post-processor (energy comp, terminal protection, etc.)
        self.command_post_processor = None
        if guidance_config.get("post_process", {}).get("enabled", False):
            self.command_post_processor = CommandPostProcessor(guidance_config)

        self.reward_calculator = RewardCalculator(config)
        self.termination_checker = TerminationChecker(self.env_config)

        # Command filter (three independent channels)
        filter_alpha = (
            config.get("guidance", {}).get("gains", {}).get("alpha_filter", 0.3)
        )
        self._command_filter = MultiChannelCommandFilter(alpha=filter_alpha)

        # Current guidance gains (default fixed)
        self.current_gains = GuidanceGains()

        # Trajectory prediction adapter
        self.trajectory_predictor_adapter = None
        tp_config = config.get("trajectory_prediction", {})
        if tp_config.get("enabled", False):
            try:
                predictor = create_predictor_from_config(tp_config)
                state_buffer = create_state_buffer_from_config(tp_config)
                self.trajectory_predictor_adapter = TrajectoryPredictorAdapter(
                    predictor=predictor,
                    state_buffer=state_buffer,
                    config=tp_config,
                )
                # Freeze predictor during RL by default
                if tp_config.get("freeze_predictor_during_rl", True):
                    predictor.freeze()
            except Exception as exc:
                print(f"WARNING: Failed to create trajectory predictor: {exc}")
                self.trajectory_predictor_adapter = None

        self.current_step = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, scenario=None, seed=None) -> dict:
        """
        Reset own aircraft and target aircraft.

        Args:
            scenario (Scenario, optional): Specific scenario to load (P2).
            seed (int, optional): Random seed for scenario sampling (P2).

        Returns:
            dict: Initial observation dictionary.
        """
        self.current_step = 0
        self._episode_count += 1
        self.reward_calculator.reset()
        self.termination_checker.reset()
        if hasattr(self.guidance, "reset"):
            self.guidance.reset()
        self._command_filter.reset()
        if self._low_level_controller is not None:
            self._low_level_controller.reset()

        if self.trajectory_predictor_adapter is not None:
            self.trajectory_predictor_adapter.reset()

        if self._backend == "jsbsim":
            self._reset_jsbsim(scenario)
        else:
            self._reset_simple(scenario)

        obs = self._get_observation()
        return obs

    def _reset_jsbsim(self, scenario=None):
        own_init = {
            "ic/long-gc-deg": 120.0,
            "ic/lat-geod-deg": 60.0,
            "ic/h-sl-ft": 20000.0,
            "ic/psi-true-deg": 0.0,
            "ic/u-fps": 900.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        }
        target_init = {
            "ic/long-gc-deg": 120.05,
            "ic/lat-geod-deg": 60.0,
            "ic/h-sl-ft": 20000.0,
            "ic/psi-true-deg": 180.0,
            "ic/u-fps": 900.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        }
        if scenario is not None:
            own_scenario = _get_scenario_attr(scenario, "own_init")
            target_scenario = _get_scenario_attr(scenario, "target_init")
            if own_scenario is not None:
                own_init = self._scenario_to_jsbsim_init(own_scenario)
            if target_scenario is not None:
                target_init = self._scenario_to_jsbsim_init(target_scenario)
        self.jsbsim_env.reset({self.own_uid: own_init, self.target_uid: target_init})

    def _reset_simple(self, scenario=None):
        own_init = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        target_init = {
            "position_m": np.array([2000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        if scenario is not None:
            own_scenario = _get_scenario_attr(scenario, "own_init")
            target_scenario = _get_scenario_attr(scenario, "target_init")
            if own_scenario is not None:
                own_init = self._scenario_to_simple_init(own_scenario)
            if target_scenario is not None:
                target_init = self._scenario_to_simple_init(target_scenario)
        self._simple_env.reset(own_init=own_init, target_init=target_init)

    def step(
        self, action: Optional[np.ndarray] = None
    ) -> Tuple[dict, float, bool, bool, dict]:
        """
        Execute one high-level step.

        step(action) 流程：
        1. 读取 own_state 和 target_state。
        2. 计算 relative_state。
        3. 更新 predictor_adapter（若启用）。
        4. 调用 VirtualPointGenerator，支持 current_target / predicted_target。
        5. 得到 Pos_Virtual = Pos_T_anchor + Δp。
        6. 调用 LOSRateGuidance.compute_command()。
        7. 调用 command_limiter 和 command_filter。
        8. 调用环境 step (SimplePointMass 或 JSBSim)。
        9. 计算 reward。
        10. 检查 done。
        11. 返回 obs, reward, done, info。

        Args:
            action (np.ndarray, optional): Policy-level action (normalized virtual pursuit point parameters).
                Shape [3] for [Δx, Δy, Δz].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # 1. 获取当前状态
        own_state, target_state = self._get_current_states()

        # 2. 计算相对态势
        rel_state = compute_relative_geometry(own_state, target_state)

        # 3. 更新 predictor_adapter（若启用）
        tp_enabled = self.config.get("trajectory_prediction", {}).get("enabled", False)
        prediction_info = {
            "prediction_enabled": tp_enabled,
            "predictor_type": None,
            "prediction_valid": False,
            "prediction_fallback_reason": None,
            "predicted_target_position": None,
            "prediction_error_m": np.nan,
        }
        if tp_enabled and self.trajectory_predictor_adapter is not None:
            try:
                self.trajectory_predictor_adapter.update(
                    own_state, target_state, rel_state
                )
            except Exception as exc:
                prediction_info["prediction_fallback_reason"] = f"update_failed: {exc}"

        # 4. 生成虚拟追踪点
        anchor_mode = self.config.get("virtual_point", {}).get(
            "anchor_mode", "current_target"
        )
        if action is None:
            action = np.zeros(3)
        action = np.asarray(action, dtype=np.float64)

        # 构建 target_state 用于 VPP 生成（统一字段名）
        target_pos = target_state.get("position_m")
        if target_pos is None:
            target_pos = target_state.get("position_neu")
        target_for_vp = {"position_neu": np.asarray(target_pos)}

        # 若 anchor_mode=predicted_target，获取预测位置
        predicted_target_pos = None
        if (
            anchor_mode == "predicted_target"
            and tp_enabled
            and self.trajectory_predictor_adapter is not None
        ):
            try:
                pred_pos, _, pred_info = self.trajectory_predictor_adapter.predict(
                    target_state
                )
                prediction_info["predictor_type"] = pred_info.get("model_type")
                prediction_info["prediction_valid"] = not pred_info.get(
                    "fallback", False
                )
                prediction_info["prediction_fallback_reason"] = pred_info.get(
                    "fallback_reason"
                )
                if pred_pos is not None and np.isfinite(pred_pos).all():
                    predicted_target_pos = np.asarray(pred_pos, dtype=np.float64)
                    prediction_info["predicted_target_position"] = (
                        predicted_target_pos.tolist()
                    )
            except Exception as exc:
                prediction_info["prediction_fallback_reason"] = f"predict_failed: {exc}"

        # 若预测不可用，回退到 current_target
        if predicted_target_pos is None:
            predicted_target_pos = target_for_vp["position_neu"]
            prediction_info["prediction_fallback_reason"] = (
                prediction_info["prediction_fallback_reason"]
                or "fallback_to_current_target"
            )
            anchor_mode = "current_target"

        vp_result = self.virtual_point_generator.action_to_virtual_point(
            action,
            own_state,
            target_for_vp,
            anchor_mode=anchor_mode,
            trajectory_predictor_adapter=None,  # generator 不直接调用 predictor
            predicted_target_position=predicted_target_pos,
            return_info=True,
        )
        virtual_point, vp_info = vp_result

        # 5. Guidance command generation
        raw_command = self.guidance.compute_command(
            own_state, target_state, virtual_point, self.current_gains
        )

        # 5b. Optional command post-processing (terminal protection, energy comp, etc.)
        if self.command_post_processor is not None:
            rel_geom = compute_relative_geometry(own_state, target_state)
            raw_command = self.command_post_processor.process(
                raw_command,
                own_state=own_state,
                target_state=target_state,
                relative_state=rel_geom,
            )

        # 6. Command clipping and filtering
        limits = self.config.get("limits", {})
        clipped_command = clip_command(raw_command, limits)
        filtered_command = self._apply_command_filter(clipped_command)

        # 7. 环境 step
        actuator_info = {}
        if self._backend == "jsbsim":
            actuator_info = self._step_jsbsim(filtered_command)
        else:
            self._step_simple(filtered_command)

        # 8. 获取 step 后的新状态（post-step）
        own_state_post, target_state_post = self._get_current_states()
        rel_state_post = compute_relative_geometry(own_state_post, target_state_post)

        # 9. 检查终止（基于 post-step 状态）
        terminated, truncated, term_info = self._check_done(
            own_state_post, target_state_post, rel_state_post
        )

        # 10. 计算 reward（基于 post-step 状态，含 terminal_reward 注入）
        reward, reward_terms = self._compute_reward(
            own_state_post,
            target_state_post,
            rel_state_post,
            filtered_command,
            term_info,
        )

        # 11. 获取观察（post-step）
        obs = self._get_observation()

        # 组装 info
        info = {
            "virtual_point": virtual_point,
            "guidance_command": filtered_command,
            "raw_command": raw_command,
            "reward_terms": reward_terms,
            "termination_info": term_info,
            "relative_state": rel_state_post,
            "anchor_mode": anchor_mode,
            "own_state": own_state_post,
            "target_state": target_state_post,
            "current_step": self.current_step,
            "episode": self._episode_count,
            "backend": self._backend,
            "range_m": rel_state_post.get("range_m", np.nan),
            "ata_deg": float(np.rad2deg(rel_state_post.get("ata_rad", np.nan))),
            "aspect_deg": float(np.rad2deg(rel_state_post.get("aa_rad", np.nan))),
            "los_rate": rel_state_post.get("range_rate_mps", np.nan),
            "nz_cmd": filtered_command.get("nz_cmd", np.nan),
            "roll_rate_cmd": filtered_command.get("roll_rate_cmd", np.nan),
            "throttle_cmd": filtered_command.get("throttle_cmd", np.nan),
            "prediction_enabled": prediction_info["prediction_enabled"],
            "predictor_type": prediction_info["predictor_type"],
            "prediction_valid": prediction_info["prediction_valid"],
            "prediction_fallback_reason": prediction_info["prediction_fallback_reason"],
            "predicted_target_position": prediction_info["predicted_target_position"],
            "prediction_error_m": prediction_info["prediction_error_m"],
        }
        info.update(actuator_info)
        info.update(term_info)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, own_state, target_state, rel_state, command, term_info):
        """
        Compute reward for the current step.

        Reward and termination are evaluated on post-step states to ensure
        consistency: the agent receives feedback about the consequence of its
        action, not the state before the action was taken.

        Args:
            own_state (dict): Post-step own aircraft state.
            target_state (dict): Post-step target state.
            rel_state (dict): Post-step relative geometry.
            command (dict): Executed command.
            term_info (dict): Termination info from _check_done.

        Returns:
            tuple: (reward, reward_terms)
        """
        info = {
            "own_state": own_state,
            "target_state": target_state,
            "relative_state": rel_state,
            "command": command,
        }

        # Inject terminal reward based on termination reason
        terminal_reward = 0.0
        if term_info.get("is_success"):
            terminal_reward = self.reward_calculator.terminal_success
        elif term_info.get("is_crash"):
            terminal_reward = self.reward_calculator.terminal_crash
        elif term_info.get("is_timeout") or term_info.get("is_out_of_bounds"):
            terminal_reward = self.reward_calculator.terminal_failure
        info["terminal_reward"] = terminal_reward

        reward, reward_terms = self.reward_calculator.compute(info)
        return reward, reward_terms

    def _check_done(self, own_state, target_state, rel_state):
        """
        Check termination conditions.

        Semantics:
          - success / crash / out_of_bounds → terminated=True, truncated=False
          - timeout → terminated=False, truncated=True

        Returns:
            tuple: (terminated, truncated, term_info)
        """
        terminated = False
        truncated = False
        done, term_info = self.termination_checker.check(
            own_state, target_state, rel_state, self.current_step
        )
        if done:
            if term_info.get("is_timeout"):
                # Timeout is a time-limit truncation, not a task termination.
                truncated = True
            else:
                # success, crash, out_of_bounds are true terminations.
                terminated = True

        return terminated, truncated, term_info

    def _apply_command_filter(self, command: dict) -> dict:
        """Apply independent first-order filters to each command channel."""
        return self._command_filter.filter(command)

    def _get_current_states(self):
        """获取当前本机和目标状态（统一格式）。"""
        if self._backend == "jsbsim":
            states = self.jsbsim_env.get_state()
            own = states[self.own_uid]
            target = states[self.target_uid]
            return own, target
        else:
            return self._simple_env.get_state()

    def _step_jsbsim(self, command):
        """在 JSBSim 后端执行控制，返回 actuator info。"""
        # Get current aircraft state for the low-level controller
        states = self.jsbsim_env.get_state()
        own_state_raw = states[self.own_uid]

        # Use low-level controller to map guidance commands to JSBSim properties
        actuator_output = self._low_level_controller.compute_actuator(
            command, own_state_raw
        )

        # Extract JSBSim properties
        jsbsim_props = {
            k: v for k, v in actuator_output.items() if k.startswith("fcs/")
        }
        control_inputs = {self.own_uid: jsbsim_props}

        for _ in range(self._sim_steps_per_decision):
            self.jsbsim_env.step(control_inputs)
            control_inputs = None

        return {
            "elevator_cmd": actuator_output.get("fcs/elevator-cmd-norm", np.nan),
            "aileron_cmd": actuator_output.get("fcs/aileron-cmd-norm", np.nan),
            "rudder_cmd": actuator_output.get("fcs/rudder-cmd-norm", np.nan),
            "throttle_actual": actuator_output.get("fcs/throttle-cmd-norm", np.nan),
            "saturation_flag": actuator_output.get("saturation_flag", False),
        }

    def _step_simple(self, command):
        """在简化后端执行控制。"""
        self._simple_env.step(own_command=command)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict:
        """
        Build the current observation from environment states.

        Returns:
            dict: Observation dictionary with relative geometry and raw states.
        """
        own_state, target_state = self._get_current_states()
        rel_state = compute_relative_geometry(own_state, target_state)

        obs_dict = {
            "own_state": own_state,
            "target_state": target_state,
            "relative_state": rel_state,
        }

        # 同时返回展平向量（供策略网络使用）
        obs_vec = build_observation(own_state, target_state)
        obs_dict["observation_vector"] = obs_vec
        return obs_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scenario_to_jsbsim_init(self, aircraft_init) -> dict:
        """Convert scenario init (dict or object) to JSBSim init_state dict."""
        pos = _get_attr(aircraft_init, "position_m", np.array([0.0, 0.0, 5000.0]))
        vel_mps = _get_attr(aircraft_init, "velocity_mps", 800.0)
        heading_deg = _get_attr(aircraft_init, "heading_deg", 0.0)
        pitch_deg = _get_attr(aircraft_init, "pitch_deg", 0.0)
        roll_deg = _get_attr(aircraft_init, "roll_deg", 0.0)

        h_sl_ft = (
            float(pos[2]) / 0.3048
            if hasattr(pos, "__len__") and len(pos) > 2
            else 20000.0
        )
        psi_deg = heading_deg
        u_fps = vel_mps / 0.3048

        return {
            "ic/h-sl-ft": h_sl_ft,
            "ic/psi-true-deg": psi_deg,
            "ic/u-fps": u_fps,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/theta-deg": pitch_deg,
            "ic/phi-deg": roll_deg,
        }

    def _scenario_to_simple_init(self, aircraft_init) -> dict:
        """Convert scenario init (dict or object) to SimplePointMassEnv init dict."""
        pos = _get_attr(aircraft_init, "position_m", np.array([0.0, 0.0, 5000.0]))
        vel_mps = _get_attr(aircraft_init, "velocity_mps", 200.0)
        heading_deg = _get_attr(aircraft_init, "heading_deg", 0.0)
        heading_rad = np.deg2rad(heading_deg)
        vel = np.array(
            [vel_mps * np.cos(heading_rad), vel_mps * np.sin(heading_rad), 0.0]
        )
        return {
            "position_m": np.asarray(pos, dtype=np.float64),
            "velocity_vector_mps": vel,
            "heading_rad": heading_rad,
            "altitude_m": (
                float(pos[2]) if hasattr(pos, "__len__") and len(pos) > 2 else 5000.0
            ),
        }

    def close(self):
        """Clean up environment resources."""
        if self.jsbsim_env is not None:
            self.jsbsim_env.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_scenario_attr(scenario, key):
    """Get attribute from scenario dict or object."""
    if isinstance(scenario, dict):
        return scenario.get(key)
    return getattr(scenario, key, None)


def _get_attr(obj, key, default):
    """Get attribute from dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
