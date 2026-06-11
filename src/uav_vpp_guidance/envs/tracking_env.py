"""
High-level close-range tracking environment.

Connects policy action -> virtual pursuit point -> LOS-rate guidance
-> low-level controller -> JSBSim dynamics -> reward/termination.

Migrated from legacy project:
  - <JSBSIM_ROOT>/envs/JSBSim/envs/env_base.py
  - <JSBSIM_ROOT>/envs/JSBSim/envs/singlecombat_env.py

P1 scope: minimal closed loop with JSBSimEnv.
P3 scope: no-prediction VPP baseline with SimplePointMassEnv fallback.
P4 scope: JSBSim high-fidelity bridge with unified backend interface.
"""

import numpy as np
from typing import Optional, Tuple

from .jsbsim_env import JSBSimEnv, neu2lla
from .simple_point_mass_env import SimplePointMassEnv
from .observation import compute_relative_geometry, build_observation
from .reward import RewardCalculator
from .termination import TerminationChecker
from ..virtual_point.generator import VirtualPointGenerator
from ..virtual_point.no_vpp_guidance import NoVPPGuidance
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
from ..trajectory_prediction.prediction_error_tracker import PredictionErrorTracker


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
        strict_backend = self.env_config.get("strict_backend", False)
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
            except Exception as exc:
                if strict_backend:
                    raise RuntimeError(
                        f"JSBSim backend initialization failed "
                        f"(strict_backend=True): {exc}"
                    ) from exc
                # Fallback to simple env if JSBSim init fails
                import warnings
                warnings.warn(
                    f"JSBSim backend initialization failed: {exc}. "
                    f"Falling back to SimplePointMassEnv.",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
        vp_config = config.get(
            "virtual_point", config.get("guidance", {}).get("virtual_point", {})
        )
        vp_enabled = vp_config.get("enabled", True)
        e2e_enabled = config.get("end_to_end", {}).get("enabled", False)

        if not vp_enabled and e2e_enabled:
            # End-to-end mode: policy outputs direct control commands
            self._use_virtual_point = False
            self.virtual_point_generator = None
        elif vp_enabled:
            # VPP layer is enabled: choose between normal offset or zero-offset (No-VPP)
            vp_mode = vp_config.get("mode", "normal")
            if vp_mode == "zero_offset":
                # No-VPP baseline: keep full guidance chain but force VPP offset to zero
                self._use_virtual_point = True
                self.virtual_point_generator = NoVPPGuidance()
            else:
                # Normal VPP mode
                self._use_virtual_point = True
                self.virtual_point_generator = VirtualPointGenerator(vp_config)
        else:
            # vp_enabled=False without e2e_enabled is an invalid configuration
            raise ValueError(
                "Invalid configuration: virtual_point.enabled=false without "
                "end_to_end.enabled=true. If you want No-VPP baseline, set "
                "virtual_point.enabled=true and virtual_point.mode='zero_offset'. "
                "If you want End-to-End, set end_to_end.enabled=true."
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

        # Mode-switch: store PN guidance for runtime switching
        self._guidance_pn = None
        self._mode_switch_config = guidance_config.get("mode_switch", {})
        self._mode_switch_latched = False
        if self._mode_switch_config.get("enabled", False):
            self._guidance_pn = ProportionalNavigationGuidance(guidance_config)

        # Optional command post-processor (energy comp, terminal protection, etc.)
        self.command_post_processor = None
        if guidance_config.get("post_process", {}).get("enabled", False):
            # Merge global limits so post-processor can read them
            processor_config = {
                **guidance_config,
                "limits": {
                    **config.get("limits", {}),
                    **guidance_config.get("limits", {}),
                },
            }
            self.command_post_processor = CommandPostProcessor(processor_config)

        self.reward_calculator = RewardCalculator(config)
        self.termination_checker = TerminationChecker(self.env_config)

        # Command filter (three independent channels)
        filter_alpha = (
            config.get("guidance", {}).get("gains", {}).get("alpha_filter", 0.3)
        )
        self._command_filter = MultiChannelCommandFilter(alpha=filter_alpha)

        # Current guidance gains (loaded from config so external gain overrides
        # actually propagate into the guidance law)
        self.current_gains = GuidanceGains(
            **config.get("guidance", {}).get("gains", {})
        )

        # Trajectory prediction adapter
        self.trajectory_predictor_adapter = None
        self._predictor_init_failed = False
        self._prediction_error_tracker = PredictionErrorTracker(
            high_level_dt=self.env_config.get("high_level_dt", 0.2)
        )
        tp_config = config.get("trajectory_prediction", {})
        if tp_config.get("enabled", False):
            strict_init = tp_config.get("strict_predictor_init", False)
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
                self._predictor_init_failed = True
                if strict_init:
                    raise RuntimeError(
                        f"strict_predictor_init=True: Failed to create trajectory predictor. "
                        f"Error: {exc}"
                    ) from exc
                print(f"WARNING: Failed to create trajectory predictor: {exc}")
                self.trajectory_predictor_adapter = None

        # Domain randomization state
        self.domain_rand_config = config.get("domain_randomization", {})
        self.domain_rand_scale = float(self.domain_rand_config.get("scale", 0.0))
        self._domain_rand_rng = np.random.default_rng(42)

        self.current_step = 0
        self._episode_count = 0
        self._sim_time_s = 0.0

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
        self._sim_time_s = 0.0
        self.reward_calculator.reset()
        self.termination_checker.reset()
        if hasattr(self.guidance, "reset"):
            self.guidance.reset()
        if self._guidance_pn is not None and hasattr(self._guidance_pn, "reset"):
            self._guidance_pn.reset()
        self._mode_switch_latched = False
        self._command_filter.reset()
        if self._low_level_controller is not None:
            self._low_level_controller.reset()

        if self.trajectory_predictor_adapter is not None:
            self.trajectory_predictor_adapter.reset()
        self._prediction_error_tracker.reset()

        if self._backend == "jsbsim":
            self._reset_jsbsim(scenario)
        else:
            self._reset_simple(scenario)

        obs = self._get_observation()
        return obs

    def set_domain_rand_scale(self, scale: float):
        """Set the current domain randomization scale (0.0 = off)."""
        self.domain_rand_scale = float(scale)

    def _apply_domain_randomization(self, scenario: dict) -> dict:
        """
        Apply domain randomization to a scenario dict.

        Perturbs own_init and target_init by:
          - position: ±position_noise_fraction of initial range
          - velocity: ±velocity_noise_fraction of nominal velocity
          - heading: ±heading_noise_deg

        The magnitude is scaled by self.domain_rand_scale.
        """
        if self.domain_rand_scale <= 0.0:
            return scenario

        cfg = self.domain_rand_config
        pos_frac = cfg.get("position_noise_fraction", 0.10)  # 10% of range
        vel_frac = cfg.get("velocity_noise_fraction", 0.10)  # 10% of speed
        head_deg = cfg.get("heading_noise_deg", 15.0)        # ±15°

        # Deep-copy scenario to avoid mutating the original config
        import copy
        scenario = copy.deepcopy(scenario)

        own = _get_scenario_attr(scenario, "own_init")
        target = _get_scenario_attr(scenario, "target_init")

        # Compute initial range for position noise magnitude
        if own is not None and target is not None:
            own_pos = _get_attr(own, "position_m", np.array([0.0, 0.0, 5000.0]))
            tgt_pos = _get_attr(target, "position_m", np.array([2000.0, 0.0, 5000.0]))
            initial_range = float(np.linalg.norm(np.asarray(own_pos) - np.asarray(tgt_pos)))
        else:
            initial_range = 2000.0

        for aircraft_key in ("own_init", "target_init"):
            ac = _get_scenario_attr(scenario, aircraft_key)
            if ac is None:
                continue

            # Perturb position
            pos = _get_attr(ac, "position_m", None)
            if pos is not None:
                pos = np.asarray(pos, dtype=np.float64)
                noise = self._domain_rand_rng.uniform(-1.0, 1.0, size=pos.shape)
                pos = pos + noise * initial_range * pos_frac * self.domain_rand_scale
                if isinstance(ac, dict):
                    ac["position_m"] = pos.tolist()
                else:
                    setattr(ac, "position_m", pos.tolist())

            # Perturb velocity
            vel = _get_attr(ac, "velocity_mps", None)
            if vel is not None:
                noise = self._domain_rand_rng.uniform(-1.0, 1.0)
                vel = float(vel) * (1.0 + noise * vel_frac * self.domain_rand_scale)
                vel = max(50.0, vel)  # prevent negative or too-small speed
                if isinstance(ac, dict):
                    ac["velocity_mps"] = vel
                else:
                    setattr(ac, "velocity_mps", vel)

            # Perturb heading
            heading = _get_attr(ac, "heading_deg", None)
            if heading is not None:
                noise = self._domain_rand_rng.uniform(-1.0, 1.0)
                heading = float(heading) + noise * head_deg * self.domain_rand_scale
                # Normalize to [-180, 180]
                heading = (heading + 180.0) % 360.0 - 180.0
                if isinstance(ac, dict):
                    ac["heading_deg"] = heading
                else:
                    setattr(ac, "heading_deg", heading)

        return scenario

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
            scenario = self._apply_domain_randomization(scenario)
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
            scenario = self._apply_domain_randomization(scenario)
            own_scenario = _get_scenario_attr(scenario, "own_init")
            target_scenario = _get_scenario_attr(scenario, "target_init")
            if own_scenario is not None:
                own_init = self._scenario_to_simple_init(own_scenario)
            if target_scenario is not None:
                target_init = self._scenario_to_simple_init(target_scenario)
        self._simple_env.reset(own_init=own_init, target_init=target_init)

    def step(
        self,
        action: Optional[np.ndarray] = None,
        command_override: Optional[dict] = None,
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
            command_override (dict, optional): If provided, bypass the policy/VPP/guidance
                pipeline and directly apply this command dict. Keys: nz_cmd, roll_rate_cmd,
                throttle_cmd. Still goes through command clipping/filtering for safety.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        high_level_dt = self.env_config.get("high_level_dt", 0.2)
        self._sim_time_s += high_level_dt

        # 1. 获取当前状态
        own_state, target_state = self._get_current_states()

        # 2. 计算相对态势
        rel_state = compute_relative_geometry(own_state, target_state)

        # 3. 更新 predictor_adapter（若启用）
        tp_enabled = self.config.get("trajectory_prediction", {}).get("enabled", False)
        prediction_info = {
            "prediction_enabled": tp_enabled,
            "predictor_init_failed": self._predictor_init_failed,
            "predictor_type": None,
            "prediction_valid": False,
            "prediction_fallback": False,
            "prediction_fallback_reason": None,
            "prediction_fallback_mode": None,
            "prediction_fallback_model": None,
            "prediction_fallback_phase": None,
            "predicted_target_position": None,
            "prediction_error_m": np.nan,
            "latest_prediction_error_m": np.nan,
            "mean_prediction_error_m": np.nan,
            "median_prediction_error_m": np.nan,
            "prediction_error_count": 0,
        }
        if tp_enabled and self.trajectory_predictor_adapter is not None:
            try:
                self.trajectory_predictor_adapter.update(
                    own_state, target_state, rel_state
                )
            except Exception as exc:
                prediction_info["prediction_fallback_reason"] = f"update_failed: {exc}"
                prediction_info["prediction_fallback_phase"] = "runtime_failure"

        # 4. 生成虚拟追踪点 (or use direct command override for diagnosis)
        anchor_mode = self.config.get("virtual_point", {}).get(
            "anchor_mode", "current_target"
        )
        use_command_override = command_override is not None
        if action is None:
            action = np.zeros(3)
        action = np.asarray(action, dtype=np.float64)

        # 构建 target_state 用于 VPP 生成（统一字段名）
        target_pos = target_state.get("position_m")
        if target_pos is None:
            target_pos = target_state.get("position_neu")
        target_for_vp = {"position_neu": np.asarray(target_pos)}
        # 传递速度信息，供 constant_velocity / oracle / rule_based 模式使用
        target_vel = target_state.get("velocity_vector_mps")
        if target_vel is not None:
            target_for_vp["velocity_vector_mps"] = np.asarray(target_vel, dtype=np.float64)
        target_vel_alt = target_state.get("velocity")
        if target_vel_alt is not None:
            target_for_vp["velocity"] = np.asarray(target_vel_alt, dtype=np.float64)

        # 若 anchor_mode=predicted_target，获取预测位置
        predicted_target_pos = None
        lookahead_time_s = self.config.get("trajectory_prediction", {}).get("prediction", {}).get("lookahead_time_s", 1.0)
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
                prediction_info["prediction_fallback"] = bool(pred_info.get("fallback", False))
                prediction_info["prediction_fallback_reason"] = pred_info.get(
                    "fallback_reason"
                )
                prediction_info["prediction_fallback_mode"] = pred_info.get(
                    "fallback_mode"
                )
                prediction_info["prediction_fallback_model"] = pred_info.get(
                    "fallback_model"
                )
                prediction_info["prediction_fallback_phase"] = pred_info.get(
                    "fallback_phase"
                )
                if pred_pos is not None and np.isfinite(pred_pos).all():
                    predicted_target_pos = np.asarray(pred_pos, dtype=np.float64)
                    prediction_info["predicted_target_position"] = (
                        predicted_target_pos.tolist()
                    )
                    # Register prediction for delayed error tracking
                    self._prediction_error_tracker.register_prediction(
                        current_time_s=self._sim_time_s,
                        lookahead_time_s=lookahead_time_s,
                        predicted_position_neu=predicted_target_pos,
                    )
            except Exception as exc:
                prediction_info["prediction_fallback_reason"] = f"predict_failed: {exc}"
                prediction_info["prediction_fallback_phase"] = "runtime_failure"

        # 若预测不可用，回退到 current_target（仅对 predicted_target 模式）
        if anchor_mode == "predicted_target" and predicted_target_pos is None:
            predicted_target_pos = target_for_vp["position_neu"]
            if prediction_info["prediction_fallback_reason"] is None:
                prediction_info["prediction_fallback_reason"] = "fallback_to_current_target"
                prediction_info["prediction_fallback_phase"] = "configured_current_target"
            prediction_info["prediction_fallback"] = True
            anchor_mode = "current_target"

        # Determine direct-track request from config (telemetry)
        direct_track_mode_requested = self.config.get("guidance", {}).get("direct_track_mode", False)

        # Mode-switch gate evaluation (if enabled)
        mode_switch_requested = self._mode_switch_config.get("enabled", False)
        mode_switch_effective = False
        mode_switch_reason = None
        effective_guidance = self.guidance
        effective_guidance_mode = self.config.get("guidance", {}).get("mode", "los_rate")

        # Diagnosis path: bypass policy/VPP/guidance and inject a command directly
        if use_command_override:
            mode_switch_requested = False
            direct_track_mode_requested = False

        if mode_switch_requested:
            gate_active, gate_reason = self._evaluate_mode_switch_gate(rel_state)
            if gate_active:
                self._mode_switch_latched = True
            if self._mode_switch_latched:
                mode_switch_effective = True
                mode_switch_reason = gate_reason if gate_active else "latched"
                direct_track_mode_requested = True  # override to bypass VPP
                effective_guidance = self._guidance_pn
                effective_guidance_mode = "proportional_navigation"

        # Direct-track mode: bypass VPP offset, track anchor directly
        if direct_track_mode_requested:
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "direct_track_mode": True,
                "action_applied": False,
            }
            direct_track_mode_effective = True
            virtual_point_source = "direct_track"
        elif self._use_virtual_point and self.virtual_point_generator is not None:
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
            direct_track_mode_effective = False
            virtual_point_source = "vpp_policy"
        else:
            # End-to-end mode: virtual_point is not used for guidance,
            # but we still populate it for telemetry consistency.
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "end_to_end_mode": True,
                "action_applied": True,
            }
            direct_track_mode_effective = False
            virtual_point_source = "end_to_end"

        # 5. Guidance command generation
        if not self._use_virtual_point:
            # Direct command mode: policy outputs normalized commands in [-1, 1]
            # which are then mapped to physical limits here.
            limits = self.config.get("limits", {})
            nz_min = limits.get("nz_min", -2.0)
            nz_max = limits.get("nz_max", 7.0)
            rr_min = limits.get("roll_rate_min", -1.5)
            rr_max = limits.get("roll_rate_max", 1.5)
            th_min = limits.get("throttle_min", 0.0)
            th_max = limits.get("throttle_max", 1.0)
            raw_command = {
                "nz_cmd": float(action[0]) * (nz_max - nz_min) / 2.0 + (nz_max + nz_min) / 2.0,
                "roll_rate_cmd": float(action[1]) * (rr_max - rr_min) / 2.0 + (rr_max + rr_min) / 2.0,
                "throttle_cmd": float(action[2]) * (th_max - th_min) / 2.0 + (th_max + th_min) / 2.0,
            }
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "direct_command_mode": True,
                "action_applied": True,
            }
            direct_track_mode_effective = False
            virtual_point_source = "direct_command"
            effective_guidance_mode = "direct_command"
        elif use_command_override:
            raw_command = dict(command_override)
            virtual_point = {"position_neu": target_for_vp["position_neu"]}
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "command_override": True,
                "action_applied": False,
            }
            direct_track_mode_effective = False
            virtual_point_source = "command_override"
            effective_guidance_mode = "command_override"
        else:
            raw_command = effective_guidance.compute_command(
                own_state, target_state, virtual_point, self.current_gains
            )

        # 5b. Optional command post-processing (terminal protection, energy comp, etc.)
        if self.command_post_processor is not None and not use_command_override:
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

        # 8a. Update prediction error tracker with actual target position
        actual_target_pos = target_state_post.get("position_m")
        if actual_target_pos is None:
            actual_target_pos = target_state_post.get("position_neu")
        if actual_target_pos is not None:
            self._prediction_error_tracker.update(
                current_time_s=self._sim_time_s,
                actual_target_position_neu=actual_target_pos,
            )
            err_stats = self._prediction_error_tracker.get_stats()
            prediction_info["latest_prediction_error_m"] = err_stats["latest_prediction_error_m"]
            prediction_info["mean_prediction_error_m"] = err_stats["mean_prediction_error_m"]
            prediction_info["median_prediction_error_m"] = err_stats["median_prediction_error_m"]
            prediction_info["prediction_error_count"] = err_stats["prediction_error_count"]
            if err_stats["latest_prediction_error_m"] is not None:
                prediction_info["prediction_error_m"] = err_stats["latest_prediction_error_m"]

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

        # Serialize virtual_point for info (convert ndarrays → lists)
        def _serialize_vp(vp):
            if isinstance(vp, dict):
                return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vp.items()}
            elif isinstance(vp, np.ndarray):
                return vp.tolist()
            return vp

        # 组装 info
        info = {
            "virtual_point": _serialize_vp(virtual_point),
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
            "predictor_init_failed": prediction_info["predictor_init_failed"],
            "predictor_type": prediction_info["predictor_type"],
            "prediction_valid": prediction_info["prediction_valid"],
            "prediction_fallback": prediction_info["prediction_fallback"],
            "prediction_fallback_reason": prediction_info["prediction_fallback_reason"],
            "prediction_fallback_mode": prediction_info["prediction_fallback_mode"],
            "prediction_fallback_model": prediction_info["prediction_fallback_model"],
            "prediction_fallback_phase": prediction_info["prediction_fallback_phase"],
            "predicted_target_position": prediction_info["predicted_target_position"],
            "prediction_error_m": prediction_info["prediction_error_m"],
            "latest_prediction_error_m": prediction_info["latest_prediction_error_m"],
            "mean_prediction_error_m": prediction_info["mean_prediction_error_m"],
            "median_prediction_error_m": prediction_info["median_prediction_error_m"],
            "prediction_error_count": prediction_info["prediction_error_count"],
            "direct_track_mode_requested": direct_track_mode_requested,
            "direct_track_mode_effective": direct_track_mode_effective,
            "virtual_point_source": virtual_point_source,
            "mode_switch_requested": mode_switch_requested,
            "mode_switch_effective": mode_switch_effective,
            "mode_switch_reason": mode_switch_reason,
            "effective_guidance_mode": effective_guidance_mode,
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

    def _evaluate_mode_switch_gate(self, rel_state):
        """Evaluate geometry-triggered mode-switch gate.

        Stage 6H.1 redesign: supports both low-aspect (tail-chase / head-on)
        and high-aspect (crossing) geometries via a dual-threshold design.

        Returns:
            tuple: (gate_active: bool, reason: str)
        """
        cfg = self._mode_switch_config
        aspect_abs_deg = abs(float(np.rad2deg(rel_state.get("aa_rad", 0.0))))
        range_m = rel_state.get("range_m", float("inf"))
        range_rate = rel_state.get("range_rate_mps", 0.0)
        closing_speed = abs(range_rate)

        aspect_thresh = cfg.get("aspect_threshold_deg", 15.0)
        crossing_thresh = cfg.get("crossing_aspect_threshold_deg", None)
        range_thresh = cfg.get("range_threshold_m", 3000.0)
        speed_thresh = cfg.get("closing_speed_threshold_mps", 100.0)

        # Common geometry-independent conditions
        if range_m > range_thresh:
            return False, f"range_{range_m:.1f}_m"
        # Must be genuinely closing (range decreasing), not opening
        if range_rate > 0:
            return False, f"opening_range_rate_{range_rate:.1f}_mps"
        if closing_speed < speed_thresh:
            return False, f"closing_speed_{closing_speed:.1f}_mps"

        # Low-aspect: tail-chase or head-on (aa near 0°)
        if aspect_abs_deg <= aspect_thresh:
            return True, "gate_active"

        # High-aspect: crossing (aa near 90°)
        if crossing_thresh is not None:
            crossing_deviation = abs(aspect_abs_deg - 90.0)
            if crossing_deviation <= crossing_thresh:
                return True, "gate_active_crossing"

        return False, f"aspect_{aspect_abs_deg:.1f}_deg"

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
        """Convert scenario init (dict or object) to JSBSim init_state dict.

        Converts NEU position_m [north, east, up] to JSBSim geodetic initial
        conditions (long-gc-deg, lat-geod-deg, h-sl-ft) relative to the env origin.
        """
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

        result = {
            "ic/h-sl-ft": h_sl_ft,
            "ic/psi-true-deg": psi_deg,
            "ic/u-fps": u_fps,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/theta-deg": pitch_deg,
            "ic/phi-deg": roll_deg,
        }

        # Convert NEU horizontal position to geodetic coordinates so scenarios
        # with non-zero x/y are placed correctly on the JSBSim spherical earth.
        if self.jsbsim_env is not None and len(pos) >= 2:
            origin = getattr(self.jsbsim_env, "origin", (120.0, 60.0, 0.0))
            lon0, lat0, alt0 = origin
            try:
                lon_deg, lat_deg, _alt_m = neu2lla(
                    float(pos[0]), float(pos[1]), float(pos[2]), lon0, lat0, alt0
                )
                result["ic/long-gc-deg"] = float(lon_deg)
                result["ic/lat-geod-deg"] = float(lat_deg)
            except Exception:
                # If pymap3d is unavailable, keep default origin position.
                pass

        return result

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
        if hasattr(self, "_simple_env") and self._simple_env is not None:
            if hasattr(self._simple_env, "close"):
                self._simple_env.close()


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
