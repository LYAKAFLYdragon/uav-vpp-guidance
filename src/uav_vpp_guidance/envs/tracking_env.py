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

import inspect
import logging
import numpy as np
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

from .jsbsim_env import JSBSimEnv, neu2lla
from .simple_point_mass_env import SimplePointMassEnv
from .attack_zone import CombatHPManager, evaluate_attack_zone
from .observation import compute_relative_geometry, build_observation
from .opponent_policy import OpponentPolicy
from .reward import RewardCalculator
from .termination import TerminationChecker
from ..common.provenance import get_config_overrides
from ..virtual_point.generator import VirtualPointGenerator
from ..virtual_point.no_vpp_guidance import NoVPPGuidance
from ..virtual_point.coordinate_transform import offset_to_world, world_to_offset_frame
from ..guidance.los_rate_guidance import LOSRateGuidance
from ..guidance.proportional_navigation import ProportionalNavigationGuidance
from ..guidance.hybrid_guidance import HybridGuidance
from ..guidance.mpc_guidance import MPCGuidance
from ..guidance.overload_rollrate import CommandPostProcessor
from ..guidance.gain_config import GuidanceGains
from ..flight_control.command_limiter import (
    clip_command,
    effective_throttle_limits,
)
from ..flight_control.command_filter import MultiChannelCommandFilter
from ..flight_control.low_level_controller import LowLevelController
from ..flight_control.pid_controllers import (
    BaselinePIDController,
    EnhancedPIDController,
    GainScheduledPIDController,
    HybridPPOPIDAdapter,
    RobustPIDController,
)
from ..utils.action_schema import ActionSchema, validate_action
from ..trajectory_prediction import (
    TrajectoryPredictorAdapter,
    create_predictor_from_config,
    create_state_buffer_from_config,
)
from ..trajectory_prediction.prediction_error_tracker import PredictionErrorTracker


VALID_ANCHOR_MODES = {
    "current_target",
    "constant_velocity",
    "oracle_future_position",
    "rule_based_pursuit",
    "predicted_target",
    "offensive_position",
}


def _normalize_runtime_specialist_profile_names(value) -> Set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item not in (None, "")}
    return {str(value)}


def _resolve_post_merge_tactical_basis_recovery_profile_effective_cfg(
    cfg: dict,
    *,
    runtime_specialist_reason: Optional[str],
    aa_deg: float = float("nan"),
    range_rate_mps: float = float("nan"),
    previous_vp_forward_bias_m: float = float("nan"),
    previous_vp_lateral_bias_m: float = float("nan"),
) -> Tuple[dict, Dict[str, Optional[str]]]:
    def _is_finite_float(value) -> bool:
        try:
            return np.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _conditions_match(conditions: dict) -> bool:
        if not isinstance(conditions, dict) or not conditions:
            return True

        def _match_single(single: dict) -> bool:
            if not isinstance(single, dict):
                return False
            for key, expected in single.items():
                if key == "any":
                    if not isinstance(expected, (list, tuple)) or not any(
                        _match_single(item) for item in expected
                    ):
                        return False
                    continue
                if key == "all":
                    if not isinstance(expected, (list, tuple)) or not all(
                        _match_single(item) for item in expected
                    ):
                        return False
                    continue
                if expected is None:
                    continue
                threshold = float(expected)
                if key == "max_aa_deg":
                    if not _is_finite_float(aa_deg) or float(aa_deg) > threshold:
                        return False
                    continue
                if key == "min_aa_deg":
                    if not _is_finite_float(aa_deg) or float(aa_deg) < threshold:
                        return False
                    continue
                if key == "max_range_rate_mps":
                    if (
                        not _is_finite_float(range_rate_mps)
                        or float(range_rate_mps) > threshold
                    ):
                        return False
                    continue
                if key == "min_range_rate_mps":
                    if (
                        not _is_finite_float(range_rate_mps)
                        or float(range_rate_mps) < threshold
                    ):
                        return False
                    continue
                if key == "max_previous_vp_forward_bias_m":
                    if (
                        not _is_finite_float(previous_vp_forward_bias_m)
                        or float(previous_vp_forward_bias_m) > threshold
                    ):
                        return False
                    continue
                if key == "min_previous_vp_forward_bias_m":
                    if (
                        not _is_finite_float(previous_vp_forward_bias_m)
                        or float(previous_vp_forward_bias_m) < threshold
                    ):
                        return False
                    continue
                if key == "max_abs_previous_vp_lateral_bias_m":
                    if (
                        not _is_finite_float(previous_vp_lateral_bias_m)
                        or abs(float(previous_vp_lateral_bias_m)) > threshold
                    ):
                        return False
                    continue
                if key == "min_abs_previous_vp_lateral_bias_m":
                    if (
                        not _is_finite_float(previous_vp_lateral_bias_m)
                        or abs(float(previous_vp_lateral_bias_m)) < threshold
                    ):
                        return False
                    continue
                return False
            return True

        return _match_single(conditions)

    effective_cfg = dict(cfg)
    normalized_reason = (
        None
        if runtime_specialist_reason in (None, "")
        else str(runtime_specialist_reason)
    )
    override_key = None
    reason_overrides = cfg.get("reason_overrides", {}) or {}
    if normalized_reason is not None and isinstance(reason_overrides, dict):
        raw_override = reason_overrides.get(normalized_reason)
        override_values = None
        override_conditions = None
        if isinstance(raw_override, dict):
            override_conditions = raw_override.get("conditions")
            nested_override_values = raw_override.get("overrides")
            if isinstance(nested_override_values, dict):
                override_values = nested_override_values
            else:
                override_values = raw_override
        if isinstance(override_values, dict) and _conditions_match(
            override_conditions
        ):
            effective_cfg.update(override_values)
            override_key = normalized_reason
    return effective_cfg, {
        "runtime_reason": normalized_reason,
        "override_key": override_key,
    }


def _shape_tactical_basis_action_with_post_merge_recovery_profile(
    action: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, Dict[str, float]]:
    shaped = np.asarray(action, dtype=np.float64).copy()
    if shaped.shape[0] < 3:
        return shaped, {
            "ll_pre": float("nan"),
            "ll_post": float("nan"),
            "io_pre": float("nan"),
            "io_post": float("nan"),
            "cd_pre": float("nan"),
            "cd_post": float("nan"),
        }

    ll_pre = float(shaped[0])
    io_pre = float(shaped[1])
    cd_pre = float(shaped[2])

    ll_scale = float(cfg.get("tactical_basis_action_ll_scale", 1.0))
    shaped[0] = ll_pre * ll_scale
    ll_min = cfg.get("tactical_basis_action_ll_min")
    if ll_min is not None:
        shaped[0] = max(float(ll_min), float(shaped[0]))
    ll_max = cfg.get("tactical_basis_action_ll_max")
    if ll_max is not None:
        shaped[0] = min(float(ll_max), float(shaped[0]))

    io_scale = float(cfg.get("tactical_basis_action_io_scale", 1.0))
    shaped[1] = io_pre * io_scale
    io_abs_max = cfg.get("tactical_basis_action_io_abs_max")
    if io_abs_max is not None:
        shaped[1] = float(
            np.clip(float(shaped[1]), -abs(float(io_abs_max)), abs(float(io_abs_max)))
        )

    cd_scale = float(cfg.get("tactical_basis_action_cd_scale", 1.0))
    cd_bias = float(cfg.get("tactical_basis_action_cd_bias", 0.0))
    shaped[2] = cd_pre * cd_scale + cd_bias
    cd_min = cfg.get("tactical_basis_action_cd_min")
    if cd_min is not None:
        shaped[2] = max(float(cd_min), float(shaped[2]))
    cd_max = cfg.get("tactical_basis_action_cd_max")
    if cd_max is not None:
        shaped[2] = min(float(cd_max), float(shaped[2]))

    shaped[:3] = np.clip(shaped[:3], -1.0, 1.0)
    return shaped, {
        "ll_pre": ll_pre,
        "ll_post": float(shaped[0]),
        "io_pre": io_pre,
        "io_post": float(shaped[1]),
        "cd_pre": cd_pre,
        "cd_post": float(shaped[2]),
    }


def _apply_post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold(
    action: np.ndarray,
    cfg: dict,
    *,
    previous_vp_lateral_bias_m: float,
    entry_lateral_sign: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    held_action = np.asarray(action, dtype=np.float64).copy()
    enabled = bool(
        cfg.get("tactical_basis_action_io_entry_lateral_sign_hold_enabled", False)
    )
    release_vp_lateral_bias_m = float(
        cfg.get(
            "tactical_basis_action_io_entry_lateral_sign_hold_release_vp_lateral_bias_m",
            0.0,
        )
    )
    normalized_entry_lateral_sign = float(np.sign(entry_lateral_sign))
    hold_active = False
    hold_applied = False
    if (
        enabled
        and held_action.shape[0] >= 2
        and normalized_entry_lateral_sign != 0.0
        and np.isfinite(previous_vp_lateral_bias_m)
        and previous_vp_lateral_bias_m * normalized_entry_lateral_sign
        > -release_vp_lateral_bias_m
    ):
        hold_active = True
        if held_action[1] * normalized_entry_lateral_sign < 0.0:
            held_action[1] = (
                abs(float(held_action[1])) * normalized_entry_lateral_sign
            )
            hold_applied = True
    return held_action, {
        "io_entry_lateral_sign_hold_enabled": enabled,
        "io_entry_lateral_sign_hold_active": hold_active,
        "io_entry_lateral_sign_hold_applied": hold_applied,
        "io_entry_lateral_sign": normalized_entry_lateral_sign,
        "io_entry_lateral_sign_hold_previous_vp_lateral_bias_m": float(
            previous_vp_lateral_bias_m
        ),
        "io_entry_lateral_sign_hold_release_vp_lateral_bias_m": (
            release_vp_lateral_bias_m
        ),
    }


def _compute_post_merge_tactical_basis_recovery_profile_entry_window_state(
    cfg: dict,
    *,
    was_active: bool,
    is_active: bool,
    previous_steps_remaining: int,
) -> Dict[str, float]:
    configured_steps = max(int(cfg.get("entry_window_steps", 0) or 0), 0)
    armed = bool(is_active and not was_active and configured_steps > 0)
    steps_remaining = 0
    next_steps_remaining = 0
    active = False
    if is_active and configured_steps > 0:
        if armed:
            steps_remaining = configured_steps
        elif previous_steps_remaining > 0:
            steps_remaining = int(previous_steps_remaining)
        if steps_remaining > 0:
            active = True
            next_steps_remaining = max(int(steps_remaining) - 1, 0)
    return {
        "entry_window_configured_steps": configured_steps,
        "entry_window_armed": armed,
        "entry_window_active": active,
        "entry_window_steps_remaining": int(steps_remaining),
        "entry_window_next_steps_remaining": int(next_steps_remaining),
    }


def _apply_post_merge_tactical_basis_recovery_profile_entry_window(
    action: np.ndarray,
    cfg: dict,
    *,
    active: bool,
    steps_remaining: int,
    entry_lateral_sign: float,
    previous_vp_forward_bias_m: float,
    previous_altitude_delta_m: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    windowed_action = np.asarray(action, dtype=np.float64).copy()
    normalized_entry_lateral_sign = float(np.sign(entry_lateral_sign))
    io_min_abs = cfg.get("tactical_basis_action_io_entry_window_min_abs")
    ll_floor = cfg.get("tactical_basis_action_ll_entry_window_floor")
    ll_forward_bias_m_max = cfg.get(
        "tactical_basis_action_ll_entry_window_overdeep_vp_forward_bias_m_max"
    )
    cd_min = cfg.get("tactical_basis_action_cd_entry_window_min")
    cd_descending_altitude_delta_m_min_abs = cfg.get(
        "tactical_basis_action_cd_entry_window_descending_altitude_delta_m_min_abs"
    )
    forward_scale_override = cfg.get(
        "predicted_target_forward_scale_entry_window_override"
    )
    forward_scale_override_forward_bias_m_max = cfg.get(
        "predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max"
    )
    io_min_abs_applied = False
    ll_floor_applied = False
    cd_min_applied = False
    forward_scale_override_applied = False
    if active and steps_remaining > 0:
        if (
            windowed_action.shape[0] >= 2
            and io_min_abs is not None
            and normalized_entry_lateral_sign != 0.0
            and float(io_min_abs) > 0.0
        ):
            desired_abs = max(abs(float(windowed_action[1])), abs(float(io_min_abs)))
            desired_io = desired_abs * normalized_entry_lateral_sign
            if not np.isclose(float(windowed_action[1]), desired_io):
                windowed_action[1] = desired_io
                io_min_abs_applied = True
        if (
            windowed_action.shape[0] >= 1
            and ll_floor is not None
            and ll_forward_bias_m_max is not None
            and np.isfinite(previous_vp_forward_bias_m)
            and previous_vp_forward_bias_m <= float(ll_forward_bias_m_max)
        ):
            clamped_ll = max(float(windowed_action[0]), float(ll_floor))
            if not np.isclose(float(windowed_action[0]), clamped_ll):
                windowed_action[0] = clamped_ll
                ll_floor_applied = True
        if (
            windowed_action.shape[0] >= 3
            and cd_min is not None
            and cd_descending_altitude_delta_m_min_abs is not None
            and np.isfinite(previous_altitude_delta_m)
            and previous_altitude_delta_m
            <= -abs(float(cd_descending_altitude_delta_m_min_abs))
        ):
            clamped_cd = max(float(windowed_action[2]), float(cd_min))
            if not np.isclose(float(windowed_action[2]), clamped_cd):
                windowed_action[2] = clamped_cd
                cd_min_applied = True
        if forward_scale_override is not None and (
            forward_scale_override_forward_bias_m_max is None
            or (
                np.isfinite(previous_vp_forward_bias_m)
                and previous_vp_forward_bias_m
                <= float(forward_scale_override_forward_bias_m_max)
            )
        ):
            forward_scale_override_applied = True
    if windowed_action.shape[0] >= 3:
        windowed_action[:3] = np.clip(windowed_action[:3], -1.0, 1.0)
    return windowed_action, {
        "entry_window_previous_vp_forward_bias_m": float(previous_vp_forward_bias_m),
        "entry_window_previous_altitude_delta_m": float(previous_altitude_delta_m),
        "entry_window_io_min_abs": (
            float(io_min_abs) if io_min_abs is not None else float("nan")
        ),
        "entry_window_io_min_abs_applied": io_min_abs_applied,
        "entry_window_ll_floor": (
            float(ll_floor) if ll_floor is not None else float("nan")
        ),
        "entry_window_ll_floor_applied": ll_floor_applied,
        "entry_window_ll_overdeep_vp_forward_bias_m_max": (
            float(ll_forward_bias_m_max)
            if ll_forward_bias_m_max is not None
            else float("nan")
        ),
        "entry_window_cd_min": (
            float(cd_min) if cd_min is not None else float("nan")
        ),
        "entry_window_cd_min_applied": cd_min_applied,
        "entry_window_cd_descending_altitude_delta_m_min_abs": (
            float(cd_descending_altitude_delta_m_min_abs)
            if cd_descending_altitude_delta_m_min_abs is not None
            else float("nan")
        ),
        "entry_window_predicted_target_forward_scale_override": (
            float(forward_scale_override)
            if forward_scale_override is not None
            else float("nan")
        ),
        "entry_window_predicted_target_forward_scale_override_applied": (
            forward_scale_override_applied
        ),
        "entry_window_predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max": (
            float(forward_scale_override_forward_bias_m_max)
            if forward_scale_override_forward_bias_m_max is not None
            else float("nan")
        ),
    }


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

    def __init__(
        self,
        config: dict,
        opponent_policy: Optional[OpponentPolicy] = None,
        opponent_config: Optional[dict] = None,
    ):
        """
        Args:
            config (dict): Full environment and guidance configuration.
            opponent_policy: Optional embedded target-aircraft controller.
            opponent_config: Optional serializable opponent metadata.
        """
        self.config = config
        self.env_config = config.get("env", {})
        raw_task_name = config.get("task", {}).get("name")
        self.task_name = (
            str(raw_task_name) if raw_task_name not in (None, "") else None
        )
        self.opponent_policy = opponent_policy
        self.opponent_config = (
            dict(opponent_config)
            if opponent_config is not None
            else dict(config.get("opponent", {}))
        )
        self._opponent_stage = self.opponent_config.get("stage", "none")
        self._runtime_specialist_key: Optional[str] = None
        self._runtime_specialist_profile: Optional[str] = None
        self._runtime_specialist_mode_name: Optional[str] = None
        self._runtime_specialist_reason: Optional[str] = None
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
        self._backend_fallback_occurred = False
        self._backend_fallback_reason: Optional[str] = None
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
                self._low_level_controller = self._build_low_level_controller(
                    config
                )
                self._target_low_level_controller = self._build_low_level_controller(
                    config
                )
            except Exception as exc:
                if strict_backend:
                    raise RuntimeError(
                        f"JSBSim backend initialization failed "
                        f"(strict_backend=True): {exc}"
                    ) from exc
                # Fallback to simple env if JSBSim init fails
                self._backend_fallback_occurred = True
                self._backend_fallback_reason = str(exc)
                import warnings
                warnings.warn(
                    f"JSBSim backend initialization failed: {exc}. "
                    f"Falling back to SimplePointMassEnv.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.jsbsim_env = None
                self._low_level_controller = None
                self._target_low_level_controller = None
                self._simple_env = SimplePointMassEnv(self.env_config)
                self._backend = "simple"
        else:
            self.jsbsim_env = None
            self._low_level_controller = None
            self._target_low_level_controller = None
            self._simple_env = SimplePointMassEnv(self.env_config)
            self._backend = "simple"

        # Observation schema configuration
        self.obs_config = config.get("observation", {})
        self._include_gains_in_observation = bool(
            self.obs_config.get("include_gains", False)
        )
        self._include_guidance_state_in_observation = bool(
            self.obs_config.get("include_guidance_state", False)
        )
        self._include_saturation_in_observation = bool(
            self.obs_config.get("include_saturation", False)
        )
        self._include_opponent_stage_in_observation = bool(
            self.obs_config.get("include_opponent_stage", False)
        )
        self._include_task_type_in_observation = bool(
            self.obs_config.get("include_task_type", False)
        )
        # Last virtual point for guidance-state features (e.g. VP tracking error)
        self._last_virtual_point: Optional[dict] = None
        # Last command saturation flags for observation
        self._last_command_saturation = {
            "nz_saturated": 0.0,
            "roll_rate_saturated": 0.0,
            "throttle_saturated": 0.0,
        }
        self._merge_min_range_so_far_m = float("inf")
        self._merge_seen_close_range = False
        self._close_range_anchor_ready = False
        self._first_pass_complete = False
        self._first_pass_completion_step = None
        self._post_merge_anchor_mode_released = False
        self._post_merge_anchor_mode_has_activated = False
        self._post_merge_anchor_mode_ego_only_streak_steps = 0
        self._post_merge_anchor_mode_lateral_world_offset_latched = None
        self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
            None
        )
        self._post_merge_predicted_target_blend_released = False
        self._post_merge_offensive_anchor_blend_released = False
        self._post_merge_offensive_anchor_blend_release_step = None
        self._post_merge_offensive_anchor_blend_has_activated = False
        self._post_merge_offensive_anchor_blend_ego_only_streak_steps = 0
        self._post_merge_offensive_anchor_lateral_world_offset_latched = None
        self._post_merge_predicted_target_forward_scale_released = False
        self._post_merge_predicted_target_forward_scale_ego_only_streak_steps = 0
        self._post_merge_tactical_basis_recovery_profile_was_active = False
        self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign = 0.0
        self._post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining = 0
        self._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m = (
            float("nan")
        )
        self._post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m = (
            float("nan")
        )
        self._post_merge_tactical_basis_recovery_profile_previous_altitude_m = (
            float("nan")
        )

        # Submodules
        vp_config = config.get(
            "virtual_point", config.get("guidance", {}).get("virtual_point", {})
        )
        vp_enabled = vp_config.get("enabled", True)
        e2e_enabled = config.get("end_to_end", {}).get("enabled", False)
        self._default_post_merge_anchor_mode = self._validate_anchor_mode_optional(
            vp_config.get("post_merge_anchor_mode"),
            "virtual_point.post_merge_anchor_mode",
        )
        self._post_merge_anchor_mode_by_task = (
            self._normalize_optional_anchor_mode_by_task(
                vp_config.get("post_merge_anchor_mode_by_task", {}),
                "virtual_point.post_merge_anchor_mode_by_task",
            )
        )
        self._default_post_merge_anchor_mode_requires_geometry_disadvantage = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_anchor_mode_requires_geometry_disadvantage",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_requires_geometry_disadvantage"
                ),
            )
        )
        self._post_merge_anchor_mode_requires_geometry_disadvantage_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_requires_geometry_disadvantage_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
                ),
            )
        )
        self._default_post_merge_anchor_mode_recovery_below_altitude_m = (
            self._validate_nonnegative_float_optional(
                vp_config.get("post_merge_anchor_mode_recovery_below_altitude_m"),
                "virtual_point.post_merge_anchor_mode_recovery_below_altitude_m",
            )
        )
        self._post_merge_anchor_mode_recovery_below_altitude_m_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_recovery_below_altitude_m_by_task",
                    {},
                ),
                "virtual_point.post_merge_anchor_mode_recovery_below_altitude_m_by_task",
            )
        )
        self._default_post_merge_anchor_mode_release_ego_only_streak_steps = (
            self._validate_nonnegative_int_optional(
                vp_config.get("post_merge_anchor_mode_release_ego_only_streak_steps"),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_release_ego_only_streak_steps"
                ),
            )
        )
        self._post_merge_anchor_mode_release_ego_only_streak_steps_by_task = (
            self._normalize_optional_nonnegative_int_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_release_ego_only_streak_steps_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_release_ego_only_streak_steps_by_task"
                ),
            )
        )
        self._default_post_merge_anchor_mode_release_reset_on_streak_break = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_anchor_mode_release_reset_on_streak_break",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_release_reset_on_streak_break"
                ),
            )
        )
        self._post_merge_anchor_mode_release_reset_on_streak_break_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_release_reset_on_streak_break_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_release_reset_on_streak_break_by_task"
                ),
            )
        )
        self._default_post_merge_anchor_mode_longitudinal_scale = (
            self._validate_nonnegative_float_optional(
                vp_config.get("post_merge_anchor_mode_longitudinal_scale"),
                "virtual_point.post_merge_anchor_mode_longitudinal_scale",
            )
        )
        self._post_merge_anchor_mode_longitudinal_scale_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_longitudinal_scale_by_task",
                    {},
                ),
                "virtual_point.post_merge_anchor_mode_longitudinal_scale_by_task",
            )
        )
        self._default_post_merge_anchor_mode_lateral_scale = (
            self._validate_nonnegative_float_optional(
                vp_config.get("post_merge_anchor_mode_lateral_scale"),
                "virtual_point.post_merge_anchor_mode_lateral_scale",
            )
        )
        self._post_merge_anchor_mode_lateral_scale_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_lateral_scale_by_task",
                    {},
                ),
                "virtual_point.post_merge_anchor_mode_lateral_scale_by_task",
            )
        )
        self._default_post_merge_anchor_mode_offensive_anchor_blend = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_anchor_mode_offensive_anchor_blend"),
                "virtual_point.post_merge_anchor_mode_offensive_anchor_blend",
            )
        )
        self._post_merge_anchor_mode_offensive_anchor_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_offensive_anchor_blend_by_task",
                    {},
                ),
                "virtual_point.post_merge_anchor_mode_offensive_anchor_blend_by_task",
            )
        )
        self._default_post_merge_anchor_mode_offensive_anchor_longitudinal_blend = (
            self._validate_unit_interval_optional(
                vp_config.get(
                    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend"
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend"
                ),
            )
        )
        self._post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"
                ),
            )
        )
        self._default_post_merge_anchor_mode_offensive_anchor_lateral_blend = (
            self._validate_unit_interval_optional(
                vp_config.get(
                    "post_merge_anchor_mode_offensive_anchor_lateral_blend"
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_offensive_anchor_lateral_blend"
                ),
            )
        )
        self._post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"
                ),
            )
        )
        self._default_post_merge_anchor_mode_lateral_world_offset_latch_on_activation = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation"
                ),
            )
        )
        self._post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    (
                        "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
                    ),
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
                ),
            )
        )
        self._default_post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min = (
            self._validate_nonnegative_float_optional(
                vp_config.get(
                    "post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min"
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min"
                ),
            )
        )
        self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    (
                        "post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min_by_task"
                    ),
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min_by_task"
                ),
            )
        )
        self._default_close_range_anchor_mode = self._validate_anchor_mode_optional(
            vp_config.get("close_range_anchor_mode"),
            "virtual_point.close_range_anchor_mode",
        )
        self._close_range_anchor_mode_by_task = (
            self._normalize_optional_anchor_mode_by_task(
                vp_config.get("close_range_anchor_mode_by_task", {}),
                "virtual_point.close_range_anchor_mode_by_task",
            )
        )
        self._default_close_range_anchor_trigger_range_m = (
            self._validate_nonnegative_float_optional(
                vp_config.get("close_range_anchor_trigger_range_m"),
                "virtual_point.close_range_anchor_trigger_range_m",
            )
        )
        self._close_range_anchor_trigger_range_m_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get("close_range_anchor_trigger_range_m_by_task", {}),
                "virtual_point.close_range_anchor_trigger_range_m_by_task",
            )
        )
        self._default_close_range_anchor_alignment_angle_deg_max = (
            self._validate_nonnegative_float_optional(
                vp_config.get("close_range_anchor_alignment_angle_deg_max"),
                "virtual_point.close_range_anchor_alignment_angle_deg_max",
            )
        )
        self._close_range_anchor_alignment_angle_deg_max_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "close_range_anchor_alignment_angle_deg_max_by_task",
                    {},
                ),
                "virtual_point.close_range_anchor_alignment_angle_deg_max_by_task",
            )
        )
        self._default_close_range_anchor_release_on_post_merge = (
            self._validate_bool(
                vp_config.get("close_range_anchor_release_on_post_merge", False),
                "virtual_point.close_range_anchor_release_on_post_merge",
            )
        )
        self._close_range_anchor_release_on_post_merge_by_task = (
            self._normalize_bool_by_task(
                vp_config.get("close_range_anchor_release_on_post_merge_by_task", {}),
                "virtual_point.close_range_anchor_release_on_post_merge_by_task",
            )
        )
        self._default_close_range_anchor_requires_first_pass = self._validate_bool(
            vp_config.get("close_range_anchor_requires_first_pass", False),
            "virtual_point.close_range_anchor_requires_first_pass",
        )
        self._close_range_anchor_requires_first_pass_by_task = (
            self._normalize_bool_by_task(
                vp_config.get("close_range_anchor_requires_first_pass_by_task", {}),
                "virtual_point.close_range_anchor_requires_first_pass_by_task",
            )
        )
        self._default_close_range_anchor_offensive_anchor_blend = (
            self._validate_unit_interval_optional(
                vp_config.get("close_range_anchor_offensive_anchor_blend"),
                "virtual_point.close_range_anchor_offensive_anchor_blend",
            )
        )
        self._close_range_anchor_offensive_anchor_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "close_range_anchor_offensive_anchor_blend_by_task",
                    {},
                ),
                "virtual_point.close_range_anchor_offensive_anchor_blend_by_task",
            )
        )
        self._default_close_range_anchor_release_alignment_angle_deg_max = (
            self._validate_nonnegative_float_optional(
                vp_config.get("close_range_anchor_release_alignment_angle_deg_max"),
                "virtual_point.close_range_anchor_release_alignment_angle_deg_max",
            )
        )
        self._close_range_anchor_release_alignment_angle_deg_max_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "close_range_anchor_release_alignment_angle_deg_max_by_task",
                    {},
                ),
                "virtual_point.close_range_anchor_release_alignment_angle_deg_max_by_task",
            )
        )
        self._default_post_merge_predicted_target_blend = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_predicted_target_blend"),
                "virtual_point.post_merge_predicted_target_blend",
            )
        )
        self._post_merge_predicted_target_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get("post_merge_predicted_target_blend_by_task", {}),
                "virtual_point.post_merge_predicted_target_blend_by_task",
            )
        )
        self._default_post_merge_offensive_anchor_blend = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_offensive_anchor_blend"),
                "virtual_point.post_merge_offensive_anchor_blend",
            )
        )
        self._post_merge_offensive_anchor_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get("post_merge_offensive_anchor_blend_by_task", {}),
                "virtual_point.post_merge_offensive_anchor_blend_by_task",
            )
        )
        self._default_post_merge_offensive_anchor_longitudinal_scale = (
            self._validate_nonnegative_float_optional(
                vp_config.get("post_merge_offensive_anchor_longitudinal_scale"),
                "virtual_point.post_merge_offensive_anchor_longitudinal_scale",
            )
        )
        self._post_merge_offensive_anchor_longitudinal_scale_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_longitudinal_scale_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_longitudinal_scale_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_lateral_scale = (
            self._validate_nonnegative_float_optional(
                vp_config.get("post_merge_offensive_anchor_lateral_scale"),
                "virtual_point.post_merge_offensive_anchor_lateral_scale",
            )
        )
        self._post_merge_offensive_anchor_lateral_scale_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_lateral_scale_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_lateral_scale_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_requires_geometry_disadvantage = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min = (
            self._validate_nonnegative_float_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min"
                ),
            )
        )
        self._post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_blend = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_offensive_anchor_blend_release_blend"),
                "virtual_point.post_merge_offensive_anchor_blend_release_blend",
            )
        )
        self._post_merge_offensive_anchor_blend_release_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_blend_by_task", {}
                ),
                "virtual_point.post_merge_offensive_anchor_blend_release_blend_by_task",
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_ego_only_streak_steps = (
            self._validate_nonnegative_int_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task = (
            self._normalize_optional_nonnegative_int_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_reset_on_streak_break = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m = (
            self._validate_nonnegative_float_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m = (
            self._validate_nonnegative_float_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend = (
            self._validate_unit_interval_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_recovery_lateral_blend = (
            self._validate_unit_interval_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max = (
            self._validate_float_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task = (
            self._normalize_optional_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min = (
            self._validate_float_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task = (
            self._normalize_optional_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m = (
            self._validate_float_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_"
                    "vp_forward_bias_clamp_negative_lateral_below_altitude_m"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task = (
            self._normalize_optional_float_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_"
                    "vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
                ),
            )
        )
        self._post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_lateral_only = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_lateral_only"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_lateral_only_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps = (
            self._validate_nonnegative_int_optional(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task = (
            self._normalize_optional_nonnegative_int_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
                ),
            )
        )
        self._default_post_merge_offensive_anchor_blend_release_direct_track_enabled = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled",
                    True,
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled"
                ),
            )
        )
        self._post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
                ),
            )
        )
        self._default_post_merge_predicted_target_forward_scale = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_predicted_target_forward_scale"),
                "virtual_point.post_merge_predicted_target_forward_scale",
            )
        )
        self._post_merge_predicted_target_forward_scale_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get("post_merge_predicted_target_forward_scale_by_task", {}),
                "virtual_point.post_merge_predicted_target_forward_scale_by_task",
            )
        )
        self._default_post_merge_predicted_target_forward_scale_release_scale = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_predicted_target_forward_scale_release_scale"),
                (
                    "virtual_point."
                    "post_merge_predicted_target_forward_scale_release_scale"
                ),
            )
        )
        self._post_merge_predicted_target_forward_scale_release_scale_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_predicted_target_forward_scale_release_scale_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_forward_scale_release_scale_by_task"
                ),
            )
        )
        self._default_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps = (
            self._validate_nonnegative_int_optional(
                vp_config.get(
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
                ),
            )
        )
        self._post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task = (
            self._normalize_optional_nonnegative_int_by_task(
                vp_config.get(
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
                ),
            )
        )
        self._default_post_merge_predicted_target_forward_scale_release_reset_on_streak_break = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
                ),
            )
        )
        self._post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
                ),
            )
        )
        self._default_post_merge_predicted_target_forward_scale_hold_steps = (
            self._validate_nonnegative_int_optional(
                vp_config.get("post_merge_predicted_target_forward_scale_hold_steps"),
                "virtual_point.post_merge_predicted_target_forward_scale_hold_steps",
            )
        )
        self._post_merge_predicted_target_forward_scale_hold_steps_by_task = (
            self._normalize_optional_nonnegative_int_by_task(
                vp_config.get(
                    "post_merge_predicted_target_forward_scale_hold_steps_by_task",
                    {},
                ),
                "virtual_point.post_merge_predicted_target_forward_scale_hold_steps_by_task",
            )
        )
        self._default_post_merge_predicted_target_blend_release_on_attack_zone = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_predicted_target_blend_release_on_attack_zone",
                    False,
                ),
                "virtual_point.post_merge_predicted_target_blend_release_on_attack_zone",
            )
        )
        self._post_merge_predicted_target_blend_release_on_attack_zone_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_predicted_target_blend_release_on_attack_zone_by_task",
                    {},
                ),
                "virtual_point.post_merge_predicted_target_blend_release_on_attack_zone_by_task",
            )
        )
        self._default_post_merge_predicted_target_blend_release_requires_target_attack_zone = (
            self._validate_bool(
                vp_config.get(
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone",
                    False,
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone"
                ),
            )
        )
        self._post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task = (
            self._normalize_bool_by_task(
                vp_config.get(
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
                ),
            )
        )
        self._default_post_merge_predicted_target_blend_release_below_altitude_m = (
            self._validate_nonnegative_float_optional(
                vp_config.get("post_merge_predicted_target_blend_release_below_altitude_m"),
                (
                    "virtual_point."
                    "post_merge_predicted_target_blend_release_below_altitude_m"
                ),
            )
        )
        self._post_merge_predicted_target_blend_release_below_altitude_m_by_task = (
            self._normalize_optional_nonnegative_float_by_task(
                vp_config.get(
                    "post_merge_predicted_target_blend_release_below_altitude_m_by_task",
                    {},
                ),
                (
                    "virtual_point."
                    "post_merge_predicted_target_blend_release_below_altitude_m_by_task"
                ),
            )
        )
        self._default_post_merge_predicted_target_blend_hold_steps = (
            self._validate_nonnegative_int_optional(
                vp_config.get("post_merge_predicted_target_blend_hold_steps"),
                "virtual_point.post_merge_predicted_target_blend_hold_steps",
            )
        )
        self._post_merge_predicted_target_blend_hold_steps_by_task = (
            self._normalize_optional_nonnegative_int_by_task(
                vp_config.get("post_merge_predicted_target_blend_hold_steps_by_task", {}),
                "virtual_point.post_merge_predicted_target_blend_hold_steps_by_task",
            )
        )
        self._default_post_merge_predicted_target_blend_release_blend = (
            self._validate_unit_interval_optional(
                vp_config.get("post_merge_predicted_target_blend_release_blend"),
                "virtual_point.post_merge_predicted_target_blend_release_blend",
            )
        )
        self._post_merge_predicted_target_blend_release_blend_by_task = (
            self._normalize_optional_unit_interval_by_task(
                vp_config.get(
                    "post_merge_predicted_target_blend_release_blend_by_task",
                    {},
                ),
                "virtual_point.post_merge_predicted_target_blend_release_blend_by_task",
            )
        )
        self._default_close_range_anchor_post_merge_hold_steps = (
            self._validate_nonnegative_int(
                vp_config.get("close_range_anchor_post_merge_hold_steps", 0),
                "virtual_point.close_range_anchor_post_merge_hold_steps",
            )
        )
        self._close_range_anchor_post_merge_hold_steps_by_task = (
            self._normalize_nonnegative_int_by_task(
                vp_config.get("close_range_anchor_post_merge_hold_steps_by_task", {}),
                "virtual_point.close_range_anchor_post_merge_hold_steps_by_task",
            )
        )
        post_merge_recovery_profile_cfg = vp_config.get(
            "post_merge_tactical_basis_recovery_profile",
            {},
        )
        if post_merge_recovery_profile_cfg in (None, ""):
            post_merge_recovery_profile_cfg = {}
        if not isinstance(post_merge_recovery_profile_cfg, dict):
            raise ValueError(
                "virtual_point.post_merge_tactical_basis_recovery_profile must be a mapping"
            )
        self._post_merge_tactical_basis_recovery_profile_cfg = dict(
            post_merge_recovery_profile_cfg
        )

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
                self.virtual_point_generator = VirtualPointGenerator(
                    vp_config,
                    task_name=self.task_name,
                )
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
        self.guidance = self._build_guidance_law(guidance_config)
        self._target_guidance = self._build_guidance_law(guidance_config)

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
                "env": config.get("env", {}),
                "reward": config.get("reward", {}),
                "limits": {
                    **config.get("limits", {}),
                    **guidance_config.get("limits", {}),
                },
            }
            self.command_post_processor = CommandPostProcessor(processor_config)

        self.reward_calculator = RewardCalculator(config)
        self.termination_checker = TerminationChecker(self.env_config)
        self.combat_hp = CombatHPManager.from_config(config)

        # Command filter (three independent channels)
        filter_alpha = (
            config.get("guidance", {}).get("gains", {}).get("alpha_filter", 0.3)
        )
        self._command_filter = MultiChannelCommandFilter(alpha=filter_alpha)
        self._target_command_filter = MultiChannelCommandFilter(alpha=filter_alpha)

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
        self._prediction_buffer_synced_for_action = False
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

        # Robustness testing: sensor noise, communication delay, wind.
        self._robustness = config.get("robustness", {})
        self._sensor_noise = self._robustness.get("sensor_noise", {})
        self._comm_delay = self._robustness.get("communication_delay", {})
        self._high_level_dt = float(self.env_config.get("high_level_dt", 1.0 / self.decision_freq))
        self._robustness_seed_base = int(self._robustness.get("seed", 0))
        self._robustness_rng = np.random.default_rng(self._robustness_seed_base)
        self._prediction_noise_rng = np.random.default_rng(
            self._robustness_seed_base + 7919
        )
        self._prev_delayed_command: Optional[dict] = None

        self.current_step = 0
        self._episode_count = 0
        self._sim_time_s = 0.0
        self._last_aggressiveness = None
        self._last_pid_gain_deltas = None
        self._last_virtual_point = None
        self._last_observation_schema = {}
        self._last_command_saturation = {
            "nz_saturated": 0.0,
            "roll_rate_saturated": 0.0,
            "throttle_saturated": 0.0,
        }
        self._merge_min_range_so_far_m = float("inf")
        self._merge_seen_close_range = False
        self._close_range_anchor_ready = False
        self._first_pass_complete = False
        self._first_pass_completion_step = None
        self._post_merge_predicted_target_blend_released = False
        self._post_merge_predicted_target_forward_scale_released = False
        self._post_merge_predicted_target_forward_scale_ego_only_streak_steps = 0

    # ------------------------------------------------------------------
    # Low-level controller factory
    # ------------------------------------------------------------------

    def _build_guidance_law(self, guidance_config: dict):
        """Construct a guidance law from the existing guidance config."""
        guidance_mode = str(guidance_config.get("mode", "los_rate")).lower()
        if guidance_mode == "los_rate":
            return LOSRateGuidance(guidance_config)
        if guidance_mode == "proportional_navigation":
            return ProportionalNavigationGuidance(guidance_config)
        if guidance_mode == "hybrid":
            return HybridGuidance(guidance_config)
        if guidance_mode == "mpc":
            return MPCGuidance(guidance_config)
        raise ValueError(f"Unknown guidance mode: {guidance_mode}")

    def _build_low_level_controller(self, config: dict):
        """
        Select the low-level controller based on the ``low_level_controller``
        config block.  Unknown / legacy configurations fall back to the original
        feed-forward ``LowLevelController`` so existing configs keep working.
        """
        ll_controller_cfg = config.get("low_level_controller", {})
        if isinstance(ll_controller_cfg, str):
            ll_controller_class = ll_controller_cfg
            ll_controller_kwargs = {}
        else:
            ll_controller_class = ll_controller_cfg.get("type", "")
            ll_controller_kwargs = {
                k: v for k, v in ll_controller_cfg.items() if k != "type"
            }

        controller_config = {
            **config.get("guidance", {}).get("gains", {}),
            **ll_controller_kwargs,
            "limits": {
                **config.get("limits", {}),
                **config.get("guidance", {}).get("limits", {}),
            },
        }

        if ll_controller_class in ("enhanced", "enhanced_pid"):
            return EnhancedPIDController(controller_config)
        if ll_controller_class in ("robust", "robust_pid"):
            return RobustPIDController(controller_config)
        if ll_controller_class in ("gain_scheduled", "gain_scheduled_pid"):
            return GainScheduledPIDController(controller_config)
        if ll_controller_class in ("baseline_pid",):
            return BaselinePIDController(controller_config)
        if ll_controller_class in ("hybrid_ppo_pid", "ppo_pid"):
            return HybridPPOPIDAdapter(controller_config)

        # Legacy / unspecified configuration: keep the original behavior.
        return LowLevelController(controller_config)

    # ------------------------------------------------------------------
    # Task-specific hooks (for MultiWaypoint / SustainedTurn subclasses)
    # ------------------------------------------------------------------

    def _task_reset(self, seed=None) -> None:
        """Subclasses may override for task-specific reset logic."""
        return None

    def _task_get_target_state(self, backend_target_state, own_state):
        """Subclasses may replace the backend target with a synthetic target."""
        return backend_target_state

    def _task_uses_backend_target(self) -> bool:
        """Return False for tasks whose target state is fully synthetic."""
        return True

    def _task_pre_step(self, own_state, target_state):
        """Subclasses may update task state before guidance is computed."""
        return own_state, target_state

    def _task_adjust_command(
        self,
        raw_command,
        own_state,
        target_state,
        rel_state,
        *,
        use_command_override: bool = False,
    ):
        """Subclasses may post-process task commands before global safety logic."""
        return dict(raw_command), {}

    def _task_post_step(
        self,
        own_state_post,
        target_state_post,
        info,
        reward,
        terminated,
        truncated,
    ):
        """Subclasses may override reward / done / info after the step."""
        return reward, terminated, truncated, info

    @staticmethod
    def _validate_anchor_mode_optional(value, field_name):
        if value in (None, "", "none"):
            return None
        mode = str(value)
        if mode not in VALID_ANCHOR_MODES:
            raise ValueError(
                f"Unknown {field_name}={mode!r}; expected one of "
                f"{sorted(VALID_ANCHOR_MODES)} or None"
            )
        return mode

    def _normalize_optional_anchor_mode_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{field_name} must be a mapping of task_name -> anchor_mode"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_anchor_mode_optional(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_nonnegative_float_optional(value, field_name):
        if value is None:
            return None
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(
                f"Invalid {field_name}={value!r}; expected a finite value >= 0"
            )
        return numeric_value

    def _normalize_optional_nonnegative_float_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{field_name} must be a mapping of task_name -> nonnegative float"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_float_optional(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_float_optional(value, field_name):
        if value is None:
            return None
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(
                f"Invalid {field_name}={value!r}; expected a finite value"
            )
        return numeric_value

    def _normalize_optional_float_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{field_name} must be a mapping of task_name -> finite float"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_float_optional(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_unit_interval_optional(value, field_name):
        if value is None:
            return None
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value < 0.0 or numeric_value > 1.0:
            raise ValueError(
                f"Invalid {field_name}={value!r}; expected a finite value in [0, 1]"
            )
        return numeric_value

    def _normalize_optional_unit_interval_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{field_name} must be a mapping of task_name -> float in [0, 1]"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_unit_interval_optional(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_nonnegative_int(value, field_name):
        numeric_value = int(value)
        if float(value) != float(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"Invalid {field_name}={value!r}; expected a nonnegative integer"
            )
        return numeric_value

    @staticmethod
    def _validate_nonnegative_int_optional(value, field_name):
        if value is None:
            return None
        return CloseRangeTrackingEnv._validate_nonnegative_int(value, field_name)

    def _normalize_optional_nonnegative_int_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{field_name} must be a mapping of task_name -> nonnegative integer"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_int_optional(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    def _normalize_nonnegative_int_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{field_name} must be a mapping of task_name -> nonnegative integer"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_int(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_bool(value, field_name):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"Invalid {field_name}={value!r}; expected a boolean")

    def _normalize_bool_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(f"{field_name} must be a mapping of task_name -> bool")
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_bool(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    def _resolve_post_merge_anchor_mode(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode
        return self._post_merge_anchor_mode_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode,
        )

    def _resolve_post_merge_anchor_mode_requires_geometry_disadvantage(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_requires_geometry_disadvantage
        return self._post_merge_anchor_mode_requires_geometry_disadvantage_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_requires_geometry_disadvantage,
        )

    def _resolve_post_merge_anchor_mode_release_ego_only_streak_steps(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_release_ego_only_streak_steps
        return self._post_merge_anchor_mode_release_ego_only_streak_steps_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_release_ego_only_streak_steps,
        )

    def _resolve_post_merge_anchor_mode_recovery_below_altitude_m(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_recovery_below_altitude_m
        return self._post_merge_anchor_mode_recovery_below_altitude_m_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_recovery_below_altitude_m,
        )

    def _resolve_post_merge_anchor_mode_release_reset_on_streak_break(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_release_reset_on_streak_break
        return self._post_merge_anchor_mode_release_reset_on_streak_break_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_release_reset_on_streak_break,
        )

    def _resolve_post_merge_anchor_mode_longitudinal_scale(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_longitudinal_scale
        return self._post_merge_anchor_mode_longitudinal_scale_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_longitudinal_scale,
        )

    def _resolve_post_merge_anchor_mode_lateral_scale(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_lateral_scale
        return self._post_merge_anchor_mode_lateral_scale_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_lateral_scale,
        )

    def _resolve_post_merge_anchor_mode_offensive_anchor_blend(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_offensive_anchor_blend
        return self._post_merge_anchor_mode_offensive_anchor_blend_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_offensive_anchor_blend,
        )

    def _resolve_post_merge_anchor_mode_offensive_anchor_longitudinal_blend(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_offensive_anchor_longitudinal_blend
        return self._post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_offensive_anchor_longitudinal_blend,
        )

    def _resolve_post_merge_anchor_mode_offensive_anchor_lateral_blend(self):
        if self.task_name is None:
            return self._default_post_merge_anchor_mode_offensive_anchor_lateral_blend
        return self._post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task.get(
            self.task_name,
            self._default_post_merge_anchor_mode_offensive_anchor_lateral_blend,
        )

    def _resolve_post_merge_anchor_mode_lateral_world_offset_latch_on_activation(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_anchor_mode_lateral_world_offset_latch_on_activation
            )
        return (
            self._post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task.get(
                self.task_name,
                self._default_post_merge_anchor_mode_lateral_world_offset_latch_on_activation,
            )
        )

    def _resolve_post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min
            )
        return (
            self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min_by_task.get(
                self.task_name,
                self._default_post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min,
            )
        )

    def _resolve_close_range_anchor_mode(self):
        if self.task_name is None:
            return self._default_close_range_anchor_mode
        return self._close_range_anchor_mode_by_task.get(
            self.task_name,
            self._default_close_range_anchor_mode,
        )

    def _resolve_close_range_anchor_trigger_range_m(self, default_range_m):
        if self.task_name is None:
            resolved_value = self._default_close_range_anchor_trigger_range_m
        else:
            resolved_value = self._close_range_anchor_trigger_range_m_by_task.get(
                self.task_name,
                self._default_close_range_anchor_trigger_range_m,
            )
        if resolved_value is None:
            return float(default_range_m)
        return float(resolved_value)

    def _resolve_close_range_anchor_alignment_angle_deg_max(self):
        if self.task_name is None:
            return self._default_close_range_anchor_alignment_angle_deg_max
        return self._close_range_anchor_alignment_angle_deg_max_by_task.get(
            self.task_name,
            self._default_close_range_anchor_alignment_angle_deg_max,
        )

    def _resolve_close_range_anchor_release_on_post_merge(self):
        if self.task_name is None:
            return self._default_close_range_anchor_release_on_post_merge
        return self._close_range_anchor_release_on_post_merge_by_task.get(
            self.task_name,
            self._default_close_range_anchor_release_on_post_merge,
        )

    def _resolve_close_range_anchor_requires_first_pass(self):
        if self.task_name is None:
            return self._default_close_range_anchor_requires_first_pass
        return self._close_range_anchor_requires_first_pass_by_task.get(
            self.task_name,
            self._default_close_range_anchor_requires_first_pass,
        )

    def _resolve_close_range_anchor_offensive_anchor_blend(self):
        if self.task_name is None:
            return self._default_close_range_anchor_offensive_anchor_blend
        return self._close_range_anchor_offensive_anchor_blend_by_task.get(
            self.task_name,
            self._default_close_range_anchor_offensive_anchor_blend,
        )

    def _resolve_close_range_anchor_release_alignment_angle_deg_max(self):
        if self.task_name is None:
            return self._default_close_range_anchor_release_alignment_angle_deg_max
        return self._close_range_anchor_release_alignment_angle_deg_max_by_task.get(
            self.task_name,
            self._default_close_range_anchor_release_alignment_angle_deg_max,
        )

    def _resolve_post_merge_predicted_target_blend(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_blend
        return self._post_merge_predicted_target_blend_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_blend,
        )

    def _resolve_post_merge_offensive_anchor_blend(self):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_blend
        return self._post_merge_offensive_anchor_blend_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend,
        )

    def _resolve_post_merge_offensive_anchor_longitudinal_scale(self):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_longitudinal_scale
        return self._post_merge_offensive_anchor_longitudinal_scale_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_longitudinal_scale,
        )

    def _resolve_post_merge_offensive_anchor_lateral_scale(self):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_lateral_scale
        return self._post_merge_offensive_anchor_lateral_scale_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_lateral_scale,
        )

    def _resolve_post_merge_offensive_anchor_blend_requires_geometry_disadvantage(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_requires_geometry_disadvantage
            )
        return self._post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_requires_geometry_disadvantage,
        )

    def _resolve_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min
            )
        return self._post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_blend(self):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_blend_release_blend
        return self._post_merge_offensive_anchor_blend_release_blend_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_blend,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_ego_only_streak_steps(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_ego_only_streak_steps
            )
        return self._post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_ego_only_streak_steps,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_reset_on_streak_break(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_reset_on_streak_break
            )
        return self._post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_reset_on_streak_break,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
            )
        return self._post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
            )
        return self._post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
            )
        return self._post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_recovery_lateral_blend(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_recovery_lateral_blend
            )
        return self._post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_recovery_lateral_blend,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
            )
        return self._post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min
            )
        return self._post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m
            )
        return self._post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m,
        )

    def _resolve_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation
            )
        return (
            self._post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task.get(
                self.task_name,
                self._default_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation,
            )
        )

    def _resolve_post_merge_offensive_anchor_blend_release_lateral_only(self):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_blend_release_lateral_only
        return self._post_merge_offensive_anchor_blend_release_lateral_only_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_lateral_only,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps(
        self,
    ):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
        return self._post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps,
        )

    def _resolve_post_merge_offensive_anchor_blend_release_direct_track_enabled(self):
        if self.task_name is None:
            return self._default_post_merge_offensive_anchor_blend_release_direct_track_enabled
        return self._post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task.get(
            self.task_name,
            self._default_post_merge_offensive_anchor_blend_release_direct_track_enabled,
        )

    def _resolve_post_merge_predicted_target_forward_scale(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_forward_scale
        return self._post_merge_predicted_target_forward_scale_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_forward_scale,
        )

    def _resolve_post_merge_predicted_target_forward_scale_release_scale(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_forward_scale_release_scale
        return self._post_merge_predicted_target_forward_scale_release_scale_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_forward_scale_release_scale,
        )

    def _resolve_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
            )
        return self._post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps,
        )

    def _resolve_post_merge_predicted_target_forward_scale_hold_steps(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_forward_scale_hold_steps
        return self._post_merge_predicted_target_forward_scale_hold_steps_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_forward_scale_hold_steps,
        )

    def _resolve_post_merge_predicted_target_forward_scale_release_reset_on_streak_break(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_predicted_target_forward_scale_release_reset_on_streak_break
            )
        return self._post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_forward_scale_release_reset_on_streak_break,
        )

    def _resolve_post_merge_predicted_target_blend_release_on_attack_zone(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_blend_release_on_attack_zone
        return self._post_merge_predicted_target_blend_release_on_attack_zone_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_blend_release_on_attack_zone,
        )

    def _resolve_post_merge_predicted_target_blend_release_requires_target_attack_zone(
        self,
    ):
        if self.task_name is None:
            return (
                self._default_post_merge_predicted_target_blend_release_requires_target_attack_zone
            )
        return self._post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_blend_release_requires_target_attack_zone,
        )

    def _resolve_post_merge_predicted_target_blend_release_below_altitude_m(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_blend_release_below_altitude_m
        return self._post_merge_predicted_target_blend_release_below_altitude_m_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_blend_release_below_altitude_m,
        )

    def _resolve_post_merge_predicted_target_blend_hold_steps(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_blend_hold_steps
        return self._post_merge_predicted_target_blend_hold_steps_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_blend_hold_steps,
        )

    def _resolve_post_merge_predicted_target_blend_release_blend(self):
        if self.task_name is None:
            return self._default_post_merge_predicted_target_blend_release_blend
        return self._post_merge_predicted_target_blend_release_blend_by_task.get(
            self.task_name,
            self._default_post_merge_predicted_target_blend_release_blend,
        )

    def _resolve_close_range_anchor_post_merge_hold_steps(self):
        if self.task_name is None:
            return self._default_close_range_anchor_post_merge_hold_steps
        return self._close_range_anchor_post_merge_hold_steps_by_task.get(
            self.task_name,
            self._default_close_range_anchor_post_merge_hold_steps,
        )

    @staticmethod
    def _extract_rel_alignment_angles_deg(rel_state: dict):
        ata_deg = rel_state.get("ata_deg")
        if ata_deg is None and rel_state.get("ata_rad") is not None:
            ata_deg = np.rad2deg(float(rel_state.get("ata_rad")))
        aa_deg = rel_state.get("aa_deg", rel_state.get("aspect_deg"))
        if aa_deg is None and rel_state.get("aa_rad") is not None:
            aa_deg = np.rad2deg(float(rel_state.get("aa_rad")))
        ata_deg = abs(float(ata_deg if ata_deg is not None else np.nan))
        aa_deg = abs(float(aa_deg if aa_deg is not None else np.nan))
        return ata_deg, aa_deg

    def _evaluate_alignment_gate(self, rel_state: dict, angle_deg_max):
        if angle_deg_max is None:
            return True, None
        ata_deg, aa_deg = self._extract_rel_alignment_angles_deg(rel_state)
        if not np.isfinite(ata_deg) or not np.isfinite(aa_deg):
            return False, float(angle_deg_max)
        return max(ata_deg, aa_deg) <= float(angle_deg_max), float(angle_deg_max)

    def _evaluate_close_range_anchor_alignment(self, rel_state: dict):
        return self._evaluate_alignment_gate(
            rel_state,
            self._resolve_close_range_anchor_alignment_angle_deg_max(),
        )

    def _evaluate_close_range_anchor_release_alignment(self, rel_state: dict):
        return self._evaluate_alignment_gate(
            rel_state,
            self._resolve_close_range_anchor_release_alignment_angle_deg_max(),
        )

    def _evaluate_post_merge_offensive_anchor_geometry_disadvantage(
        self,
        own_state: dict,
        target_state: dict,
        rel_state: Optional[dict] = None,
        *,
        force_geometry_gate: bool = False,
        gate_requires_geometry_disadvantage_override: Optional[bool] = None,
    ) -> dict:
        if gate_requires_geometry_disadvantage_override is None:
            blend_gate_requires_geometry_disadvantage = (
                self._resolve_post_merge_offensive_anchor_blend_requires_geometry_disadvantage()
            )
            gate_requires_geometry_disadvantage = bool(
                force_geometry_gate or blend_gate_requires_geometry_disadvantage
            )
        else:
            gate_requires_geometry_disadvantage = bool(
                force_geometry_gate
                or gate_requires_geometry_disadvantage_override
            )
        aa_deg_min = (
            self._resolve_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min()
        )
        aa_deg = np.nan
        range_rate_mps = np.nan
        range_opening = False
        alignment_disadvantage = False
        if rel_state is not None:
            _, aa_deg = self._extract_rel_alignment_angles_deg(rel_state)
            range_rate_raw = rel_state.get("range_rate_mps")
            if range_rate_raw is not None:
                range_rate_mps = float(range_rate_raw)
                range_opening = bool(
                    np.isfinite(range_rate_mps) and range_rate_mps > 0.0
                )
        if aa_deg_min is not None and np.isfinite(aa_deg):
            alignment_disadvantage = bool(
                range_opening and aa_deg >= float(aa_deg_min)
            )
        base_result = {
            "requires_geometry_disadvantage": bool(
                gate_requires_geometry_disadvantage
            ),
            "condition_met": not bool(gate_requires_geometry_disadvantage),
            "target_only_attack_zone_disadvantage": False,
            "geometry_disadvantage": False,
            "ego_attack_score": 0.0,
            "target_attack_score": 0.0,
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
            "alignment_disadvantage": bool(alignment_disadvantage),
            "aa_deg_min": (
                float(aa_deg_min) if aa_deg_min is not None else np.nan
            ),
            "aa_deg": float(aa_deg),
            "range_rate_mps": float(range_rate_mps),
            "range_opening": bool(range_opening),
        }
        if not gate_requires_geometry_disadvantage or not self.combat_hp.enabled:
            if gate_requires_geometry_disadvantage and not self.combat_hp.enabled:
                base_result["condition_met"] = False
            return base_result

        ego_attack = evaluate_attack_zone(
            own_state,
            target_state,
            config=self.combat_hp.attack_zone_config,
        )
        target_attack = evaluate_attack_zone(
            target_state,
            own_state,
            config=self.combat_hp.attack_zone_config,
        )
        ego_attack_score = float(ego_attack["score"])
        target_attack_score = float(target_attack["score"])
        ego_in_attack_zone = bool(ego_attack["in_attack_zone"])
        target_in_attack_zone = bool(target_attack["in_attack_zone"])
        target_only_attack_zone_disadvantage = bool(
            target_in_attack_zone and not ego_in_attack_zone
        )
        geometry_disadvantage = bool(target_attack_score > ego_attack_score)
        condition_met = bool(
            target_only_attack_zone_disadvantage
            or geometry_disadvantage
            or alignment_disadvantage
        )
        return {
            "requires_geometry_disadvantage": True,
            "condition_met": condition_met,
            "target_only_attack_zone_disadvantage": target_only_attack_zone_disadvantage,
            "geometry_disadvantage": geometry_disadvantage,
            "ego_attack_score": ego_attack_score,
            "target_attack_score": target_attack_score,
            "ego_in_attack_zone": ego_in_attack_zone,
            "target_in_attack_zone": target_in_attack_zone,
            "alignment_disadvantage": bool(alignment_disadvantage),
            "aa_deg_min": (
                float(aa_deg_min) if aa_deg_min is not None else np.nan
            ),
            "aa_deg": float(aa_deg),
            "range_rate_mps": float(range_rate_mps),
            "range_opening": bool(range_opening),
        }

    def _evaluate_post_merge_anchor_mode_geometry_disadvantage(
        self,
        own_state: dict,
        target_state: dict,
        rel_state: Optional[dict] = None,
    ) -> dict:
        requires_geometry_disadvantage = (
            self._resolve_post_merge_anchor_mode_requires_geometry_disadvantage()
        )
        gate = self._evaluate_post_merge_offensive_anchor_geometry_disadvantage(
            own_state,
            target_state,
            rel_state=rel_state,
            gate_requires_geometry_disadvantage_override=(
                requires_geometry_disadvantage
            ),
        )
        recovery_below_altitude_m = (
            self._resolve_post_merge_anchor_mode_recovery_below_altitude_m()
        )
        recovery_active = False
        if recovery_below_altitude_m is not None:
            own_altitude_m = float(
                own_state.get(
                    "altitude_m",
                    self._extract_position_neu(own_state)[2],
                )
            )
            recovery_active = bool(
                np.isfinite(own_altitude_m)
                and own_altitude_m <= float(recovery_below_altitude_m)
                and bool(gate["range_opening"])
                and not bool(gate["ego_in_attack_zone"])
                and not bool(gate["target_in_attack_zone"])
            )
            gate["condition_met"] = bool(gate["condition_met"] and recovery_active)
        gate["recovery_below_altitude_m"] = (
            float(recovery_below_altitude_m)
            if recovery_below_altitude_m is not None
            else np.nan
        )
        gate["recovery_active"] = bool(recovery_active)
        return gate

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
        self._last_aggressiveness = None
        self._last_pid_gain_deltas = None
        self._prev_delayed_command = None
        # Deterministic sensor-noise stream per episode.
        self._robustness_rng = np.random.default_rng(
            self._robustness_seed_base + (seed if seed is not None else self._episode_count)
        )
        self._prediction_noise_rng = np.random.default_rng(
            self._robustness_seed_base
            + 7919
            + (seed if seed is not None else self._episode_count)
        )
        # Re-seed domain randomization RNG per episode for true seed-variance
        self._domain_rand_rng = np.random.default_rng(
            42 + (seed if seed is not None else self._episode_count)
        )
        self.reward_calculator.reset()
        self.termination_checker.reset()
        self._task_reset(seed)
        if hasattr(self.guidance, "reset"):
            self.guidance.reset()
        if self._guidance_pn is not None and hasattr(self._guidance_pn, "reset"):
            self._guidance_pn.reset()
        self._mode_switch_latched = False
        self._command_filter.reset()
        self._target_command_filter.reset()
        if self.command_post_processor is not None and hasattr(
            self.command_post_processor, "reset"
        ):
            self.command_post_processor.reset()
        if self._low_level_controller is not None:
            self._low_level_controller.reset()
        if self._target_low_level_controller is not None:
            self._target_low_level_controller.reset()
        if self.opponent_policy is not None:
            self.opponent_policy.reset()
        self.combat_hp.reset()
        self._merge_min_range_so_far_m = float("inf")
        self._merge_seen_close_range = False
        self._close_range_anchor_ready = False
        self._first_pass_complete = False
        self._first_pass_completion_step = None
        self._post_merge_anchor_mode_released = False
        self._post_merge_anchor_mode_has_activated = False
        self._post_merge_anchor_mode_ego_only_streak_steps = 0
        self._post_merge_anchor_mode_lateral_world_offset_latched = None
        self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
            None
        )
        self._post_merge_predicted_target_blend_released = False
        self._post_merge_offensive_anchor_blend_released = False
        self._post_merge_offensive_anchor_blend_release_step = None
        self._post_merge_offensive_anchor_blend_ego_only_streak_steps = 0
        self._post_merge_offensive_anchor_lateral_world_offset_latched = None
        self._post_merge_predicted_target_forward_scale_released = False
        self._post_merge_predicted_target_forward_scale_ego_only_streak_steps = 0
        self._post_merge_tactical_basis_recovery_profile_was_active = False
        self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign = 0.0
        self._post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining = 0
        self._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m = (
            float("nan")
        )
        self._post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m = (
            float("nan")
        )
        self._post_merge_tactical_basis_recovery_profile_previous_altitude_m = (
            float("nan")
        )
        self.clear_runtime_specialist_context()

        if self.trajectory_predictor_adapter is not None:
            self.trajectory_predictor_adapter.reset()
        self._prediction_buffer_synced_for_action = False
        self._prediction_error_tracker.reset()

        if self._backend == "jsbsim":
            self._reset_jsbsim(scenario)
        else:
            self._reset_simple(scenario)

        obs = self._get_observation()
        return obs

    def set_runtime_specialist_context(
        self,
        *,
        specialist_key=None,
        specialist_profile=None,
        specialist_mode_name=None,
        specialist_reason=None,
    ) -> None:
        self._runtime_specialist_key = (
            None if specialist_key in (None, "") else str(specialist_key)
        )
        self._runtime_specialist_profile = (
            None if specialist_profile in (None, "") else str(specialist_profile)
        )
        self._runtime_specialist_mode_name = (
            None if specialist_mode_name in (None, "") else str(specialist_mode_name)
        )
        self._runtime_specialist_reason = (
            None if specialist_reason in (None, "") else str(specialist_reason)
        )

    def clear_runtime_specialist_context(self) -> None:
        self._runtime_specialist_key = None
        self._runtime_specialist_profile = None
        self._runtime_specialist_mode_name = None
        self._runtime_specialist_reason = None

    def set_domain_rand_scale(self, scale: float):
        """Set the current domain randomization scale (0.0 = off)."""
        self.domain_rand_scale = float(scale)

    def set_success_criteria(
        self,
        success_range_m: float = None,
        success_ata_deg: float = None,
        success_hold_time_s: float = None,
        hysteresis_range_m: float = None,
        hysteresis_ata_deg: float = None,
    ):
        """
        Update the success criteria used by the termination checker.

        This is the public entry point for curriculum learning: the training
        loop can call it with progressively tighter thresholds and longer
        hold times.  A value of None preserves the current threshold.
        """
        self.termination_checker.set_success_criteria(
            success_range_m=success_range_m,
            success_ata_deg=success_ata_deg,
            success_hold_time_s=success_hold_time_s,
            hysteresis_range_m=hysteresis_range_m,
            hysteresis_ata_deg=hysteresis_ata_deg,
        )

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

    def set_opponent_policy(
        self,
        opponent_policy: Optional[OpponentPolicy],
        opponent_config: Optional[dict] = None,
    ) -> None:
        """Replace the embedded opponent policy at runtime."""
        self.opponent_policy = opponent_policy
        self.opponent_config = dict(opponent_config or {})
        self._opponent_stage = self.opponent_config.get("stage", "custom")
        if self.opponent_policy is not None:
            self.opponent_policy.reset()

    def set_command_post_processor_safety_mode(self, enabled: bool) -> None:
        if self.command_post_processor is not None:
            self.command_post_processor.set_safety_mode(enabled)

    def _get_opponent_obs(
        self,
        own_state: Optional[dict] = None,
        target_state: Optional[dict] = None,
    ) -> dict:
        """Build role-reversed observation for the target-aircraft opponent."""
        if own_state is None or target_state is None:
            own_state, target_state = self._get_current_states(noisy=True)

        rel_state = compute_relative_geometry(own_state, target_state)
        physical_rel_state = compute_relative_geometry(target_state, own_state)

        gains = self.current_gains if self._include_gains_in_observation else None
        prediction_features = None
        guidance_state = None
        if self._include_guidance_state_in_observation:
            guidance_state = {"vp_tracking_error": np.zeros(3, dtype=np.float64)}
        saturation_features = (
            self._last_command_saturation
            if self._include_saturation_in_observation
            else None
        )
        ego_vec, feature_names = build_observation(
            own_state,
            target_state,
            guidance_state=guidance_state,
            gains=gains,
            prediction_features=prediction_features,
            saturation_features=saturation_features,
            return_feature_names=True,
        )
        opponent_vec = self._invert_observation_vector(ego_vec, feature_names)
        opponent_rel_state = self._invert_relative_state(rel_state)

        return {
            "own_state": self._copy_state(target_state),
            "target_state": self._copy_state(own_state),
            "relative_state": opponent_rel_state,
            "physical_relative_state": physical_rel_state,
            "observation_vector": opponent_vec,
            "observation_schema": {
                "dim": int(opponent_vec.shape[0]),
                "feature_names": list(feature_names),
                "role_reversed": True,
            },
            "dt": float(self.env_config.get("high_level_dt", 0.2)),
        }

    @staticmethod
    def _invert_observation_vector(obs_vec: np.ndarray, feature_names: list) -> np.ndarray:
        """Invert the ordered policy observation according to the combat role table."""
        values = {name: float(obs_vec[i]) for i, name in enumerate(feature_names)}
        inverted = dict(values)
        sign_flip = (
            "range_rate_mps",
            "altitude_diff_m",
            "speed_diff_mps",
            "los_azimuth_sin",
            "los_elevation_sin",
        )
        for key in sign_flip:
            if key in inverted:
                inverted[key] = -inverted[key]
        swaps = (
            ("ata_sin", "aa_sin"),
            ("ata_cos", "aa_cos"),
            ("own_speed", "target_speed"),
            ("own_altitude", "target_altitude"),
        )
        for left, right in swaps:
            if left in inverted and right in inverted:
                inverted[left], inverted[right] = values[right], values[left]
        return np.array([inverted[name] for name in feature_names], dtype=np.float32)

    @staticmethod
    def _invert_relative_state(rel_state: dict) -> dict:
        inverted = dict(rel_state)
        if "range_rate_mps" in inverted:
            inverted["range_rate_mps"] = -float(inverted["range_rate_mps"])
        if "altitude_diff_m" in inverted:
            inverted["altitude_diff_m"] = -float(inverted["altitude_diff_m"])
        if "speed_diff_mps" in inverted:
            inverted["speed_diff_mps"] = -float(inverted["speed_diff_mps"])
        if "los_azimuth_rad" in inverted:
            inverted["los_azimuth_rad"] = -float(inverted["los_azimuth_rad"])
        if "los_elevation_rad" in inverted:
            inverted["los_elevation_rad"] = -float(inverted["los_elevation_rad"])
        if "ata_rad" in rel_state and "aa_rad" in rel_state:
            inverted["ata_rad"] = rel_state["aa_rad"]
            inverted["aa_rad"] = rel_state["ata_rad"]
        if "relative_position" in inverted:
            inverted["relative_position"] = -np.asarray(inverted["relative_position"])
        if "relative_velocity" in inverted:
            inverted["relative_velocity"] = -np.asarray(inverted["relative_velocity"])
        return inverted

    def _compute_opponent_command(self, own_state: dict, target_state: dict):
        """Run the embedded opponent and convert its action into target commands."""
        if self.opponent_policy is None:
            return None, {}

        opponent_obs = self._get_opponent_obs(own_state, target_state)
        action = np.asarray(self.opponent_policy.act(opponent_obs), dtype=np.float64)
        action_mode = getattr(self.opponent_policy, "action_mode", "direct_command")
        if action_mode == "vpp":
            raw_command = self._build_vpp_command_for_actor(
                actor_state=target_state,
                target_state=own_state,
                action=action,
            )
        else:
            raw_command = self._normalized_direct_action_to_command(action)

        clipped_command = clip_command(raw_command, self.config.get("limits", {}))
        filtered_command = self._target_command_filter.filter(clipped_command)
        diagnostics = {
            "opponent_stage": self._opponent_stage,
            "opponent_action_mode": action_mode,
            "opponent_action": action.tolist(),
            "opponent_raw_command": raw_command,
            "opponent_command": filtered_command,
        }
        diagnostics.update(self.opponent_policy.get_diagnostics())
        return filtered_command, diagnostics

    def _normalized_direct_action_to_command(self, action: np.ndarray) -> dict:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] < 3:
            raise ValueError(f"Opponent direct action must have at least 3 dims, got {action.shape}")
        limits = self.config.get("limits", {})
        nz_min = limits.get("nz_min", -2.0)
        nz_max = limits.get("nz_max", 7.0)
        rr_min = limits.get("roll_rate_min", -1.5)
        rr_max = limits.get("roll_rate_max", 1.5)
        th_min, th_max = effective_throttle_limits(limits)
        action = np.clip(action[:3], -1.0, 1.0)
        return {
            "nz_cmd": float(action[0]) * (nz_max - nz_min) / 2.0 + (nz_max + nz_min) / 2.0,
            "roll_rate_cmd": float(action[1]) * (rr_max - rr_min) / 2.0 + (rr_max + rr_min) / 2.0,
            "throttle_cmd": float(action[2]) * (th_max - th_min) / 2.0 + (th_max + th_min) / 2.0,
        }

    def _build_vpp_command_for_actor(
        self,
        actor_state: dict,
        target_state: dict,
        action: np.ndarray,
    ) -> dict:
        target_pos = target_state.get("position_m")
        if target_pos is None:
            target_pos = target_state.get("position_neu")
        target_for_vp = {"position_neu": np.asarray(target_pos, dtype=np.float64)}
        target_vel = target_state.get("velocity_vector_mps")
        if target_vel is not None:
            target_for_vp["velocity_vector_mps"] = np.asarray(target_vel, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64).reshape(-1)[:3]
        if self.virtual_point_generator is not None:
            virtual_point, _vp_info = self.virtual_point_generator.action_to_virtual_point(
                action,
                actor_state,
                target_for_vp,
                anchor_mode="current_target",
                trajectory_predictor_adapter=None,
                predicted_target_position=None,
                return_info=True,
            )
        else:
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
        return self._target_guidance.compute_command(
            actor_state,
            target_state,
            virtual_point,
            self.current_gains,
        )

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

        # 1. 获取当前状态（带传感器噪声的测量值用于闭环）
        own_state, target_state = self._get_current_states(noisy=True)

        # 1b. 任务钩子：允许子类替换/合成目标状态
        target_state = self._task_get_target_state(target_state, own_state)

        # 1c. 任务钩子：允许子类在 guidance 前更新任务状态
        own_state, target_state = self._task_pre_step(own_state, target_state)
        target_command, opponent_info = self._compute_opponent_command(
            own_state, target_state
        )

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
            "prediction_noise_std_m": 0.0,
            "prediction_noise_applied": False,
            "prediction_error_m": np.nan,
            "latest_prediction_error_m": np.nan,
            "mean_prediction_error_m": np.nan,
            "median_prediction_error_m": np.nan,
            "prediction_error_count": 0,
        }
        if tp_enabled and self.trajectory_predictor_adapter is not None:
            self._sync_prediction_buffer_for_current_state(
                own_state, target_state, rel_state, prediction_info
            )

        target_pos = target_state.get("position_m")
        if target_pos is None:
            target_pos = target_state.get("position_neu")
        target_for_vp = {"position_neu": np.asarray(target_pos, dtype=np.float64)}
        target_vel = target_state.get("velocity_vector_mps")
        if target_vel is not None:
            target_for_vp["velocity_vector_mps"] = np.asarray(
                target_vel,
                dtype=np.float64,
            )
        target_vel_alt = target_state.get("velocity")
        if target_vel_alt is not None:
            target_for_vp["velocity"] = np.asarray(
                target_vel_alt,
                dtype=np.float64,
            )

        # 4. 生成虚拟追踪点 (or use direct command override for diagnosis)
        anchor_mode = self.config.get("virtual_point", {}).get(
            "anchor_mode", "current_target"
        )
        requested_anchor_mode = anchor_mode
        close_range_anchor_mode = self._resolve_close_range_anchor_mode()
        close_range_anchor_trigger_range_m = self._resolve_close_range_anchor_trigger_range_m(
            self.config.get("combat_diagnostics", {}).get("merge_range_m", 1000.0)
        )
        close_range_anchor_release_on_post_merge = (
            self._resolve_close_range_anchor_release_on_post_merge()
        )
        close_range_anchor_requires_first_pass = (
            self._resolve_close_range_anchor_requires_first_pass()
        )
        close_range_anchor_offensive_anchor_blend = (
            self._resolve_close_range_anchor_offensive_anchor_blend()
        )
        (
            close_range_anchor_release_alignment_satisfied,
            close_range_anchor_release_alignment_angle_deg_max,
        ) = self._evaluate_close_range_anchor_release_alignment(rel_state)
        close_range_anchor_post_merge_hold_steps = (
            self._resolve_close_range_anchor_post_merge_hold_steps()
        )
        (
            close_range_anchor_alignment_satisfied,
            close_range_anchor_alignment_angle_deg_max,
        ) = self._evaluate_close_range_anchor_alignment(rel_state)
        close_range_anchor_mode_active = False
        close_range_anchor_post_merge_steps_since_first_pass = np.nan
        close_range_anchor_post_merge_hold_remaining_steps = np.nan
        close_range_anchor_window_open = True
        close_range_anchor_release_ready = False
        close_range_anchor_offensive_anchor_blend_active = False
        if close_range_anchor_release_on_post_merge and self._first_pass_complete:
            if self._first_pass_completion_step is None:
                close_range_anchor_window_open = False
                close_range_anchor_post_merge_hold_remaining_steps = 0.0
                close_range_anchor_release_ready = True
            else:
                close_range_anchor_post_merge_steps_since_first_pass = float(
                    max(0, self.current_step - self._first_pass_completion_step)
                )
                close_range_anchor_post_merge_hold_remaining_steps = float(
                    max(
                        0,
                        close_range_anchor_post_merge_hold_steps
                        - int(close_range_anchor_post_merge_steps_since_first_pass)
                        + 1,
                    )
                )
                close_range_anchor_release_ready = bool(
                    int(close_range_anchor_post_merge_steps_since_first_pass)
                    > close_range_anchor_post_merge_hold_steps
                    and close_range_anchor_release_alignment_satisfied
                )
                close_range_anchor_window_open = (
                    not close_range_anchor_release_ready
                )
        if (
            self._close_range_anchor_ready
            and close_range_anchor_window_open
            and close_range_anchor_mode is not None
            and (
                not close_range_anchor_requires_first_pass
                or self._first_pass_complete
            )
        ):
            close_range_anchor_mode_active = True
            if (
                close_range_anchor_mode == "offensive_position"
                and close_range_anchor_offensive_anchor_blend is not None
                and anchor_mode == "predicted_target"
            ):
                close_range_anchor_offensive_anchor_blend_active = True
            else:
                anchor_mode = close_range_anchor_mode
        post_merge_offensive_anchor_gate = (
            self._evaluate_post_merge_offensive_anchor_geometry_disadvantage(
                own_state,
                target_state,
                rel_state=rel_state,
            )
        )
        post_merge_anchor_mode = self._resolve_post_merge_anchor_mode()
        post_merge_anchor_mode_release_ego_only_streak_steps = (
            self._resolve_post_merge_anchor_mode_release_ego_only_streak_steps()
        )
        post_merge_anchor_mode_release_reset_on_streak_break = (
            self._resolve_post_merge_anchor_mode_release_reset_on_streak_break()
        )
        post_merge_anchor_mode_longitudinal_scale = (
            self._resolve_post_merge_anchor_mode_longitudinal_scale()
        )
        post_merge_anchor_mode_lateral_scale = (
            self._resolve_post_merge_anchor_mode_lateral_scale()
        )
        post_merge_anchor_mode_offensive_anchor_blend = (
            self._resolve_post_merge_anchor_mode_offensive_anchor_blend()
        )
        post_merge_anchor_mode_offensive_anchor_longitudinal_blend = (
            self._resolve_post_merge_anchor_mode_offensive_anchor_longitudinal_blend()
        )
        post_merge_anchor_mode_offensive_anchor_lateral_blend = (
            self._resolve_post_merge_anchor_mode_offensive_anchor_lateral_blend()
        )
        post_merge_anchor_mode_lateral_world_offset_latch_on_activation = (
            self._resolve_post_merge_anchor_mode_lateral_world_offset_latch_on_activation()
        )
        post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min = (
            self._resolve_post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min()
        )
        post_merge_anchor_mode_requires_geometry_disadvantage = False
        post_merge_anchor_mode_condition_met = True
        post_merge_anchor_mode_active = False
        post_merge_anchor_mode_offensive_anchor_blend_active = False
        post_merge_anchor_mode_offensive_anchor_component_blend_active = False
        post_merge_anchor_mode_recovery_below_altitude_m = np.nan
        post_merge_anchor_mode_recovery_active = False
        post_merge_anchor_mode_release_triggered = False
        post_merge_anchor_mode_release_reset_triggered = False
        if post_merge_anchor_mode is not None:
            post_merge_anchor_mode_gate = (
                self._evaluate_post_merge_anchor_mode_geometry_disadvantage(
                    own_state,
                    target_state,
                    rel_state=rel_state,
                )
            )
            post_merge_anchor_mode_requires_geometry_disadvantage = bool(
                post_merge_anchor_mode_gate["requires_geometry_disadvantage"]
            )
            post_merge_anchor_mode_recovery_below_altitude_m = float(
                post_merge_anchor_mode_gate.get(
                    "recovery_below_altitude_m",
                    np.nan,
                )
            )
            post_merge_anchor_mode_recovery_active = bool(
                post_merge_anchor_mode_gate.get("recovery_active", False)
            )
            post_merge_anchor_mode_recovery_enabled = bool(
                np.isfinite(post_merge_anchor_mode_recovery_below_altitude_m)
            )
            post_merge_anchor_mode_condition_met = bool(
                post_merge_anchor_mode_gate["condition_met"]
                if (
                    post_merge_anchor_mode_requires_geometry_disadvantage
                    or post_merge_anchor_mode_recovery_enabled
                )
                else True
            )
            if (
                self._first_pass_complete
                and self._post_merge_anchor_mode_has_activated
                and post_merge_anchor_mode_release_ego_only_streak_steps is not None
                and self._post_merge_anchor_mode_ego_only_streak_steps
                >= post_merge_anchor_mode_release_ego_only_streak_steps
            ):
                self._post_merge_anchor_mode_released = True
            if self._first_pass_complete and post_merge_anchor_mode_condition_met:
                post_merge_anchor_mode_active = not self._post_merge_anchor_mode_released
                if post_merge_anchor_mode_active:
                    self._post_merge_anchor_mode_has_activated = True
                    post_merge_anchor_mode_soft_blend_requested = bool(
                        (
                            post_merge_anchor_mode_offensive_anchor_blend is not None
                            and float(post_merge_anchor_mode_offensive_anchor_blend) > 0.0
                        )
                        or (
                            post_merge_anchor_mode_offensive_anchor_longitudinal_blend
                            is not None
                            and float(
                                post_merge_anchor_mode_offensive_anchor_longitudinal_blend
                            )
                            > 0.0
                        )
                        or (
                            post_merge_anchor_mode_offensive_anchor_lateral_blend
                            is not None
                            and float(
                                post_merge_anchor_mode_offensive_anchor_lateral_blend
                            )
                            > 0.0
                        )
                    )
                    if (
                        post_merge_anchor_mode == "offensive_position"
                        and post_merge_anchor_mode_soft_blend_requested
                        and anchor_mode == "predicted_target"
                    ):
                        post_merge_anchor_mode_offensive_anchor_blend_active = True
                        post_merge_anchor_mode_offensive_anchor_component_blend_active = (
                            bool(
                                (
                                    post_merge_anchor_mode_offensive_anchor_longitudinal_blend
                                    is not None
                                )
                                or (
                                    post_merge_anchor_mode_offensive_anchor_lateral_blend
                                    is not None
                                )
                            )
                        )
                    else:
                        anchor_mode = post_merge_anchor_mode
        longitudinal_scale_override = (
            float(post_merge_anchor_mode_longitudinal_scale)
            if (
                post_merge_anchor_mode_active
                and post_merge_anchor_mode_longitudinal_scale is not None
            )
            else None
        )
        lateral_scale_override = (
            float(post_merge_anchor_mode_lateral_scale)
            if (
                post_merge_anchor_mode_active
                and post_merge_anchor_mode_lateral_scale is not None
            )
            else None
        )
        post_merge_predicted_target_blend = self._resolve_post_merge_predicted_target_blend()
        post_merge_offensive_anchor_blend = (
            self._resolve_post_merge_offensive_anchor_blend()
        )
        post_merge_offensive_anchor_longitudinal_scale = (
            self._resolve_post_merge_offensive_anchor_longitudinal_scale()
        )
        post_merge_offensive_anchor_lateral_scale = (
            self._resolve_post_merge_offensive_anchor_lateral_scale()
        )
        post_merge_offensive_anchor_blend_release_blend = (
            self._resolve_post_merge_offensive_anchor_blend_release_blend()
        )
        post_merge_offensive_anchor_blend_release_ego_only_streak_steps = (
            self._resolve_post_merge_offensive_anchor_blend_release_ego_only_streak_steps()
        )
        post_merge_offensive_anchor_blend_release_reset_on_streak_break = (
            self._resolve_post_merge_offensive_anchor_blend_release_reset_on_streak_break()
        )
        post_merge_offensive_anchor_blend_release_direct_track_enabled = (
            self._resolve_post_merge_offensive_anchor_blend_release_direct_track_enabled()
        )
        post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m = (
            self._resolve_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m()
        )
        post_merge_offensive_anchor_blend_release_recovery_below_altitude_m = (
            self._resolve_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m()
        )
        post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend = (
            self._resolve_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend()
        )
        post_merge_offensive_anchor_blend_release_recovery_lateral_blend = (
            self._resolve_post_merge_offensive_anchor_blend_release_recovery_lateral_blend()
        )
        post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max = (
            self._resolve_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max()
        )
        post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min = (
            self._resolve_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min()
        )
        post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m = (
            self._resolve_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m()
        )
        post_merge_offensive_anchor_lateral_world_offset_latch_on_activation = (
            self._resolve_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation()
        )
        post_merge_offensive_anchor_blend_release_lateral_only = (
            self._resolve_post_merge_offensive_anchor_blend_release_lateral_only()
        )
        post_merge_offensive_anchor_blend_release_lateral_only_hold_steps = (
            self._resolve_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps()
        )
        post_merge_predicted_target_forward_scale = (
            self._resolve_post_merge_predicted_target_forward_scale()
        )
        post_merge_predicted_target_forward_scale_release_scale = (
            self._resolve_post_merge_predicted_target_forward_scale_release_scale()
        )
        post_merge_predicted_target_forward_scale_release_ego_only_streak_steps = (
            self._resolve_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps()
        )
        post_merge_predicted_target_forward_scale_release_reset_on_streak_break = (
            self._resolve_post_merge_predicted_target_forward_scale_release_reset_on_streak_break()
        )
        post_merge_predicted_target_forward_scale_hold_steps = (
            self._resolve_post_merge_predicted_target_forward_scale_hold_steps()
        )
        post_merge_predicted_target_blend_release_on_attack_zone = (
            self._resolve_post_merge_predicted_target_blend_release_on_attack_zone()
        )
        post_merge_predicted_target_blend_release_requires_target_attack_zone = (
            self._resolve_post_merge_predicted_target_blend_release_requires_target_attack_zone()
        )
        post_merge_predicted_target_blend_release_below_altitude_m = (
            self._resolve_post_merge_predicted_target_blend_release_below_altitude_m()
        )
        post_merge_predicted_target_blend_hold_steps = (
            self._resolve_post_merge_predicted_target_blend_hold_steps()
        )
        post_merge_predicted_target_blend_release_blend = (
            self._resolve_post_merge_predicted_target_blend_release_blend()
        )
        use_command_override = command_override is not None
        post_merge_predicted_target_blend_steps_since_first_pass = np.nan
        post_merge_predicted_target_blend_hold_remaining_steps = np.nan
        post_merge_predicted_target_blend_hold_window_open = True
        post_merge_predicted_target_blend_hold_expired = False
        if self._first_pass_complete and post_merge_predicted_target_blend_hold_steps is not None:
            if self._first_pass_completion_step is None:
                post_merge_predicted_target_blend_hold_window_open = False
                post_merge_predicted_target_blend_hold_remaining_steps = 0.0
                post_merge_predicted_target_blend_hold_expired = True
            else:
                post_merge_predicted_target_blend_steps_since_first_pass = float(
                    max(0, self.current_step - self._first_pass_completion_step)
                )
                post_merge_predicted_target_blend_hold_remaining_steps = float(
                    max(
                        0,
                        post_merge_predicted_target_blend_hold_steps
                        - int(post_merge_predicted_target_blend_steps_since_first_pass)
                        + 1,
                    )
                )
                post_merge_predicted_target_blend_hold_expired = bool(
                    int(post_merge_predicted_target_blend_steps_since_first_pass)
                    > post_merge_predicted_target_blend_hold_steps
                )
                post_merge_predicted_target_blend_hold_window_open = (
                    not post_merge_predicted_target_blend_hold_expired
                )
        post_merge_predicted_target_blend_active = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and post_merge_predicted_target_blend is not None
            and not self._post_merge_predicted_target_blend_released
            and post_merge_predicted_target_blend_hold_window_open
        )
        post_merge_predicted_target_blend_release_blend_active = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and self._post_merge_predicted_target_blend_released
            and post_merge_predicted_target_blend_release_blend is not None
        )
        predicted_target_blend_override = (
            post_merge_predicted_target_blend
            if post_merge_predicted_target_blend_active
            else (
                post_merge_predicted_target_blend_release_blend
                if post_merge_predicted_target_blend_release_blend_active
                else None
            )
        )
        post_merge_offensive_anchor_blend_release_triggered = False
        post_merge_offensive_anchor_blend_release_reset_triggered = False
        post_merge_offensive_anchor_lateral_world_offset_latch_active = False
        post_merge_offensive_anchor_blend_release_recovery_active = False
        post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active = False
        post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active = False
        post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m = (
            np.nan
        )
        post_merge_offensive_anchor_blend_release_lateral_only_active = False
        post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release = (
            np.nan
        )
        post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps = (
            np.nan
        )
        post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open = True
        post_merge_offensive_anchor_blend_enabled = (
            post_merge_offensive_anchor_blend is not None
            and float(post_merge_offensive_anchor_blend) > 0.0
        )
        post_merge_offensive_anchor_blend_release_blend_enabled = (
            post_merge_offensive_anchor_blend_release_blend is not None
            and float(post_merge_offensive_anchor_blend_release_blend) > 0.0
        )
        if (
            self._first_pass_complete
            and self._post_merge_offensive_anchor_blend_has_activated
            and post_merge_offensive_anchor_blend_release_ego_only_streak_steps is not None
            and self._post_merge_offensive_anchor_blend_ego_only_streak_steps
            >= post_merge_offensive_anchor_blend_release_ego_only_streak_steps
        ):
            self._post_merge_offensive_anchor_blend_released = True
            if self._post_merge_offensive_anchor_blend_release_step is None:
                self._post_merge_offensive_anchor_blend_release_step = max(
                    self.current_step - 1,
                    0,
                )
        if (
            self._post_merge_offensive_anchor_blend_released
            and post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
            is not None
        ):
            if self._post_merge_offensive_anchor_blend_release_step is None:
                post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps = float(
                    post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
                )
            else:
                post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release = (
                    float(
                        max(
                            0,
                            self.current_step
                            - self._post_merge_offensive_anchor_blend_release_step,
                        )
                    )
                )
                post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps = float(
                    max(
                        0,
                        post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
                        - int(
                            post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release
                        )
                        + 1,
                    )
                )
                post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open = bool(
                    int(
                        post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release
                    )
                    <= post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
                )
        post_merge_offensive_anchor_blend_active = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and post_merge_offensive_anchor_blend_enabled
            and post_merge_offensive_anchor_gate["condition_met"]
            and not self._post_merge_offensive_anchor_blend_released
        )
        if post_merge_offensive_anchor_blend_active:
            self._post_merge_offensive_anchor_blend_has_activated = True
            if (
                bool(post_merge_offensive_anchor_lateral_world_offset_latch_on_activation)
                and self.virtual_point_generator is not None
                and self._post_merge_offensive_anchor_lateral_world_offset_latched
                is None
            ):
                (
                    _latched_anchor_pos,
                    _latched_anchor_offset,
                    _latched_anchor_world_offset,
                    _latched_anchor_lateral_local_offset,
                    latched_lateral_world_offset,
                    _latched_anchor_lateral_sign,
                ) = self.virtual_point_generator.compute_offensive_anchor_components(
                    target_for_vp,
                    own_state=own_state,
                )
                self._post_merge_offensive_anchor_lateral_world_offset_latched = (
                    np.asarray(
                        latched_lateral_world_offset,
                        dtype=np.float64,
                    )
                )
        if self._post_merge_offensive_anchor_lateral_world_offset_latched is not None:
            post_merge_offensive_anchor_lateral_world_offset_latch_active = True
        post_merge_offensive_anchor_blend_release_blend_active = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and self._post_merge_offensive_anchor_blend_released
            and post_merge_offensive_anchor_blend_release_blend_enabled
        )
        offensive_anchor_blend_override = (
            close_range_anchor_offensive_anchor_blend
            if close_range_anchor_offensive_anchor_blend_active
            else (
                post_merge_anchor_mode_offensive_anchor_blend
                if post_merge_anchor_mode_offensive_anchor_blend_active
                else (
                    post_merge_offensive_anchor_blend
                    if post_merge_offensive_anchor_blend_active
                    else (
                        post_merge_offensive_anchor_blend_release_blend
                        if post_merge_offensive_anchor_blend_release_blend_active
                        else None
                    )
                )
            )
        )
        if post_merge_offensive_anchor_blend_active:
            if (
                longitudinal_scale_override is None
                and post_merge_offensive_anchor_longitudinal_scale is not None
            ):
                longitudinal_scale_override = float(
                    post_merge_offensive_anchor_longitudinal_scale
                )
            if (
                lateral_scale_override is None
                and post_merge_offensive_anchor_lateral_scale is not None
            ):
                lateral_scale_override = float(
                    post_merge_offensive_anchor_lateral_scale
                )
        offensive_anchor_longitudinal_blend_override = (
            float(post_merge_anchor_mode_offensive_anchor_longitudinal_blend)
            if (
                post_merge_anchor_mode_offensive_anchor_blend_active
                and post_merge_anchor_mode_offensive_anchor_longitudinal_blend
                is not None
            )
            else None
        )
        offensive_anchor_lateral_blend_override = (
            float(post_merge_anchor_mode_offensive_anchor_lateral_blend)
            if (
                post_merge_anchor_mode_offensive_anchor_blend_active
                and post_merge_anchor_mode_offensive_anchor_lateral_blend is not None
            )
            else None
        )
        own_altitude_pre_m = float(
            own_state.get(
                "altitude_m",
                self._extract_position_neu(own_state)[2],
            )
        )
        post_merge_offensive_anchor_blend_release_recovery_components_configured = (
            post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
            is not None
            or post_merge_offensive_anchor_blend_release_recovery_lateral_blend
            is not None
        )
        pre_recovery_vp_result = None
        recovery_gate_ready = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and self._post_merge_offensive_anchor_blend_released
            and post_merge_offensive_anchor_blend_release_recovery_components_configured
            and bool(post_merge_offensive_anchor_gate["range_opening"])
            and not bool(post_merge_offensive_anchor_gate["ego_in_attack_zone"])
            and not bool(post_merge_offensive_anchor_gate["target_in_attack_zone"])
        )
        if recovery_gate_ready:
            if (
                post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
                is not None
                and np.isfinite(own_altitude_pre_m)
                and own_altitude_pre_m
                <= float(
                    post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
                )
            ):
                post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active = True
        if (
            recovery_gate_ready
            and (
                post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active
                or post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active
            )
        ):
            post_merge_offensive_anchor_blend_release_recovery_active = True
            offensive_anchor_blend_override = 0.0
            if (
                post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
                is not None
            ):
                offensive_anchor_longitudinal_blend_override = float(
                    post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
                )
            if (
                post_merge_offensive_anchor_blend_release_recovery_lateral_blend
                is not None
            ):
                offensive_anchor_lateral_blend_override = float(
                    post_merge_offensive_anchor_blend_release_recovery_lateral_blend
                )
            if (
                self._post_merge_offensive_anchor_lateral_world_offset_latched
                is None
                and bool(
                    post_merge_offensive_anchor_lateral_world_offset_latch_on_activation
                )
                and self.virtual_point_generator is not None
            ):
                (
                    _latched_anchor_pos,
                    _latched_anchor_offset,
                    _latched_anchor_world_offset,
                    _latched_anchor_lateral_local_offset,
                    latched_lateral_world_offset,
                    _latched_anchor_lateral_sign,
                ) = self.virtual_point_generator.compute_offensive_anchor_components(
                    target_for_vp,
                    own_state=own_state,
                )
                self._post_merge_offensive_anchor_lateral_world_offset_latched = (
                    np.asarray(
                        latched_lateral_world_offset,
                        dtype=np.float64,
                    )
                )
            if self._post_merge_offensive_anchor_lateral_world_offset_latched is not None:
                offensive_anchor_lateral_world_offset_override = (
                    self._post_merge_offensive_anchor_lateral_world_offset_latched.copy()
                )
                post_merge_offensive_anchor_lateral_world_offset_latch_active = True
        post_merge_predicted_target_forward_scale_steps_since_first_pass = np.nan
        post_merge_predicted_target_forward_scale_hold_remaining_steps = np.nan
        post_merge_predicted_target_forward_scale_hold_window_open = True
        post_merge_predicted_target_forward_scale_hold_expired = False
        post_merge_predicted_target_forward_scale_release_triggered = False
        post_merge_predicted_target_forward_scale_release_reset_triggered = False
        if (
            self._first_pass_complete
            and post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
            is not None
            and self._post_merge_predicted_target_forward_scale_ego_only_streak_steps
            >= post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
        ):
            self._post_merge_predicted_target_forward_scale_released = True
        if (
            self._first_pass_complete
            and post_merge_predicted_target_forward_scale_hold_steps is not None
        ):
            if self._first_pass_completion_step is None:
                post_merge_predicted_target_forward_scale_hold_window_open = False
                post_merge_predicted_target_forward_scale_hold_remaining_steps = 0.0
                post_merge_predicted_target_forward_scale_hold_expired = True
            else:
                post_merge_predicted_target_forward_scale_steps_since_first_pass = float(
                    max(0, self.current_step - self._first_pass_completion_step)
                )
                post_merge_predicted_target_forward_scale_hold_remaining_steps = float(
                    max(
                        0,
                        post_merge_predicted_target_forward_scale_hold_steps
                        - int(
                            post_merge_predicted_target_forward_scale_steps_since_first_pass
                        )
                        + 1,
                    )
                )
                post_merge_predicted_target_forward_scale_hold_expired = bool(
                    int(
                        post_merge_predicted_target_forward_scale_steps_since_first_pass
                    )
                    > post_merge_predicted_target_forward_scale_hold_steps
                )
                post_merge_predicted_target_forward_scale_hold_window_open = (
                    not post_merge_predicted_target_forward_scale_hold_expired
                )
        post_merge_predicted_target_forward_scale_active = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and post_merge_predicted_target_forward_scale is not None
            and not self._post_merge_predicted_target_forward_scale_released
            and post_merge_predicted_target_forward_scale_hold_window_open
        )
        post_merge_predicted_target_forward_scale_release_scale_active = bool(
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and self._post_merge_predicted_target_forward_scale_released
            and post_merge_predicted_target_forward_scale_release_scale is not None
        )
        predicted_target_forward_scale_override = (
            post_merge_predicted_target_forward_scale
            if post_merge_predicted_target_forward_scale_active
            else (
                post_merge_predicted_target_forward_scale_release_scale
                if post_merge_predicted_target_forward_scale_release_scale_active
                else None
            )
        )
        if action is None:
            action = np.zeros(3)
        action = np.asarray(action, dtype=np.float64)

        # --- Validate action-space schema (3D/4D/6D) ---
        # Legacy anchor-mode variants may use other action shapes; pass those
        # through unchanged so existing tests and configs keep working.
        action_dim = (
            int(action.shape[0]) if action.ndim == 1 else int(action.shape[-1])
        )
        if action_dim in (3, 4, 6):
            action, schema = validate_action(action)
        else:
            schema = None

        # --- Detect APIC mode (policy outputs PID gain deltas) ---
        ll_cfg = self.config.get("low_level_controller", {})
        if isinstance(ll_cfg, str):
            ll_cfg = {}
        apic_enabled = bool(ll_cfg.get("apic", {}).get("enabled", False))

        if apic_enabled or schema == ActionSchema.APIC_PID_6D:
            # APIC action = [delta_Kp_nz, delta_Ki_nz, delta_Kd_nz,
            #                delta_Kp_roll, delta_Ki_roll, delta_Kd_roll]
            pid_gain_deltas = np.clip(action, -1.0, 1.0)
            self._last_pid_gain_deltas = pid_gain_deltas
            action = np.zeros(3, dtype=np.float64)
            aggressiveness = None
        else:
            aggressiveness = None
            pid_gain_deltas = None
            if schema == ActionSchema.PPO_PID_4D:
                aggressiveness = float(np.clip(action[3], -1.0, 1.0))
                action = action[:3].copy()
            self._last_pid_gain_deltas = None
        self._last_aggressiveness = aggressiveness

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
                    noise_std_m = float(
                        self.config.get("trajectory_prediction", {})
                        .get("integration", {})
                        .get("prediction_noise_std_m", 0.0)
                    )
                    if noise_std_m > 0.0:
                        predicted_target_pos = predicted_target_pos + self._prediction_noise_rng.normal(
                            loc=0.0, scale=noise_std_m, size=3
                        )
                        prediction_info["prediction_noise_std_m"] = noise_std_m
                        prediction_info["prediction_noise_applied"] = True
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

        if self._post_merge_anchor_mode_released:
            self._post_merge_anchor_mode_lateral_world_offset_latched = None
            self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
                None
            )
        post_merge_anchor_mode_lateral_world_offset_latch_active = False
        post_merge_anchor_mode_lateral_world_offset_latch_range_gate_satisfied = True
        post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = np.nan
        offensive_anchor_lateral_world_offset_override = None
        if (
            post_merge_anchor_mode_active
            and post_merge_anchor_mode == "offensive_position"
            and bool(post_merge_anchor_mode_lateral_world_offset_latch_on_activation)
            and self.virtual_point_generator is not None
        ):
            current_range_m = float(rel_state.get("range_m", np.nan))
            if self._post_merge_anchor_mode_lateral_world_offset_latched is None:
                if (
                    post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min
                    is not None
                ):
                    post_merge_anchor_mode_lateral_world_offset_latch_range_gate_satisfied = bool(
                        np.isfinite(current_range_m)
                        and current_range_m
                        >= float(
                            post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min
                        )
                    )
                if post_merge_anchor_mode_lateral_world_offset_latch_range_gate_satisfied:
                    (
                        _latched_anchor_pos,
                        _latched_anchor_offset,
                        _latched_anchor_world_offset,
                        _latched_anchor_lateral_local_offset,
                        latched_lateral_world_offset,
                        _latched_anchor_lateral_sign,
                    ) = self.virtual_point_generator.compute_offensive_anchor_components(
                        target_for_vp,
                        own_state=own_state,
                    )
                    self._post_merge_anchor_mode_lateral_world_offset_latched = (
                        np.asarray(
                            latched_lateral_world_offset,
                            dtype=np.float64,
                        )
                    )
                    self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
                        current_range_m
                    )
            if self._post_merge_anchor_mode_lateral_world_offset_latched is not None:
                offensive_anchor_lateral_world_offset_override = (
                    self._post_merge_anchor_mode_lateral_world_offset_latched.copy()
                )
                post_merge_anchor_mode_lateral_world_offset_latch_active = True
                if (
                    self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m
                    is not None
                ):
                    post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = float(
                        self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m
                    )
            else:
                post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
                    current_range_m
                )

        if (
            recovery_gate_ready
            and not post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active
            and not post_merge_offensive_anchor_blend_release_recovery_active
            and post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
            is not None
            and anchor_mode == "predicted_target"
            and self._use_virtual_point
            and self.virtual_point_generator is not None
            and not use_command_override
        ):
            pre_recovery_vp_result = self.virtual_point_generator.action_to_virtual_point(
                action,
                own_state,
                target_for_vp,
                anchor_mode=anchor_mode,
                trajectory_predictor_adapter=None,
                predicted_target_position=predicted_target_pos,
                predicted_target_blend_override=predicted_target_blend_override,
                predicted_target_forward_scale_override=predicted_target_forward_scale_override,
                offensive_anchor_blend_override=offensive_anchor_blend_override,
                offensive_anchor_longitudinal_blend_override=(
                    offensive_anchor_longitudinal_blend_override
                ),
                offensive_anchor_lateral_blend_override=(
                    offensive_anchor_lateral_blend_override
                ),
                offensive_anchor_lateral_world_offset_override=(
                    offensive_anchor_lateral_world_offset_override
                ),
                longitudinal_scale_override=longitudinal_scale_override,
                lateral_scale_override=lateral_scale_override,
                return_info=True,
            )
            preview_virtual_point, preview_vp_info = pre_recovery_vp_result
            preview_geometry = self._build_vpp_geometry_diagnostics(
                preview_virtual_point,
                preview_vp_info,
                own_state,
                target_state,
            )
            post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m = float(
                preview_geometry["vp_forward_bias_m"]
            )
            post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active = bool(
                np.isfinite(
                    post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m
                )
                and post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m
                <= float(
                    post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
                )
            )
            if post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active:
                post_merge_offensive_anchor_blend_release_recovery_active = True
                offensive_anchor_blend_override = 0.0
                if (
                    post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
                    is not None
                ):
                    offensive_anchor_longitudinal_blend_override = float(
                        post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
                    )
                if (
                    post_merge_offensive_anchor_blend_release_recovery_lateral_blend
                    is not None
                ):
                    offensive_anchor_lateral_blend_override = float(
                        post_merge_offensive_anchor_blend_release_recovery_lateral_blend
                    )
                if (
                    self._post_merge_offensive_anchor_lateral_world_offset_latched is None
                    and bool(
                        post_merge_offensive_anchor_lateral_world_offset_latch_on_activation
                    )
                    and self.virtual_point_generator is not None
                ):
                    (
                        _latched_anchor_pos,
                        _latched_anchor_offset,
                        _latched_anchor_world_offset,
                        _latched_anchor_lateral_local_offset,
                        latched_lateral_world_offset,
                        _latched_anchor_lateral_sign,
                    ) = self.virtual_point_generator.compute_offensive_anchor_components(
                        target_for_vp,
                        own_state=own_state,
                    )
                    self._post_merge_offensive_anchor_lateral_world_offset_latched = (
                        np.asarray(
                            latched_lateral_world_offset,
                            dtype=np.float64,
                        )
                    )
        if (
            post_merge_offensive_anchor_blend_release_recovery_active
            and self._post_merge_offensive_anchor_lateral_world_offset_latched
            is not None
        ):
            offensive_anchor_lateral_world_offset_override = (
                self._post_merge_offensive_anchor_lateral_world_offset_latched.copy()
            )
            post_merge_offensive_anchor_lateral_world_offset_latch_active = True

        # Determine direct-track request from config (telemetry)
        direct_track_mode_requested = self.config.get("guidance", {}).get("direct_track_mode", False)
        post_merge_offensive_anchor_blend_release_direct_track_active = bool(
            post_merge_offensive_anchor_blend_release_direct_track_enabled
            and self._first_pass_complete
            and anchor_mode == "predicted_target"
            and self._post_merge_offensive_anchor_blend_released
            and not post_merge_offensive_anchor_blend_release_recovery_active
            and post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
            is not None
            and own_altitude_pre_m
            <= float(
                post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
            )
            and bool(post_merge_offensive_anchor_gate["range_opening"])
            and not bool(post_merge_offensive_anchor_gate["ego_in_attack_zone"])
            and not bool(post_merge_offensive_anchor_gate["target_in_attack_zone"])
        )
        if post_merge_offensive_anchor_blend_release_direct_track_active:
            direct_track_mode_requested = True
            self._post_merge_offensive_anchor_lateral_world_offset_latched = None
            post_merge_offensive_anchor_lateral_world_offset_latch_active = False
        elif (
            self._first_pass_complete
            and anchor_mode == "predicted_target"
            and self._post_merge_offensive_anchor_blend_released
            and not post_merge_offensive_anchor_blend_release_recovery_active
            and bool(post_merge_offensive_anchor_blend_release_lateral_only)
            and post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open
            and not bool(direct_track_mode_requested)
        ):
            if (
                self._post_merge_offensive_anchor_lateral_world_offset_latched is None
                and self.virtual_point_generator is not None
            ):
                (
                    _latched_anchor_pos,
                    _latched_anchor_offset,
                    _latched_anchor_world_offset,
                    _latched_anchor_lateral_local_offset,
                    latched_lateral_world_offset,
                    _latched_anchor_lateral_sign,
                ) = self.virtual_point_generator.compute_offensive_anchor_components(
                    target_for_vp,
                    own_state=own_state,
                )
                self._post_merge_offensive_anchor_lateral_world_offset_latched = (
                    np.asarray(
                        latched_lateral_world_offset,
                        dtype=np.float64,
                    )
                )
            if self._post_merge_offensive_anchor_lateral_world_offset_latched is not None:
                offensive_anchor_lateral_world_offset_override = (
                    self._post_merge_offensive_anchor_lateral_world_offset_latched.copy()
                )
                offensive_anchor_blend_override = 0.0
                offensive_anchor_longitudinal_blend_override = 0.0
                offensive_anchor_lateral_blend_override = 1.0
                post_merge_offensive_anchor_lateral_world_offset_latch_active = True
                post_merge_offensive_anchor_blend_release_lateral_only_active = True

        post_merge_tactical_basis_recovery_profile_state = {
            "configured": False,
            "active": False,
            "reason": "disabled",
            "source": None,
            "requested_profile": self._runtime_specialist_profile,
            "specialist_profile_match": False,
            "blend_release_recovery_active": bool(
                post_merge_offensive_anchor_blend_release_recovery_active
            ),
            "predicted_target_forward_scale_override": None,
            "ll_pre": float("nan"),
            "ll_post": float("nan"),
            "io_pre": float("nan"),
            "io_post": float("nan"),
            "cd_pre": float("nan"),
            "cd_post": float("nan"),
            "io_entry_lateral_sign_hold_enabled": False,
            "io_entry_lateral_sign_hold_active": False,
            "io_entry_lateral_sign_hold_applied": False,
            "io_entry_lateral_sign": float("nan"),
            "io_entry_lateral_sign_hold_previous_vp_lateral_bias_m": float("nan"),
            "io_entry_lateral_sign_hold_release_vp_lateral_bias_m": float("nan"),
            "entry_window_configured_steps": 0,
            "entry_window_armed": False,
            "entry_window_active": False,
            "entry_window_steps_remaining": 0,
            "entry_window_next_steps_remaining": 0,
            "entry_window_previous_vp_forward_bias_m": float("nan"),
            "entry_window_previous_altitude_delta_m": float("nan"),
            "entry_window_io_min_abs": float("nan"),
            "entry_window_io_min_abs_applied": False,
            "entry_window_ll_floor": float("nan"),
            "entry_window_ll_floor_applied": False,
            "entry_window_ll_overdeep_vp_forward_bias_m_max": float("nan"),
            "entry_window_cd_min": float("nan"),
            "entry_window_cd_min_applied": False,
            "entry_window_cd_descending_altitude_delta_m_min_abs": float("nan"),
            "entry_window_predicted_target_forward_scale_override": float("nan"),
            "entry_window_predicted_target_forward_scale_override_applied": False,
            "entry_window_predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max": float(
                "nan"
            ),
            "action": np.asarray(action, dtype=np.float64).copy(),
            "shaped_action": np.asarray(action, dtype=np.float64).copy(),
        }
        action_for_vpp = np.asarray(action, dtype=np.float64).copy()
        if not use_command_override:
            post_merge_tactical_basis_recovery_profile_state = (
                self._evaluate_post_merge_tactical_basis_recovery_profile_state(
                    action=action_for_vpp,
                    own_state=own_state,
                    anchor_mode=anchor_mode,
                    post_merge_offensive_anchor_gate=post_merge_offensive_anchor_gate,
                    post_merge_offensive_anchor_blend_release_recovery_active=(
                        post_merge_offensive_anchor_blend_release_recovery_active
                    ),
                )
            )
            if bool(post_merge_tactical_basis_recovery_profile_state.get("active", False)):
                action_for_vpp = np.asarray(
                    post_merge_tactical_basis_recovery_profile_state["shaped_action"],
                    dtype=np.float64,
                ).copy()
                recovery_forward_scale_override = (
                    post_merge_tactical_basis_recovery_profile_state.get(
                        "predicted_target_forward_scale_override"
                    )
                )
                if recovery_forward_scale_override is not None:
                    predicted_target_forward_scale_override = float(
                        recovery_forward_scale_override
                    )

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
            zero_offset = np.zeros(3, dtype=np.float64)
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "anchor_pos": np.asarray(target_for_vp["position_neu"], dtype=np.float64),
                "offset": zero_offset,
                "world_offset": zero_offset,
                "offset_frame": "world_neu",
                "direct_track_mode": True,
                "action_applied": False,
            }
            direct_track_mode_effective = True
            virtual_point_source = "direct_track"
        elif use_command_override:
            # Bridge / diagnosis path: skip VPP generation entirely when a
            # command override is injected directly into the environment.
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
            zero_offset = np.zeros(3, dtype=np.float64)
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "anchor_pos": np.asarray(target_for_vp["position_neu"], dtype=np.float64),
                "offset": zero_offset,
                "world_offset": zero_offset,
                "offset_frame": "world_neu",
                "command_override": True,
                "action_applied": False,
            }
            direct_track_mode_effective = False
            virtual_point_source = "command_override"
        elif self._use_virtual_point and self.virtual_point_generator is not None:
            if (
                pre_recovery_vp_result is not None
                and not post_merge_offensive_anchor_blend_release_recovery_active
                and not bool(
                    post_merge_tactical_basis_recovery_profile_state.get(
                        "active", False
                    )
                )
            ):
                vp_result = pre_recovery_vp_result
            else:
                vp_result = self.virtual_point_generator.action_to_virtual_point(
                    action_for_vpp,
                    own_state,
                    target_for_vp,
                    anchor_mode=anchor_mode,
                    trajectory_predictor_adapter=None,  # generator 不直接调用 predictor
                    predicted_target_position=predicted_target_pos,
                    predicted_target_blend_override=predicted_target_blend_override,
                    predicted_target_forward_scale_override=predicted_target_forward_scale_override,
                    offensive_anchor_blend_override=offensive_anchor_blend_override,
                    offensive_anchor_longitudinal_blend_override=(
                        offensive_anchor_longitudinal_blend_override
                    ),
                    offensive_anchor_lateral_blend_override=(
                        offensive_anchor_lateral_blend_override
                    ),
                    offensive_anchor_lateral_world_offset_override=(
                        offensive_anchor_lateral_world_offset_override
                    ),
                    longitudinal_scale_override=longitudinal_scale_override,
                    lateral_scale_override=lateral_scale_override,
                    return_info=True,
                )
            virtual_point, vp_info = vp_result
            direct_track_mode_effective = False
            virtual_point_source = "vpp_policy"
        else:
            # End-to-end mode: virtual_point is not used for guidance,
            # but we still populate it for telemetry consistency.
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
            zero_offset = np.zeros(3, dtype=np.float64)
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "anchor_pos": np.asarray(target_for_vp["position_neu"], dtype=np.float64),
                "offset": zero_offset,
                "world_offset": zero_offset,
                "offset_frame": "world_neu",
                "end_to_end_mode": True,
                "action_applied": True,
            }
            direct_track_mode_effective = False
            virtual_point_source = "end_to_end"

        (
            virtual_point,
            vp_info,
            post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active,
            post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m,
        ) = self._apply_post_merge_offensive_anchor_release_forward_bias_clamp(
            virtual_point,
            vp_info,
            own_state,
            target_state,
            anchor_mode=anchor_mode,
            recovery_active=post_merge_offensive_anchor_blend_release_recovery_active,
            direct_track_mode_effective=direct_track_mode_effective,
            use_command_override=use_command_override,
            post_merge_offensive_anchor_gate=post_merge_offensive_anchor_gate,
        )

        # 5. Guidance command generation
        if not self._use_virtual_point:
            # Direct command mode: policy outputs normalized commands in [-1, 1]
            # which are then mapped to physical limits here.
            limits = self.config.get("limits", {})
            nz_min = limits.get("nz_min", -2.0)
            nz_max = limits.get("nz_max", 7.0)
            rr_min = limits.get("roll_rate_min", -1.5)
            rr_max = limits.get("roll_rate_max", 1.5)
            th_min, th_max = effective_throttle_limits(limits)
            raw_command = {
                "nz_cmd": float(action[0]) * (nz_max - nz_min) / 2.0 + (nz_max + nz_min) / 2.0,
                "roll_rate_cmd": float(action[1]) * (rr_max - rr_min) / 2.0 + (rr_max + rr_min) / 2.0,
                "throttle_cmd": float(action[2]) * (th_max - th_min) / 2.0 + (th_max + th_min) / 2.0,
            }
            virtual_point = {"position_neu": np.asarray(target_for_vp["position_neu"], dtype=np.float64)}
            zero_offset = np.zeros(3, dtype=np.float64)
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "anchor_pos": np.asarray(target_for_vp["position_neu"], dtype=np.float64),
                "offset": zero_offset,
                "world_offset": zero_offset,
                "offset_frame": "world_neu",
                "direct_command_mode": True,
                "action_applied": True,
            }
            direct_track_mode_effective = False
            virtual_point_source = "direct_command"
            effective_guidance_mode = "direct_command"
        elif use_command_override:
            raw_command = dict(command_override)
            virtual_point = {"position_neu": target_for_vp["position_neu"]}
            zero_offset = np.zeros(3, dtype=np.float64)
            vp_info = {
                "virtual_point": virtual_point["position_neu"],
                "anchor_mode": anchor_mode,
                "anchor_pos": np.asarray(target_for_vp["position_neu"], dtype=np.float64),
                "offset": zero_offset,
                "world_offset": zero_offset,
                "offset_frame": "world_neu",
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

        raw_command, task_command_info = self._task_adjust_command(
            raw_command,
            own_state,
            target_state,
            rel_state,
            use_command_override=use_command_override,
        )
        raw_command = dict(raw_command)
        vp_info = self._ensure_vpp_semantic_defaults(vp_info)

        # Persist virtual point for observation features (e.g. VP tracking error)
        self._last_virtual_point = virtual_point

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
        eps = 1e-6
        self._last_command_saturation = {
            "nz_saturated": float(
                abs(raw_command.get("nz_cmd", 1.0) - clipped_command["nz_cmd"]) > eps
            ),
            "roll_rate_saturated": float(
                abs(raw_command.get("roll_rate_cmd", 0.0) - clipped_command["roll_rate_cmd"]) > eps
            ),
            "throttle_saturated": float(
                abs(raw_command.get("throttle_cmd", 0.5) - clipped_command["throttle_cmd"]) > eps
            ),
        }
        filtered_command = self._apply_command_filter(
            clipped_command,
            reset=bool(
                task_command_info.get("task_supervisor_recovery_active", False)
                or task_command_info.get("reset_command_filter", False)
            ),
        )

        # 6b. 通信延迟：对滤波后的指令做一步滞后近似
        delayed_command = self._apply_communication_delay(filtered_command)

        # 7. 环境 step
        actuator_info = {}
        if self._backend == "jsbsim":
            actuator_info = self._step_jsbsim(
                delayed_command,
                aggressiveness=self._last_aggressiveness,
                pid_gain_deltas=self._last_pid_gain_deltas,
                target_command=target_command,
            )
        else:
            self._step_simple(delayed_command, target_command=target_command)
        self._prediction_buffer_synced_for_action = False

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
        if self.combat_hp.enabled and term_info.get("is_success"):
            terminated = False
            truncated = False
            term_info["tracking_success_suppressed_by_combat"] = True
            term_info["reason"] = None
            term_info["is_success"] = False
            if self.current_step >= self.max_steps:
                truncated = True
                term_info["reason"] = "timeout"
                term_info["is_timeout"] = True
        raw_term_info = dict(term_info)
        term_info["raw_termination_reason"] = raw_term_info.get("reason")
        term_info["raw_is_crash"] = bool(raw_term_info.get("is_crash", False))
        term_info["raw_is_out_of_bounds"] = bool(raw_term_info.get("is_out_of_bounds", False))
        term_info["raw_is_timeout"] = bool(raw_term_info.get("is_timeout", False))
        combat_step_info = self.combat_hp.step(own_state_post, target_state_post)
        release_on_attack_zone_triggered = bool(
            combat_step_info.get("target_in_attack_zone", False)
            if post_merge_predicted_target_blend_release_requires_target_attack_zone
            else (
                combat_step_info.get("ego_in_attack_zone", False)
                or combat_step_info.get("target_in_attack_zone", False)
            )
        )
        release_below_altitude_triggered = bool(
            post_merge_predicted_target_blend_release_below_altitude_m is not None
            and float(own_state_post["altitude_m"])
            <= float(post_merge_predicted_target_blend_release_below_altitude_m)
        )
        if (
            self._first_pass_complete
            and post_merge_predicted_target_blend is not None
            and not self._post_merge_predicted_target_blend_released
            and (
                post_merge_predicted_target_blend_hold_expired
                or (
                (
                    post_merge_predicted_target_blend_release_on_attack_zone
                    and release_on_attack_zone_triggered
                )
                or release_below_altitude_triggered
                )
            )
        ):
            self._post_merge_predicted_target_blend_released = True
        target_failed = self._aircraft_failed(target_state_post)
        ego_failed = bool(term_info.get("is_crash") or term_info.get("is_out_of_bounds"))
        combat_terminated, combat_truncated, combat_info = self.combat_hp.resolve_terminal(
            term_info,
            current_time_s=self._sim_time_s,
            ego_failed=ego_failed,
            target_failed=target_failed,
        )
        if combat_terminated or combat_truncated:
            terminated = combat_terminated
            truncated = combat_truncated
            term_info.update(combat_info)
            term_info["reason"] = combat_info.get("combat_reason", term_info.get("reason"))
            term_info["is_success"] = bool(combat_info.get("combat_success", False))
            term_info["is_crash"] = ego_failed or bool(term_info.get("is_crash", False))
            term_info["is_timeout"] = bool(combat_truncated)
        else:
            term_info.update(combat_step_info)

        # 10. 计算 reward（基于 post-step 状态，含 terminal_reward 注入）
        reward, reward_terms = self._compute_reward(
            own_state_post,
            target_state_post,
            rel_state_post,
            filtered_command,
            term_info,
        )

        # Serialize virtual_point for info (convert ndarrays → lists)
        def _serialize_vp(vp):
            if isinstance(vp, dict):
                return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vp.items()}
            elif isinstance(vp, np.ndarray):
                return vp.tolist()
            return vp

        vpp_geometry_info = self._build_vpp_geometry_diagnostics(
            virtual_point,
            vp_info,
            own_state_post,
            target_state_post,
        )
        self._post_merge_tactical_basis_recovery_profile_was_active = bool(
            post_merge_tactical_basis_recovery_profile_state.get("active", False)
        )
        if not self._post_merge_tactical_basis_recovery_profile_was_active:
            self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign = 0.0
            self._post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining = 0
        else:
            self._post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining = int(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_next_steps_remaining", 0
                )
            )
        self._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m = (
            float(vpp_geometry_info["vp_lateral_bias_m"])
        )
        self._post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m = (
            float(vpp_geometry_info["vp_forward_bias_m"])
        )
        self._post_merge_tactical_basis_recovery_profile_previous_altitude_m = float(
            own_state.get("altitude_m", np.nan)
        )
        merge_info = self._update_merge_diagnostics(rel_state_post)
        attack_zone_info = self._build_attack_zone_diagnostics(term_info)
        if merge_info.get("post_merge", False):
            ego_only_attack_zone = bool(
                attack_zone_info.get("ego_in_attack_zone", False)
                and not attack_zone_info.get("target_in_attack_zone", False)
            )
            if ego_only_attack_zone:
                self._post_merge_anchor_mode_ego_only_streak_steps += 1
            else:
                self._post_merge_anchor_mode_ego_only_streak_steps = 0
            if ego_only_attack_zone:
                self._post_merge_offensive_anchor_blend_ego_only_streak_steps += 1
            else:
                self._post_merge_offensive_anchor_blend_ego_only_streak_steps = 0
            if ego_only_attack_zone:
                self._post_merge_predicted_target_forward_scale_ego_only_streak_steps += 1
            else:
                self._post_merge_predicted_target_forward_scale_ego_only_streak_steps = 0
        else:
            self._post_merge_anchor_mode_ego_only_streak_steps = 0
            self._post_merge_offensive_anchor_blend_ego_only_streak_steps = 0
            self._post_merge_predicted_target_forward_scale_ego_only_streak_steps = 0
        if (
            merge_info.get("post_merge", False)
            and not self._post_merge_anchor_mode_released
            and self._post_merge_anchor_mode_has_activated
            and post_merge_anchor_mode_release_ego_only_streak_steps is not None
            and self._post_merge_anchor_mode_ego_only_streak_steps
            >= post_merge_anchor_mode_release_ego_only_streak_steps
        ):
            self._post_merge_anchor_mode_released = True
            post_merge_anchor_mode_release_triggered = True
            self._post_merge_anchor_mode_lateral_world_offset_latched = None
            self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
                None
            )
        if (
            merge_info.get("post_merge", False)
            and self._post_merge_anchor_mode_released
            and post_merge_anchor_mode_release_reset_on_streak_break
            and not ego_only_attack_zone
        ):
            self._post_merge_anchor_mode_released = False
            post_merge_anchor_mode_release_reset_triggered = True
            self._post_merge_anchor_mode_lateral_world_offset_latched = None
            self._post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m = (
                None
            )
        if (
            merge_info.get("post_merge", False)
            and self._post_merge_offensive_anchor_blend_has_activated
            and not self._post_merge_offensive_anchor_blend_released
            and post_merge_offensive_anchor_blend_release_ego_only_streak_steps
            is not None
            and self._post_merge_offensive_anchor_blend_ego_only_streak_steps
            >= post_merge_offensive_anchor_blend_release_ego_only_streak_steps
        ):
            self._post_merge_offensive_anchor_blend_released = True
            self._post_merge_offensive_anchor_blend_release_step = self.current_step
            post_merge_offensive_anchor_blend_release_triggered = True
        if (
            merge_info.get("post_merge", False)
            and self._post_merge_offensive_anchor_blend_released
            and post_merge_offensive_anchor_blend_release_reset_on_streak_break
            and not ego_only_attack_zone
        ):
            self._post_merge_offensive_anchor_blend_released = False
            self._post_merge_offensive_anchor_blend_release_step = None
            post_merge_offensive_anchor_blend_release_reset_triggered = True
            self._post_merge_offensive_anchor_lateral_world_offset_latched = None
        if (
            merge_info.get("post_merge", False)
            and not self._post_merge_predicted_target_forward_scale_released
            and post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
            is not None
            and self._post_merge_predicted_target_forward_scale_ego_only_streak_steps
            >= post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
        ):
            self._post_merge_predicted_target_forward_scale_released = True
            post_merge_predicted_target_forward_scale_release_triggered = True
        if (
            merge_info.get("post_merge", False)
            and self._post_merge_predicted_target_forward_scale_released
            and post_merge_predicted_target_forward_scale_release_reset_on_streak_break
            and not ego_only_attack_zone
        ):
            self._post_merge_predicted_target_forward_scale_released = False
            post_merge_predicted_target_forward_scale_release_reset_triggered = True

        # 组装 info
        configured_offset_frame = "world_neu"
        if self.virtual_point_generator is not None:
            configured_offset_frame = str(
                getattr(self.virtual_point_generator, "offset_frame", "world_neu")
            )
        info = {
            "virtual_point": _serialize_vp(virtual_point),
            "anchor_pos": vpp_geometry_info["anchor_pos"].tolist(),
            "vp_position_neu": vpp_geometry_info["vp_position_neu"].tolist(),
            "vp_offset": vpp_geometry_info["offset"].tolist(),
            "vp_world_offset": vpp_geometry_info["world_offset"].tolist(),
            "offset_frame": vpp_geometry_info["offset_frame"],
            "configured_offset_frame": configured_offset_frame,
            "action_semantics": str(
                vp_info.get("action_semantics", "cartesian_offset")
            ),
            "configured_action_semantics": str(
                vp_info.get("configured_action_semantics", "cartesian_offset")
            ),
            "tactical_basis_enabled": bool(
                vp_info.get("tactical_basis_enabled", False)
            ),
            "tactical_basis_action_ll": float(
                vp_info.get("tactical_basis_action_ll", np.nan)
            ),
            "tactical_basis_action_io": float(
                vp_info.get("tactical_basis_action_io", np.nan)
            ),
            "tactical_basis_action_cd": float(
                vp_info.get("tactical_basis_action_cd", np.nan)
            ),
            "tactical_basis_lead_lag_extent_m": float(
                vp_info.get("tactical_basis_lead_lag_extent_m", np.nan)
            ),
            "tactical_basis_inside_outside_extent_m": float(
                vp_info.get("tactical_basis_inside_outside_extent_m", np.nan)
            ),
            "tactical_basis_climb_descent_extent_m": float(
                vp_info.get("tactical_basis_climb_descent_extent_m", np.nan)
            ),
            "tactical_basis_longitudinal_frame": str(
                vp_info.get("tactical_basis_longitudinal_frame", "target_velocity")
            ),
            "tactical_basis_lateral_frame": str(
                vp_info.get("tactical_basis_lateral_frame", "encounter_stable")
            ),
            "tactical_basis_vertical_frame": str(
                vp_info.get("tactical_basis_vertical_frame", "world_neu")
            ),
            "tactical_basis_lateral_sign_mode": str(
                vp_info.get("tactical_basis_lateral_sign_mode", "same_side")
            ),
            "tactical_basis_lateral_sign": float(
                vp_info.get("tactical_basis_lateral_sign", np.nan)
            ),
            "tactical_basis_ll_world": np.asarray(
                vp_info.get("tactical_basis_ll_world", np.full(3, np.nan)),
                dtype=np.float64,
            ).tolist(),
            "tactical_basis_io_world": np.asarray(
                vp_info.get("tactical_basis_io_world", np.full(3, np.nan)),
                dtype=np.float64,
            ).tolist(),
            "tactical_basis_cd_world": np.asarray(
                vp_info.get("tactical_basis_cd_world", np.full(3, np.nan)),
                dtype=np.float64,
            ).tolist(),
            "tactical_basis_world_offset": np.asarray(
                vp_info.get("tactical_basis_world_offset", np.full(3, np.nan)),
                dtype=np.float64,
            ).tolist(),
            "runtime_specialist_key": self._runtime_specialist_key,
            "runtime_specialist_profile": self._runtime_specialist_profile,
            "runtime_specialist_mode_name": self._runtime_specialist_mode_name,
            "runtime_specialist_reason": self._runtime_specialist_reason,
            "post_merge_tactical_basis_recovery_profile_active": bool(
                post_merge_tactical_basis_recovery_profile_state.get("active", False)
            ),
            "post_merge_tactical_basis_recovery_profile_reason": (
                post_merge_tactical_basis_recovery_profile_state.get("reason")
            ),
            "post_merge_tactical_basis_recovery_profile_source": (
                post_merge_tactical_basis_recovery_profile_state.get("source")
            ),
            "post_merge_tactical_basis_recovery_profile_requested": (
                post_merge_tactical_basis_recovery_profile_state.get(
                    "requested_profile"
                )
            ),
            "post_merge_tactical_basis_recovery_profile_runtime_reason": (
                post_merge_tactical_basis_recovery_profile_state.get(
                    "runtime_reason"
                )
            ),
            "post_merge_tactical_basis_recovery_profile_override_key": (
                post_merge_tactical_basis_recovery_profile_state.get("override_key")
            ),
            "post_merge_tactical_basis_recovery_profile_specialist_profile_match": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "specialist_profile_match", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_blend_release_recovery_active": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "blend_release_recovery_active", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_ll_pre": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "ll_pre", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_ll_post": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "ll_post", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_pre": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_pre", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_post": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_post", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_cd_pre": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "cd_pre", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_cd_post": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "cd_post", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold_enabled": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_entry_lateral_sign_hold_enabled", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold_active": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_entry_lateral_sign_hold_active", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold_applied": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_entry_lateral_sign_hold_applied", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_entry_lateral_sign", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold_previous_vp_lateral_bias_m": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_entry_lateral_sign_hold_previous_vp_lateral_bias_m",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold_release_vp_lateral_bias_m": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "io_entry_lateral_sign_hold_release_vp_lateral_bias_m",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_configured_steps": int(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_configured_steps", 0
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_armed": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_armed", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_active": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_active", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining": int(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_steps_remaining", 0
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_previous_vp_forward_bias_m": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_previous_vp_forward_bias_m",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_previous_altitude_delta_m": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_previous_altitude_delta_m",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_io_min_abs": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_io_min_abs", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_io_min_abs_applied": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_io_min_abs_applied", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_ll_floor": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_ll_floor", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_ll_floor_applied": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_ll_floor_applied", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_ll_overdeep_vp_forward_bias_m_max": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_ll_overdeep_vp_forward_bias_m_max",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_cd_min": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_cd_min", np.nan
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_cd_min_applied": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_cd_min_applied", False
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_cd_descending_altitude_delta_m_min_abs": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_cd_descending_altitude_delta_m_min_abs",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_predicted_target_forward_scale_override": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_predicted_target_forward_scale_override",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_predicted_target_forward_scale_override_applied": bool(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_predicted_target_forward_scale_override_applied",
                    False,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_entry_window_predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max": float(
                post_merge_tactical_basis_recovery_profile_state.get(
                    "entry_window_predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_predicted_target_forward_scale_override": (
                float(
                    post_merge_tactical_basis_recovery_profile_state.get(
                        "predicted_target_forward_scale_override"
                    )
                )
                if post_merge_tactical_basis_recovery_profile_state.get(
                    "predicted_target_forward_scale_override"
                )
                is not None
                else np.nan
            ),
            "anchor_mode_requested": requested_anchor_mode,
            "predicted_target_blend": float(vp_info.get("predicted_target_blend", 1.0)),
            "predicted_target_forward_scale": float(
                vp_info.get("predicted_target_forward_scale", 1.0)
            ),
            "offensive_anchor_blend": float(
                vp_info.get("offensive_anchor_blend", 0.0)
            ),
            "post_merge_predicted_target_blend": (
                float(post_merge_predicted_target_blend)
                if post_merge_predicted_target_blend is not None
                else np.nan
            ),
            "post_merge_predicted_target_forward_scale": (
                float(post_merge_predicted_target_forward_scale)
                if post_merge_predicted_target_forward_scale is not None
                else np.nan
            ),
            "post_merge_predicted_target_forward_scale_release_scale": (
                float(post_merge_predicted_target_forward_scale_release_scale)
                if post_merge_predicted_target_forward_scale_release_scale is not None
                else np.nan
            ),
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": (
                float(
                    post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
                )
                if post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
                is not None
                else np.nan
            ),
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": bool(
                post_merge_predicted_target_forward_scale_release_reset_on_streak_break
            ),
            "post_merge_predicted_target_forward_scale_hold_steps": (
                float(post_merge_predicted_target_forward_scale_hold_steps)
                if post_merge_predicted_target_forward_scale_hold_steps is not None
                else np.nan
            ),
            "post_merge_anchor_mode_release_ego_only_streak_steps": (
                float(post_merge_anchor_mode_release_ego_only_streak_steps)
                if post_merge_anchor_mode_release_ego_only_streak_steps is not None
                else np.nan
            ),
            "post_merge_anchor_mode_recovery_below_altitude_m": (
                float(post_merge_anchor_mode_recovery_below_altitude_m)
                if np.isfinite(post_merge_anchor_mode_recovery_below_altitude_m)
                else np.nan
            ),
            "post_merge_anchor_mode_release_reset_on_streak_break": bool(
                post_merge_anchor_mode_release_reset_on_streak_break
            ),
            "post_merge_anchor_mode_offensive_anchor_blend": (
                float(post_merge_anchor_mode_offensive_anchor_blend)
                if post_merge_anchor_mode_offensive_anchor_blend is not None
                else np.nan
            ),
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend": (
                float(post_merge_anchor_mode_offensive_anchor_longitudinal_blend)
                if post_merge_anchor_mode_offensive_anchor_longitudinal_blend
                is not None
                else np.nan
            ),
            "post_merge_anchor_mode_offensive_anchor_lateral_blend": (
                float(post_merge_anchor_mode_offensive_anchor_lateral_blend)
                if post_merge_anchor_mode_offensive_anchor_lateral_blend is not None
                else np.nan
            ),
            "post_merge_anchor_mode_offensive_anchor_blend_active": bool(
                post_merge_anchor_mode_offensive_anchor_blend_active
            ),
            "post_merge_anchor_mode_offensive_anchor_component_blend_active": bool(
                post_merge_anchor_mode_offensive_anchor_component_blend_active
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": bool(
                post_merge_anchor_mode_lateral_world_offset_latch_on_activation
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min": (
                float(
                    post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min
                )
                if post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m_min
                is not None
                else np.nan
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_range_gate_satisfied": bool(
                post_merge_anchor_mode_lateral_world_offset_latch_range_gate_satisfied
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m": (
                float(post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m)
                if np.isfinite(
                    post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m
                )
                else np.nan
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_active": bool(
                post_merge_anchor_mode_lateral_world_offset_latch_active
            ),
            "post_merge_offensive_anchor_blend": (
                float(post_merge_offensive_anchor_blend)
                if post_merge_offensive_anchor_blend is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_longitudinal_scale": (
                float(post_merge_offensive_anchor_longitudinal_scale)
                if post_merge_offensive_anchor_longitudinal_scale is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_lateral_scale": (
                float(post_merge_offensive_anchor_lateral_scale)
                if post_merge_offensive_anchor_lateral_scale is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_blend": (
                float(post_merge_offensive_anchor_blend_release_blend)
                if post_merge_offensive_anchor_blend_release_blend is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": (
                float(post_merge_offensive_anchor_blend_release_ego_only_streak_steps)
                if post_merge_offensive_anchor_blend_release_ego_only_streak_steps
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break": bool(
                post_merge_offensive_anchor_blend_release_reset_on_streak_break
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_enabled": bool(
                post_merge_offensive_anchor_blend_release_direct_track_enabled
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": (
                float(
                    post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
                )
                if post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": (
                float(
                    post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
                )
                if post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": (
                float(
                    post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
                )
                if post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": (
                float(
                    post_merge_offensive_anchor_blend_release_recovery_lateral_blend
                )
                if post_merge_offensive_anchor_blend_release_recovery_lateral_blend
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": (
                float(
                    post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
                )
                if post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": (
                float(post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min)
                if post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m": (
                float(
                    post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m
                )
                if (
                    post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m
                    is not None
                )
                else np.nan
            ),
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": bool(
                post_merge_offensive_anchor_lateral_world_offset_latch_on_activation
            ),
            "post_merge_offensive_anchor_lateral_world_offset_latch_active": bool(
                post_merge_offensive_anchor_lateral_world_offset_latch_active
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only": bool(
                post_merge_offensive_anchor_blend_release_lateral_only
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": (
                float(post_merge_offensive_anchor_blend_release_lateral_only_hold_steps)
                if post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
                is not None
                else np.nan
            ),
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage": bool(
                post_merge_offensive_anchor_gate[
                    "requires_geometry_disadvantage"
                ]
            ),
            "post_merge_offensive_anchor_condition_met": bool(
                post_merge_offensive_anchor_gate["condition_met"]
            ),
            "post_merge_offensive_anchor_target_only_attack_zone_disadvantage": bool(
                post_merge_offensive_anchor_gate[
                    "target_only_attack_zone_disadvantage"
                ]
            ),
            "post_merge_offensive_anchor_geometry_disadvantage": bool(
                post_merge_offensive_anchor_gate["geometry_disadvantage"]
            ),
            "post_merge_offensive_anchor_alignment_disadvantage": bool(
                post_merge_offensive_anchor_gate["alignment_disadvantage"]
            ),
            "post_merge_offensive_anchor_gate_ego_attack_score": float(
                post_merge_offensive_anchor_gate["ego_attack_score"]
            ),
            "post_merge_offensive_anchor_gate_target_attack_score": float(
                post_merge_offensive_anchor_gate["target_attack_score"]
            ),
            "post_merge_offensive_anchor_gate_ego_in_attack_zone": bool(
                post_merge_offensive_anchor_gate["ego_in_attack_zone"]
            ),
            "post_merge_offensive_anchor_gate_target_in_attack_zone": bool(
                post_merge_offensive_anchor_gate["target_in_attack_zone"]
            ),
            "post_merge_offensive_anchor_gate_aa_deg_min": float(
                post_merge_offensive_anchor_gate["aa_deg_min"]
            ),
            "post_merge_offensive_anchor_gate_aa_deg": float(
                post_merge_offensive_anchor_gate["aa_deg"]
            ),
            "post_merge_offensive_anchor_gate_range_rate_mps": float(
                post_merge_offensive_anchor_gate["range_rate_mps"]
            ),
            "post_merge_offensive_anchor_gate_range_opening": bool(
                post_merge_offensive_anchor_gate["range_opening"]
            ),
            "post_merge_predicted_target_blend_release_on_attack_zone": bool(
                post_merge_predicted_target_blend_release_on_attack_zone
            ),
            "post_merge_predicted_target_blend_release_requires_target_attack_zone": bool(
                post_merge_predicted_target_blend_release_requires_target_attack_zone
            ),
            "post_merge_predicted_target_blend_release_below_altitude_m": (
                float(post_merge_predicted_target_blend_release_below_altitude_m)
                if post_merge_predicted_target_blend_release_below_altitude_m is not None
                else np.nan
            ),
            "post_merge_predicted_target_blend_hold_steps": (
                float(post_merge_predicted_target_blend_hold_steps)
                if post_merge_predicted_target_blend_hold_steps is not None
                else np.nan
            ),
            "post_merge_predicted_target_blend_release_blend": (
                float(post_merge_predicted_target_blend_release_blend)
                if post_merge_predicted_target_blend_release_blend is not None
                else np.nan
            ),
            "post_merge_predicted_target_blend_steps_since_first_pass": float(
                post_merge_predicted_target_blend_steps_since_first_pass
            ),
            "post_merge_predicted_target_blend_hold_remaining_steps": float(
                post_merge_predicted_target_blend_hold_remaining_steps
            ),
            "post_merge_predicted_target_blend_hold_window_open": bool(
                post_merge_predicted_target_blend_hold_window_open
            ),
            "post_merge_predicted_target_forward_scale_steps_since_first_pass": float(
                post_merge_predicted_target_forward_scale_steps_since_first_pass
            ),
            "post_merge_predicted_target_forward_scale_hold_remaining_steps": float(
                post_merge_predicted_target_forward_scale_hold_remaining_steps
            ),
            "post_merge_predicted_target_forward_scale_hold_window_open": bool(
                post_merge_predicted_target_forward_scale_hold_window_open
            ),
            "post_merge_predicted_target_forward_scale_ego_only_streak_steps": float(
                self._post_merge_predicted_target_forward_scale_ego_only_streak_steps
            ),
            "post_merge_predicted_target_blend_active": bool(
                post_merge_predicted_target_blend_active
            ),
            "post_merge_predicted_target_forward_scale_active": bool(
                post_merge_predicted_target_forward_scale_active
            ),
            "post_merge_predicted_target_forward_scale_release_scale_active": bool(
                post_merge_predicted_target_forward_scale_release_scale_active
            ),
            "post_merge_predicted_target_forward_scale_hold_expired": bool(
                post_merge_predicted_target_forward_scale_hold_expired
            ),
            "post_merge_predicted_target_forward_scale_release_triggered": bool(
                post_merge_predicted_target_forward_scale_release_triggered
            ),
            "post_merge_predicted_target_forward_scale_release_reset_triggered": bool(
                post_merge_predicted_target_forward_scale_release_reset_triggered
            ),
            "post_merge_predicted_target_forward_scale_released": bool(
                self._post_merge_predicted_target_forward_scale_released
            ),
            "post_merge_offensive_anchor_blend_active": bool(
                post_merge_offensive_anchor_blend_active
            ),
            "post_merge_offensive_anchor_blend_release_blend_active": bool(
                post_merge_offensive_anchor_blend_release_blend_active
            ),
            "post_merge_offensive_anchor_blend_ego_only_streak_steps": float(
                self._post_merge_offensive_anchor_blend_ego_only_streak_steps
            ),
            "post_merge_offensive_anchor_blend_release_triggered": bool(
                post_merge_offensive_anchor_blend_release_triggered
            ),
            "post_merge_offensive_anchor_blend_release_reset_triggered": bool(
                post_merge_offensive_anchor_blend_release_reset_triggered
            ),
            "post_merge_offensive_anchor_blend_released": bool(
                self._post_merge_offensive_anchor_blend_released
            ),
            "post_merge_offensive_anchor_blend_release_recovery_active": bool(
                post_merge_offensive_anchor_blend_release_recovery_active
            ),
            "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active": bool(
                post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active
            ),
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active": bool(
                post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active
            ),
            "post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m": float(
                post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active": bool(
                post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active
            ),
            "post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m": float(
                post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_active": bool(
                post_merge_offensive_anchor_blend_release_direct_track_active
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_active": bool(
                post_merge_offensive_anchor_blend_release_lateral_only_active
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release": float(
                post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps": float(
                post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open": bool(
                post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open
            ),
            "post_merge_predicted_target_blend_release_blend_active": bool(
                post_merge_predicted_target_blend_release_blend_active
            ),
            "post_merge_predicted_target_blend_released": bool(
                self._post_merge_predicted_target_blend_released
            ),
            "longitudinal_scale": float(vp_info.get("longitudinal_scale", 1.0)),
            "lateral_scale": float(vp_info.get("lateral_scale", 1.0)),
            "close_range_anchor_mode": close_range_anchor_mode,
            "close_range_anchor_trigger_range_m": float(
                close_range_anchor_trigger_range_m
            ),
            "close_range_anchor_release_on_post_merge": bool(
                close_range_anchor_release_on_post_merge
            ),
            "close_range_anchor_requires_first_pass": bool(
                close_range_anchor_requires_first_pass
            ),
            "close_range_anchor_offensive_anchor_blend": (
                float(close_range_anchor_offensive_anchor_blend)
                if close_range_anchor_offensive_anchor_blend is not None
                else np.nan
            ),
            "close_range_anchor_offensive_anchor_blend_active": bool(
                close_range_anchor_offensive_anchor_blend_active
            ),
            "close_range_anchor_release_alignment_angle_deg_max": (
                float(close_range_anchor_release_alignment_angle_deg_max)
                if close_range_anchor_release_alignment_angle_deg_max is not None
                else np.nan
            ),
            "close_range_anchor_release_alignment_satisfied": bool(
                close_range_anchor_release_alignment_satisfied
            ),
            "close_range_anchor_release_ready": bool(close_range_anchor_release_ready),
            "close_range_anchor_post_merge_hold_steps": int(
                close_range_anchor_post_merge_hold_steps
            ),
            "close_range_anchor_post_merge_steps_since_first_pass": float(
                close_range_anchor_post_merge_steps_since_first_pass
            ),
            "close_range_anchor_post_merge_hold_remaining_steps": float(
                close_range_anchor_post_merge_hold_remaining_steps
            ),
            "close_range_anchor_window_open": bool(close_range_anchor_window_open),
            "close_range_anchor_alignment_angle_deg_max": (
                float(close_range_anchor_alignment_angle_deg_max)
                if close_range_anchor_alignment_angle_deg_max is not None
                else np.nan
            ),
            "close_range_anchor_alignment_satisfied": bool(
                close_range_anchor_alignment_satisfied
            ),
            "close_range_anchor_mode_active": bool(close_range_anchor_mode_active),
            "post_merge_anchor_mode": post_merge_anchor_mode,
            "post_merge_anchor_mode_requires_geometry_disadvantage": bool(
                post_merge_anchor_mode_requires_geometry_disadvantage
            ),
            "post_merge_anchor_mode_condition_met": bool(
                post_merge_anchor_mode_condition_met
            ),
            "post_merge_anchor_mode_active": bool(post_merge_anchor_mode_active),
            "post_merge_anchor_mode_recovery_active": bool(
                post_merge_anchor_mode_recovery_active
            ),
            "post_merge_anchor_mode_ego_only_streak_steps": float(
                self._post_merge_anchor_mode_ego_only_streak_steps
            ),
            "post_merge_anchor_mode_release_triggered": bool(
                post_merge_anchor_mode_release_triggered
            ),
            "post_merge_anchor_mode_release_reset_triggered": bool(
                post_merge_anchor_mode_release_reset_triggered
            ),
            "post_merge_anchor_mode_released": bool(
                self._post_merge_anchor_mode_released
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": bool(
                post_merge_anchor_mode_lateral_world_offset_latch_on_activation
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_active": bool(
                post_merge_anchor_mode_lateral_world_offset_latch_active
            ),
            "vp_forward_bias_m": vpp_geometry_info["vp_forward_bias_m"],
            "vp_lateral_bias_m": vpp_geometry_info["vp_lateral_bias_m"],
            "guidance_command": filtered_command,
            "raw_command": raw_command,
            "command_override_active": use_command_override,
            "opponent_info": opponent_info,
            "opponent_stage": self._opponent_stage,
            "opponent_command": opponent_info.get("opponent_command"),
            "opponent_action": opponent_info.get("opponent_action"),
            "task_command_adjustment": task_command_info,
            "task_supervisor_state": task_command_info.get("task_supervisor_state"),
            "task_supervisor_active": bool(
                task_command_info.get("task_supervisor_active", False)
            ),
            "reward_terms": reward_terms,
            "termination_info": term_info,
            "relative_state": rel_state_post,
            "anchor_mode": anchor_mode,
            "own_state": own_state_post,
            "target_state": target_state_post,
            "current_step": self.current_step,
            "episode": self._episode_count,
            "backend": self._backend,
            "backend_fallback_occurred": self._backend_fallback_occurred,
            "backend_fallback_reason": self._backend_fallback_reason,
            "provenance": self._build_provenance(
                self._last_observation_schema if hasattr(self, "_last_observation_schema") else {}
            ),
            "range_m": rel_state_post.get("range_m", np.nan),
            "range_rate_mps": rel_state_post.get("range_rate_mps", np.nan),
            "ata_deg": float(np.rad2deg(rel_state_post.get("ata_rad", np.nan))),
            "aa_deg": float(np.rad2deg(rel_state_post.get("aa_rad", np.nan))),
            "aspect_deg": float(np.rad2deg(rel_state_post.get("aa_rad", np.nan))),
            "los_rate": rel_state_post.get("range_rate_mps", np.nan),
            "nz_cmd": filtered_command.get("nz_cmd", np.nan),
            "roll_rate_cmd": filtered_command.get("roll_rate_cmd", np.nan),
            "throttle_cmd": filtered_command.get("throttle_cmd", np.nan),
            "nz_saturated": self._last_command_saturation["nz_saturated"],
            "roll_rate_saturated": self._last_command_saturation["roll_rate_saturated"],
            "throttle_saturated": self._last_command_saturation["throttle_saturated"],
            "lookahead_time_s": lookahead_time_s,
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
            "prediction_noise_std_m": prediction_info["prediction_noise_std_m"],
            "prediction_noise_applied": prediction_info["prediction_noise_applied"],
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
            "aggressiveness": self._last_aggressiveness,
            "pid_gain_deltas": self._last_pid_gain_deltas,
        }
        info.update(merge_info)
        info.update(attack_zone_info)
        if "offensive_anchor_longitudinal_m" in vp_info:
            info["offensive_anchor_longitudinal_m"] = float(
                vp_info["offensive_anchor_longitudinal_m"]
            )
            info["offensive_anchor_frame"] = str(
                vp_info.get("offensive_anchor_frame", "target_velocity")
            )
            info["offensive_anchor_lateral_frame"] = str(
                vp_info.get(
                    "offensive_anchor_lateral_frame",
                    vp_info.get("offensive_anchor_frame", "target_velocity"),
                )
            )
            info["offensive_anchor_encounter_stable_max_heading_delta_deg"] = float(
                vp_info.get(
                    "offensive_anchor_encounter_stable_max_heading_delta_deg",
                    np.nan,
                )
            )
            info["offensive_anchor_lateral_m"] = float(
                vp_info.get("offensive_anchor_lateral_m", np.nan)
            )
            info["offensive_anchor_vertical_m"] = float(
                vp_info.get("offensive_anchor_vertical_m", np.nan)
            )
        if "offensive_anchor_longitudinal_blend" in vp_info:
            info["offensive_anchor_longitudinal_blend"] = float(
                vp_info.get("offensive_anchor_longitudinal_blend", np.nan)
            )
        if "offensive_anchor_lateral_blend" in vp_info:
            info["offensive_anchor_lateral_blend"] = float(
                vp_info.get("offensive_anchor_lateral_blend", np.nan)
            )
        if "offensive_anchor_lateral_sign_mode" in vp_info:
            info["offensive_anchor_lateral_sign_mode"] = str(
                vp_info.get("offensive_anchor_lateral_sign_mode", "same_side")
            )
        if "offensive_anchor_lateral_sign" in vp_info:
            info["offensive_anchor_lateral_sign"] = float(
                vp_info.get("offensive_anchor_lateral_sign", np.nan)
            )
        if "offensive_anchor_pos" in vp_info:
            info["offensive_anchor_pos"] = np.asarray(
                vp_info["offensive_anchor_pos"], dtype=np.float64
            ).tolist()
        if "offensive_anchor_offset" in vp_info:
            info["offensive_anchor_offset"] = np.asarray(
                vp_info["offensive_anchor_offset"], dtype=np.float64
            ).tolist()
        if "offensive_anchor_world_offset" in vp_info:
            info["offensive_anchor_world_offset"] = np.asarray(
                vp_info["offensive_anchor_world_offset"], dtype=np.float64
            ).tolist()
        if "offensive_anchor_lateral_world_offset" in vp_info:
            info["offensive_anchor_lateral_world_offset"] = np.asarray(
                vp_info["offensive_anchor_lateral_world_offset"],
                dtype=np.float64,
            ).tolist()
        if "offensive_anchor_lateral_local_offset" in vp_info:
            info["offensive_anchor_lateral_local_offset"] = np.asarray(
                vp_info["offensive_anchor_lateral_local_offset"],
                dtype=np.float64,
            ).tolist()
        if "pre_offensive_blend_anchor_pos" in vp_info:
            info["pre_offensive_blend_anchor_pos"] = np.asarray(
                vp_info["pre_offensive_blend_anchor_pos"], dtype=np.float64
            ).tolist()
        if "pre_offensive_blend_anchor_local" in vp_info:
            info["pre_offensive_blend_anchor_local"] = np.asarray(
                vp_info["pre_offensive_blend_anchor_local"], dtype=np.float64
            ).tolist()
        if "post_offensive_blend_anchor_local" in vp_info:
            info["post_offensive_blend_anchor_local"] = np.asarray(
                vp_info["post_offensive_blend_anchor_local"], dtype=np.float64
            ).tolist()
        if "pre_predicted_target_forward_scale_anchor_pos" in vp_info:
            info["pre_predicted_target_forward_scale_anchor_pos"] = np.asarray(
                vp_info["pre_predicted_target_forward_scale_anchor_pos"],
                dtype=np.float64,
            ).tolist()
        info.update(actuator_info)
        info.update(term_info)

        # 12. 任务钩子：允许子类覆盖 reward / done / info
        reward, terminated, truncated, info = self._task_post_step(
            own_state_post, target_state_post, info, reward, terminated, truncated
        )

        # 11. 获取观察（post-step）
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _as_neu_vector(value, default=None):
        if value is None:
            return default
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return default
        if arr.shape[0] < 3 or not np.isfinite(arr[:3]).all():
            return default
        return arr[:3]

    def _position_from_state(self, state: dict):
        for key in ("position_m", "position_neu", "position"):
            value = self._as_neu_vector(state.get(key))
            if value is not None:
                return value
        return np.full(3, np.nan, dtype=np.float64)

    def _ensure_vpp_semantic_defaults(self, vp_info: Optional[dict]) -> dict:
        info = dict(vp_info or {})
        generator = self.virtual_point_generator
        configured_action_semantics = str(
            getattr(
                generator,
                "configured_action_semantics",
                getattr(generator, "action_semantics", "cartesian_offset"),
            )
        )
        if info.get("action_semantics") in (None, ""):
            info["action_semantics"] = configured_action_semantics
        if info.get("configured_action_semantics") in (None, ""):
            info["configured_action_semantics"] = configured_action_semantics
        info.setdefault("tactical_basis_enabled", False)
        info.setdefault("tactical_basis_action_ll", np.nan)
        info.setdefault("tactical_basis_action_io", np.nan)
        info.setdefault("tactical_basis_action_cd", np.nan)
        info.setdefault(
            "tactical_basis_lead_lag_extent_m",
            float(getattr(generator, "tactical_basis_lead_lag_extent_m", np.nan)),
        )
        info.setdefault(
            "tactical_basis_inside_outside_extent_m",
            float(
                getattr(
                    generator,
                    "tactical_basis_inside_outside_extent_m",
                    np.nan,
                )
            ),
        )
        info.setdefault(
            "tactical_basis_climb_descent_extent_m",
            float(
                getattr(
                    generator,
                    "tactical_basis_climb_descent_extent_m",
                    np.nan,
                )
            ),
        )
        info.setdefault(
            "tactical_basis_longitudinal_frame",
            str(getattr(generator, "tactical_basis_longitudinal_frame", "target_velocity")),
        )
        info.setdefault(
            "tactical_basis_lateral_frame",
            str(getattr(generator, "tactical_basis_lateral_frame", "encounter_stable")),
        )
        info.setdefault(
            "tactical_basis_vertical_frame",
            str(getattr(generator, "tactical_basis_vertical_frame", "world_neu")),
        )
        info.setdefault(
            "tactical_basis_lateral_sign_mode",
            str(getattr(generator, "tactical_basis_lateral_sign_mode", "same_side")),
        )
        info.setdefault("tactical_basis_lateral_sign", np.nan)
        nan_vec = np.full(3, np.nan, dtype=np.float64)
        for key in (
            "tactical_basis_ll_world",
            "tactical_basis_io_world",
            "tactical_basis_cd_world",
            "tactical_basis_world_offset",
        ):
            value = self._as_neu_vector(info.get(key), default=nan_vec.copy())
            info[key] = value.copy() if value is not None else nan_vec.copy()
        return info

    def _update_merge_diagnostics(self, rel_state: dict) -> dict:
        range_m = float(rel_state.get("range_m", np.nan))
        range_rate_mps = float(rel_state.get("range_rate_mps", np.nan))
        cfg = self.config.get("combat_diagnostics", {})
        merge_range_m = float(cfg.get("merge_range_m", 1000.0))
        close_range_anchor_trigger_range_m = (
            self._resolve_close_range_anchor_trigger_range_m(merge_range_m)
        )
        (
            close_range_anchor_alignment_satisfied,
            _,
        ) = self._evaluate_close_range_anchor_alignment(rel_state)
        post_merge_hysteresis_m = float(cfg.get("post_merge_hysteresis_m", 50.0))

        if np.isfinite(range_m):
            self._merge_min_range_so_far_m = min(
                self._merge_min_range_so_far_m,
                range_m,
            )
            if range_m <= merge_range_m:
                self._merge_seen_close_range = True
            if (
                range_m <= close_range_anchor_trigger_range_m
                and close_range_anchor_alignment_satisfied
            ):
                self._close_range_anchor_ready = True
            if (
                self._merge_seen_close_range
                and not self._first_pass_complete
                and range_m > self._merge_min_range_so_far_m + post_merge_hysteresis_m
                and range_rate_mps > 0.0
            ):
                self._first_pass_complete = True
                if self._first_pass_completion_step is None:
                    self._first_pass_completion_step = int(self.current_step)

        min_range = (
            self._merge_min_range_so_far_m
            if np.isfinite(self._merge_min_range_so_far_m)
            else np.nan
        )
        return {
            "pre_merge": not self._first_pass_complete,
            "post_merge": bool(self._first_pass_complete),
            "min_range_so_far_m": float(min_range),
            "first_pass_complete": bool(self._first_pass_complete),
        }

    def _build_vpp_geometry_diagnostics(
        self,
        virtual_point: dict,
        vp_info: dict,
        own_state: dict,
        target_state: dict,
    ) -> dict:
        target_pos = self._position_from_state(target_state)
        vp_pos = self._as_neu_vector(
            virtual_point.get("position_neu", virtual_point.get("position")),
            default=target_pos,
        )
        anchor_pos = self._as_neu_vector(vp_info.get("anchor_pos"), default=target_pos)
        offset = self._as_neu_vector(vp_info.get("offset"), default=np.zeros(3))
        world_offset = self._as_neu_vector(
            vp_info.get("world_offset"),
            default=vp_pos - anchor_pos,
        )
        offset_frame = str(
            vp_info.get(
                "offset_frame",
                virtual_point.get("offset_frame", "world_neu"),
            )
        )
        target_relative_vp = world_to_offset_frame(
            vp_pos - target_pos,
            "target_velocity",
            own_state=own_state,
            target_state=target_state,
        )
        return {
            "vp_position_neu": vp_pos,
            "anchor_pos": anchor_pos,
            "offset": offset,
            "world_offset": world_offset,
            "offset_frame": offset_frame,
            "vp_forward_bias_m": float(target_relative_vp[0]),
            "vp_lateral_bias_m": float(target_relative_vp[1]),
        }

    def _evaluate_post_merge_tactical_basis_recovery_profile_state(
        self,
        *,
        action: np.ndarray,
        own_state: dict,
        anchor_mode: str,
        post_merge_offensive_anchor_gate: dict,
        post_merge_offensive_anchor_blend_release_recovery_active: bool,
    ) -> dict:
        cfg = self._post_merge_tactical_basis_recovery_profile_cfg
        enabled = bool(cfg.get("enabled", False))
        configured_profiles = _normalize_runtime_specialist_profile_names(
            cfg.get("specialist_profile_names")
        )
        current_profile = (
            None
            if self._runtime_specialist_profile in (None, "")
            else str(self._runtime_specialist_profile)
        )
        active_via_specialist_profile = (
            current_profile is not None and current_profile in configured_profiles
        )
        active_via_release_recovery = bool(
            cfg.get("activate_on_blend_release_recovery", True)
        ) and bool(post_merge_offensive_anchor_blend_release_recovery_active)
        state = {
            "configured": enabled,
            "active": False,
            "reason": "disabled" if not enabled else "inactive",
            "source": None,
            "requested_profile": current_profile,
            "runtime_reason": self._runtime_specialist_reason,
            "override_key": None,
            "specialist_profile_match": bool(active_via_specialist_profile),
            "blend_release_recovery_active": bool(
                post_merge_offensive_anchor_blend_release_recovery_active
            ),
            "predicted_target_forward_scale_override": None,
            "ll_pre": float("nan"),
            "ll_post": float("nan"),
            "io_pre": float("nan"),
            "io_post": float("nan"),
            "cd_pre": float("nan"),
            "cd_post": float("nan"),
            "io_entry_lateral_sign_hold_enabled": False,
            "io_entry_lateral_sign_hold_active": False,
            "io_entry_lateral_sign_hold_applied": False,
            "io_entry_lateral_sign": float("nan"),
            "io_entry_lateral_sign_hold_previous_vp_lateral_bias_m": float("nan"),
            "io_entry_lateral_sign_hold_release_vp_lateral_bias_m": float("nan"),
            "entry_window_configured_steps": 0,
            "entry_window_armed": False,
            "entry_window_active": False,
            "entry_window_steps_remaining": 0,
            "entry_window_next_steps_remaining": 0,
            "entry_window_previous_vp_forward_bias_m": float("nan"),
            "entry_window_previous_altitude_delta_m": float("nan"),
            "entry_window_io_min_abs": float("nan"),
            "entry_window_io_min_abs_applied": False,
            "entry_window_ll_floor": float("nan"),
            "entry_window_ll_floor_applied": False,
            "entry_window_ll_overdeep_vp_forward_bias_m_max": float("nan"),
            "entry_window_cd_min": float("nan"),
            "entry_window_cd_min_applied": False,
            "entry_window_cd_descending_altitude_delta_m_min_abs": float("nan"),
            "entry_window_predicted_target_forward_scale_override": float("nan"),
            "entry_window_predicted_target_forward_scale_override_applied": False,
            "entry_window_predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max": float(
                "nan"
            ),
            "action": np.asarray(action, dtype=np.float64).copy(),
            "shaped_action": np.asarray(action, dtype=np.float64).copy(),
        }
        if not enabled:
            return state
        if getattr(self.virtual_point_generator, "action_semantics", None) != "tactical_basis_v1":
            state["reason"] = "non_tactical_basis"
            return state
        if str(anchor_mode) != str(cfg.get("anchor_mode", "predicted_target")):
            state["reason"] = "anchor_mode_mismatch"
            return state
        if bool(cfg.get("require_first_pass_complete", True)) and not self._first_pass_complete:
            state["reason"] = "pre_merge"
            return state
        task_name = str(cfg.get("task_name", "head_on"))
        if self.task_name is not None and str(self.task_name) != task_name:
            state["reason"] = "task_mismatch"
            return state
        range_opening = bool(post_merge_offensive_anchor_gate.get("range_opening", False))
        require_range_opening = bool(cfg.get("require_range_opening", True))
        require_range_opening_for_specialist_profile = bool(
            cfg.get("require_range_opening_for_specialist_profile", False)
        )
        if require_range_opening and not range_opening:
            specialist_profile_bypasses_range_opening = bool(
                active_via_specialist_profile
                and not require_range_opening_for_specialist_profile
            )
            if not specialist_profile_bypasses_range_opening:
                state["reason"] = "range_not_opening"
                return state

        if bool(cfg.get("require_no_attack_zone", True)) and (
            bool(post_merge_offensive_anchor_gate.get("ego_in_attack_zone", False))
            or bool(post_merge_offensive_anchor_gate.get("target_in_attack_zone", False))
        ):
            state["reason"] = "attack_zone_active"
            return state

        max_altitude_m = cfg.get("max_altitude_m")
        own_altitude_m = float(own_state.get("altitude_m", np.nan))
        if (
            max_altitude_m is not None
            and np.isfinite(own_altitude_m)
            and own_altitude_m > float(max_altitude_m)
        ):
            state["reason"] = "above_max_altitude"
            return state

        if active_via_specialist_profile:
            state["source"] = "specialist_profile"
        elif active_via_release_recovery:
            state["source"] = "blend_release_recovery"
        else:
            state["reason"] = "inactive_source"
            return state

        effective_cfg, recovery_profile_override_diag = (
            _resolve_post_merge_tactical_basis_recovery_profile_effective_cfg(
                cfg,
                runtime_specialist_reason=self._runtime_specialist_reason,
                aa_deg=post_merge_offensive_anchor_gate.get("aa_deg", np.nan),
                range_rate_mps=post_merge_offensive_anchor_gate.get(
                    "range_rate_mps", np.nan
                ),
                previous_vp_forward_bias_m=(
                    self._post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m
                ),
                previous_vp_lateral_bias_m=(
                    self._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m
                ),
            )
        )
        state.update(recovery_profile_override_diag)

        shaped_action, action_diag = _shape_tactical_basis_action_with_post_merge_recovery_profile(
            action,
            effective_cfg,
        )
        if not self._post_merge_tactical_basis_recovery_profile_was_active:
            previous_vp_lateral_bias_m = (
                self._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m
            )
            if np.isfinite(previous_vp_lateral_bias_m):
                self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign = (
                    float(np.sign(previous_vp_lateral_bias_m))
                )
            else:
                self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign = (
                    float(np.sign(action_diag["io_post"]))
                )
        (
            shaped_action,
            io_sign_hold_diag,
        ) = _apply_post_merge_tactical_basis_recovery_profile_io_entry_lateral_sign_hold(
            shaped_action,
            effective_cfg,
            previous_vp_lateral_bias_m=(
                self._post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m
            ),
            entry_lateral_sign=(
                self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign
            ),
        )
        previous_altitude_delta_m = float("nan")
        if (
            np.isfinite(self._post_merge_tactical_basis_recovery_profile_previous_altitude_m)
            and np.isfinite(own_altitude_m)
        ):
            previous_altitude_delta_m = float(
                own_altitude_m
                - self._post_merge_tactical_basis_recovery_profile_previous_altitude_m
            )
        entry_window_state = (
            _compute_post_merge_tactical_basis_recovery_profile_entry_window_state(
                effective_cfg,
                was_active=self._post_merge_tactical_basis_recovery_profile_was_active,
                is_active=True,
                previous_steps_remaining=(
                    self._post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining
                ),
            )
        )
        shaped_action, entry_window_diag = (
            _apply_post_merge_tactical_basis_recovery_profile_entry_window(
                shaped_action,
                effective_cfg,
                active=bool(entry_window_state["entry_window_active"]),
                steps_remaining=int(entry_window_state["entry_window_steps_remaining"]),
                entry_lateral_sign=(
                    self._post_merge_tactical_basis_recovery_profile_entry_lateral_sign
                ),
                previous_vp_forward_bias_m=(
                    self._post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m
                ),
                previous_altitude_delta_m=previous_altitude_delta_m,
            )
        )
        action_diag["ll_post"] = float(shaped_action[0])
        action_diag["io_post"] = float(shaped_action[1])
        action_diag["cd_post"] = float(shaped_action[2])
        state.update(action_diag)
        state.update(io_sign_hold_diag)
        state.update(entry_window_state)
        state.update(entry_window_diag)
        state["active"] = True
        state["reason"] = "active"
        state["shaped_action"] = shaped_action
        predicted_target_forward_scale_override = effective_cfg.get(
            "predicted_target_forward_scale_override",
            cfg.get("predicted_target_forward_scale_override"),
        )
        if bool(
            entry_window_diag.get(
                "entry_window_predicted_target_forward_scale_override_applied",
                False,
            )
        ):
            predicted_target_forward_scale_override = float(
                entry_window_diag[
                    "entry_window_predicted_target_forward_scale_override"
                ]
            )
        if predicted_target_forward_scale_override is not None:
            state["predicted_target_forward_scale_override"] = float(
                predicted_target_forward_scale_override
            )
        return state

    def _apply_post_merge_offensive_anchor_release_forward_bias_clamp(
        self,
        virtual_point: dict,
        vp_info: dict,
        own_state: dict,
        target_state: dict,
        *,
        anchor_mode: str,
        recovery_active: bool,
        direct_track_mode_effective: bool,
        use_command_override: bool,
        post_merge_offensive_anchor_gate: dict,
    ) -> Tuple[dict, dict, bool, float]:
        clamp_min_m = (
            self._resolve_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min()
        )
        negative_lateral_below_altitude_m = (
            self._resolve_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m()
        )
        clamp_active = False
        preclamp_vp_forward_bias_m = float("nan")
        if (
            clamp_min_m is None
            or not self._first_pass_complete
            or anchor_mode != "predicted_target"
            or direct_track_mode_effective
            or not self._use_virtual_point
            or self.virtual_point_generator is None
            or use_command_override
            or recovery_active
            or not self._post_merge_offensive_anchor_blend_released
            or not bool(post_merge_offensive_anchor_gate["range_opening"])
            or bool(post_merge_offensive_anchor_gate["ego_in_attack_zone"])
            or bool(post_merge_offensive_anchor_gate["target_in_attack_zone"])
        ):
            return virtual_point, vp_info, clamp_active, preclamp_vp_forward_bias_m

        target_pos = self._position_from_state(target_state)
        vp_pos = self._as_neu_vector(
            virtual_point.get("position_neu", virtual_point.get("position")),
            default=target_pos,
        )
        target_relative_vp = world_to_offset_frame(
            vp_pos - target_pos,
            "target_velocity",
            own_state=own_state,
            target_state=target_state,
        )
        preclamp_vp_forward_bias_m = float(target_relative_vp[0])
        if (
            not np.isfinite(preclamp_vp_forward_bias_m)
            or preclamp_vp_forward_bias_m >= float(clamp_min_m)
        ):
            return virtual_point, vp_info, clamp_active, preclamp_vp_forward_bias_m

        preclamp_vp_lateral_bias_m = float(target_relative_vp[1])
        own_altitude_m = float(own_state.get("altitude_m", np.nan))
        if (
            negative_lateral_below_altitude_m is not None
            and np.isfinite(preclamp_vp_lateral_bias_m)
            and preclamp_vp_lateral_bias_m < 0.0
            and np.isfinite(own_altitude_m)
            and own_altitude_m
            > float(negative_lateral_below_altitude_m)
        ):
            return virtual_point, vp_info, clamp_active, preclamp_vp_forward_bias_m

        target_relative_vp[0] = float(clamp_min_m)
        clamped_world_offset = offset_to_world(
            target_relative_vp,
            "target_velocity",
            own_state=own_state,
            target_state=target_state,
        )
        clamped_vp_pos = target_pos + clamped_world_offset

        updated_virtual_point = dict(virtual_point)
        updated_virtual_point["position_neu"] = clamped_vp_pos.copy()
        updated_virtual_point["position"] = clamped_vp_pos.copy()

        updated_vp_info = dict(vp_info)
        anchor_pos = self._as_neu_vector(updated_vp_info.get("anchor_pos"), default=target_pos)
        actual_world_offset = clamped_vp_pos - anchor_pos
        updated_vp_info["world_offset"] = actual_world_offset.copy()
        offset_frame = str(
            updated_vp_info.get(
                "offset_frame",
                updated_virtual_point.get("offset_frame", "world_neu"),
            )
        )
        updated_vp_info["offset"] = world_to_offset_frame(
            actual_world_offset,
            offset_frame,
            own_state=own_state,
            target_state=target_state,
        )

        clamp_active = True
        return (
            updated_virtual_point,
            updated_vp_info,
            clamp_active,
            preclamp_vp_forward_bias_m,
        )

    @staticmethod
    def _build_attack_zone_diagnostics(term_info: dict) -> dict:
        ego_attack = term_info.get("ego_attack") or {}
        target_attack = term_info.get("target_attack") or {}
        return {
            "ego_attack_aoa_deg": float(ego_attack.get("aoa_deg", np.nan)),
            "target_attack_aoa_deg": float(target_attack.get("aoa_deg", np.nan)),
            "ego_attack_range_m": float(ego_attack.get("range_m", np.nan)),
            "target_attack_range_m": float(target_attack.get("range_m", np.nan)),
            "ego_attack_score": float(term_info.get("ego_attack_score", 0.0)),
            "target_attack_score": float(term_info.get("target_attack_score", 0.0)),
            "ego_in_attack_zone": bool(term_info.get("ego_in_attack_zone", False)),
            "target_in_attack_zone": bool(term_info.get("target_in_attack_zone", False)),
        }

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
        if term_info.get("combat_outcome") == "win":
            terminal_reward = self.reward_calculator.terminal_success
        elif term_info.get("combat_outcome") == "loss":
            terminal_reward = self.reward_calculator.terminal_failure
        elif term_info.get("combat_outcome") == "draw":
            terminal_reward = 0.0
        elif term_info.get("is_success"):
            terminal_reward = self.reward_calculator.terminal_success
        elif term_info.get("is_crash"):
            terminal_reward = self.reward_calculator.terminal_crash
        elif term_info.get("is_timeout") or term_info.get("is_out_of_bounds"):
            terminal_reward = self.reward_calculator.terminal_failure
        info["terminal_reward"] = terminal_reward

        reward, reward_terms = self.reward_calculator.compute(info)
        return reward, reward_terms

    def _aircraft_failed(self, state: dict) -> bool:
        """Return True if an aircraft has crashed or left configured bounds."""
        altitude_m = float(state.get("altitude_m", 5000.0))
        min_alt = float(self.env_config.get("min_altitude_m", 500.0))
        max_alt = float(self.env_config.get("max_altitude_m", 15000.0))
        if altitude_m < min_alt or altitude_m > max_alt:
            return True
        pos = np.asarray(
            state.get("position_m", state.get("position_neu", [0.0, 0.0, altitude_m])),
            dtype=np.float64,
        )
        xy_limit = self.env_config.get("xy_limit_m")
        if xy_limit is not None and pos.shape[0] >= 2:
            limit = float(xy_limit)
            if abs(float(pos[0])) > limit or abs(float(pos[1])) > limit:
                return True
        return False

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

    def _apply_command_filter(self, command: dict, *, reset: bool = False) -> dict:
        """Apply independent first-order filters to each command channel."""
        if reset:
            self._command_filter.reset()
            return {key: float(value) for key, value in command.items()}
        return self._command_filter.filter(command)

    def _sync_prediction_buffer_for_current_state(
        self,
        own_state: dict,
        target_state: dict,
        rel_state: dict,
        prediction_info: Optional[dict] = None,
    ) -> bool:
        """Push the current state into the predictor history at most once.

        When prediction features are included in the policy observation, the
        observation builder updates the history before the action is selected.
        The step path then reuses that same history instead of pushing a
        duplicate frame.
        """
        if self.trajectory_predictor_adapter is None:
            return False
        if self._prediction_buffer_synced_for_action:
            return True
        try:
            self.trajectory_predictor_adapter.update(
                own_state, target_state, rel_state
            )
            self._prediction_buffer_synced_for_action = True
            return True
        except Exception as exc:
            self._prediction_buffer_synced_for_action = True
            if prediction_info is not None:
                prediction_info["prediction_fallback_reason"] = (
                    f"update_failed: {exc}"
                )
                prediction_info["prediction_fallback_phase"] = "runtime_failure"
            return False

    @staticmethod
    def _extract_position_neu(state: dict) -> np.ndarray:
        pos = state.get("position_neu")
        if pos is None:
            pos = state.get("position_m")
        if pos is None:
            pos = state.get("position")
        if pos is None:
            raise ValueError(
                "State missing position field (position_neu, position_m, or position)"
            )
        return np.asarray(pos, dtype=np.float64)

    def _prediction_observation_enabled(self) -> bool:
        tp_cfg = self.config.get("trajectory_prediction", {})
        int_cfg = tp_cfg.get("integration", {})
        return bool(
            int_cfg.get("add_prediction_to_observation", False)
            or int_cfg.get("add_uncertainty_to_observation", False)
        )

    def _build_prediction_observation_features(
        self, own_state: dict, target_state: dict, rel_state: dict
    ) -> Optional[dict]:
        """Build normalized prediction features for the high-level policy."""
        if not self._prediction_observation_enabled():
            return None

        keys = (
            "pred_rel_x",
            "pred_rel_y",
            "pred_rel_z",
            "pred_disp_x",
            "pred_disp_y",
            "pred_disp_z",
            "pred_vel_x",
            "pred_vel_y",
            "pred_vel_z",
            "pred_var_x",
            "pred_var_y",
            "pred_var_z",
            "pred_valid",
            "pred_fallback",
        )
        features = {key: 0.0 for key in keys}

        tp_cfg = self.config.get("trajectory_prediction", {})
        if not tp_cfg.get("enabled", False) or self.trajectory_predictor_adapter is None:
            features["pred_fallback"] = 1.0
            return features

        self._sync_prediction_buffer_for_current_state(
            own_state, target_state, rel_state
        )

        try:
            pred_pos, pred_var, pred_info = self.trajectory_predictor_adapter.predict(
                target_state
            )
        except Exception as exc:
            logger.warning(
                "Prediction observation feature build failed: %s", exc
            )
            features["pred_fallback"] = 1.0
            return features

        pred_pos = np.asarray(pred_pos, dtype=np.float64).reshape(-1)[:3]
        if pred_pos.shape[0] != 3 or not np.isfinite(pred_pos).all():
            features["pred_fallback"] = 1.0
            return features

        own_pos = self._extract_position_neu(own_state)
        target_pos = self._extract_position_neu(target_state)
        pred_rel = pred_pos - own_pos
        pred_disp = pred_pos - target_pos

        pred_cfg = tp_cfg.get("prediction", {})
        lookahead_time_s = max(float(pred_cfg.get("lookahead_time_s", 1.0)), 1e-6)
        pred_vel = pred_disp / lookahead_time_s

        norm_cfg = tp_cfg.get("normalization", {})
        pos_scale = max(float(norm_cfg.get("position_scale_m", 1000.0)), 1e-6)
        vel_scale = max(float(norm_cfg.get("velocity_scale_mps", 300.0)), 1e-6)

        features["pred_rel_x"], features["pred_rel_y"], features["pred_rel_z"] = (
            pred_rel / pos_scale
        )
        features["pred_disp_x"], features["pred_disp_y"], features["pred_disp_z"] = (
            pred_disp / pos_scale
        )
        features["pred_vel_x"], features["pred_vel_y"], features["pred_vel_z"] = (
            pred_vel / vel_scale
        )

        if pred_var is not None:
            pred_var_arr = np.asarray(pred_var, dtype=np.float64).reshape(-1)[:3]
            if pred_var_arr.shape[0] == 3 and np.isfinite(pred_var_arr).all():
                var_scale = pos_scale * pos_scale
                (
                    features["pred_var_x"],
                    features["pred_var_y"],
                    features["pred_var_z"],
                ) = pred_var_arr / var_scale

        fallback = bool(pred_info.get("fallback", False))
        features["pred_valid"] = 0.0 if fallback else 1.0
        features["pred_fallback"] = 1.0 if fallback else 0.0
        return features

    def _get_current_states(self, noisy: bool = False):
        """获取当前本机和目标状态（统一格式）。

        Args:
            noisy: If True, apply sensor-noise perturbations to the returned
                measurement copy.  The true environment state is never mutated.
        """
        if self._backend == "jsbsim":
            states = self.jsbsim_env.get_state()
            own = states[self.own_uid]
            target = states[self.target_uid]
        else:
            own, target = self._simple_env.get_state()
        own = self._copy_state(own)
        target = self._copy_state(target)
        if noisy and self._sensor_noise.get("enabled", False):
            own, target = self._apply_sensor_noise(own, target)
        return own, target

    @staticmethod
    def _copy_state(state: dict) -> dict:
        """Return a shallow copy with array fields duplicated."""
        copied = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                copied[k] = v.copy()
            else:
                copied[k] = v
        return copied

    def _apply_sensor_noise(self, own_state: dict, target_state: dict):
        """Apply Gaussian IMU/GPS noise to a measurement copy."""
        rng = self._robustness_rng
        dt = self._high_level_dt

        imu_std_deg_s = float(self._sensor_noise.get("imu_noise_std_deg_s", 0.0))
        imu_std_rad = np.deg2rad(imu_std_deg_s) * dt
        gps_std_m = float(self._sensor_noise.get("gps_position_noise_std_m", 0.0))

        def _add_position_noise(state):
            if "position_m" in state and gps_std_m > 0.0:
                state["position_m"] = state["position_m"] + rng.normal(0.0, gps_std_m, size=3)
            return state

        def _add_attitude_noise(state):
            if imu_std_rad <= 0.0:
                return state
            for key in ("heading_rad", "pitch_rad", "roll_rad", "yaw_rad"):
                if key in state:
                    state[key] = _wrap_angle(float(state[key]) + rng.normal(0.0, imu_std_rad))
            return state

        own_state = _add_position_noise(own_state)
        own_state = _add_attitude_noise(own_state)
        target_state = _add_position_noise(target_state)
        target_state = _add_attitude_noise(target_state)
        return own_state, target_state

    def _step_jsbsim(
        self,
        command,
        aggressiveness: Optional[float] = None,
        pid_gain_deltas=None,
        target_command: Optional[dict] = None,
    ):
        """在 JSBSim 后端执行控制，返回 actuator info。"""
        # Get current aircraft state for the low-level controller
        states = self.jsbsim_env.get_state()
        own_state_raw = states[self.own_uid]
        target_state_raw = states[self.target_uid]

        # Use low-level controller to map guidance commands to JSBSim properties.
        # New PID controllers accept aggressiveness/pid_gain_deltas; the legacy
        # LowLevelController does not, so fall back gracefully.
        sig = inspect.signature(self._low_level_controller.compute_actuator)
        supports_extras = any(p in sig.parameters for p in ("aggressiveness", "pid_gain_deltas"))
        if supports_extras:
            actuator_output = self._low_level_controller.compute_actuator(
                command,
                own_state_raw,
                aggressiveness=aggressiveness,
                pid_gain_deltas=pid_gain_deltas,
            )
        else:
            actuator_output = self._low_level_controller.compute_actuator(
                command, own_state_raw
            )

        # Extract JSBSim properties
        jsbsim_props = {
            k: v for k, v in actuator_output.items() if k.startswith("fcs/")
        }
        control_inputs = {self.own_uid: jsbsim_props}
        target_actuator_output = {}
        if target_command is not None and self._target_low_level_controller is not None:
            target_actuator_output = self._compute_actuator_output(
                self._target_low_level_controller,
                target_command,
                target_state_raw,
            )
            control_inputs[self.target_uid] = {
                k: v
                for k, v in target_actuator_output.items()
                if k.startswith("fcs/")
            }

        active_uids = [self.own_uid]
        if target_command is not None or self._task_uses_backend_target():
            active_uids.append(self.target_uid)

        for _ in range(self._sim_steps_per_decision):
            self.jsbsim_env.step(control_inputs, active_uids=active_uids)
            control_inputs = None

        return {
            "elevator_cmd": actuator_output.get("fcs/elevator-cmd-norm", np.nan),
            "aileron_cmd": actuator_output.get("fcs/aileron-cmd-norm", np.nan),
            "rudder_cmd": actuator_output.get("fcs/rudder-cmd-norm", np.nan),
            "throttle_actual": actuator_output.get("fcs/throttle-cmd-norm", np.nan),
            "saturation_flag": actuator_output.get("saturation_flag", False),
            "target_elevator_cmd": target_actuator_output.get("fcs/elevator-cmd-norm", np.nan),
            "target_aileron_cmd": target_actuator_output.get("fcs/aileron-cmd-norm", np.nan),
            "target_rudder_cmd": target_actuator_output.get("fcs/rudder-cmd-norm", np.nan),
            "target_throttle_actual": target_actuator_output.get("fcs/throttle-cmd-norm", np.nan),
            "target_saturation_flag": target_actuator_output.get("saturation_flag", False),
        }

    def _compute_actuator_output(
        self,
        controller,
        command,
        state,
        aggressiveness: Optional[float] = None,
        pid_gain_deltas=None,
    ):
        sig = inspect.signature(controller.compute_actuator)
        supports_extras = any(
            p in sig.parameters for p in ("aggressiveness", "pid_gain_deltas")
        )
        if supports_extras:
            return controller.compute_actuator(
                command,
                state,
                aggressiveness=aggressiveness,
                pid_gain_deltas=pid_gain_deltas,
            )
        return controller.compute_actuator(command, state)

    def _step_simple(self, command, target_command: Optional[dict] = None):
        """在简化后端执行控制。"""
        self._simple_env.step(own_command=command, target_command=target_command)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_guidance_state_for_observation(
        self, target_state: dict
    ) -> Optional[dict]:
        """Build guidance internal state for observation when enabled.

        Currently exposes the virtual-point tracking error (VP - target).
        On reset or before any step, the VP defaults to the target position so
        the error is zero rather than undefined.
        """
        if not self._include_guidance_state_in_observation:
            return None

        target_pos = target_state.get("position_m")
        if target_pos is None:
            target_pos = target_state.get("position_neu")
        if target_pos is None:
            return None
        target_pos = np.asarray(target_pos, dtype=np.float64)

        if self._last_virtual_point is not None:
            vp_pos = self._last_virtual_point.get("position_neu")
            if vp_pos is None:
                vp_pos = self._last_virtual_point.get("position")
            if vp_pos is not None:
                vp_pos = np.asarray(vp_pos, dtype=np.float64)
                return {"vp_tracking_error": vp_pos - target_pos}

        # Default: zero tracking error before first step
        return {"vp_tracking_error": np.zeros(3, dtype=np.float64)}

    def _build_provenance(self, observation_schema: dict) -> dict:
        """Build a structured provenance record for this environment instance.

        Records the final backend choice, whether a backend fallback occurred,
        the observation schema actually seen by the policy, and any config
        overrides that were declared before env construction.
        """
        provenance = {
            "backend": self._backend,
            "backend_fallback_occurred": self._backend_fallback_occurred,
            "observation_schema": observation_schema,
            "config_overrides": list(get_config_overrides(self.config)),
            "opponent_stage": self._opponent_stage,
            "opponent_config": self.opponent_config,
            "attack_zone_enabled": bool(self.combat_hp.enabled),
        }
        if self._backend_fallback_occurred:
            provenance["backend_fallback_reason"] = self._backend_fallback_reason
        return provenance

    def _get_observation(self) -> dict:
        """
        Build the current observation from environment states.

        Returns:
            dict: Observation dictionary with relative geometry, raw states,
            flattened observation vector, observation schema metadata, and
            structured provenance.
        """
        own_state, target_state = self._get_current_states(noisy=True)
        target_state = self._task_get_target_state(target_state, own_state)
        rel_state = compute_relative_geometry(own_state, target_state)

        obs_dict = {
            "own_state": own_state,
            "target_state": target_state,
            "relative_state": rel_state,
        }

        prediction_features = self._build_prediction_observation_features(
            own_state, target_state, rel_state
        )

        # Optionally expose gains / guidance internal state to the policy.
        # Default is off for backward compatibility with existing checkpoints.
        gains = self.current_gains if self._include_gains_in_observation else None
        guidance_state = (
            self._build_guidance_state_for_observation(target_state)
            if self._include_guidance_state_in_observation
            else None
        )
        saturation_features = (
            self._last_command_saturation
            if self._include_saturation_in_observation
            else None
        )
        opponent_stage = (
            self._opponent_stage
            if self._include_opponent_stage_in_observation
            else None
        )
        task_type = (
            self.task_name
            if self._include_task_type_in_observation
            else None
        )

        # 同时返回展平向量（供策略网络使用）
        obs_vec, feature_names = build_observation(
            own_state,
            target_state,
            guidance_state=guidance_state,
            gains=gains,
            prediction_features=prediction_features,
            saturation_features=saturation_features,
            opponent_stage=opponent_stage,
            task_type=task_type,
            return_feature_names=True,
        )
        obs_dict["observation_vector"] = obs_vec
        observation_schema = {
            "include_gains": self._include_gains_in_observation,
            "include_guidance_state": self._include_guidance_state_in_observation,
            "include_prediction_features": prediction_features is not None,
            "include_saturation": self._include_saturation_in_observation,
            "include_opponent_stage": self._include_opponent_stage_in_observation,
            "include_task_type": self._include_task_type_in_observation,
            "dim": int(obs_vec.shape[0]),
            "feature_names": feature_names,
        }
        obs_dict["observation_schema"] = observation_schema
        self._last_observation_schema = observation_schema
        obs_dict["provenance"] = self._build_provenance(observation_schema)
        return obs_dict

    def _apply_communication_delay(self, command: dict) -> dict:
        """Apply a communication delay to the filtered command.

        The delay is modeled as a convex combination of the current command and
        the previous one: alpha = delay_s / high_level_dt.  This is equivalent
        to a one-step zero-order hold interpolation for delays shorter than the
        decision interval.
        """
        if not self._comm_delay.get("enabled", False):
            return command
        delay_s = float(self._comm_delay.get("delay_s", 0.0))
        if delay_s <= 0.0:
            return command
        alpha = min(delay_s / self._high_level_dt, 1.0)
        if self._prev_delayed_command is None:
            self._prev_delayed_command = dict(command)
        prev = self._prev_delayed_command
        delayed = {
            k: (1.0 - alpha) * float(command[k]) + alpha * float(prev.get(k, command[k]))
            for k in command.keys()
        }
        self._prev_delayed_command = dict(command)
        return delayed

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
            except Exception as exc:
                # If pymap3d is unavailable, keep default origin position.
                logger.warning(
                    "Could not convert NEU position to geodetic coordinates: %s", exc
                )

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
