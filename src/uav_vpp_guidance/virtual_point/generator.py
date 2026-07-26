"""
Virtual Pursuit Point (VPP) generator.

Converts a normalized policy action into a virtual pursuit point
relative to the target aircraft.

升级：支持将策略输出的相对偏移量叠加到预测未来位置，
而不仅仅是目标当前位置。
"""

import numpy as np

from uav_vpp_guidance.utils.vpp_action_contract import (
    resolve_vpp_action_dimension,
    validate_vpp_action,
)
from .coordinate_transform import (
    VALID_OFFSET_FRAMES,
    offset_to_world,
    world_to_offset_frame,
)

VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES = {
    "same_side",
    "fixed_positive",
    "fixed_negative",
}
VALID_ACTION_SEMANTICS = {
    "cartesian_offset",
    "tactical_basis_v1",
}


def _stable_angle_diff(a, b):
    """Signed smallest angle difference in radians."""
    delta = a - b
    if not np.isfinite(delta):
        return float(delta)
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


class VirtualPointGenerator:
    """
    Convert policy action into a virtual pursuit point.

    Policy action should be normalized in [-1, 1].
    It is mapped to:
    - longitudinal offset
    - lateral offset
    - vertical offset
    - prediction time
    - speed bias

    支持三种锚点模式：
    - current_target: 目标当前位置（旧逻辑）
    - constant_velocity: 匀速外推预测位置
    - predicted_target: 通过 trajectory_predictor_adapter 获取预测位置
    """

    def __init__(self, config, task_name=None):
        """
        Args:
            config (dict): Virtual point configuration dictionary.
        """
        self.config = config
        self.task_name = str(task_name) if task_name not in (None, "") else None
        self.legacy_compatibility_mode = config.get("legacy_compatibility_mode")
        self.action_dim = resolve_vpp_action_dimension(config)
        self.d_long_range = config.get("d_long_range", [-1500.0, 1500.0])
        self.d_lat_range = config.get("d_lat_range", [-800.0, 800.0])
        self.d_vert_range = config.get("d_vert_range", [-500.0, 500.0])
        self.tau_pred_range = config.get("tau_pred_range", [0.0, 3.0])
        self.speed_bias_range = config.get("speed_bias_range", [-80.0, 80.0])
        self.smoothing_alpha = config.get("smoothing_alpha", 0.3)
        self.lead_distance_m = config.get("lead_distance_m", 500.0)
        self.default_offset_frame = str(config.get("offset_frame", "world_neu"))
        self.offset_frame_by_task = self._normalize_offset_frame_by_task(
            config.get("offset_frame_by_task", {})
        )
        self.offset_frame = self._resolve_offset_frame(self.task_name)
        self.configured_action_semantics = self._validate_action_semantics(
            config.get("action_semantics", "cartesian_offset"),
            "virtual_point.action_semantics",
        )
        self.action_semantics = self.configured_action_semantics
        self.default_predicted_target_blend = self._validate_predicted_target_blend(
            config.get("predicted_target_blend", 1.0),
            "virtual_point.predicted_target_blend",
        )
        self.predicted_target_blend_by_task = (
            self._normalize_predicted_target_blend_by_task(
                config.get("predicted_target_blend_by_task", {})
            )
        )
        self.predicted_target_blend = self._resolve_predicted_target_blend(
            self.task_name
        )
        self.default_predicted_target_forward_scale = (
            self._validate_predicted_target_blend(
                config.get("predicted_target_forward_scale", 1.0),
                "virtual_point.predicted_target_forward_scale",
            )
        )
        self.predicted_target_forward_scale_by_task = (
            self._normalize_predicted_target_forward_scale_by_task(
                config.get("predicted_target_forward_scale_by_task", {})
            )
        )
        self.predicted_target_forward_scale = (
            self._resolve_predicted_target_forward_scale(self.task_name)
        )
        self.default_longitudinal_scale = self._validate_nonnegative_scale(
            config.get("longitudinal_scale", 1.0),
            "virtual_point.longitudinal_scale",
        )
        self.longitudinal_scale_by_task = self._normalize_longitudinal_scale_by_task(
            config.get("longitudinal_scale_by_task", {})
        )
        self.longitudinal_scale = self._resolve_longitudinal_scale(self.task_name)
        self.default_lateral_scale = self._validate_nonnegative_scale(
            config.get("lateral_scale", 1.0),
            "virtual_point.lateral_scale",
        )
        self.lateral_scale_by_task = self._normalize_lateral_scale_by_task(
            config.get("lateral_scale_by_task", {})
        )
        self.lateral_scale = self._resolve_lateral_scale(self.task_name)
        self.default_offensive_anchor_longitudinal_m = self._validate_nonnegative_scale(
            config.get("offensive_anchor_longitudinal_m", 800.0),
            "virtual_point.offensive_anchor_longitudinal_m",
        )
        self.offensive_anchor_longitudinal_m_by_task = (
            self._normalize_offensive_anchor_longitudinal_m_by_task(
                config.get("offensive_anchor_longitudinal_m_by_task", {})
            )
        )
        self.offensive_anchor_longitudinal_m = (
            self._resolve_offensive_anchor_longitudinal_m(self.task_name)
        )
        self.default_offensive_anchor_frame = self._validate_offset_frame(
            config.get("offensive_anchor_frame", "target_velocity"),
            "virtual_point.offensive_anchor_frame",
        )
        self.offensive_anchor_frame_by_task = (
            self._normalize_offset_frame_by_task(
                config.get("offensive_anchor_frame_by_task", {})
            )
        )
        self.offensive_anchor_frame = self._resolve_offensive_anchor_frame(
            self.task_name
        )
        self.default_offensive_anchor_encounter_stable_max_heading_delta_deg = (
            self._validate_nonnegative_scale(
                config.get(
                    "offensive_anchor_encounter_stable_max_heading_delta_deg",
                    45.0,
                ),
                (
                    "virtual_point."
                    "offensive_anchor_encounter_stable_max_heading_delta_deg"
                ),
            )
        )
        self.offensive_anchor_encounter_stable_max_heading_delta_deg_by_task = (
            self._normalize_offensive_anchor_encounter_stable_max_heading_delta_deg_by_task(
                config.get(
                    "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task",
                    {},
                )
            )
        )
        self.offensive_anchor_encounter_stable_max_heading_delta_deg = (
            self._resolve_offensive_anchor_encounter_stable_max_heading_delta_deg(
                self.task_name
            )
        )
        self.default_offensive_anchor_lateral_frame = self._validate_offset_frame(
            config.get(
                "offensive_anchor_lateral_frame",
                self.default_offensive_anchor_frame,
            ),
            "virtual_point.offensive_anchor_lateral_frame",
        )
        self.offensive_anchor_lateral_frame_by_task = (
            self._normalize_offset_frame_by_task(
                config.get("offensive_anchor_lateral_frame_by_task", {})
            )
        )
        self.offensive_anchor_lateral_frame = (
            self._resolve_offensive_anchor_lateral_frame(self.task_name)
        )
        self.default_offensive_anchor_lateral_m = self._validate_nonnegative_scale(
            config.get("offensive_anchor_lateral_m", 0.0),
            "virtual_point.offensive_anchor_lateral_m",
        )
        self.offensive_anchor_lateral_m_by_task = (
            self._normalize_offensive_anchor_lateral_m_by_task(
                config.get("offensive_anchor_lateral_m_by_task", {})
            )
        )
        self.offensive_anchor_lateral_m = (
            self._resolve_offensive_anchor_lateral_m(self.task_name)
        )
        self.default_offensive_anchor_lateral_sign_mode = (
            self._validate_offensive_anchor_lateral_sign_mode(
                config.get("offensive_anchor_lateral_sign_mode", "same_side"),
                "virtual_point.offensive_anchor_lateral_sign_mode",
            )
        )
        self.offensive_anchor_lateral_sign_mode_by_task = (
            self._normalize_offensive_anchor_lateral_sign_mode_by_task(
                config.get("offensive_anchor_lateral_sign_mode_by_task", {})
            )
        )
        self.offensive_anchor_lateral_sign_mode = (
            self._resolve_offensive_anchor_lateral_sign_mode(self.task_name)
        )
        self.default_offensive_anchor_vertical_m = self._validate_scale(
            config.get("offensive_anchor_vertical_m", 0.0),
            "virtual_point.offensive_anchor_vertical_m",
        )
        self.offensive_anchor_vertical_m_by_task = (
            self._normalize_offensive_anchor_vertical_m_by_task(
                config.get("offensive_anchor_vertical_m_by_task", {})
            )
        )
        self.offensive_anchor_vertical_m = (
            self._resolve_offensive_anchor_vertical_m(self.task_name)
        )
        self.default_offensive_anchor_blend = self._validate_predicted_target_blend(
            config.get("offensive_anchor_blend", 0.0),
            "virtual_point.offensive_anchor_blend",
        )
        self.offensive_anchor_blend_by_task = (
            self._normalize_offensive_anchor_blend_by_task(
                config.get("offensive_anchor_blend_by_task", {})
            )
        )
        self.offensive_anchor_blend = self._resolve_offensive_anchor_blend(
            self.task_name
        )
        self.default_tactical_basis_longitudinal_frame = self._validate_offset_frame(
            config.get("tactical_basis_longitudinal_frame", "target_velocity"),
            "virtual_point.tactical_basis_longitudinal_frame",
        )
        self.default_tactical_basis_lateral_frame = self._validate_offset_frame(
            config.get("tactical_basis_lateral_frame", "encounter_stable"),
            "virtual_point.tactical_basis_lateral_frame",
        )
        self.default_tactical_basis_vertical_frame = self._validate_offset_frame(
            config.get("tactical_basis_vertical_frame", "world_neu"),
            "virtual_point.tactical_basis_vertical_frame",
        )
        self.default_tactical_basis_lead_lag_extent_m = (
            self._validate_nonnegative_scale(
                config.get("tactical_basis_lead_lag_extent_m", 1200.0),
                "virtual_point.tactical_basis_lead_lag_extent_m",
            )
        )
        self.tactical_basis_lead_lag_extent_m_by_task = (
            self._normalize_tactical_basis_extent_by_task(
                config.get("tactical_basis_lead_lag_extent_m_by_task", {}),
                "virtual_point.tactical_basis_lead_lag_extent_m_by_task",
            )
        )
        self.tactical_basis_lead_lag_extent_m = (
            self._resolve_tactical_basis_lead_lag_extent_m(self.task_name)
        )
        self.default_tactical_basis_inside_outside_extent_m = (
            self._validate_nonnegative_scale(
                config.get("tactical_basis_inside_outside_extent_m", 0.0),
                "virtual_point.tactical_basis_inside_outside_extent_m",
            )
        )
        self.tactical_basis_inside_outside_extent_m_by_task = (
            self._normalize_tactical_basis_extent_by_task(
                config.get("tactical_basis_inside_outside_extent_m_by_task", {}),
                "virtual_point.tactical_basis_inside_outside_extent_m_by_task",
            )
        )
        self.tactical_basis_inside_outside_extent_m = (
            self._resolve_tactical_basis_inside_outside_extent_m(self.task_name)
        )
        self.default_tactical_basis_climb_descent_extent_m = (
            self._validate_nonnegative_scale(
                config.get("tactical_basis_climb_descent_extent_m", 600.0),
                "virtual_point.tactical_basis_climb_descent_extent_m",
            )
        )
        self.tactical_basis_climb_descent_extent_m_by_task = (
            self._normalize_tactical_basis_extent_by_task(
                config.get("tactical_basis_climb_descent_extent_m_by_task", {}),
                "virtual_point.tactical_basis_climb_descent_extent_m_by_task",
            )
        )
        self.tactical_basis_climb_descent_extent_m = (
            self._resolve_tactical_basis_climb_descent_extent_m(self.task_name)
        )
        self.default_tactical_basis_lateral_sign_mode = (
            self._validate_offensive_anchor_lateral_sign_mode(
                config.get("tactical_basis_lateral_sign_mode", "same_side"),
                "virtual_point.tactical_basis_lateral_sign_mode",
            )
        )
        self.tactical_basis_lateral_sign_mode = (
            self._resolve_tactical_basis_lateral_sign_mode(self.task_name)
        )
        self.default_tactical_basis_encounter_stable_max_heading_delta_deg = (
            self._validate_nonnegative_scale(
                config.get(
                    "tactical_basis_encounter_stable_max_heading_delta_deg",
                    45.0,
                ),
                (
                    "virtual_point."
                    "tactical_basis_encounter_stable_max_heading_delta_deg"
                ),
            )
        )
        self.tactical_basis_encounter_stable_max_heading_delta_deg = (
            self._resolve_tactical_basis_encounter_stable_max_heading_delta_deg(
                self.task_name
            )
        )
        self.tactical_basis_longitudinal_frame = (
            self._resolve_tactical_basis_longitudinal_frame(self.task_name)
        )
        self.tactical_basis_lateral_frame = self._resolve_tactical_basis_lateral_frame(
            self.task_name
        )
        self.tactical_basis_vertical_frame = self._resolve_tactical_basis_vertical_frame(
            self.task_name
        )
        if self.offset_frame not in VALID_OFFSET_FRAMES:
            raise ValueError(
                f"Unknown virtual_point.offset_frame={self.offset_frame!r}; "
                f"expected one of {sorted(VALID_OFFSET_FRAMES)}"
            )
        # Dynamics-aware constraint: clip virtual points to feasible heading sector
        self.dynamics_aware = config.get("dynamics_aware", False)
        # F-16 max feasible heading change per step (high_level_dt=0.2s, max rate ≈ 0.3 rad/s)
        self.max_heading_rate = config.get("max_heading_rate", 0.3)
        self.lookahead_steps = config.get("lookahead_steps", 5)
        self._prev_action = None

    @staticmethod
    def _validate_offset_frame(frame_value, field_name):
        frame = str(frame_value or "world_neu")
        if frame not in VALID_OFFSET_FRAMES:
            raise ValueError(
                f"Unknown {field_name}={frame!r}; "
                f"expected one of {sorted(VALID_OFFSET_FRAMES)}"
            )
        return frame

    @staticmethod
    def _validate_action_semantics(semantics_value, field_name):
        semantics = str(semantics_value or "cartesian_offset")
        if semantics not in VALID_ACTION_SEMANTICS:
            raise ValueError(
                f"Unknown {field_name}={semantics!r}; "
                f"expected one of {sorted(VALID_ACTION_SEMANTICS)}"
            )
        return semantics

    def _normalize_offset_frame_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offset_frame_by_task must be a mapping of "
                "task_name -> offset_frame"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_offset_frame(
                value,
                f"virtual_point.offset_frame_by_task[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_predicted_target_blend(blend_value, field_name):
        blend = float(blend_value)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                f"Invalid {field_name}={blend_value!r}; expected a finite value in [0, 1]"
            )
        return blend

    def _normalize_predicted_target_blend_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.predicted_target_blend_by_task must be a mapping "
                "of task_name -> blend"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_predicted_target_blend(
                value,
                f"virtual_point.predicted_target_blend_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_predicted_target_forward_scale_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.predicted_target_forward_scale_by_task must be a mapping "
                "of task_name -> scale"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_predicted_target_blend(
                value,
                f"virtual_point.predicted_target_forward_scale_by_task[{task_key!r}]",
            )
        return normalized

    @staticmethod
    def _validate_nonnegative_scale(scale_value, field_name):
        scale = float(scale_value)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"Invalid {field_name}={scale_value!r}; expected a finite value >= 0"
            )
        return scale

    @staticmethod
    def _validate_scale(scale_value, field_name):
        scale = float(scale_value)
        if not np.isfinite(scale):
            raise ValueError(
                f"Invalid {field_name}={scale_value!r}; expected a finite value"
            )
        return scale

    @staticmethod
    def _validate_offensive_anchor_lateral_sign_mode(mode_value, field_name):
        mode = str(mode_value or "same_side")
        if mode not in VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES:
            raise ValueError(
                f"Unknown {field_name}={mode!r}; expected one of "
                f"{sorted(VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES)}"
            )
        return mode

    def _normalize_lateral_scale_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.lateral_scale_by_task must be a mapping of "
                "task_name -> scale"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_scale(
                value,
                f"virtual_point.lateral_scale_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_longitudinal_scale_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.longitudinal_scale_by_task must be a mapping of "
                "task_name -> scale"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_scale(
                value,
                f"virtual_point.longitudinal_scale_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_offensive_anchor_longitudinal_m_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offensive_anchor_longitudinal_m_by_task must be "
                "a mapping of task_name -> distance_m"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_scale(
                value,
                f"virtual_point.offensive_anchor_longitudinal_m_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_offensive_anchor_lateral_m_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offensive_anchor_lateral_m_by_task must be a "
                "mapping of task_name -> distance_m"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_scale(
                value,
                f"virtual_point.offensive_anchor_lateral_m_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_offensive_anchor_vertical_m_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offensive_anchor_vertical_m_by_task must be a "
                "mapping of task_name -> distance_m"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_scale(
                value,
                f"virtual_point.offensive_anchor_vertical_m_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_offensive_anchor_encounter_stable_max_heading_delta_deg_by_task(
        self,
        mapping,
    ):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offensive_anchor_encounter_stable_max_heading_delta_deg_by_task "
                "must be a mapping of task_name -> degrees"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_scale(
                value,
                (
                    "virtual_point."
                    "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
                    f"[{task_key!r}]"
                ),
            )
        return normalized

    def _normalize_offensive_anchor_lateral_sign_mode_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offensive_anchor_lateral_sign_mode_by_task must be a "
                "mapping of task_name -> sign_mode"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_offensive_anchor_lateral_sign_mode(
                value,
                (
                    "virtual_point.offensive_anchor_lateral_sign_mode_by_task"
                    f"[{task_key!r}]"
                ),
            )
        return normalized

    def _normalize_offensive_anchor_blend_by_task(self, mapping):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(
                "virtual_point.offensive_anchor_blend_by_task must be a mapping "
                "of task_name -> blend"
            )
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_predicted_target_blend(
                value,
                f"virtual_point.offensive_anchor_blend_by_task[{task_key!r}]",
            )
        return normalized

    def _normalize_tactical_basis_extent_by_task(self, mapping, field_name):
        if mapping in (None, {}):
            return {}
        if not isinstance(mapping, dict):
            raise ValueError(f"{field_name} must be a mapping of task_name -> extent_m")
        normalized = {}
        for key, value in mapping.items():
            task_key = str(key)
            normalized[task_key] = self._validate_nonnegative_scale(
                value,
                f"{field_name}[{task_key!r}]",
            )
        return normalized

    def _resolve_offset_frame(self, task_name):
        default_frame = self._validate_offset_frame(
            self.default_offset_frame,
            "virtual_point.offset_frame",
        )
        if task_name is None:
            return default_frame
        return self.offset_frame_by_task.get(str(task_name), default_frame)

    def _resolve_predicted_target_blend(self, task_name):
        if task_name is None:
            return self.default_predicted_target_blend
        return self.predicted_target_blend_by_task.get(
            str(task_name),
            self.default_predicted_target_blend,
        )

    def _resolve_predicted_target_forward_scale(self, task_name):
        if task_name is None:
            return self.default_predicted_target_forward_scale
        return self.predicted_target_forward_scale_by_task.get(
            str(task_name),
            self.default_predicted_target_forward_scale,
        )

    def _resolve_lateral_scale(self, task_name):
        if task_name is None:
            return self.default_lateral_scale
        return self.lateral_scale_by_task.get(
            str(task_name),
            self.default_lateral_scale,
        )

    def _resolve_longitudinal_scale(self, task_name):
        if task_name is None:
            return self.default_longitudinal_scale
        return self.longitudinal_scale_by_task.get(
            str(task_name),
            self.default_longitudinal_scale,
        )

    def _resolve_offensive_anchor_longitudinal_m(self, task_name):
        if task_name is None:
            return self.default_offensive_anchor_longitudinal_m
        return self.offensive_anchor_longitudinal_m_by_task.get(
            str(task_name),
            self.default_offensive_anchor_longitudinal_m,
        )

    def _resolve_offensive_anchor_frame(self, task_name):
        default_frame = self._validate_offset_frame(
            self.default_offensive_anchor_frame,
            "virtual_point.offensive_anchor_frame",
        )
        if task_name is None:
            return default_frame
        return self.offensive_anchor_frame_by_task.get(str(task_name), default_frame)

    def _resolve_offensive_anchor_lateral_frame(self, task_name):
        if task_name is None:
            return self.default_offensive_anchor_lateral_frame
        return self.offensive_anchor_lateral_frame_by_task.get(
            str(task_name),
            self._resolve_offensive_anchor_frame(task_name),
        )

    def _resolve_offensive_anchor_encounter_stable_max_heading_delta_deg(
        self,
        task_name,
    ):
        if task_name is None:
            return self.default_offensive_anchor_encounter_stable_max_heading_delta_deg
        return self.offensive_anchor_encounter_stable_max_heading_delta_deg_by_task.get(
            str(task_name),
            self.default_offensive_anchor_encounter_stable_max_heading_delta_deg,
        )

    def _resolve_offensive_anchor_lateral_m(self, task_name):
        if task_name is None:
            return self.default_offensive_anchor_lateral_m
        return self.offensive_anchor_lateral_m_by_task.get(
            str(task_name),
            self.default_offensive_anchor_lateral_m,
        )

    def _resolve_offensive_anchor_vertical_m(self, task_name):
        if task_name is None:
            return self.default_offensive_anchor_vertical_m
        return self.offensive_anchor_vertical_m_by_task.get(
            str(task_name),
            self.default_offensive_anchor_vertical_m,
        )

    def _resolve_offensive_anchor_lateral_sign_mode(self, task_name):
        if task_name is None:
            return self.default_offensive_anchor_lateral_sign_mode
        return self.offensive_anchor_lateral_sign_mode_by_task.get(
            str(task_name),
            self.default_offensive_anchor_lateral_sign_mode,
        )

    def _resolve_offensive_anchor_blend(self, task_name):
        if task_name is None:
            return self.default_offensive_anchor_blend
        return self.offensive_anchor_blend_by_task.get(
            str(task_name),
            self.default_offensive_anchor_blend,
        )

    def _resolve_tactical_basis_lead_lag_extent_m(self, task_name):
        if task_name is None:
            return self.default_tactical_basis_lead_lag_extent_m
        return self.tactical_basis_lead_lag_extent_m_by_task.get(
            str(task_name),
            self.default_tactical_basis_lead_lag_extent_m,
        )

    def _resolve_tactical_basis_inside_outside_extent_m(self, task_name):
        if task_name is None:
            return self.default_tactical_basis_inside_outside_extent_m
        return self.tactical_basis_inside_outside_extent_m_by_task.get(
            str(task_name),
            self.default_tactical_basis_inside_outside_extent_m,
        )

    def _resolve_tactical_basis_climb_descent_extent_m(self, task_name):
        if task_name is None:
            return self.default_tactical_basis_climb_descent_extent_m
        return self.tactical_basis_climb_descent_extent_m_by_task.get(
            str(task_name),
            self.default_tactical_basis_climb_descent_extent_m,
        )

    def _resolve_tactical_basis_longitudinal_frame(self, task_name):
        del task_name
        return self.default_tactical_basis_longitudinal_frame

    def _resolve_tactical_basis_lateral_frame(self, task_name):
        del task_name
        return self.default_tactical_basis_lateral_frame

    def _resolve_tactical_basis_vertical_frame(self, task_name):
        del task_name
        return self.default_tactical_basis_vertical_frame

    def _resolve_tactical_basis_lateral_sign_mode(self, task_name):
        del task_name
        return self.default_tactical_basis_lateral_sign_mode

    def _resolve_tactical_basis_encounter_stable_max_heading_delta_deg(
        self,
        task_name,
    ):
        del task_name
        return self.default_tactical_basis_encounter_stable_max_heading_delta_deg

    def action_to_virtual_point(
        self,
        action,
        own_state,
        target_state,
        anchor_mode: str = "current_target",
        lookahead_time_s: float = 1.0,
        trajectory_predictor_adapter=None,
        predicted_target_position=None,
        predicted_target_blend_override=None,
        predicted_target_forward_scale_override=None,
        offensive_anchor_blend_override=None,
        offensive_anchor_longitudinal_blend_override=None,
        offensive_anchor_lateral_blend_override=None,
        offensive_anchor_lateral_world_offset_override=None,
        longitudinal_scale_override=None,
        lateral_scale_override=None,
        return_info: bool = False,
    ):
        """
        Convert normalized policy action to a virtual pursuit point.

        Args:
            action (np.ndarray): Normalized action vector in [-1, 1].
                前 3 维映射为 [Δx, Δy, Δz]（纵向、横向、垂直偏移）。
            own_state (dict): Own aircraft state.
            target_state (dict): Target aircraft state.
            anchor_mode (str): 锚点模式，"current_target" / "constant_velocity" / "predicted_target" / "oracle_future_position" / "rule_based_pursuit"。
            lookahead_time_s (float): 前瞻时间（用于 constant_velocity 模式）。
            trajectory_predictor_adapter (TrajectoryPredictorAdapter, optional):
                轨迹预测适配器（用于 predicted_target 模式，向后兼容）。
            predicted_target_position (np.ndarray, optional):
                外部传入的预测目标位置（优先于 adapter）。
            return_info (bool): 若为 True，额外返回 info 字典。

        Returns:
            dict or tuple: 默认返回 virtual_point dict；
                若 return_info=True，返回 (virtual_point, info)。
        """
        action = validate_vpp_action(
            action,
            expected_dim=self.action_dim,
            legacy_compatibility_mode=self.legacy_compatibility_mode,
        )
        current_target_pos = self._get_target_position(target_state)
        nan_vec = np.full(3, np.nan, dtype=np.float64)
        tactical_basis_info = {
            "action_semantics": self.action_semantics,
            "configured_action_semantics": self.configured_action_semantics,
            "tactical_basis_enabled": False,
            "tactical_basis_action_ll": np.nan,
            "tactical_basis_action_io": np.nan,
            "tactical_basis_action_cd": np.nan,
            "tactical_basis_lead_lag_extent_m": (
                self.tactical_basis_lead_lag_extent_m
            ),
            "tactical_basis_inside_outside_extent_m": (
                self.tactical_basis_inside_outside_extent_m
            ),
            "tactical_basis_climb_descent_extent_m": (
                self.tactical_basis_climb_descent_extent_m
            ),
            "tactical_basis_longitudinal_frame": (
                self.tactical_basis_longitudinal_frame
            ),
            "tactical_basis_lateral_frame": self.tactical_basis_lateral_frame,
            "tactical_basis_vertical_frame": self.tactical_basis_vertical_frame,
            "tactical_basis_lateral_sign_mode": (
                self.tactical_basis_lateral_sign_mode
            ),
            "tactical_basis_lateral_sign": np.nan,
            "tactical_basis_ll_world": nan_vec.copy(),
            "tactical_basis_io_world": nan_vec.copy(),
            "tactical_basis_cd_world": nan_vec.copy(),
            "tactical_basis_world_offset": nan_vec.copy(),
        }

        # 将 action 前 3 维映射为实际偏移量
        if self.action_semantics == "tactical_basis_v1":
            world_offset, tactical_basis_active_info = (
                self._action_to_tactical_basis_world_offset(
                    action,
                    own_state,
                    target_state,
                )
            )
            tactical_basis_info.update(tactical_basis_active_info)
            offset = world_to_offset_frame(
                world_offset,
                self.offset_frame,
                own_state=own_state,
                target_state=target_state,
            )
        else:
            offset = np.array(
                [
                    self._rescale(action[0], self.d_long_range),
                    self._rescale(action[1], self.d_lat_range),
                    self._rescale(action[2], self.d_vert_range),
                ],
                dtype=np.float64,
            )
        longitudinal_scale = self.longitudinal_scale
        if longitudinal_scale_override is not None:
            longitudinal_scale = self._validate_nonnegative_scale(
                longitudinal_scale_override,
                "longitudinal_scale_override",
            )
        lateral_scale = self.lateral_scale
        if lateral_scale_override is not None:
            lateral_scale = self._validate_nonnegative_scale(
                lateral_scale_override,
                "lateral_scale_override",
            )
        offset[0] *= longitudinal_scale
        offset[1] *= lateral_scale
        offensive_anchor_frame_kwargs = self._offensive_anchor_frame_transform_kwargs(
            self.offensive_anchor_frame
        )
        offensive_anchor_lateral_frame_kwargs = (
            self._offensive_anchor_frame_transform_kwargs(
                self.offensive_anchor_lateral_frame
            )
        )

        # 确定锚点位置
        if anchor_mode == "current_target":
            anchor_pos = current_target_pos
            pred_var = None
            prediction_info = {"anchor_mode": "current_target"}

        elif anchor_mode == "constant_velocity":
            anchor_pos = self._constant_velocity_prediction(
                target_state, lookahead_time_s
            )
            pred_var = None
            prediction_info = {
                "anchor_mode": "constant_velocity",
                "lookahead_time_s": lookahead_time_s,
            }

        elif anchor_mode == "oracle_future_position":
            # Oracle: perfect future-position prediction using true velocity
            anchor_pos = self._constant_velocity_prediction(
                target_state, lookahead_time_s
            )
            pred_var = None
            prediction_info = {
                "anchor_mode": "oracle_future_position",
                "lookahead_time_s": lookahead_time_s,
            }

        elif anchor_mode == "rule_based_pursuit":
            # Rule-based pure pursuit with fixed lead distance
            lead_distance_m = getattr(self, "lead_distance_m", 500.0)
            anchor_pos = self._rule_based_pursuit_anchor(
                own_state, target_state, lead_distance_m
            )
            pred_var = None
            prediction_info = {
                "anchor_mode": "rule_based_pursuit",
                "lead_distance_m": lead_distance_m,
            }

        elif anchor_mode == "predicted_target":
            if predicted_target_position is not None:
                anchor_pos = np.asarray(predicted_target_position, dtype=np.float64)
                pred_var = None
                prediction_info = {
                    "anchor_mode": "predicted_target",
                    "source": "external",
                }
            elif trajectory_predictor_adapter is not None:
                anchor_pos, pred_var, prediction_info = (
                    trajectory_predictor_adapter.predict(target_state)
                )
                prediction_info["anchor_mode"] = "predicted_target"
            else:
                raise ValueError(
                    "anchor_mode='predicted_target' requires either predicted_target_position or trajectory_predictor_adapter"
                )

            predicted_target_blend = self.predicted_target_blend
            if predicted_target_blend_override is not None:
                predicted_target_blend = self._validate_predicted_target_blend(
                    predicted_target_blend_override,
                    "predicted_target_blend_override",
                )
            unblended_anchor_pos = anchor_pos.copy()
            anchor_pos = current_target_pos + predicted_target_blend * (
                anchor_pos - current_target_pos
            )
            prediction_info["predicted_target_blend"] = predicted_target_blend
            prediction_info["unblended_anchor_pos"] = unblended_anchor_pos
            predicted_target_forward_scale = self.predicted_target_forward_scale
            if predicted_target_forward_scale_override is not None:
                predicted_target_forward_scale = (
                    self._validate_predicted_target_blend(
                        predicted_target_forward_scale_override,
                        "predicted_target_forward_scale_override",
                    )
                )
            if predicted_target_forward_scale != 1.0:
                prediction_info[
                    "pre_predicted_target_forward_scale_anchor_pos"
                ] = anchor_pos.copy()
                anchor_local = world_to_offset_frame(
                    anchor_pos - current_target_pos,
                    "target_velocity",
                    target_state=target_state,
                )
                anchor_local[0] *= predicted_target_forward_scale
                anchor_pos = current_target_pos + offset_to_world(
                    anchor_local,
                    "target_velocity",
                    target_state=target_state,
                )
            prediction_info["predicted_target_forward_scale"] = (
                predicted_target_forward_scale
            )

        elif anchor_mode == "offensive_position":
            (
                anchor_pos,
                offensive_anchor_offset,
                offensive_anchor_world_offset,
                offensive_anchor_lateral_local_offset,
                offensive_anchor_lateral_world_offset,
                offensive_anchor_lateral_sign,
            ) = self._offensive_anchor_components(
                target_state,
                own_state=own_state,
                lateral_world_offset_override=(
                    offensive_anchor_lateral_world_offset_override
                ),
            )
            pred_var = None
            prediction_info = {
                "anchor_mode": "offensive_position",
                "offensive_anchor_blend": 1.0,
                "offensive_anchor_longitudinal_m": self.offensive_anchor_longitudinal_m,
                "offensive_anchor_frame": self.offensive_anchor_frame,
                "offensive_anchor_lateral_frame": self.offensive_anchor_lateral_frame,
                "offensive_anchor_lateral_m": self.offensive_anchor_lateral_m,
                "offensive_anchor_lateral_sign_mode": (
                    self.offensive_anchor_lateral_sign_mode
                ),
                "offensive_anchor_lateral_sign": offensive_anchor_lateral_sign,
                "offensive_anchor_vertical_m": self.offensive_anchor_vertical_m,
                "offensive_anchor_encounter_stable_max_heading_delta_deg": (
                    self.offensive_anchor_encounter_stable_max_heading_delta_deg
                ),
                "offensive_anchor_pos": anchor_pos.copy(),
                "offensive_anchor_offset": offensive_anchor_offset,
                "offensive_anchor_world_offset": offensive_anchor_world_offset,
                "offensive_anchor_lateral_world_offset": (
                    offensive_anchor_lateral_world_offset
                ),
                "offensive_anchor_lateral_local_offset": (
                    offensive_anchor_lateral_local_offset
                ),
            }

        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

        offensive_anchor_blend = self.offensive_anchor_blend
        if offensive_anchor_blend_override is not None:
            offensive_anchor_blend = self._validate_predicted_target_blend(
                offensive_anchor_blend_override,
                "offensive_anchor_blend_override",
            )
        offensive_anchor_longitudinal_blend = offensive_anchor_blend
        if offensive_anchor_longitudinal_blend_override is not None:
            offensive_anchor_longitudinal_blend = self._validate_predicted_target_blend(
                offensive_anchor_longitudinal_blend_override,
                "offensive_anchor_longitudinal_blend_override",
            )
        offensive_anchor_lateral_blend = offensive_anchor_blend
        if offensive_anchor_lateral_blend_override is not None:
            offensive_anchor_lateral_blend = self._validate_predicted_target_blend(
                offensive_anchor_lateral_blend_override,
                "offensive_anchor_lateral_blend_override",
            )
        if anchor_mode != "offensive_position":
            prediction_info["offensive_anchor_blend"] = offensive_anchor_blend
            prediction_info["offensive_anchor_longitudinal_blend"] = (
                offensive_anchor_longitudinal_blend
            )
            prediction_info["offensive_anchor_lateral_blend"] = (
                offensive_anchor_lateral_blend
            )
            if (
                offensive_anchor_blend > 0.0
                or offensive_anchor_longitudinal_blend > 0.0
                or offensive_anchor_lateral_blend > 0.0
            ):
                (
                    offensive_anchor_pos,
                    offensive_anchor_offset,
                    offensive_anchor_world_offset,
                    offensive_anchor_lateral_local_offset,
                    offensive_anchor_lateral_world_offset,
                    offensive_anchor_lateral_sign,
                ) = self._offensive_anchor_components(
                    target_state,
                    own_state=own_state,
                    lateral_world_offset_override=(
                        offensive_anchor_lateral_world_offset_override
                    ),
                )
                prediction_info["pre_offensive_blend_anchor_pos"] = anchor_pos.copy()
                anchor_local = world_to_offset_frame(
                    anchor_pos - current_target_pos,
                    self.offensive_anchor_frame,
                    own_state=own_state,
                    target_state=target_state,
                    **offensive_anchor_frame_kwargs,
                )
                anchor_lateral_local = world_to_offset_frame(
                    anchor_pos - current_target_pos,
                    self.offensive_anchor_lateral_frame,
                    own_state=own_state,
                    target_state=target_state,
                    **offensive_anchor_lateral_frame_kwargs,
                )
                prediction_info["pre_offensive_blend_anchor_local"] = anchor_local.copy()
                prediction_info["pre_offensive_blend_anchor_lateral_local"] = (
                    anchor_lateral_local.copy()
                )
                blended_anchor_local = anchor_local.copy()
                blended_anchor_lateral_local = anchor_lateral_local.copy()
                blended_anchor_local[0] = (
                    (1.0 - offensive_anchor_longitudinal_blend) * anchor_local[0]
                    + offensive_anchor_longitudinal_blend * offensive_anchor_offset[0]
                )
                blended_anchor_lateral_local[1] = (
                    (1.0 - offensive_anchor_lateral_blend) * anchor_lateral_local[1]
                    + offensive_anchor_lateral_blend
                    * offensive_anchor_lateral_local_offset[1]
                )
                blended_anchor_local[2] = (
                    (1.0 - offensive_anchor_blend) * anchor_local[2]
                    + offensive_anchor_blend * offensive_anchor_offset[2]
                )
                blended_longitudinal_vertical_world = offset_to_world(
                    np.array(
                        [
                            blended_anchor_local[0],
                            0.0,
                            blended_anchor_local[2],
                        ],
                        dtype=np.float64,
                    ),
                    self.offensive_anchor_frame,
                    own_state=own_state,
                    target_state=target_state,
                    **offensive_anchor_frame_kwargs,
                )
                blended_lateral_world = offset_to_world(
                    np.array(
                        [
                            0.0,
                            blended_anchor_lateral_local[1],
                            0.0,
                        ],
                        dtype=np.float64,
                    ),
                    self.offensive_anchor_lateral_frame,
                    own_state=own_state,
                    target_state=target_state,
                    **offensive_anchor_lateral_frame_kwargs,
                )
                anchor_pos = (
                    current_target_pos
                    + blended_longitudinal_vertical_world
                    + blended_lateral_world
                )
                prediction_info["post_offensive_blend_anchor_local"] = (
                    world_to_offset_frame(
                        anchor_pos - current_target_pos,
                        self.offensive_anchor_frame,
                        own_state=own_state,
                        target_state=target_state,
                        **offensive_anchor_frame_kwargs,
                    )
                )
                prediction_info["post_offensive_blend_anchor_lateral_local"] = (
                    world_to_offset_frame(
                        anchor_pos - current_target_pos,
                        self.offensive_anchor_lateral_frame,
                        own_state=own_state,
                        target_state=target_state,
                        **offensive_anchor_lateral_frame_kwargs,
                    )
                )
                prediction_info["offensive_anchor_longitudinal_m"] = (
                    self.offensive_anchor_longitudinal_m
                )
                prediction_info["offensive_anchor_frame"] = (
                    self.offensive_anchor_frame
                )
                prediction_info["offensive_anchor_lateral_frame"] = (
                    self.offensive_anchor_lateral_frame
                )
                prediction_info["offensive_anchor_lateral_m"] = (
                    self.offensive_anchor_lateral_m
                )
                prediction_info["offensive_anchor_lateral_sign_mode"] = (
                    self.offensive_anchor_lateral_sign_mode
                )
                prediction_info["offensive_anchor_lateral_sign"] = (
                    offensive_anchor_lateral_sign
                )
                prediction_info["offensive_anchor_vertical_m"] = (
                    self.offensive_anchor_vertical_m
                )
                prediction_info[
                    "offensive_anchor_encounter_stable_max_heading_delta_deg"
                ] = self.offensive_anchor_encounter_stable_max_heading_delta_deg
                prediction_info["offensive_anchor_pos"] = offensive_anchor_pos
                prediction_info["offensive_anchor_offset"] = offensive_anchor_offset
                prediction_info["offensive_anchor_world_offset"] = (
                    offensive_anchor_world_offset
                )
                prediction_info["offensive_anchor_lateral_world_offset"] = (
                    offensive_anchor_lateral_world_offset
                )
                prediction_info["offensive_anchor_lateral_local_offset"] = (
                    offensive_anchor_lateral_local_offset
                )

        # 虚拟追踪点 = 锚点 + 偏移
        world_offset = offset_to_world(
            offset,
            self.offset_frame,
            own_state=own_state,
            target_state=target_state,
        )
        virtual_point_pos = anchor_pos + world_offset

        # 动力学感知约束：将虚拟点限制在当前飞机的可行航向扇区内
        if self.dynamics_aware and own_state is not None:
            virtual_point_pos = self._apply_dynamics_constraint(
                virtual_point_pos, own_state
            )
            world_offset = virtual_point_pos - anchor_pos
            if self.action_semantics == "tactical_basis_v1":
                offset = world_to_offset_frame(
                    world_offset,
                    self.offset_frame,
                    own_state=own_state,
                    target_state=target_state,
                )

        virtual_point = {
            "position": virtual_point_pos,
            "offset": offset,
            "world_offset": world_offset,
            "offset_frame": self.offset_frame,
        }

        if not return_info:
            return virtual_point

        info = {
            "anchor_mode": anchor_mode,
            "anchor_pos": anchor_pos,
            "offset": offset,
            "world_offset": world_offset,
            "offset_frame": self.offset_frame,
            "predicted_target_blend": prediction_info.get(
                "predicted_target_blend",
                self.predicted_target_blend,
            ),
            "predicted_target_forward_scale": prediction_info.get(
                "predicted_target_forward_scale",
                1.0,
            ),
            "offensive_anchor_blend": prediction_info.get(
                "offensive_anchor_blend",
                self.offensive_anchor_blend,
            ),
            "offensive_anchor_longitudinal_blend": prediction_info.get(
                "offensive_anchor_longitudinal_blend",
                self.offensive_anchor_blend,
            ),
            "offensive_anchor_lateral_blend": prediction_info.get(
                "offensive_anchor_lateral_blend",
                self.offensive_anchor_blend,
            ),
            "offensive_anchor_longitudinal_m": self.offensive_anchor_longitudinal_m,
            "offensive_anchor_frame": self.offensive_anchor_frame,
            "offensive_anchor_lateral_frame": prediction_info.get(
                "offensive_anchor_lateral_frame",
                self.offensive_anchor_lateral_frame,
            ),
            "offensive_anchor_lateral_m": self.offensive_anchor_lateral_m,
            "offensive_anchor_lateral_sign_mode": prediction_info.get(
                "offensive_anchor_lateral_sign_mode",
                self.offensive_anchor_lateral_sign_mode,
            ),
            "offensive_anchor_lateral_sign": prediction_info.get(
                "offensive_anchor_lateral_sign",
                1.0,
            ),
            "offensive_anchor_vertical_m": self.offensive_anchor_vertical_m,
            "offensive_anchor_encounter_stable_max_heading_delta_deg": (
                prediction_info.get(
                    "offensive_anchor_encounter_stable_max_heading_delta_deg",
                    self.offensive_anchor_encounter_stable_max_heading_delta_deg,
                )
            ),
            "longitudinal_scale": longitudinal_scale,
            "lateral_scale": lateral_scale,
            "pred_var": pred_var,
            "prediction_info": prediction_info,
        }
        info.update(tactical_basis_info)
        if "unblended_anchor_pos" in prediction_info:
            info["unblended_anchor_pos"] = prediction_info["unblended_anchor_pos"]
        if "pre_predicted_target_forward_scale_anchor_pos" in prediction_info:
            info["pre_predicted_target_forward_scale_anchor_pos"] = prediction_info[
                "pre_predicted_target_forward_scale_anchor_pos"
            ]
        if "pre_offensive_blend_anchor_pos" in prediction_info:
            info["pre_offensive_blend_anchor_pos"] = prediction_info[
                "pre_offensive_blend_anchor_pos"
            ]
        if "pre_offensive_blend_anchor_local" in prediction_info:
            info["pre_offensive_blend_anchor_local"] = prediction_info[
                "pre_offensive_blend_anchor_local"
            ]
        if "pre_offensive_blend_anchor_lateral_local" in prediction_info:
            info["pre_offensive_blend_anchor_lateral_local"] = prediction_info[
                "pre_offensive_blend_anchor_lateral_local"
            ]
        if "post_offensive_blend_anchor_local" in prediction_info:
            info["post_offensive_blend_anchor_local"] = prediction_info[
                "post_offensive_blend_anchor_local"
            ]
        if "post_offensive_blend_anchor_lateral_local" in prediction_info:
            info["post_offensive_blend_anchor_lateral_local"] = prediction_info[
                "post_offensive_blend_anchor_lateral_local"
            ]
        if "offensive_anchor_longitudinal_m" in prediction_info:
            info["offensive_anchor_pos"] = prediction_info["offensive_anchor_pos"]
            info["offensive_anchor_offset"] = prediction_info["offensive_anchor_offset"]
            info["offensive_anchor_world_offset"] = prediction_info[
                "offensive_anchor_world_offset"
            ]
            info["offensive_anchor_lateral_world_offset"] = prediction_info.get(
                "offensive_anchor_lateral_world_offset"
            )
        return virtual_point, info

    @staticmethod
    def _rescale(val, range_limits):
        """将 [-1, 1] 映射到 [min, max]。"""
        min_val, max_val = range_limits
        return 0.5 * (val + 1.0) * (max_val - min_val) + min_val

    @staticmethod
    def _get_own_position(own_state):
        """从 own_state 中提取位置。支持 position_neu, position_m, position。"""
        pos = own_state.get("position_neu")
        if pos is None:
            pos = own_state.get("position_m")
        if pos is None:
            pos = own_state.get("position")
        if pos is None:
            raise ValueError(
                "own_state must contain 'position_neu', 'position_m', or 'position'"
            )
        arr = np.asarray(pos, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"Own position must be a 3-element vector, got shape {arr.shape}"
            )
        return arr

    @staticmethod
    def _rule_based_pursuit_anchor(own_state, target_state, lead_distance_m: float):
        """纯追踪法则：沿 LOS 方向在目标前方放置固定距离的锚点。"""
        own_pos = VirtualPointGenerator._get_own_position(own_state)
        target_pos = VirtualPointGenerator._get_target_position(target_state)
        los = target_pos - own_pos
        distance = np.linalg.norm(los)
        if distance < 1e-6:
            los_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            los_unit = los / distance
        return target_pos + los_unit * lead_distance_m

    def compute_offensive_anchor_components(
        self,
        target_state,
        own_state=None,
        *,
        lateral_world_offset_override=None,
    ):
        current_target_pos = self._get_target_position(target_state)
        frame = self.offensive_anchor_frame
        lateral_frame = self.offensive_anchor_lateral_frame
        frame_kwargs = self._offensive_anchor_frame_transform_kwargs(frame)
        lateral_frame_kwargs = self._offensive_anchor_frame_transform_kwargs(
            lateral_frame
        )
        lateral_sign = 1.0
        longitudinal_vertical_local_offset = np.array(
            [-self.offensive_anchor_longitudinal_m, 0.0, self.offensive_anchor_vertical_m],
            dtype=np.float64,
        )
        longitudinal_vertical_world_offset = offset_to_world(
            longitudinal_vertical_local_offset,
            frame,
            own_state=own_state,
            target_state=target_state,
            **frame_kwargs,
        )
        if lateral_world_offset_override is not None:
            lateral_world_offset = np.asarray(
                lateral_world_offset_override,
                dtype=np.float64,
            ).reshape(3)
            offensive_anchor_lateral_local_offset = world_to_offset_frame(
                lateral_world_offset,
                lateral_frame,
                own_state=own_state,
                target_state=target_state,
                **lateral_frame_kwargs,
            )
            lateral_component = float(offensive_anchor_lateral_local_offset[1])
            if np.isfinite(lateral_component) and abs(lateral_component) > 1e-6:
                lateral_sign = float(np.sign(lateral_component))
            else:
                lateral_sign = 1.0
        else:
            if self.offensive_anchor_lateral_sign_mode == "fixed_positive":
                lateral_sign = 1.0
            elif self.offensive_anchor_lateral_sign_mode == "fixed_negative":
                lateral_sign = -1.0
            elif self.offensive_anchor_lateral_m > 0.0 and own_state is not None:
                own_pos = self._get_own_position(own_state)
                if own_pos is not None:
                    own_local = world_to_offset_frame(
                        own_pos - current_target_pos,
                        lateral_frame,
                        own_state=own_state,
                        target_state=target_state,
                        **lateral_frame_kwargs,
                    )
                    lateral_component = float(own_local[1])
                    if np.isfinite(lateral_component) and abs(lateral_component) > 1e-6:
                        lateral_sign = float(np.sign(lateral_component))
            offensive_anchor_lateral_local_offset = np.array(
                [0.0, lateral_sign * self.offensive_anchor_lateral_m, 0.0],
                dtype=np.float64,
            )
            lateral_world_offset = offset_to_world(
                offensive_anchor_lateral_local_offset,
                lateral_frame,
                own_state=own_state,
                target_state=target_state,
                **lateral_frame_kwargs,
            )
        offensive_anchor_world_offset = (
            longitudinal_vertical_world_offset + lateral_world_offset
        )
        offensive_anchor_offset = world_to_offset_frame(
            offensive_anchor_world_offset,
            frame,
            own_state=own_state,
            target_state=target_state,
            **frame_kwargs,
        )
        return (
            current_target_pos + offensive_anchor_world_offset,
            offensive_anchor_offset,
            offensive_anchor_world_offset,
            offensive_anchor_lateral_local_offset,
            lateral_world_offset,
            float(lateral_sign),
        )

    def _offensive_anchor_components(
        self,
        target_state,
        own_state=None,
        *,
        lateral_world_offset_override=None,
    ):
        return self.compute_offensive_anchor_components(
            target_state,
            own_state=own_state,
            lateral_world_offset_override=lateral_world_offset_override,
        )

    def _tactical_basis_frame_transform_kwargs(self, frame_name):
        if str(frame_name) != "encounter_stable":
            return {}
        return {
            "encounter_stable_max_heading_delta_deg": (
                self.tactical_basis_encounter_stable_max_heading_delta_deg
            )
        }

    def _compute_tactical_basis_lateral_sign(self, own_state, target_state):
        if self.tactical_basis_lateral_sign_mode == "fixed_positive":
            return 1.0
        if self.tactical_basis_lateral_sign_mode == "fixed_negative":
            return -1.0
        if own_state is None:
            return 1.0
        current_target_pos = self._get_target_position(target_state)
        own_pos = self._get_own_position(own_state)
        lateral_frame_kwargs = self._tactical_basis_frame_transform_kwargs(
            self.tactical_basis_lateral_frame
        )
        own_local = world_to_offset_frame(
            own_pos - current_target_pos,
            self.tactical_basis_lateral_frame,
            own_state=own_state,
            target_state=target_state,
            **lateral_frame_kwargs,
        )
        lateral_component = float(own_local[1])
        if np.isfinite(lateral_component) and abs(lateral_component) > 1e-6:
            return float(np.sign(lateral_component))
        return 1.0

    def _build_tactical_basis_vectors(self, own_state, target_state):
        longitudinal_frame_kwargs = self._tactical_basis_frame_transform_kwargs(
            self.tactical_basis_longitudinal_frame
        )
        lateral_frame_kwargs = self._tactical_basis_frame_transform_kwargs(
            self.tactical_basis_lateral_frame
        )
        vertical_frame_kwargs = self._tactical_basis_frame_transform_kwargs(
            self.tactical_basis_vertical_frame
        )
        tactical_basis_lateral_sign = self._compute_tactical_basis_lateral_sign(
            own_state,
            target_state,
        )
        tactical_basis_ll_world = offset_to_world(
            np.array([self.tactical_basis_lead_lag_extent_m, 0.0, 0.0]),
            self.tactical_basis_longitudinal_frame,
            own_state=own_state,
            target_state=target_state,
            **longitudinal_frame_kwargs,
        )
        tactical_basis_io_world = offset_to_world(
            np.array(
                [
                    0.0,
                    tactical_basis_lateral_sign
                    * self.tactical_basis_inside_outside_extent_m,
                    0.0,
                ]
            ),
            self.tactical_basis_lateral_frame,
            own_state=own_state,
            target_state=target_state,
            **lateral_frame_kwargs,
        )
        tactical_basis_cd_world = offset_to_world(
            np.array([0.0, 0.0, self.tactical_basis_climb_descent_extent_m]),
            self.tactical_basis_vertical_frame,
            own_state=own_state,
            target_state=target_state,
            **vertical_frame_kwargs,
        )
        return (
            tactical_basis_ll_world,
            tactical_basis_io_world,
            tactical_basis_cd_world,
            float(tactical_basis_lateral_sign),
        )

    def _action_to_tactical_basis_world_offset(self, action, own_state, target_state):
        tactical_basis_action_ll = float(action[0])
        tactical_basis_action_io = float(action[1])
        tactical_basis_action_cd = float(action[2])
        (
            tactical_basis_ll_world,
            tactical_basis_io_world,
            tactical_basis_cd_world,
            tactical_basis_lateral_sign,
        ) = self._build_tactical_basis_vectors(own_state, target_state)
        tactical_basis_world_offset = (
            tactical_basis_action_ll * tactical_basis_ll_world
            + tactical_basis_action_io * tactical_basis_io_world
            + tactical_basis_action_cd * tactical_basis_cd_world
        )
        return tactical_basis_world_offset, {
            "action_semantics": "tactical_basis_v1",
            "configured_action_semantics": self.configured_action_semantics,
            "tactical_basis_enabled": True,
            "tactical_basis_action_ll": tactical_basis_action_ll,
            "tactical_basis_action_io": tactical_basis_action_io,
            "tactical_basis_action_cd": tactical_basis_action_cd,
            "tactical_basis_lead_lag_extent_m": (
                self.tactical_basis_lead_lag_extent_m
            ),
            "tactical_basis_inside_outside_extent_m": (
                self.tactical_basis_inside_outside_extent_m
            ),
            "tactical_basis_climb_descent_extent_m": (
                self.tactical_basis_climb_descent_extent_m
            ),
            "tactical_basis_longitudinal_frame": (
                self.tactical_basis_longitudinal_frame
            ),
            "tactical_basis_lateral_frame": self.tactical_basis_lateral_frame,
            "tactical_basis_vertical_frame": self.tactical_basis_vertical_frame,
            "tactical_basis_lateral_sign_mode": (
                self.tactical_basis_lateral_sign_mode
            ),
            "tactical_basis_lateral_sign": tactical_basis_lateral_sign,
            "tactical_basis_ll_world": tactical_basis_ll_world,
            "tactical_basis_io_world": tactical_basis_io_world,
            "tactical_basis_cd_world": tactical_basis_cd_world,
            "tactical_basis_world_offset": tactical_basis_world_offset,
        }

    def _offensive_anchor_frame_transform_kwargs(self, frame_name):
        if str(frame_name) != "encounter_stable":
            return {}
        return {
            "encounter_stable_max_heading_delta_deg": (
                self.offensive_anchor_encounter_stable_max_heading_delta_deg
            )
        }

    def _apply_dynamics_constraint(self, virtual_point_pos, own_state):
        """
        Clip virtual point to a feasible heading sector based on aircraft dynamics.

        F-16 cannot instantaneously change heading. If the virtual point demands a
        heading change beyond the aircraft's physical capability, clip it to the
        edge of the feasible sector while preserving distance.

        Args:
            virtual_point_pos (np.ndarray): Proposed virtual point position [3].
            own_state (dict): Own aircraft state.

        Returns:
            np.ndarray: Constrained virtual point position [3].
        """
        own_pos = self._get_own_position(own_state)
        los = virtual_point_pos - own_pos
        distance = float(np.linalg.norm(los))
        if distance < 1e-6:
            return virtual_point_pos

        los_heading = float(np.arctan2(los[1], los[0]))

        # Extract own heading from velocity
        own_vel = own_state.get("velocity_vector_mps")
        if own_vel is None:
            own_vel = own_state.get("velocity_ned")
        if own_vel is not None:
            own_vel_arr = np.asarray(own_vel, dtype=np.float64)
            own_speed = float(np.linalg.norm(own_vel_arr))
            if own_speed > 1e-6:
                own_heading = float(np.arctan2(own_vel_arr[1], own_vel_arr[0]))
            else:
                own_heading = float(own_state.get("yaw_rad", 0.0))
        else:
            own_heading = float(own_state.get("yaw_rad", 0.0))

        heading_error = _stable_angle_diff(los_heading, own_heading)
        abs_error = abs(heading_error)

        # Max feasible heading change per step (assuming high_level_dt ≈ 0.2s)
        # Allow a small buffer so the policy still learns to turn aggressively
        # when it is physically possible.
        max_feasible = self.max_heading_rate * 0.2 * self.lookahead_steps

        if abs_error > max_feasible:
            sign = 1.0 if heading_error > 0 else -1.0
            constrained_heading = own_heading + sign * max_feasible
            los[0] = distance * np.cos(constrained_heading)
            los[1] = distance * np.sin(constrained_heading)
            return own_pos + los

        return virtual_point_pos

    @staticmethod
    def _get_target_position(target_state):
        """从 target_state 中提取位置。支持 position_neu, position_m, position。"""
        pos = target_state.get("position_neu")
        if pos is None:
            pos = target_state.get("position_m")
        if pos is None:
            pos = target_state.get("position")
        if pos is None:
            raise ValueError(
                "target_state must contain 'position_neu', 'position_m', or 'position'"
            )
        arr = np.asarray(pos, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"Target position must be a 3-element vector, got shape {arr.shape}"
            )
        return arr

    @staticmethod
    def _constant_velocity_prediction(target_state, lookahead_time_s):
        """匀速外推预测目标未来位置。支持 velocity_vector_mps, velocity, velocity_ned。

        velocity_ned 会被转换为 NEU frame（垂直速度取反）。
        """
        pos = VirtualPointGenerator._get_target_position(target_state)

        vel = target_state.get("velocity_vector_mps")
        if vel is not None:
            vel_arr = np.asarray(vel, dtype=np.float64)
            if vel_arr.shape != (3,):
                raise ValueError(
                    f"velocity_vector_mps must be a 3-element vector, got shape {vel_arr.shape}"
                )
            return pos + vel_arr * lookahead_time_s

        vel = target_state.get("velocity")
        if vel is not None:
            vel_arr = np.asarray(vel, dtype=np.float64)
            if vel_arr.shape != (3,):
                raise ValueError(
                    f"velocity must be a 3-element vector, got shape {vel_arr.shape}"
                )
            return pos + vel_arr * lookahead_time_s

        vel_ned = target_state.get("velocity_ned")
        if vel_ned is not None:
            v = np.asarray(vel_ned, dtype=np.float64)
            if v.shape != (3,):
                raise ValueError(
                    f"velocity_ned must be a 3-element vector, got shape {v.shape}"
                )
            vel_neu = np.array([v[0], v[1], -v[2]], dtype=np.float64)
            return pos + vel_neu * lookahead_time_s

        # 缺少速度信息时返回当前位置
        return pos
