"""
Virtual Pursuit Point (VPP) generator.

Converts a normalized policy action into a virtual pursuit point
relative to the target aircraft.

升级：支持将策略输出的相对偏移量叠加到预测未来位置，
而不仅仅是目标当前位置。
"""

import numpy as np


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
    The canonical action space (action_dim == 3) is mapped to:
    - longitudinal offset
    - lateral offset
    - vertical offset

    The legacy 5-D action space also included prediction time and speed bias;
    those extra dimensions are deprecated and ignored when action_dim == 3.

    支持三种锚点模式：
    - current_target: 目标当前位置（旧逻辑）
    - constant_velocity: 匀速外推预测位置
    - predicted_target: 通过 trajectory_predictor_adapter 获取预测位置
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Virtual point configuration dictionary.
        """
        self.config = config
        self.action_dim = config.get("action_dim", 3)
        self.d_long_range = config.get("d_long_range", [-1500.0, 1500.0])
        self.d_lat_range = config.get("d_lat_range", [-800.0, 800.0])
        self.d_vert_range = config.get("d_vert_range", [-500.0, 500.0])
        self.tau_pred_range = config.get("tau_pred_range", [0.0, 3.0])
        self.speed_bias_range = config.get("speed_bias_range", [-80.0, 80.0])
        self.smoothing_alpha = config.get("smoothing_alpha", 0.3)
        self.lead_distance_m = config.get("lead_distance_m", 500.0)
        # Dynamics-aware constraint: clip virtual points to feasible heading sector
        self.dynamics_aware = config.get("dynamics_aware", False)
        # F-16 max feasible heading change per step (high_level_dt=0.2s, max rate ≈ 0.3 rad/s)
        self.max_heading_rate = config.get("max_heading_rate", 0.3)
        self.lookahead_steps = config.get("lookahead_steps", 5)
        self._prev_action = None

    def action_to_virtual_point(
        self,
        action,
        own_state,
        target_state,
        anchor_mode: str = "current_target",
        lookahead_time_s: float = 1.0,
        trajectory_predictor_adapter=None,
        predicted_target_position=None,
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
        action = np.asarray(action, dtype=np.float64)

        # 将 action 前 3 维映射为实际偏移量
        offset = np.array(
            [
                self._rescale(action[0], self.d_long_range),
                self._rescale(action[1], self.d_lat_range),
                self._rescale(action[2], self.d_vert_range),
            ],
            dtype=np.float64,
        )

        # 确定锚点位置
        if anchor_mode == "current_target":
            anchor_pos = self._get_target_position(target_state)
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

        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

        # 虚拟追踪点 = 锚点 + 偏移
        virtual_point_pos = anchor_pos + offset

        # 动力学感知约束：将虚拟点限制在当前飞机的可行航向扇区内
        if self.dynamics_aware and own_state is not None:
            virtual_point_pos = self._apply_dynamics_constraint(
                virtual_point_pos, own_state
            )

        virtual_point = {
            "position": virtual_point_pos,
            "offset": offset,
        }

        if not return_info:
            return virtual_point

        info = {
            "anchor_mode": anchor_mode,
            "anchor_pos": anchor_pos,
            "offset": offset,
            "pred_var": pred_var,
            "prediction_info": prediction_info,
        }
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
