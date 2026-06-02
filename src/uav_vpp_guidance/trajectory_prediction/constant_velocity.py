"""
匀速外推目标轨迹预测模型。

物理含义：
假设目标机在短时预测窗口内速度保持不变，
则未来位置 = 当前目标位置 + 当前目标速度 * T_lookahead。
"""

import numpy as np
from .base_predictor import BaseTrajectoryPredictor


class ConstantVelocityPredictor(BaseTrajectoryPredictor):
    """
    基于匀速假设的目标轨迹预测模型。

    作为 fallback baseline，无需训练，计算开销极低。
    """

    def __init__(self, lookahead_time_s: float = 1.0):
        """
        Args:
            lookahead_time_s (float): 预测前瞻时间（秒）。
        """
        self.lookahead_time_s = lookahead_time_s

    def predict(self, history_seq=None, current_target_state=None):
        """
        匀速外推预测。

        Args:
            history_seq: 不使用，为接口兼容保留。
            current_target_state (dict): 目标当前状态。
                必须包含以下字段之一：
                - "velocity_ned" / "velocity_vector_mps": [vx, vy, vz] in m/s
                - "velocity_mps": 标量速度 + "heading_rad" 等（简化转换）
                - "position_neu" / "position_m": 当前位置 [x, y, z] in m

        Returns:
            tuple: (pred_pos, pred_var, info)
        """
        info = {"model": "constant_velocity", "fallback": False, "output_is_absolute": True}

        if current_target_state is None:
            info["fallback"] = True
            info["reason"] = "current_target_state is None"
            return None, None, info

        # 提取位置
        pos = current_target_state.get("position_neu")
        if pos is None:
            pos = current_target_state.get("position_m")
        if pos is None:
            info["fallback"] = True
            info["reason"] = "position missing"
            return None, None, info
        pos = np.asarray(pos, dtype=np.float64)

        # 提取速度向量（优先使用 velocity_ned / velocity_vector_mps）
        vel = current_target_state.get("velocity_ned")
        if vel is None:
            vel = current_target_state.get("velocity_vector_mps")
        if vel is not None:
            vel = np.asarray(vel, dtype=np.float64)
        else:
            # 如果没有速度向量，尝试用速度和航向做简化转换
            speed = current_target_state.get("velocity_mps") or current_target_state.get("speed_mps") or current_target_state.get("vt_mps")
            heading = current_target_state.get("heading_rad") or current_target_state.get("attitude_rpy", [0, 0, 0])[2]
            if speed is not None and heading is not None:
                vel = np.array([
                    speed * np.cos(heading),
                    speed * np.sin(heading),
                    0.0
                ], dtype=np.float64)
            else:
                info["fallback"] = True
                info["reason"] = "velocity information insufficient"
                return pos, None, info

        pred_pos = pos + vel * self.lookahead_time_s
        return pred_pos, None, info
