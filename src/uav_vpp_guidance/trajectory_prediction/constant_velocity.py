"""
匀速外推目标轨迹预测模型。

物理含义：
假设目标机在短时预测窗口内速度保持不变，
则未来位置 = 当前目标位置 + 当前目标速度 * T_lookahead。

所有坐标统一使用 NEU (North-East-Up)。
"""

import numpy as np
from .base_predictor import BaseTrajectoryPredictor
from .coordinate_utils import get_position_neu, get_velocity_neu


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
                必须包含位置字段 (position_neu / position_m) 和
                速度字段 (velocity_vector_mps / velocity_ned / velocity_mps+heading)。

        Returns:
            tuple: (pred_pos, pred_var, info)
        """
        info = {
            "model": "constant_velocity",
            "fallback": False,
            "output_is_absolute": True,
        }

        if current_target_state is None:
            info["fallback"] = True
            info["reason"] = "current_target_state is None"
            return None, None, info

        # 提取位置 (NEU)
        try:
            pos = get_position_neu(current_target_state)
        except ValueError as exc:
            info["fallback"] = True
            info["reason"] = str(exc)
            return None, None, info

        # 提取速度 (NEU)
        try:
            vel = get_velocity_neu(current_target_state)
        except ValueError as exc:
            info["fallback"] = True
            info["reason"] = str(exc)
            return pos, None, info

        pred_pos = pos + vel * self.lookahead_time_s
        return pred_pos, None, info
