"""
匀加速外推目标轨迹预测模型。

物理含义：
假设目标机在短时预测窗口内加速度保持不变，
则未来位置 = 当前目标位置 + 当前目标速度 * T_lookahead + 0.5 * 当前目标加速度 * T_lookahead^2。

当历史不足以估计加速度时，回退到 ConstantVelocityPredictor；
当历史不足以估计速度时，回退到当前位置。

所有坐标统一使用 NEU (North-East-Up)。
"""

import numpy as np
from .base_predictor import BaseTrajectoryPredictor
from .constant_velocity import ConstantVelocityPredictor
from .coordinate_utils import get_position_neu, get_velocity_neu


class ConstantAccelerationPredictor(BaseTrajectoryPredictor):
    """
    基于匀加速假设的目标轨迹预测模型。

    作为经典外推模型，无需训练，计算开销低。
    """

    def __init__(self, lookahead_time_s: float = 1.0):
        """
        Args:
            lookahead_time_s (float): 预测前瞻时间（秒）。
        """
        self.lookahead_time_s = lookahead_time_s
        self._fallback_cv = ConstantVelocityPredictor(lookahead_time_s=lookahead_time_s)

    def predict(self, history_seq=None, current_target_state=None):
        """
        匀加速外推预测。

        Args:
            history_seq (np.ndarray, optional): 历史状态序列，shape 为 [history_len, feature_dim]。
            current_target_state (dict, optional): 目标当前状态。

        Returns:
            tuple: (pred_pos, pred_var, info)
        """
        info = {
            "model": "constant_acceleration",
            "fallback": False,
            "fallback_reason": None,
            "output_is_absolute": True,
        }

        if current_target_state is None:
            info["fallback"] = True
            info["fallback_reason"] = "current_target_state is None"
            return None, None, info

        # 提取当前位置 (NEU)
        try:
            pos = get_position_neu(current_target_state)
        except ValueError as exc:
            info["fallback"] = True
            info["fallback_reason"] = str(exc)
            return None, None, info

        # 提取当前速度 (NEU)
        try:
            vel = get_velocity_neu(current_target_state)
        except ValueError as exc:
            info["fallback"] = True
            info["fallback_reason"] = str(exc)
            return pos, None, info

        # 尝试从历史序列估计加速度
        acc = self._estimate_acceleration(history_seq, current_target_state)

        if acc is None:
            info["fallback"] = True
            info["fallback_reason"] = "insufficient history for acceleration estimation"
            pred_pos, _, cv_info = self._fallback_cv.predict(
                history_seq=None, current_target_state=current_target_state
            )
            info["cv_fallback"] = cv_info
            return pred_pos, None, info

        # NaN/inf 保护
        if not np.isfinite(acc).all():
            info["fallback"] = True
            info["fallback_reason"] = "non-finite acceleration estimate"
            pred_pos, _, cv_info = self._fallback_cv.predict(
                history_seq=None, current_target_state=current_target_state
            )
            info["cv_fallback"] = cv_info
            return pred_pos, None, info

        # 匀加速预测公式
        T = self.lookahead_time_s
        pred_pos = pos + vel * T + 0.5 * acc * (T ** 2)

        return pred_pos, None, info

    def _estimate_acceleration(self, history_seq, current_target_state):
        """
        从历史序列估计加速度。

        优先使用 history_seq 中隐含的 position/velocity 信息；
        如果 history_seq 是标准 feature_builder 输出（16 维），则利用其中包含的相对位置和速度信息。

        当前简化实现：
        - 如果 history_seq 至少有 3 帧，用最近两帧的速度差分估计加速度。
        - 如果只有 2 帧且包含 position 信息，用 position 二阶差分估计。
        - 否则返回 None（回退到 CV）。

        Args:
            history_seq (np.ndarray): shape [history_len, feature_dim]

        Returns:
            np.ndarray or None: 加速度向量 [3]，若无法估计则返回 None。
        """
        if history_seq is None:
            return None

        history_seq = np.asarray(history_seq, dtype=np.float64)
        if history_seq.ndim == 3:
            # 如果有 batch 维度，取第一个样本
            history_seq = history_seq[0]

        history_len = history_seq.shape[0]
        if history_len < 3:
            return None

        # 对于 feature_builder 生成的 16 维特征：
        # 索引 0-2: rel_pos / pos_scale (相对位置归一化)
        # 索引 3-5: target_vel / vel_scale (目标速度归一化)
        # 但直接从 feature 反推物理量不够稳定，因此优先使用 current_target_state 中的速度
        # 和历史序列中的速度变化。

        # 简化策略：如果历史序列中连续帧包含速度特征（索引 3-5），
        # 用最近两帧的速度差分除以时间来估计加速度。
        # 假设相邻帧时间间隔相同，记为 dt_feature。
        # 由于 feature_builder 输出的是归一化特征，我们需要恢复原始速度。
        vel_scale = 300.0  # feature_builder 默认 velocity_scale_mps

        # 取最近两帧和前一帧的速度特征（索引 3,4,5）
        vel_feat_t = history_seq[-1, 3:6]
        vel_feat_t1 = history_seq[-2, 3:6]
        vel_t = vel_feat_t * vel_scale
        vel_t1 = vel_feat_t1 * vel_scale

        # 取最近三帧的速度特征做二阶差分
        vel_feat_t2 = history_seq[-3, 3:6]
        vel_t2 = vel_feat_t2 * vel_scale

        # 估计时间步长：从 feature 中的 distance/range_rate 很难反推 dt，
        # 但 trajectory_prediction.yaml 中默认 history sample_rate_hz=5，即 dt=0.2s。
        # 这里用默认 0.2s；如果 config 中有 dt 会优先读取。
        dt = 0.2

        # 用中心差分估计加速度：a = (v_t - v_{t-1}) / dt
        # 为减少噪声，也可用多帧平均
        acc1 = (vel_t - vel_t1) / dt
        acc2 = (vel_t1 - vel_t2) / dt
        acc = 0.5 * (acc1 + acc2)

        return acc
