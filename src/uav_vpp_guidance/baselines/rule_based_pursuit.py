"""
规则基线 pursuit 策略。

不依赖策略网络，直接根据几何关系生成虚拟追踪点偏移量。

支持三种模式：
1. pure_pursuit: 虚拟点直接放在目标当前位置（offset = 0）
2. lag_pursuit: 虚拟点放在目标后方
3. lead_pursuit: 虚拟点放在目标前方

输出格式与策略网络一致：归一化的 3 维偏移 action。
"""

import numpy as np


class RuleBasedPursuitPolicy:
    """
    规则 pursuit 基线策略。

    根据本机与目标的相对几何关系，直接计算虚拟追踪点偏移，
    不经过神经网络前向传播。
    """

    def __init__(self, mode: str = "pure_pursuit", lag_distance_m: float = 500.0):
        """
        Args:
            mode (str): "pure_pursuit", "lag_pursuit", 或 "lead_pursuit"。
            lag_distance_m (float): lag/lead 模式下的前后距离（米）。
        """
        assert mode in ("pure_pursuit", "lag_pursuit", "lead_pursuit")
        self.mode = mode
        self.lag_distance_m = lag_distance_m

    def get_action(self, own_state, target_state, rel_state) -> np.ndarray:
        """
        计算规则 pursuit 动作。

        Args:
            own_state (dict): 本机状态。
            target_state (dict): 目标状态。
            rel_state (dict): 相对态势（由 compute_relative_geometry 输出）。

        Returns:
            np.ndarray: 3 维偏移 action（已归一化到 [-1, 1] 范围）。
        """
        if self.mode == "pure_pursuit":
            return np.zeros(3, dtype=np.float64)

        # 获取目标速度方向（水平面）
        target_vel = target_state.get("velocity_vector_mps")
        if target_vel is None:
            target_vel = target_state.get("velocity_ned")
        if target_vel is None:
            return np.zeros(3, dtype=np.float64)
        target_vel = np.asarray(target_vel, dtype=np.float64)
        speed = float(np.linalg.norm(target_vel))
        if speed < 1e-6:
            return np.zeros(3, dtype=np.float64)

        heading = np.arctan2(target_vel[1], target_vel[0])

        if self.mode == "lag_pursuit":
            # 虚拟点放在目标后方 lag_distance_m 处
            offset = np.array([
                -self.lag_distance_m * np.cos(heading),
                -self.lag_distance_m * np.sin(heading),
                0.0,
            ])
        elif self.mode == "lead_pursuit":
            # 虚拟点放在目标前方 lag_distance_m 处
            offset = np.array([
                self.lag_distance_m * np.cos(heading),
                self.lag_distance_m * np.sin(heading),
                0.0,
            ])
        else:
            offset = np.zeros(3)

        # 归一化到 [-1, 1]（假设最大偏移量 1500m）
        max_offset = 1500.0
        action = np.clip(offset / max_offset, -1.0, 1.0)
        return action
