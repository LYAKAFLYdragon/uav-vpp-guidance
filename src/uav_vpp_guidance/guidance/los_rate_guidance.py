"""
LOS-rate-based guidance law.

输入：
- own aircraft state
- target aircraft state
- virtual pursuit point
- guidance gains

输出：
- normal overload command (nz_cmd)
- roll-rate command (roll_rate_cmd)
- throttle command (throttle_cmd)

物理含义：
1. 根据本机到虚拟追踪点的方向，计算航向误差和俯仰误差；
2. roll_rate_cmd 与航向误差成正比（滚转用于改变航向）；
3. nz_cmd 与俯仰误差或 LOS 俯仰角速率成正比（俯仰/法向过载用于改变高度）；
4. throttle_cmd 先固定或根据距离简单调整。

注意：
- 本模块只负责计算原始制导指令；
- command_limiter 和 command_filter 应由调用方（如 tracking_env）在 step 中调用。
"""

import numpy as np


class LOSRateGuidance:
    """
    LOS-rate-based guidance law.

    Input:
    - own aircraft state
    - target aircraft state
    - virtual pursuit point
    - guidance gains

    Output:
    - normal overload command
    - roll-rate command
    - optional throttle or speed command
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Guidance configuration dictionary.
        """
        self.config = config
        self.prev_command = None
        # 默认增益
        self.gains = config.get("gains", {})
        self.k_los = self.gains.get("k_los", 1.0)
        self.k_pos = self.gains.get("k_pos", 0.5)
        self.k_damp = self.gains.get("k_damp", 0.2)
        self.k_roll = self.gains.get("k_roll", 1.0)
        self.k_speed = self.gains.get("k_speed", 0.2)

    def reset(self):
        """Reset internal guidance state."""
        self.prev_command = None

    def compute_command(self, own_state, target_state, virtual_point, gains=None):
        """
        Compute guidance commands.

        简化逻辑：
        1. 计算 own 到 virtual_point 的方向；
        2. 计算 heading_error；
        3. roll_rate_cmd = k_roll * heading_error；
        4. 计算 elevation_error 或 vertical_error；
        5. nz_cmd = 1.0 + k_pos * elevation_error + k_damp * vertical_error_rate；
        6. throttle_cmd 根据距离或固定为 0.7。

        Args:
            own_state (dict): Own aircraft state.
            target_state (dict): Target aircraft state.
            virtual_point (dict): Virtual pursuit point state.
                至少包含 "position" 或 "position_neu" 或 "position_m" 字段（NEU 坐标）。
            gains (GuidanceGains, optional): Current guidance gains.

        Returns:
            dict: Command dictionary with keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
        """
        # 使用外部传入的 gains 或默认增益
        if gains is not None:
            k_los = getattr(gains, "k_los", self.k_los)
            k_pos = getattr(gains, "k_pos", self.k_pos)
            k_damp = getattr(gains, "k_damp", self.k_damp)
            k_roll = getattr(gains, "k_roll", self.k_roll)
            k_speed = getattr(gains, "k_speed", self.k_speed)
        else:
            k_los = self.k_los
            k_pos = self.k_pos
            k_damp = self.k_damp
            k_roll = self.k_roll
            k_speed = self.k_speed

        # 提取位置
        own_pos = _get_position(own_state)
        vp_pos = _get_position(virtual_point)

        # 相对位置（虚拟追踪点 - 本机）
        rel_pos = vp_pos - own_pos
        distance = float(np.linalg.norm(rel_pos))

        # 提取本机速度
        own_vel = _get_velocity(own_state)
        own_speed = float(np.linalg.norm(own_vel))
        own_heading = np.arctan2(own_vel[1], own_vel[0]) if own_speed > 1e-6 else 0.0

        # LOS 方向（水平面）
        los_horizontal = rel_pos.copy()
        los_horizontal[2] = 0.0
        los_h_dist = float(np.linalg.norm(los_horizontal))
        if los_h_dist > 1e-6:
            los_heading = np.arctan2(los_horizontal[1], los_horizontal[0])
        else:
            los_heading = own_heading

        # 1. 航向误差 -> roll_rate_cmd
        # roll_rate_cmd 用于让飞机朝虚拟追踪点转向
        heading_error = _normalize_angle(los_heading - own_heading)
        current_roll = own_state.get("roll_rad", 0.0)
        roll_rate_cmd = k_roll * heading_error - k_damp * current_roll

        # 2. 垂直误差 -> nz_cmd
        # nz_cmd 用于调整垂向机动
        if distance > 1e-6:
            los_elevation = np.arcsin(np.clip(rel_pos[2] / distance, -1.0, 1.0))
        else:
            los_elevation = 0.0

        # 法向过载：基础 1g（平飞）+ 俯仰误差修正 + 距离比例项
        nz_cmd = 1.0 + k_los * los_elevation + k_pos * (distance / 2000.0)

        # 3. 速度/油门指令
        # throttle_cmd 用于维持或调整接近速度
        target_speed = 250.0  # 目标速度参考值 m/s
        speed_error = target_speed - own_speed
        throttle_cmd = 0.7 + k_speed * (speed_error / 100.0)
        throttle_cmd = np.clip(throttle_cmd, 0.0, 1.0)

        command = {
            "nz_cmd": float(nz_cmd),
            "roll_rate_cmd": float(roll_rate_cmd),
            "throttle_cmd": float(throttle_cmd),
        }
        self.prev_command = command
        return command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_position(state):
    pos = state.get("position_neu")
    if pos is None:
        pos = state.get("position_m")
    if pos is None:
        pos = state.get("position")
    if pos is None:
        raise ValueError("State missing position field")
    return np.asarray(pos, dtype=np.float64)


def _get_velocity(state):
    vel = state.get("velocity_ned")
    if vel is None:
        vel = state.get("velocity_vector_mps")
    if vel is None:
        vel = state.get("velocity")
    if vel is None:
        raise ValueError("State missing velocity field")
    return np.asarray(vel, dtype=np.float64)


def _normalize_angle(angle):
    """将角度归一化到 [-pi, pi]。"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle
