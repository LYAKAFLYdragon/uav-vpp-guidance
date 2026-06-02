"""
轨迹预测特征构造器。

将本机状态、目标状态和相对态势构造成固定长度的特征向量，
供轨迹预测模型（LSTM/GRU）作为时序输入。

兼容字段名：
  - position: position_neu → position_m
  - velocity: velocity_ned → velocity_vector_mps
  - attitude: attitude_rpy → roll_rad/pitch_rad/yaw_rad
  - distance: distance → range_m
"""

import numpy as np


def _get_position(state):
    """提取位置，兼容 position_neu 和 position_m。"""
    pos = state.get("position_neu")
    if pos is None:
        pos = state.get("position_m")
    if pos is None:
        raise ValueError("State missing position field (position_neu or position_m)")
    return np.asarray(pos, dtype=np.float64)


def _get_velocity(state):
    """提取速度，兼容 velocity_ned 和 velocity_vector_mps。

    注意：
      - velocity_ned 是 NED 坐标系 [north, east, down]
      - velocity_vector_mps 是 NEU 坐标系 [north, east, up]
      这里统一转换为 NEU 格式输出。
    """
    vel = state.get("velocity_ned")
    if vel is not None:
        vel = np.asarray(vel, dtype=np.float64)
        # NED → NEU: [n, e, -d]
        return np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
    vel = state.get("velocity_vector_mps")
    if vel is not None:
        return np.asarray(vel, dtype=np.float64)
    raise ValueError("State missing velocity field (velocity_ned or velocity_vector_mps)")


def _get_attitude_rpy(state):
    """提取姿态角，兼容 attitude_rpy 和 roll_rad/pitch_rad/yaw_rad。"""
    rpy = state.get("attitude_rpy")
    if rpy is not None:
        return np.asarray(rpy, dtype=np.float64)
    roll = state.get("roll_rad", 0.0)
    pitch = state.get("pitch_rad", 0.0)
    yaw = state.get("yaw_rad", 0.0)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _get_distance(relative_state, rel_pos):
    """提取距离，兼容 distance 和 range_m。"""
    d = relative_state.get("distance")
    if d is not None:
        return float(d)
    d = relative_state.get("range_m")
    if d is not None:
        return float(d)
    return float(np.linalg.norm(rel_pos))


def build_target_prediction_feature(own_state, target_state, relative_state, config) -> np.ndarray:
    """
    构造目标轨迹预测所需的 16 维特征向量。

    Args:
        own_state (dict): 本机状态。
        target_state (dict): 目标状态。
        relative_state (dict): 相对态势。
        config (dict): 配置字典，需包含 normalization 参数。

    Returns:
        np.ndarray: 16 维特征向量。
    """
    norm_cfg = config.get("normalization", {})
    pos_scale = norm_cfg.get("position_scale_m", 1000.0)
    vel_scale = norm_cfg.get("velocity_scale_mps", 300.0)
    overload_scale = norm_cfg.get("overload_scale", 9.0)

    # 1-3. 目标相对本机的位置
    own_pos = _get_position(own_state)
    target_pos = _get_position(target_state)
    rel_pos = target_pos - own_pos

    # 4-6. 目标速度 (统一为 NEU)
    target_vel = _get_velocity(target_state)

    # 7-12. 目标姿态的 sin/cos
    target_rpy = _get_attitude_rpy(target_state)
    target_roll, target_pitch, target_yaw = target_rpy[0], target_rpy[1], target_rpy[2]

    # 13. 目标法向过载
    target_nz = target_state.get("nz", 0.0)

    # 14. 相对距离
    distance = _get_distance(relative_state, rel_pos)

    # 15. 距离变化率 (range rate)
    own_vel = _get_velocity(own_state)
    rel_vel = relative_state.get("relative_velocity", target_vel - own_vel)
    rel_vel = np.asarray(rel_vel, dtype=np.float64)
    rel_pos_norm = rel_pos / (np.linalg.norm(rel_pos) + 1e-8)
    range_rate = float(np.dot(rel_vel, rel_pos_norm))

    # 16. 高度差
    altitude_diff = target_pos[2] - own_pos[2]

    feature = np.array([
        rel_pos[0] / pos_scale,
        rel_pos[1] / pos_scale,
        rel_pos[2] / pos_scale,
        target_vel[0] / vel_scale,
        target_vel[1] / vel_scale,
        target_vel[2] / vel_scale,
        np.sin(target_roll),
        np.cos(target_roll),
        np.sin(target_pitch),
        np.cos(target_pitch),
        np.sin(target_yaw),
        np.cos(target_yaw),
        target_nz / overload_scale,
        distance / pos_scale,
        range_rate / vel_scale,
        altitude_diff / pos_scale,
    ], dtype=np.float32)

    return feature
