"""
轨迹预测特征构造器。

将本机状态、目标状态和相对态势构造成固定长度的特征向量，
供轨迹预测模型（LSTM/GRU）作为时序输入。
"""

import numpy as np


def build_target_prediction_feature(own_state, target_state, relative_state, config) -> np.ndarray:
    """
    构造目标轨迹预测所需的 16 维特征向量。

    Args:
        own_state (dict): 本机状态，至少包含 position_neu, velocity_ned, attitude_rpy。
        target_state (dict): 目标状态，至少包含 position_neu, velocity_ned, attitude_rpy。
        relative_state (dict): 相对态势，至少包含 relative_position, relative_velocity, distance。
        config (dict): 配置字典，需包含 normalization 参数。

    Returns:
        np.ndarray: 16 维特征向量。
    """
    norm_cfg = config.get("normalization", {})
    pos_scale = norm_cfg.get("position_scale_m", 1000.0)
    vel_scale = norm_cfg.get("velocity_scale_mps", 300.0)
    overload_scale = norm_cfg.get("overload_scale", 9.0)

    # 1-3. 目标相对本机的位置 (NEU)
    own_pos = own_state.get("position_neu", np.zeros(3))
    target_pos = target_state.get("position_neu", np.zeros(3))
    rel_pos = target_pos - own_pos

    # 4-6. 目标速度 (NED)
    target_vel = target_state.get("velocity_ned", np.zeros(3))

    # 7-12. 目标姿态的 sin/cos
    target_rpy = target_state.get("attitude_rpy", np.zeros(3))
    target_roll, target_pitch, target_yaw = target_rpy[0], target_rpy[1], target_rpy[2]

    # 13. 目标法向过载 (若缺失则使用 0.0)
    target_nz = target_state.get("nz", 0.0)

    # 14. 相对距离
    distance = relative_state.get("distance", float(np.linalg.norm(rel_pos)))

    # 15. 距离变化率 (range rate)
    rel_vel = relative_state.get("relative_velocity", target_vel - own_state.get("velocity_ned", np.zeros(3)))
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
