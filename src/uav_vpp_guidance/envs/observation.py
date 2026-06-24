"""
Observation construction utilities.

Migrated from legacy project:
  <JSBSIM_ROOT>/envs/JSBSim/envs/env_wrappers.py

第一版使用简化几何计算，支持 NEU 坐标系下的相对态势特征提取。
"""

import numpy as np


def compute_relative_geometry(own_state, target_state):
    """
    Compute range, line-of-sight angles, ATA, AA, range rate, speed difference,
    altitude difference, and other relative-motion features.

    Args:
        own_state (dict): Own aircraft state. Must contain at least:
            position_m or position_neu (np.ndarray [3])
            velocity_vector_mps or velocity_ned (np.ndarray [3])
        target_state (dict): Target aircraft state. Same fields as own_state.

    Returns:
        dict: Relative geometry features.
    """
    # Extract positions
    own_pos = _get_position(own_state)
    target_pos = _get_position(target_state)

    # Extract velocities
    own_vel = _get_velocity(own_state)
    target_vel = _get_velocity(target_state)

    # Relative position (target - own), NEU
    rel_pos = target_pos - own_pos
    range_m = float(np.linalg.norm(rel_pos))

    # Relative velocity
    rel_vel = target_vel - own_vel
    range_rate_mps = float(np.dot(rel_vel, rel_pos) / (range_m + 1e-8))

    # LOS vector (unit vector from own to target)
    los_vector = rel_pos / (range_m + 1e-8)

    # LOS angles
    los_azimuth_rad = np.arctan2(los_vector[1], los_vector[0])
    los_elevation_rad = np.arcsin(np.clip(los_vector[2], -1.0, 1.0))

    # ATA (Aspect Target Angle): angle between target velocity and line-of-sight
    # 即目标视线角，描述目标相对于本机的方位
    target_speed = float(np.linalg.norm(target_vel))
    if target_speed > 1e-6:
        cos_ata = np.dot(-los_vector, target_vel) / target_speed
        cos_ata = np.clip(cos_ata, -1.0, 1.0)
        ata_rad = np.arccos(cos_ata)
    else:
        ata_rad = 0.0

    # AA (Antenna Train Angle / Angle off): angle between own velocity and line-of-sight
    # 即本机天线方位角，描述本机速度方向与目标视线的夹角。
    # AA 越小表示本机越正对目标。
    own_speed = float(np.linalg.norm(own_vel))
    if own_speed > 1e-6:
        cos_aa = np.dot(los_vector, own_vel) / own_speed
        cos_aa = np.clip(cos_aa, -1.0, 1.0)
        aa_rad = np.arccos(cos_aa)
    else:
        aa_rad = 0.0

    # Altitude and speed differences
    own_alt = _get_altitude(own_state)
    target_alt = _get_altitude(target_state)
    altitude_diff_m = target_alt - own_alt
    speed_diff_mps = target_speed - own_speed

    return {
        "range_m": range_m,
        "range_rate_mps": range_rate_mps,
        "altitude_diff_m": altitude_diff_m,
        "speed_diff_mps": speed_diff_mps,
        "los_vector": los_vector,
        "los_unit": los_vector,
        "los_azimuth_rad": los_azimuth_rad,
        "los_elevation_rad": los_elevation_rad,
        "ata_rad": ata_rad,
        "aa_rad": aa_rad,
        "relative_position": rel_pos,
        "relative_velocity": rel_vel,
    }


def build_observation(
    own_state,
    target_state,
    guidance_state=None,
    gains=None,
    prediction_features=None,
    saturation_features=None,
    return_feature_names=False,
):
    """
    Build policy observation.

    Observation schema (default + opt-in extensions):

    Base (always present, 16 dims):
      - relative geometry: range, range_rate, alt_diff, speed_diff
      - LOS angles: azimuth/elevation sin/cos
      - aspect geometry: ATA/AA sin/cos
      - own/target speed and altitude

    Optional extensions (controlled by caller / config):
      - gains (+2): k_los, k_pos
      - guidance_state (+3): VP tracking error [x, y, z]
      - prediction_features (+14): prediction-aware relative/displacement/velocity/variance/valid/fallback
      - saturation_features (+3): per-channel command saturation flags [nz, roll_rate, throttle]

    Args:
        own_state (dict): Own aircraft state.
        target_state (dict): Target aircraft state.
        guidance_state (dict, optional): Guidance internal state (e.g. VP tracking error).
        gains (GuidanceGains, optional): Current guidance gains.
        prediction_features (dict, optional): Normalized prediction-aware features.
        saturation_features (dict, optional): Per-channel command saturation indicators.
        return_feature_names (bool): If True, also return the ordered list of feature names.

    Returns:
        np.ndarray: Flattened observation vector.
        If return_feature_names is True, returns (obs_vec, feature_names).
    """
    rel = compute_relative_geometry(own_state, target_state)

    own_speed = float(np.linalg.norm(_get_velocity(own_state)))
    target_speed = float(np.linalg.norm(_get_velocity(target_state)))
    own_alt = _get_altitude(own_state)
    target_alt = _get_altitude(target_state)

    # 归一化参考值
    ref_range = 5000.0
    ref_range_rate = 200.0
    ref_alt = 10000.0
    ref_speed = 400.0

    # 基础观察：相对几何 + 速度/高度
    obs_dict = {
        "range_m": rel["range_m"] / ref_range,
        "range_rate_mps": rel["range_rate_mps"] / ref_range_rate,
        "altitude_diff_m": rel["altitude_diff_m"] / ref_alt,
        "speed_diff_mps": rel["speed_diff_mps"] / ref_speed,
        "los_azimuth_sin": np.sin(rel["los_azimuth_rad"]),
        "los_azimuth_cos": np.cos(rel["los_azimuth_rad"]),
        "los_elevation_sin": np.sin(rel["los_elevation_rad"]),
        "los_elevation_cos": np.cos(rel["los_elevation_rad"]),
        "ata_sin": np.sin(rel["ata_rad"]),
        "ata_cos": np.cos(rel["ata_rad"]),
        "aa_sin": np.sin(rel["aa_rad"]),
        "aa_cos": np.cos(rel["aa_rad"]),
        "own_speed": own_speed / ref_speed,
        "target_speed": target_speed / ref_speed,
        "own_altitude": own_alt / ref_alt,
        "target_altitude": target_alt / ref_alt,
    }

    # 可选：guidance gains
    if gains is not None:
        obs_dict["gain_k_los"] = float(getattr(gains, "k_los", 1.0))
        obs_dict["gain_k_pos"] = float(getattr(gains, "k_pos", 0.5))

    # 可选：guidance state（虚拟点跟踪误差等）
    if guidance_state is not None:
        vp_error = guidance_state.get("vp_tracking_error", np.zeros(3))
        obs_dict["vp_error_x"] = vp_error[0] / ref_range
        obs_dict["vp_error_y"] = vp_error[1] / ref_range
        obs_dict["vp_error_z"] = vp_error[2] / ref_alt

    if prediction_features is not None:
        for key in (
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
        ):
            obs_dict[key] = float(prediction_features.get(key, 0.0))

    # 可选：指令饱和指示器
    if saturation_features is not None:
        for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated"):
            obs_dict[key] = float(saturation_features.get(key, 0.0))

    # 展平为向量，并对极端值 / NaN / Inf 做保护，避免 float32 溢出或策略输入异常
    safe_values = []
    for v in obs_dict.values():
        if v is None:
            v = 0.0
        v = float(v)
        if not np.isfinite(v):
            v = 0.0
        # 限制在 float32 安全范围内，保留足够动态范围
        v = float(np.clip(v, -1e6, 1e6))
        safe_values.append(v)
    obs_vec = np.array(safe_values, dtype=np.float32)
    if return_feature_names:
        return obs_vec, list(obs_dict.keys())
    return obs_vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_position(state):
    pos = state.get("position_neu")
    if pos is None:
        pos = state.get("position_m")
    if pos is None:
        raise ValueError("State missing position field (position_neu or position_m)")
    return np.asarray(pos, dtype=np.float64)


def _get_velocity(state):
    """Extract velocity in NEU frame.

    Priority:
      1. velocity_vector_mps (assumed NEU: [vn, ve, vu])
      2. velocity_ned (converted to NEU: [vn, ve, -vd])
      3. velocity (assumed NEU)
    """
    vel = state.get("velocity_vector_mps")
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"velocity_vector_mps must be a 3-element vector, got shape {arr.shape}"
            )
        return arr

    vel_ned = state.get("velocity_ned")
    if vel_ned is not None:
        v = np.asarray(vel_ned, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(
                f"velocity_ned must be a 3-element vector, got shape {v.shape}"
            )
        return np.array([v[0], v[1], -v[2]], dtype=np.float64)

    vel = state.get("velocity")
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"velocity must be a 3-element vector, got shape {arr.shape}"
            )
        return arr

    raise ValueError(
        "State missing velocity field (velocity_vector_mps, velocity_ned, or velocity)"
    )


def _get_altitude(state):
    alt = state.get("altitude_m")
    if alt is not None:
        return float(alt)
    pos = _get_position(state)
    return float(pos[2])
