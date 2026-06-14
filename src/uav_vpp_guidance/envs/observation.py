"""
Observation construction utilities.

Self-contained JSBSim integration.

第一版使用简化几何计算，支持 NEU 坐标系下的相对态势特征提取。
"""

from collections import deque
import numpy as np


def _stable_angle_diff(a, b):
    """Signed smallest angle difference in radians."""
    delta = a - b
    if not np.isfinite(delta):
        return float(delta)
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


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
    scenario_type=None,
    prev_rel_state=None,
    dt=0.2,
    include_gains=False,
    include_vp_error=False,
    include_scenario=False,
    include_temporal=False,
):
    """
    Build policy observation.

    Supports optional enhancements:
    - relative geometry
    - own/target speed and altitude
    - temporal rates (range acceleration, ATA/AA rates)
    - virtual point tracking error
    - current guidance gains
    - scenario type one-hot encoding

    Args:
        own_state (dict): Own aircraft state.
        target_state (dict): Target aircraft state.
        guidance_state (dict, optional): Guidance internal state (e.g. vp_tracking_error).
        gains (GuidanceGains, optional): Current guidance gains.
        scenario_type (str, optional): Scenario type name (e.g. 'favorable').
        prev_rel_state (dict, optional): Previous relative geometry for temporal features.
        dt (float): Time step for temporal derivative computation.
        include_gains (bool): Include guidance gains in observation.
        include_vp_error (bool): Include VPP tracking error in observation.
        include_scenario (bool): Include scenario one-hot encoding in observation.
        include_temporal (bool): Include temporal derivative features.

    Returns:
        np.ndarray: Flattened observation vector.
    """
    rel = compute_relative_geometry(own_state, target_state)

    own_speed = float(np.linalg.norm(_get_velocity(own_state)))
    target_speed = float(np.linalg.norm(_get_velocity(target_state)))
    own_alt = _get_altitude(own_state)
    target_alt = _get_altitude(target_state)

    # 归一化参考值
    # ref_range 从 5000 改为 3000，避免近距离场景（800-2000m）信号过度压缩。
    ref_range = 3000.0
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

    # 可选：时序特征。即使缺少历史也保留槽位，保证 reset/step 观察维度一致。
    if include_temporal:
        if prev_rel_state is not None:
            obs_dict["range_rate_delta"] = (
                rel["range_rate_mps"] - prev_rel_state["range_rate_mps"]
            ) / dt / ref_range_rate
            obs_dict["ata_rate"] = _stable_angle_diff(
                rel["ata_rad"], prev_rel_state["ata_rad"]
            ) / dt
            obs_dict["aa_rate"] = _stable_angle_diff(
                rel["aa_rad"], prev_rel_state["aa_rad"]
            ) / dt
        else:
            obs_dict["range_rate_delta"] = 0.0
            obs_dict["ata_rate"] = 0.0
            obs_dict["aa_rate"] = 0.0

    # 可选：guidance gains
    if include_gains and gains is not None:
        try:
            obs_dict["gain_k_los"] = getattr(gains, "k_los", 1.0)
            obs_dict["gain_k_pos"] = getattr(gains, "k_pos", 0.5)
        except Exception:
            pass

    # 可选：guidance state（虚拟点跟踪误差等）。缺少时补零，保持维度一致。
    if include_vp_error:
        if guidance_state is not None:
            vp_error = guidance_state.get("vp_tracking_error", np.zeros(3))
            obs_dict["vp_error_x"] = vp_error[0] / ref_range
            obs_dict["vp_error_y"] = vp_error[1] / ref_range
            obs_dict["vp_error_z"] = vp_error[2] / ref_alt
        else:
            obs_dict["vp_error_x"] = 0.0
            obs_dict["vp_error_y"] = 0.0
            obs_dict["vp_error_z"] = 0.0

    # 可选：场景类型 one-hot 编码
    if include_scenario and scenario_type is not None:
        scenario_types = ["favorable", "neutral", "disadvantage", "challenging"]
        for st in scenario_types:
            obs_dict[f"scenario_{st}"] = 1.0 if scenario_type == st else 0.0

    # 展平为向量
    obs_vec = np.array(list(obs_dict.values()), dtype=np.float32)
    return obs_vec


class ObservationBuilder:
    """
    Stateful observation builder with temporal history caching.

    This wrapper is useful when the policy needs access to historical relative
    geometry (e.g. target turn rate, range acceleration). The history is reset
    via :meth:`reset` at the beginning of each episode.
    """

    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): Observation configuration. Expected keys:
                observation:
                  temporal:
                    enabled: bool
                    history_length: int (default 2)
                    dt: float (default 0.2)
                  include_gains: bool
                  include_vp_error: bool
                  include_scenario: bool
        """
        self.config = config or {}
        self.obs_config = self.config.get("observation", {})
        temporal_cfg = self.obs_config.get("temporal", {})
        self.include_temporal = bool(temporal_cfg.get("enabled", False))
        self.history_length = int(temporal_cfg.get("history_length", 2))
        self.dt = float(temporal_cfg.get("dt", 0.2))
        self.include_gains = bool(self.obs_config.get("include_gains", False))
        self.include_vp_error = bool(self.obs_config.get("include_vp_error", False))
        self.include_scenario = bool(self.obs_config.get("include_scenario", False))
        self._history = deque(maxlen=max(1, self.history_length))

    def reset(self):
        """Clear history (call at episode reset)."""
        self._history.clear()

    def build(
        self,
        own_state,
        target_state,
        guidance_state=None,
        gains=None,
        scenario_type=None,
    ):
        """
        Build observation with optional temporal features.

        Args:
            own_state (dict): Own aircraft state.
            target_state (dict): Target aircraft state.
            guidance_state (dict, optional): Guidance internal state.
            gains (GuidanceGains, optional): Current guidance gains.
            scenario_type (str, optional): Scenario type name.

        Returns:
            np.ndarray: Flattened observation vector.
        """
        rel = compute_relative_geometry(own_state, target_state)
        prev_rel = self._history[-1] if self._history else None
        self._history.append(rel)

        return build_observation(
            own_state,
            target_state,
            guidance_state=guidance_state,
            gains=gains,
            scenario_type=scenario_type,
            prev_rel_state=prev_rel,
            dt=self.dt,
            include_gains=self.include_gains,
            include_vp_error=self.include_vp_error,
            include_scenario=self.include_scenario,
            include_temporal=self.include_temporal,
        )


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
