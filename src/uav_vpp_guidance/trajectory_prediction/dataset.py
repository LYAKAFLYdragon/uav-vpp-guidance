"""
轨迹预测监督学习数据集。

每个样本：
  输入：过去 K 帧目标/相对态势特征
  标签：未来 T_lookahead 时刻目标相对位移
"""

from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .feature_builder import build_target_prediction_feature
from .coordinate_utils import get_position_neu, get_velocity_neu, get_acceleration_neu


def _compute_velocity_from_position(positions: np.ndarray, dt: float) -> np.ndarray:
    """通过中心差分估算速度。"""
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    return velocities


def _compute_acceleration_from_velocity(velocities: np.ndarray, dt: float) -> np.ndarray:
    """通过中心差分估算加速度。"""
    accelerations = np.zeros_like(velocities)
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2.0 * dt)
    accelerations[0] = (velocities[1] - velocities[0]) / dt
    accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    return accelerations


def _build_states_from_trajectory(df: pd.DataFrame, dt: float) -> tuple:
    """从轨迹 DataFrame 构造 own_state、target_state、relative_state 序列。"""
    own_pos = df[["ego_x", "ego_y", "ego_z"]].to_numpy(dtype=np.float64)
    target_pos = df[["target_x", "target_y", "target_z"]].to_numpy(dtype=np.float64)

    own_vel = _compute_velocity_from_position(own_pos, dt)
    target_vel = _compute_velocity_from_position(target_pos, dt)

    own_acc = _compute_acceleration_from_velocity(own_vel, dt)
    target_acc = _compute_acceleration_from_velocity(target_vel, dt)

    rel_pos = target_pos - own_pos
    rel_vel = target_vel - own_vel

    # 从速度估算航向角（简化）
    heading = np.arctan2(own_vel[:, 1], own_vel[:, 0])
    pitch = np.where(
        np.linalg.norm(own_vel[:, :2], axis=1) > 1e-6,
        np.arctan2(own_vel[:, 2], np.linalg.norm(own_vel[:, :2], axis=1)),
        0.0,
    )

    target_heading = np.arctan2(target_vel[:, 1], target_vel[:, 0])
    target_pitch = np.where(
        np.linalg.norm(target_vel[:, :2], axis=1) > 1e-6,
        np.arctan2(target_vel[:, 2], np.linalg.norm(target_vel[:, :2], axis=1)),
        0.0,
    )

    # range_m 优先从 CSV 读取，否则计算
    if "range_m" in df.columns:
        range_m = df["range_m"].to_numpy(dtype=np.float64)
    else:
        range_m = np.linalg.norm(rel_pos, axis=1)

    n = len(df)
    own_states = []
    target_states = []
    relative_states = []

    for i in range(n):
        own_states.append(
            {
                "position_m": own_pos[i],
                "velocity_vector_mps": own_vel[i],
                "acceleration_vector_mps2": own_acc[i],
                "attitude_rpy": np.array([0.0, pitch[i], heading[i]], dtype=np.float64),
                "nz": 0.0,
            }
        )
        target_states.append(
            {
                "position_m": target_pos[i],
                "velocity_vector_mps": target_vel[i],
                "acceleration_vector_mps2": target_acc[i],
                "attitude_rpy": np.array(
                    [0.0, target_pitch[i], target_heading[i]], dtype=np.float64
                ),
                "nz": 0.0,
            }
        )
        relative_states.append(
            {
                "range_m": float(range_m[i]),
                "relative_velocity": rel_vel[i],
            }
        )

    return own_states, target_states, relative_states


def build_lstm_prediction_feature(own_state, target_state, relative_state, config) -> np.ndarray:
    """
    构造 LSTM 轨迹预测所需的 16 维特征向量。

    特征组成（按顺序）：
      1-3.  目标相对本机的位置 (NEU)
      4-6.  相对速度
      7-9.  目标速度
      10-12. 目标加速度
      13.   相对距离
      14.   距离变化率
      15.   高度差
      16.   目标法向过载

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
    acc_scale = norm_cfg.get("acceleration_scale_mps2", 50.0)
    overload_scale = norm_cfg.get("overload_scale", 9.0)

    own_pos = get_position_neu(own_state)
    target_pos = get_position_neu(target_state)
    rel_pos = target_pos - own_pos

    own_vel = get_velocity_neu(own_state)
    target_vel = get_velocity_neu(target_state)
    rel_vel = target_vel - own_vel

    target_acc = get_acceleration_neu(target_state)
    if target_acc is None:
        target_acc = np.zeros(3, dtype=np.float64)

    distance = relative_state.get("range_m")
    if distance is None:
        distance = float(np.linalg.norm(rel_pos))

    rel_pos_norm = rel_pos / (np.linalg.norm(rel_pos) + 1e-8)
    range_rate = float(np.dot(rel_vel, rel_pos_norm))

    altitude_diff = target_pos[2] - own_pos[2]
    target_nz = target_state.get("nz", 0.0)

    feature = np.array([
        rel_pos[0] / pos_scale,
        rel_pos[1] / pos_scale,
        rel_pos[2] / pos_scale,
        rel_vel[0] / vel_scale,
        rel_vel[1] / vel_scale,
        rel_vel[2] / vel_scale,
        target_vel[0] / vel_scale,
        target_vel[1] / vel_scale,
        target_vel[2] / vel_scale,
        target_acc[0] / acc_scale,
        target_acc[1] / acc_scale,
        target_acc[2] / acc_scale,
        distance / pos_scale,
        range_rate / vel_scale,
        altitude_diff / pos_scale,
        target_nz / overload_scale,
    ], dtype=np.float32)

    return feature


class TrajectoryPredictionDataset(Dataset):
    """
    用于目标轨迹预测模型的监督学习数据集。
    """

    def __init__(self, samples):
        """
        Args:
            samples (list): 样本列表，每个样本为 (history_seq, target) 元组。
                history_seq: np.ndarray, shape [history_len, feature_dim]
                target: np.ndarray, shape [3] (未来相对位移)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history_seq, target = self.samples[idx]
        return (
            torch.from_numpy(history_seq).float(),
            torch.from_numpy(target).float(),
        )

    @classmethod
    def from_episode_logs(
        cls,
        episode_logs: Union[List[str], List[pd.DataFrame], List[List[dict]]],
        config: dict,
    ):
        """
        从 episode 日志构建数据集。

        支持三种输入格式：
          - list of str: CSV 文件路径列表
          - list of pd.DataFrame: 已加载的轨迹 DataFrame 列表
          - list of list of dict: 轨迹记录字典列表的列表

        每行/每条记录至少需要包含以下列/键：
          ego_x, ego_y, ego_z, target_x, target_y, target_z, time（或 step）
        可选列：range_m

        Args:
            episode_logs: episode 轨迹数据。
            config (dict): 配置参数，需包含：
                - history_len (int): 历史序列长度
                - prediction.lookahead_time_s (float): 预测前瞻时间
                - env.high_level_dt (float, optional): 决策时间步长，默认 0.2
                - normalization (dict): feature_builder 所需的归一化参数

        Returns:
            TrajectoryPredictionDataset
        """
        history_len = config.get("history", {}).get("history_len", 10)
        pred_cfg = config.get("prediction", {})
        lookahead_time_s = pred_cfg.get("lookahead_time_s", 1.0)
        high_level_dt = config.get("env", {}).get("high_level_dt", 0.2)
        lookahead_steps = max(1, int(round(lookahead_time_s / high_level_dt)))

        samples = []

        for episode in episode_logs:
            # 统一加载为 DataFrame
            if isinstance(episode, str):
                df = pd.read_csv(episode)
            elif isinstance(episode, pd.DataFrame):
                df = episode.copy()
            elif isinstance(episode, list):
                df = pd.DataFrame(episode)
            else:
                raise TypeError(
                    f"Unsupported episode type: {type(episode)}. "
                    "Expected str (path), pd.DataFrame, or list of dict."
                )

            if len(df) < history_len + lookahead_steps:
                continue

            # 确定时间步长
            if "time" in df.columns:
                dt = float(df["time"].iloc[1] - df["time"].iloc[0])
            else:
                dt = high_level_dt

            own_states, target_states, relative_states = _build_states_from_trajectory(
                df, dt
            )

            # 预计算所有时刻的特征
            features = []
            for i in range(len(df)):
                feat = build_target_prediction_feature(
                    own_states[i], target_states[i], relative_states[i], config
                )
                features.append(feat)
            features = np.stack(features, axis=0)  # [T, feature_dim]

            # 提取目标位置用于计算相对位移标签
            target_pos = df[["target_x", "target_y", "target_z"]].to_numpy(
                dtype=np.float64
            )

            # 滑窗构造样本
            for t in range(history_len - 1, len(df) - lookahead_steps):
                history_seq = features[t - history_len + 1 : t + 1]
                future_pos = target_pos[t + lookahead_steps]
                current_pos = target_pos[t]
                target_disp = future_pos - current_pos

                samples.append((history_seq, target_disp))

        if not samples:
            raise ValueError(
                "No valid samples constructed from episode logs. "
                "Check that episodes are long enough (>= history_len + lookahead_steps)."
            )

        return cls(samples)

    @classmethod
    def from_tracking_env(
        cls,
        env,
        num_episodes: int = 100,
        max_steps_per_episode: int = 512,
        history_len: int = 10,
        prediction_horizon: int = 5,
        config: dict = None,
        feature_builder=None,
        seed: int = None,
    ):
        """
        从 CloseRangeTrackingEnv 运行 episode 收集训练数据。

        通过运行 env 采集目标轨迹，利用滑窗构造 (history_seq, future_disp) 样本。
        适合在训练前批量生成 LSTM/GRU 预测器的训练集。

        Args:
            env: CloseRangeTrackingEnv 实例。
            num_episodes (int): 采集 episode 数量。
            max_steps_per_episode (int): 每 episode 最大步数。
            history_len (int): 历史序列长度（默认 10）。
            prediction_horizon (int): 预测前瞻步数（默认 5）。
            config (dict): 配置字典，需包含 normalization 参数。
                若未提供，使用空 dict（此时 feature_builder 需自行处理默认值）。
            feature_builder (callable): 特征构造函数，签名为
                (own_state, target_state, relative_state, config) -> np.ndarray。
                默认使用 build_lstm_prediction_feature（16 维：相对位置、速度、加速度）。
            seed (int, optional): 随机种子，用于 env.reset() 的场景采样。

        Returns:
            TrajectoryPredictionDataset
        """
        if config is None:
            config = {}
        if feature_builder is None:
            feature_builder = build_lstm_prediction_feature

        high_level_dt = config.get("env", {}).get("high_level_dt", 0.2)
        samples = []

        rng = np.random.default_rng(seed)

        for ep in range(num_episodes):
            episode_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None
            obs = env.reset(seed=episode_seed)

            own_states_ep = []
            target_states_ep = []
            relative_states_ep = []

            for _ in range(max_steps_per_episode):
                own_state = obs.get("own_state")
                target_state = obs.get("target_state")
                rel_state = obs.get("relative_state")

                if own_state is None or target_state is None or rel_state is None:
                    break

                own_states_ep.append(own_state)
                target_states_ep.append(target_state)
                relative_states_ep.append(rel_state)

                # 零动作采集，目标按自身动力学运动
                action = np.zeros(3, dtype=np.float64)
                obs, _reward, terminated, truncated, _info = env.step(action)

                if terminated or truncated:
                    break

            if len(own_states_ep) < history_len + prediction_horizon:
                continue

            # 从位置序列重新计算速度和加速度（保证数值一致性）
            own_positions = np.stack(
                [get_position_neu(s) for s in own_states_ep]
            )
            target_positions = np.stack(
                [get_position_neu(s) for s in target_states_ep]
            )

            own_velocities = _compute_velocity_from_position(own_positions, high_level_dt)
            target_velocities = _compute_velocity_from_position(target_positions, high_level_dt)
            own_accelerations = _compute_acceleration_from_velocity(own_velocities, high_level_dt)
            target_accelerations = _compute_acceleration_from_velocity(target_velocities, high_level_dt)

            # 更新状态字典中的速度/加速度字段
            for i in range(len(own_states_ep)):
                own_states_ep[i]["velocity_vector_mps"] = own_velocities[i]
                own_states_ep[i]["acceleration_vector_mps2"] = own_accelerations[i]
                target_states_ep[i]["velocity_vector_mps"] = target_velocities[i]
                target_states_ep[i]["acceleration_vector_mps2"] = target_accelerations[i]

            # 构造特征序列
            features = []
            for i in range(len(own_states_ep)):
                feat = feature_builder(
                    own_states_ep[i],
                    target_states_ep[i],
                    relative_states_ep[i],
                    config,
                )
                features.append(feat)
            features = np.stack(features, axis=0)  # [T, feature_dim]

            # 滑窗构造样本
            for t in range(history_len - 1, len(own_states_ep) - prediction_horizon):
                history_seq = features[t - history_len + 1 : t + 1]
                future_pos = target_positions[t + prediction_horizon]
                current_pos = target_positions[t]
                target_disp = future_pos - current_pos

                samples.append((history_seq, target_disp))

        if not samples:
            raise ValueError(
                "No valid samples constructed from tracking_env. "
                f"Check that episodes are long enough (>= {history_len + prediction_horizon} steps)."
            )

        return cls(samples)
