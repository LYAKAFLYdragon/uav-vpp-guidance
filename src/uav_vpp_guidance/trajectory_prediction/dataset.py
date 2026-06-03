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


def _compute_velocity_from_position(positions: np.ndarray, dt: float) -> np.ndarray:
    """通过中心差分估算速度。"""
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    return velocities


def _build_states_from_trajectory(df: pd.DataFrame, dt: float) -> tuple:
    """从轨迹 DataFrame 构造 own_state、target_state、relative_state 序列。"""
    own_pos = df[["ego_x", "ego_y", "ego_z"]].to_numpy(dtype=np.float64)
    target_pos = df[["target_x", "target_y", "target_z"]].to_numpy(dtype=np.float64)

    own_vel = _compute_velocity_from_position(own_pos, dt)
    target_vel = _compute_velocity_from_position(target_pos, dt)

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
                "attitude_rpy": np.array([0.0, pitch[i], heading[i]], dtype=np.float64),
                "nz": 0.0,
            }
        )
        target_states.append(
            {
                "position_m": target_pos[i],
                "velocity_vector_mps": target_vel[i],
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
