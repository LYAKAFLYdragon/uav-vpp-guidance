"""
轨迹预测监督学习数据集骨架。

每个样本：
  输入：过去 K 帧目标/相对态势特征
  标签：未来 T_lookahead 时刻目标位置或位移
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryPredictionDataset(Dataset):
    """
    用于目标轨迹预测模型的监督学习数据集。
    """

    def __init__(self, samples):
        """
        Args:
            samples (list): 样本列表，每个样本为 (history_seq, target) 元组。
                history_seq: np.ndarray, shape [history_len, feature_dim]
                target: np.ndarray, shape [3] (未来位置或位移)
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
    def from_episode_logs(cls, episode_logs, config):
        """
        从 episode 日志构建数据集。

        TODO: 实现从仿真记录中构造 (history_seq, target) 样本的逻辑。

        Args:
            episode_logs (list): 多个 episode 的状态序列记录。
            config (dict): 配置参数（history_len, lookahead_time_s 等）。

        Returns:
            TrajectoryPredictionDataset
        """
        raise NotImplementedError("from_episode_logs is not yet implemented.")
