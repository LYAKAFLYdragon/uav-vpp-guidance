"""
轨迹预测历史状态滑动窗口缓冲区。

固定长度历史窗口用于为轨迹预测模型提供时序输入。
"""

import numpy as np
from collections import deque
from typing import Optional


class TrajectoryStateBuffer:
    """
    维护固定长度的目标历史状态序列。

    使用 collections.deque(maxlen=history_len) 实现自动滑动窗口，
    历史不足时根据 padding_mode 进行填充处理。
    """

    def __init__(self, history_len: int, feature_dim: int, padding_mode: str = "repeat_first"):
        """
        Args:
            history_len (int): 历史序列长度（时间步数）。
            feature_dim (int): 单个时间步的特征维度。
            padding_mode (str): 填充模式，"repeat_first" 或 "zero"。
        """
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.padding_mode = padding_mode
        self._buffer: deque = deque(maxlen=history_len)

    def reset(self):
        """清空缓冲区。"""
        self._buffer.clear()

    def push(self, feature: np.ndarray):
        """
        将新一帧特征推入缓冲区。

        Args:
            feature (np.ndarray): 特征向量，shape 为 [feature_dim] 或可广播为一维向量。
        """
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, got {feature.shape[0]}"
            )
        self._buffer.append(feature)

    def is_ready(self) -> bool:
        """
        判断缓冲区是否已积累足够的历史帧数。

        Returns:
            bool: True 当且仅当缓冲区长度达到 history_len。
        """
        return len(self._buffer) >= self.history_len

    def get_sequence(self) -> np.ndarray:
        """
        获取固定长度的历史序列。

        如果缓冲区为空，抛出 ValueError。
        如果缓冲区未满，根据 padding_mode 进行填充：
          - "repeat_first": 用第一帧重复填充左侧。
          - "zero": 用零向量填充左侧。

        Returns:
            np.ndarray: shape 为 [history_len, feature_dim]。
        """
        if len(self._buffer) == 0:
            # Return zero-padded sequence when buffer is empty.
            # This allows predictors that do not require history (e.g. CV)
            # to operate before any push() calls.
            return np.zeros((self.history_len, self.feature_dim), dtype=np.float32)

        seq = np.array(list(self._buffer), dtype=np.float32)
        current_len = seq.shape[0]

        if current_len < self.history_len:
            pad_len = self.history_len - current_len
            if self.padding_mode == "repeat_first":
                pad = np.tile(seq[0:1, :], (pad_len, 1))
            elif self.padding_mode == "zero":
                pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            else:
                raise ValueError(f"Unknown padding_mode: {self.padding_mode}")
            seq = np.concatenate([pad, seq], axis=0)

        return seq
