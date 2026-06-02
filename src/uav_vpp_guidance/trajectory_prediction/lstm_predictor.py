"""
LSTM 目标轨迹预测模型。

输入：目标历史状态序列 [batch, history_len, input_dim]
输出：目标未来相对当前位置的位移 [batch, 3]

预测的是未来位移，后续需要叠加到目标当前位置。
"""

import numpy as np
import torch
import torch.nn as nn
from .base_predictor import BaseTrajectoryPredictor


class LSTMTrajectoryPredictor(nn.Module, BaseTrajectoryPredictor):
    """
    基于 LSTM 的时序轨迹预测模型。

    使用最后一层最后一个时间步的 hidden state，
    经过 MLP head 映射为 3 维相对位移（或均值+方差）。
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_variance: bool = False,
    ):
        """
        Args:
            input_dim (int): 输入特征维度。
            hidden_dim (int): LSTM 隐藏层维度。
            num_layers (int): LSTM 层数。
            dropout (float): Dropout 概率。
            predict_variance (bool): 是否同时预测位移方差。
        """
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_variance = predict_variance

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_dim = 6 if predict_variance else 3
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, history_seq: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            history_seq (torch.Tensor): [batch, history_len, input_dim]

        Returns:
            torch.Tensor: [batch, 3] 相对位移均值；若 predict_variance=True，
                输出 [batch, 6]，前 3 维为均值，后 3 维为 log-variance。
        """
        # lstm_out: [batch, history_len, hidden_dim]
        # hidden: (h_n, c_n), h_n shape [num_layers, batch, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(history_seq)

        # 取最后一层最后一个时间步的 hidden state
        # h_n[-1]: [batch, hidden_dim]
        last_hidden = h_n[-1]
        output = self.head(last_hidden)
        return output

    def predict(self, history_seq, current_target_state=None):
        """
        预测接口，支持 numpy 输入并自动转 torch。

        Args:
            history_seq (np.ndarray or torch.Tensor): [batch, history_len, input_dim] 或 [history_len, input_dim]。
            current_target_state: 不使用，为接口兼容保留。

        Returns:
            tuple: (pred_disp, pred_var, info)
                pred_disp (np.ndarray): [batch, 3] 或 [3]。
                pred_var (np.ndarray or None): 位移方差。
                info (dict): 附加信息。
        """
        self.eval()
        with torch.no_grad():
            if isinstance(history_seq, np.ndarray):
                history_seq = torch.from_numpy(history_seq).float()
            if history_seq.dim() == 2:
                history_seq = history_seq.unsqueeze(0)

            device = next(self.parameters()).device
            history_seq = history_seq.to(device)
            output = self.forward(history_seq)
            output = output.cpu().numpy()

        pred_disp = output[:, :3]
        if self.predict_variance:
            # 使用 softplus 保证方差为正
            log_var = output[:, 3:6]
            pred_var = np.log(1 + np.exp(log_var))
        else:
            pred_var = None

        # 若输入是单样本，去掉 batch 维度
        if pred_disp.shape[0] == 1:
            pred_disp = pred_disp[0]
            if pred_var is not None:
                pred_var = pred_var[0]

        info = {"model": "lstm", "fallback": False}
        return pred_disp, pred_var, info
