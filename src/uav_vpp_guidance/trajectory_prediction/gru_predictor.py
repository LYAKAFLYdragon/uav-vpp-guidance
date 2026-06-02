"""
GRU 目标轨迹预测模型。

接口与 LSTM 完全一致，仅将核心网络替换为 nn.GRU。
"""

import numpy as np
import torch
import torch.nn as nn
from .base_predictor import BaseTrajectoryPredictor


class GRUTrajectoryPredictor(nn.Module, BaseTrajectoryPredictor):
    """
    基于 GRU 的时序轨迹预测模型。

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
            hidden_dim (int): GRU 隐藏层维度。
            num_layers (int): GRU 层数。
            dropout (float): Dropout 概率。
            predict_variance (bool): 是否同时预测位移方差。
        """
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_variance = predict_variance

        self.gru = nn.GRU(
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
            torch.Tensor: [batch, 3] 或 [batch, 6]。
        """
        gru_out, h_n = self.gru(history_seq)
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
            log_var = output[:, 3:6]
            pred_var = np.log(1 + np.exp(log_var))
        else:
            pred_var = None

        if pred_disp.shape[0] == 1:
            pred_disp = pred_disp[0]
            if pred_var is not None:
                pred_var = pred_var[0]

        info = {"model": "gru", "fallback": False}
        return pred_disp, pred_var, info
