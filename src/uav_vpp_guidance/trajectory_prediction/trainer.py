"""
轨迹预测模型独立训练器骨架。

用于在收集到的 episode 数据上监督训练 LSTM/GRU 预测模型。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TrajectoryPredictorTrainer:
    """
    独立训练目标轨迹预测模型。
    """

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model (nn.Module): 轨迹预测模型（LSTM 或 GRU）。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            config (dict): 训练配置（learning_rate, epochs, device 等）。
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.get("device", "cpu"))
        self.model.to(self.device)

        self.lr = config.get("learning_rate", 1.0e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train_one_epoch(self):
        """
        训练一个 epoch。

        TODO: 实现完整的训练循环。
        """
        raise NotImplementedError("train_one_epoch is not yet implemented.")

    def validate(self):
        """
        在验证集上评估。

        TODO: 实现完整的验证循环。
        """
        raise NotImplementedError("validate is not yet implemented.")

    def fit(self):
        """
        完整训练流程。

        TODO: 实现多 epoch 训练和早停逻辑。
        """
        raise NotImplementedError("fit is not yet implemented.")
