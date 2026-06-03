"""
轨迹预测模型独立训练器。

用于在收集到的 episode 数据上监督训练 LSTM/GRU 预测模型。
"""

import os
import time

import torch

from .losses import relative_displacement_mse_loss


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
        self.weight_decay = config.get("weight_decay", 1.0e-5)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # 使用相对位移 MSE 损失
        self.loss_fn = relative_displacement_mse_loss

        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 10)

        # 输出目录
        self.output_dir = config.get("output_dir", "outputs/trajectory_prediction")
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.history = {"train_loss": [], "val_loss": []}

    def train_one_epoch(self):
        """
        训练一个 epoch。

        Returns:
            float: 该 epoch 的平均训练损失。
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for history_seq, target in self.train_loader:
            history_seq = history_seq.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(history_seq)

            # 若模型输出 variance（6 维），只取前 3 维均值计算训练损失
            if pred.shape[-1] == 6:
                pred = pred[:, :3]

            loss = self.loss_fn(pred, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("grad_clip", 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def validate(self):
        """
        在验证集上评估。

        Returns:
            float: 验证集平均损失。
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for history_seq, target in self.val_loader:
                history_seq = history_seq.to(self.device)
                target = target.to(self.device)

                pred = self.model(history_seq)
                if pred.shape[-1] == 6:
                    pred = pred[:, :3]

                loss = self.loss_fn(pred, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(self):
        """
        完整训练流程。

        执行多 epoch 训练，每 epoch 结束后在验证集上评估，
        保存验证损失最低的模型权重，支持早停。

        Returns:
            dict: 训练历史，包含 train_loss 和 val_loss 列表。
        """
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # 保存最优模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1

            # 每个 epoch 都保存当前模型
            self.save_checkpoint("latest_model.pt")

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best={self.best_val_loss:.6f}@{self.best_epoch} | "
                f"patience={patience_counter}/{self.patience}"
            )

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        elapsed = time.time() - start_time
        print(
            f"Training finished in {elapsed:.1f}s. Best val_loss={self.best_val_loss:.6f} at epoch {self.best_epoch}."
        )
        return self.history

    def save_checkpoint(self, filename):
        """保存模型权重到输出目录。"""
        path = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename):
        """从输出目录加载模型权重。"""
        path = os.path.join(self.output_dir, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
