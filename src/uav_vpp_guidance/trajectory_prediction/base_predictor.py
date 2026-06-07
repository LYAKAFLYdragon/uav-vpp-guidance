"""
目标轨迹预测模型基类。

定义所有预测模型（ConstantVelocity、LSTM、GRU、后续 Transformer）
必须实现的统一接口。
"""

import torch


class BaseTrajectoryPredictor:
    """
    目标轨迹预测模型基类。

    输入：
        history_seq: [batch, history_len, feature_dim]

    输出：
        pred_pos: [batch, 3]
        pred_var: Optional[[batch, 3]]
    """

    def predict(self, history_seq, current_target_state=None):
        """
        预测目标未来位置或相对位移。

        Args:
            history_seq (np.ndarray or torch.Tensor): 历史状态序列，
                shape [batch, history_len, feature_dim] 或 [history_len, feature_dim]。
            current_target_state (dict, optional): 目标当前状态，用于某些 baseline。

        Returns:
            tuple: (pred_pos, pred_var, info)
                pred_pos (np.ndarray): 预测位置或位移，shape [batch, 3] 或 [3]。
                pred_var (np.ndarray or None): 预测方差，若不支持则返回 None。
                info (dict): 附加信息（如 fallback 标志）。
        """
        raise NotImplementedError

    def freeze(self):
        """
        冻结模型参数，使其在 RL 训练期间不更新。
        若模型继承自 torch.nn.Module，则设置 requires_grad=False。
        """
        if isinstance(self, torch.nn.Module):
            for param in self.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """
        解冻模型参数，允许在监督学习阶段更新。
        """
        if isinstance(self, torch.nn.Module):
            for param in self.parameters():
                param.requires_grad = True
