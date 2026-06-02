"""
轨迹预测模型损失函数。
"""

import torch
import torch.nn.functional as F


def position_mse_loss(pred_pos, target_pos):
    """
    预测位置与目标位置的均方误差。

    Args:
        pred_pos (torch.Tensor): [batch, 3]
        target_pos (torch.Tensor): [batch, 3]

    Returns:
        torch.Tensor: scalar loss
    """
    return F.mse_loss(pred_pos, target_pos)


def relative_displacement_mse_loss(pred_disp, target_disp):
    """
    预测相对位移与目标相对位移的均方误差。

    Args:
        pred_disp (torch.Tensor): [batch, 3]
        target_disp (torch.Tensor): [batch, 3]

    Returns:
        torch.Tensor: scalar loss
    """
    return F.mse_loss(pred_disp, target_disp)


def gaussian_nll_loss(pred_mean, pred_var, target_pos):
    """
    高斯负对数似然损失，用于同时学习均值和不确定性。

    Args:
        pred_mean (torch.Tensor): [batch, 3]
        pred_var (torch.Tensor): [batch, 3]，必须为正。
        target_pos (torch.Tensor): [batch, 3]

    Returns:
        torch.Tensor: scalar loss
    """
    # log(2*pi*var)/2 + (target - mean)^2 / (2*var)
    loss = 0.5 * torch.log(2 * torch.pi * pred_var) + (target_pos - pred_mean) ** 2 / (2 * pred_var)
    return loss.mean()
