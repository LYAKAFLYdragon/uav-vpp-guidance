"""
预测模型适配器。

连接历史状态缓冲区、特征构造器和预测模型，
为虚拟追踪点生成器提供统一的目标未来位置预测接口。

支持 predictor_type:
- constant_velocity
- constant_acceleration
- lstm
- gru
"""

import os

import numpy as np
import torch
from typing import Optional, Tuple

from .state_buffer import TrajectoryStateBuffer
from .feature_builder import build_target_prediction_feature
from .constant_velocity import ConstantVelocityPredictor
from .constant_acceleration import ConstantAccelerationPredictor
from .coordinate_utils import get_position_neu
from .device_utils import load_checkpoint_to_model, resolve_torch_device


# ---------------------------------------------------------------------------
# Fallback predictor classes
# ---------------------------------------------------------------------------

class _CurrentTargetFallback:
    """Fallback that returns the current target position (no motion extrapolation)."""

    def predict(self, history_seq=None, current_target_state=None):
        if current_target_state is None:
            return None, None, {
                "model": "current_target",
                "fallback": True,
                "fallback_reason": "current_target_state is None",
            }
        pos = get_position_neu(current_target_state)
        return pos, None, {
            "model": "current_target",
            "fallback": False,
            "output_is_absolute": True,
        }


# ---------------------------------------------------------------------------
# Fallback predictor factory
# ---------------------------------------------------------------------------

_FALLBACK_REGISTRY = {
    "constant_velocity": ConstantVelocityPredictor,
    "constant_acceleration": ConstantAccelerationPredictor,
    "current_target": _CurrentTargetFallback,
}


def _create_fallback_predictor(fallback_mode: str, lookahead_time_s: float):
    """根据 fallback_mode 创建对应的 fallback predictor。"""
    if fallback_mode == "none":
        return None
    cls = _FALLBACK_REGISTRY.get(fallback_mode)
    if cls is None:
        raise ValueError(
            f"Unknown fallback_mode: {fallback_mode}. "
            f"Supported: {list(_FALLBACK_REGISTRY.keys()) + ['none']}"
        )
    if fallback_mode == "current_target":
        return cls()
    return cls(lookahead_time_s=lookahead_time_s)


# ---------------------------------------------------------------------------
# Predictor factory
# ---------------------------------------------------------------------------

def _resolve_checkpoint_strict(config: dict) -> bool:
    """Resolve checkpoint_strict canonical key with backward-compatible alias.

    Canonical key: checkpoint_strict
    Legacy alias: strict_checkpoint

    Raises:
        ValueError: if both keys are present with different values.
    """
    canonical = config.get("checkpoint_strict")
    alias = config.get("strict_checkpoint")
    if canonical is not None and alias is not None and canonical != alias:
        raise ValueError(
            f"Conflicting checkpoint strict keys: "
            f"checkpoint_strict={canonical} != strict_checkpoint={alias}. "
            f"Please use only 'checkpoint_strict'."
        )
    if canonical is not None:
        return bool(canonical)
    if alias is not None:
        return bool(alias)
    return True


def create_predictor_from_config(config: dict):
    """
    根据配置创建对应的轨迹预测器。

    Args:
        config (dict): trajectory_prediction 配置字典。

    Returns:
        BaseTrajectoryPredictor: 预测器实例。
    """
    predictor_type = config.get("predictor_type", "constant_velocity")
    pred_cfg = config.get("prediction", {})
    lookahead_time_s = pred_cfg.get("lookahead_time_s", 1.0)

    if predictor_type == "constant_velocity":
        return ConstantVelocityPredictor(lookahead_time_s=lookahead_time_s)
    elif predictor_type == "constant_acceleration":
        return ConstantAccelerationPredictor(lookahead_time_s=lookahead_time_s)
    elif predictor_type == "lstm":
        from .lstm_predictor import LSTMTrajectoryPredictor

        model_cfg = config.get("model", {})
        model = LSTMTrajectoryPredictor(
            input_dim=model_cfg.get("input_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            predict_variance=model_cfg.get("predict_variance", False),
        )
        ckpt = config.get("checkpoint_path")
        if ckpt:
            device_str = config.get("device", "cpu")
            allow_fallback = config.get("allow_device_fallback", True)
            strict = _resolve_checkpoint_strict(config)
            load_checkpoint_to_model(
                model,
                ckpt,
                device_str=device_str,
                allow_device_fallback=allow_fallback,
                strict=strict,
            )
        return model
    elif predictor_type == "gru":
        from .gru_predictor import GRUTrajectoryPredictor

        model_cfg = config.get("model", {})
        model = GRUTrajectoryPredictor(
            input_dim=model_cfg.get("input_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            predict_variance=model_cfg.get("predict_variance", False),
        )
        ckpt = config.get("checkpoint_path")
        if ckpt:
            device_str = config.get("device", "cpu")
            allow_fallback = config.get("allow_device_fallback", True)
            strict = _resolve_checkpoint_strict(config)
            load_checkpoint_to_model(
                model,
                ckpt,
                device_str=device_str,
                allow_device_fallback=allow_fallback,
                strict=strict,
            )
        return model
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")


def create_state_buffer_from_config(config: dict):
    """
    根据配置创建历史状态缓冲区。

    Args:
        config (dict): trajectory_prediction 配置字典。

    Returns:
        TrajectoryStateBuffer: 状态缓冲区实例。
    """
    history_cfg = config.get("history", {})
    history_len = history_cfg.get("history_len", 10)
    padding_mode = history_cfg.get("padding_mode", "repeat_first")

    # feature_dim 由 feature_builder 决定，当前固定为 16
    feature_dim = config.get("model", {}).get("input_dim", 16)

    return TrajectoryStateBuffer(
        history_len=history_len,
        feature_dim=feature_dim,
        padding_mode=padding_mode,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TrajectoryPredictorAdapter:
    """
    连接历史状态缓冲区、特征构造器和预测模型的适配器。
    """

    def __init__(self, predictor, state_buffer: TrajectoryStateBuffer, config: dict):
        """
        Args:
            predictor (BaseTrajectoryPredictor): 轨迹预测模型实例。
            state_buffer (TrajectoryStateBuffer): 历史状态缓冲区。
            config (dict): 配置字典，需包含 prediction 和 integration 参数。
        """
        self.predictor = predictor
        self.state_buffer = state_buffer
        self.config = config

        pred_cfg = config.get("prediction", {})
        self.lookahead_time_s = pred_cfg.get("lookahead_time_s", 1.0)
        self.output_mode = pred_cfg.get("output_mode", "relative_displacement")
        self.fallback_mode = pred_cfg.get("fallback_mode", "constant_velocity")

        int_cfg = config.get("integration", {})
        self.anchor_mode = int_cfg.get("anchor_mode", "predicted_target")

        # 初始化 fallback 预测器
        self._fallback_predictor = _create_fallback_predictor(
            self.fallback_mode, self.lookahead_time_s
        )

    def reset(self):
        """重置缓冲区。"""
        self.state_buffer.reset()

    def update(self, own_state, target_state, relative_state):
        """
        构建当前时刻的特征向量并推入历史缓冲区。

        Args:
            own_state (dict): 本机状态。
            target_state (dict): 目标状态。
            relative_state (dict): 相对态势。
        """
        feature = build_target_prediction_feature(
            own_state, target_state, relative_state, self.config
        )
        self.state_buffer.push(feature)

    def predict(
        self, current_target_state
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """
        预测目标未来位置。

        若预测器为 LSTM/GRU（输出相对位移），则：
            pred_pos = current_target_pos + pred_relative_disp
        若预测器为 ConstantVelocityPredictor / ConstantAccelerationPredictor，
        则直接返回其输出（绝对位置）。

        若历史不足或模型调用失败，回退到 fallback 模式。

        Args:
            current_target_state (dict): 目标当前状态，需包含位置信息。

        Returns:
            tuple: (pred_pos, pred_var, info)
                pred_pos (np.ndarray): 预测的未来目标位置 [3]。
                pred_var (np.ndarray or None): 预测方差。
                info (dict): 包含 anchor_mode、model_type、fallback、
                    fallback_reason、fallback_mode、prediction_valid 等。
        """
        info = {
            "anchor_mode": self.anchor_mode,
            "model_type": getattr(self.predictor, "__class__", object).__name__,
            "fallback": False,
            "fallback_reason": None,
            "fallback_mode": None,
            "fallback_phase": None,
            "fallback_model": None,
            "prediction_valid": True,
        }

        # 尝试使用主预测器
        try:
            # Neural predictors (LSTM/GRU) require a fully populated buffer.
            # Classical predictors (CV/CA) can tolerate padded short history.
            if isinstance(self.predictor, torch.nn.Module):
                if not self.state_buffer.is_ready():
                    raise ValueError(
                        f"Neural predictor requires {self.state_buffer.history_len} "
                        f"history frames, got {len(self.state_buffer._buffer)}"
                    )

            history_seq = self.state_buffer.get_sequence()
            # 增加 batch 维度 -> [1, history_len, feature_dim]
            if history_seq.ndim == 2:
                history_seq = np.expand_dims(history_seq, axis=0)

            pred_disp, pred_var, pred_info = self.predictor.predict(
                history_seq, current_target_state
            )
            info.update(pred_info)

            # 获取当前目标位置（兼容 position_neu / position_m）
            target_pos = get_position_neu(current_target_state)

            # 若 predictor 输出的是相对位移，叠加到当前位置。
            # CV/CA 经典模型直接返回绝对位置，不需要叠加。
            pred_is_absolute = pred_info.get("output_is_absolute", False)
            if (
                pred_disp is not None
                and self.output_mode == "relative_displacement"
                and not pred_is_absolute
            ):
                pred_pos = target_pos + np.asarray(pred_disp, dtype=np.float64)
            elif pred_disp is not None:
                pred_pos = np.asarray(pred_disp, dtype=np.float64)
            else:
                raise ValueError("Predictor returned None for displacement")

            return pred_pos, pred_var, info

        except Exception as exc:
            # 回退到 fallback（不覆盖主 info 的关键字段）
            info["fallback"] = True
            info["fallback_reason"] = str(exc)
            info["prediction_valid"] = False

            # Determine fallback_phase
            if isinstance(self.predictor, torch.nn.Module) and not self.state_buffer.is_ready():
                info["fallback_phase"] = "warmup"
            else:
                info["fallback_phase"] = "runtime_failure"

            if self._fallback_predictor is None:
                info["fallback_mode"] = "none"
                raise RuntimeError(
                    f"Primary predictor failed and fallback_mode is 'none'. "
                    f"Error: {exc}"
                ) from exc

            info["fallback_mode"] = self.fallback_mode
            pred_pos, pred_var, fallback_info = self._fallback_predictor.predict(
                current_target_state=current_target_state
            )
            # 只提取 fallback_info 中的非冲突诊断字段
            for key in ("fallback_model", "model_type", "model"):
                if key in fallback_info and info.get("fallback_model") is None:
                    info["fallback_model"] = fallback_info[key]
                    break
            return pred_pos, pred_var, info
