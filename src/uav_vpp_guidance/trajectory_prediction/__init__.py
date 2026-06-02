"""
目标轨迹预测模块 (Trajectory Prediction Module)

为虚拟追踪点生成器提供目标未来位置预测能力，
将虚拟追踪点锚点从目标当前位置升级为目标预测未来位置。
"""

from .state_buffer import TrajectoryStateBuffer
from .base_predictor import BaseTrajectoryPredictor
from .constant_velocity import ConstantVelocityPredictor
from .constant_acceleration import ConstantAccelerationPredictor
from .lstm_predictor import LSTMTrajectoryPredictor
from .gru_predictor import GRUTrajectoryPredictor
from .predictor_adapter import TrajectoryPredictorAdapter, create_predictor_from_config, create_state_buffer_from_config
from .feature_builder import build_target_prediction_feature

__all__ = [
    "TrajectoryStateBuffer",
    "BaseTrajectoryPredictor",
    "ConstantVelocityPredictor",
    "ConstantAccelerationPredictor",
    "LSTMTrajectoryPredictor",
    "GRUTrajectoryPredictor",
    "TrajectoryPredictorAdapter",
    "create_predictor_from_config",
    "create_state_buffer_from_config",
    "build_target_prediction_feature",
]
