"""
虚拟追踪点生成器与轨迹预测模块的集成测试。
"""

import pytest
import numpy as np

from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator
from uav_vpp_guidance.trajectory_prediction.predictor_adapter import TrajectoryPredictorAdapter
from uav_vpp_guidance.trajectory_prediction.constant_velocity import ConstantVelocityPredictor
from uav_vpp_guidance.trajectory_prediction.state_buffer import TrajectoryStateBuffer


class TestVirtualPointPredictionIntegration:
    def _make_adapter(self, lookahead=1.0):
        config = {
            "prediction": {
                "lookahead_time_s": lookahead,
                "output_mode": "relative_displacement",
                "fallback_mode": "constant_velocity",
            },
            "integration": {
                "anchor_mode": "predicted_target",
            },
            "normalization": {
                "position_scale_m": 1000.0,
                "velocity_scale_mps": 300.0,
                "overload_scale": 9.0,
            },
        }
        predictor = ConstantVelocityPredictor(lookahead_time_s=lookahead)
        buffer = TrajectoryStateBuffer(history_len=5, feature_dim=16)
        adapter = TrajectoryPredictorAdapter(predictor, buffer, config)
        return adapter

    def test_anchor_mode_current_target(self):
        """anchor_mode=current_target 时与旧逻辑一致（当前位置作为锚点）。"""
        vp_config = {
            "action_dim": 5,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "tau_pred_range": [0.0, 3.0],
            "speed_bias_range": [-80.0, 80.0],
            "smoothing_alpha": 0.3,
        }
        gen = VirtualPointGenerator(vp_config)
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0])  # 只偏移 x
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 0.0])}

        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state, anchor_mode="current_target", return_info=True
        )

        assert info["anchor_mode"] == "current_target"
        assert np.allclose(info["anchor_pos"], target_state["position_neu"])
        assert "offset" in info

    def test_anchor_mode_constant_velocity(self):
        """anchor_mode=constant_velocity 时虚拟点锚点前移。"""
        vp_config = {
            "action_dim": 5,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "tau_pred_range": [0.0, 3.0],
            "speed_bias_range": [-80.0, 80.0],
            "smoothing_alpha": 0.3,
        }
        gen = VirtualPointGenerator(vp_config)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_ned": np.array([100.0, 0.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="constant_velocity", lookahead_time_s=2.0, return_info=True
        )

        assert info["anchor_mode"] == "constant_velocity"
        # 锚点应为 1000 + 100*2 = 1200
        assert np.allclose(info["anchor_pos"], np.array([1200.0, 0.0, 0.0]))

    def test_anchor_mode_predicted_target_with_adapter(self):
        """anchor_mode=predicted_target 时通过 adapter 获取预测位置。"""
        vp_config = {
            "action_dim": 5,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "tau_pred_range": [0.0, 3.0],
            "speed_bias_range": [-80.0, 80.0],
            "smoothing_alpha": 0.3,
        }
        gen = VirtualPointGenerator(vp_config)
        adapter = self._make_adapter(lookahead=1.0)

        # 给 adapter 填充一些历史
        for _ in range(5):
            adapter.update(
                own_state={"position_neu": np.zeros(3), "velocity_ned": np.zeros(3), "attitude_rpy": np.zeros(3)},
                target_state={"position_neu": np.array([1000.0, 0.0, 0.0]), "velocity_ned": np.array([50.0, 0.0, 0.0]), "attitude_rpy": np.zeros(3)},
                relative_state={"distance": 1000.0, "relative_velocity": np.array([50.0, 0.0, 0.0])},
            )

        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_ned": np.array([50.0, 0.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="predicted_target",
            trajectory_predictor_adapter=adapter,
            return_info=True,
        )

        assert info["anchor_mode"] == "predicted_target"
        assert "prediction_info" in info
        # 由于使用的是 ConstantVelocityPredictor，预测位置应为 1000 + 50*1 = 1050
        assert np.allclose(info["anchor_pos"], np.array([1050.0, 0.0, 0.0]))

    def test_return_info_false_backward_compatible(self):
        """默认 return_info=False 时保持向后兼容，只返回 virtual_point。"""
        vp_config = {
            "action_dim": 5,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "tau_pred_range": [0.0, 3.0],
            "speed_bias_range": [-80.0, 80.0],
            "smoothing_alpha": 0.3,
        }
        gen = VirtualPointGenerator(vp_config)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 0.0])}

        result = gen.action_to_virtual_point(action, own_state, target_state)
        # 默认只返回一个对象（dict），不是 tuple
        assert not isinstance(result, tuple)
        assert isinstance(result, dict)
