"""
虚拟追踪点生成器与轨迹预测模块的集成测试。
"""

import os

import numpy as np
import torch

from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator
from uav_vpp_guidance.trajectory_prediction.predictor_adapter import (
    TrajectoryPredictorAdapter,
)
from uav_vpp_guidance.trajectory_prediction.constant_velocity import (
    ConstantVelocityPredictor,
)
from uav_vpp_guidance.trajectory_prediction.state_buffer import TrajectoryStateBuffer
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import (
    LSTMTrajectoryPredictor,
)
from uav_vpp_guidance.trajectory_prediction.gru_predictor import (
    GRUTrajectoryPredictor,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


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
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            return_info=True,
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
            action,
            own_state,
            target_state,
            anchor_mode="constant_velocity",
            lookahead_time_s=2.0,
            return_info=True,
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
                own_state={
                    "position_neu": np.zeros(3),
                    "velocity_ned": np.zeros(3),
                    "attitude_rpy": np.zeros(3),
                },
                target_state={
                    "position_neu": np.array([1000.0, 0.0, 0.0]),
                    "velocity_ned": np.array([50.0, 0.0, 0.0]),
                    "attitude_rpy": np.zeros(3),
                },
                relative_state={
                    "distance": 1000.0,
                    "relative_velocity": np.array([50.0, 0.0, 0.0]),
                },
            )

        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_ned": np.array([50.0, 0.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            action,
            own_state,
            target_state,
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

    def test_lstm_predictor_closed_loop(self, tmpdir):
        """LSTM predictor 在线集成到 TrackingEnv 中，闭环步进不崩溃，
        且 virtual point 坐标符合 current_target + predicted_displacement + action_offset。"""
        # 创建 dummy LSTM checkpoint
        dummy_model = LSTMTrajectoryPredictor(
            input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0
        )
        ckpt_path = os.path.join(str(tmpdir), "lstm_dummy.pt")
        torch.save(dummy_model.state_dict(), ckpt_path)

        config = {
            "experiment": {
                "name": "test_lstm_cl",
                "seed": 42,
                "output_root": str(tmpdir),
            },
            "env": {
                "use_jsbsim": False,
                "decision_freq": 5,
                "sim_freq": 60,
                "max_high_level_steps": 512,
                "success_range_m": 900.0,
                "success_ata_deg": 25.0,
                "success_hold_time_s": 0.2,
                "hysteresis_range_m": 950.0,
                "hysteresis_ata_deg": 30.0,
                "min_altitude_m": 500.0,
                "max_altitude_m": 15000.0,
                "max_range_m": 8000.0,
                "target_mode": "constant_velocity",
            },
            "virtual_point": {
                "anchor_mode": "predicted_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
                "smoothing_alpha": 0.3,
            },
            "trajectory_prediction": {
                "enabled": True,
                "predictor_type": "lstm",
                "checkpoint_path": ckpt_path,
                "freeze_predictor_during_rl": True,
                "prediction": {
                    "lookahead_time_s": 1.0,
                    "output_mode": "relative_displacement",
                    "fallback_mode": "constant_velocity",
                },
                "history": {"history_len": 5, "padding_mode": "repeat_first"},
                "model": {
                    "input_dim": 16,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "predict_variance": False,
                },
                "integration": {"anchor_mode": "predicted_target"},
                "normalization": {
                    "position_scale_m": 1000.0,
                    "velocity_scale_mps": 300.0,
                    "overload_scale": 9.0,
                },
            },
            "limits": {
                "nz_min": -2.0,
                "nz_max": 7.0,
                "roll_rate_min": -1.5,
                "roll_rate_max": 1.5,
                "throttle_min": 0.0,
                "throttle_max": 1.0,
            },
            "reward": {
                "w_range": 0.5,
                "w_angle": 0.8,
                "w_energy": 0.2,
                "w_safety": 2.0,
                "w_saturation": 1.0,
                "w_smooth": 0.1,
                "terminal_success": 200.0,
                "terminal_failure": -200.0,
                "terminal_crash": -300.0,
                "min_altitude_m": 500.0,
            },
            "guidance": {
                "mode": "los_rate",
                "use_gain_adapter": False,
                "gains": {
                    "k_los": 1.0,
                    "k_pos": 0.5,
                    "k_damp": 0.2,
                    "k_roll": 1.0,
                    "k_speed": 0.2,
                    "alpha_filter": 0.3,
                },
            },
        }

        env = CloseRangeTrackingEnv(config)
        assert env.trajectory_predictor_adapter is not None
        obs = env.reset(seed=0)

        # Step enough times to fill the neural predictor buffer (history_len=5)
        for _ in range(10):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Environment should report LSTM predictor type
        assert info["prediction_enabled"] is True
        assert info["predictor_type"] == "LSTMTrajectoryPredictor"

        # Virtual point and predicted target position must exist
        vp = info.get("virtual_point", {})
        vp_pos = np.asarray(vp.get("position", np.zeros(3)))
        pred_target_pos = np.asarray(info.get("predicted_target_position", np.zeros(3)))

        assert vp_pos is not None
        assert pred_target_pos is not None
        assert np.isfinite(vp_pos).all()
        assert np.isfinite(pred_target_pos).all()

        # With zero action, virtual point should be very close to predicted_target_position
        # (allowing for small numerical differences)
        shift = np.linalg.norm(vp_pos - pred_target_pos)
        assert (
            shift < 100.0
        ), f"virtual_point too far from predicted_target: {shift:.1f} m"

        env.close()


    def test_gru_predictor_closed_loop(self, tmpdir):
        """GRU predictor 在线集成到 TrackingEnv 中，闭环步进不崩溃。"""
        dummy_model = GRUTrajectoryPredictor(
            input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0
        )
        ckpt_path = os.path.join(str(tmpdir), "gru_dummy.pt")
        torch.save(dummy_model.state_dict(), ckpt_path)

        config = {
            "experiment": {
                "name": "test_gru_cl",
                "seed": 42,
                "output_root": str(tmpdir),
            },
            "env": {
                "use_jsbsim": False,
                "decision_freq": 5,
                "sim_freq": 60,
                "max_high_level_steps": 512,
                "success_range_m": 900.0,
                "success_ata_deg": 25.0,
                "success_hold_time_s": 0.2,
                "hysteresis_range_m": 950.0,
                "hysteresis_ata_deg": 30.0,
                "min_altitude_m": 500.0,
                "max_altitude_m": 15000.0,
                "max_range_m": 8000.0,
                "target_mode": "constant_velocity",
            },
            "virtual_point": {
                "anchor_mode": "predicted_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
                "smoothing_alpha": 0.3,
            },
            "trajectory_prediction": {
                "enabled": True,
                "predictor_type": "gru",
                "checkpoint_path": ckpt_path,
                "freeze_predictor_during_rl": True,
                "strict_predictor_init": True,
                "device": "cpu",
                "allow_device_fallback": True,
                "prediction": {
                    "lookahead_time_s": 1.0,
                    "output_mode": "relative_displacement",
                    "fallback_mode": "constant_velocity",
                },
                "history": {"history_len": 5, "padding_mode": "repeat_first"},
                "model": {
                    "input_dim": 16,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "predict_variance": False,
                },
                "integration": {"anchor_mode": "predicted_target"},
                "normalization": {
                    "position_scale_m": 1000.0,
                    "velocity_scale_mps": 300.0,
                    "overload_scale": 9.0,
                },
            },
            "limits": {
                "nz_min": -2.0,
                "nz_max": 7.0,
                "roll_rate_min": -1.5,
                "roll_rate_max": 1.5,
                "throttle_min": 0.0,
                "throttle_max": 1.0,
            },
            "reward": {
                "w_range": 0.5,
                "w_angle": 0.8,
                "w_energy": 0.2,
                "w_safety": 2.0,
                "w_saturation": 1.0,
                "w_smooth": 0.1,
                "terminal_success": 200.0,
                "terminal_failure": -200.0,
                "terminal_crash": -300.0,
                "min_altitude_m": 500.0,
            },
            "guidance": {
                "mode": "los_rate",
                "use_gain_adapter": False,
                "gains": {
                    "k_los": 1.0,
                    "k_pos": 0.5,
                    "k_damp": 0.2,
                    "k_roll": 1.0,
                    "k_speed": 0.2,
                    "alpha_filter": 0.3,
                },
            },
        }

        env = CloseRangeTrackingEnv(config)
        assert env.trajectory_predictor_adapter is not None
        obs = env.reset(seed=0)

        for _ in range(10):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert info["prediction_enabled"] is True
        assert info["predictor_type"] == "GRUTrajectoryPredictor"
        assert info["predictor_init_failed"] is False

        vp = info.get("virtual_point", {})
        vp_pos = np.asarray(vp.get("position", np.zeros(3)))
        pred_target_pos = np.asarray(info.get("predicted_target_position", np.zeros(3)))

        assert np.isfinite(vp_pos).all()
        assert np.isfinite(pred_target_pos).all()

        shift = np.linalg.norm(vp_pos - pred_target_pos)
        assert shift < 100.0, f"virtual_point too far from predicted_target: {shift:.1f} m"

        env.close()
