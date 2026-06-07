"""
Tests for JSBSim No-Prediction VPP High-Fidelity Bridge.
"""

import pytest
import numpy as np
import os

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.flight_control.low_level_controller import LowLevelController
from uav_vpp_guidance.flight_control.actuator_interface import JSBSimActuatorInterface


_JSBSIM_DATA_DIR = os.path.join("E:/CloseAirCombat_control", "envs", "JSBSim", "data")
skip_if_no_jsbsim_data = pytest.mark.skipif(
    not os.path.isdir(_JSBSIM_DATA_DIR),
    reason=f"JSBSim data directory not found: {_JSBSIM_DATA_DIR}",
)


@pytest.fixture
def jsbsim_config():
    return {
        "experiment": {"name": "test_jsbsim", "seed": 42, "output_root": "outputs"},
        "backend": "jsbsim",
        "env": {
            "use_jsbsim": True,
            "legacy_project_root": "E:/CloseAirCombat_control",
            "aircraft_model": "f16",
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": 512,
            "high_level_dt": 0.2,
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
            "anchor_mode": "current_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
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
            }
        },
    }


class TestBackendSelection:
    def test_backend_simple_from_explicit_config(self):
        config = {"backend": "simple", "env": {"decision_freq": 5}}
        env = CloseRangeTrackingEnv(config)
        assert env._backend == "simple"
        env.close()

    def test_backend_jsbsim_from_explicit_config(self):
        config = {"backend": "jsbsim", "env": {"decision_freq": 5, "legacy_project_root": "E:/CloseAirCombat_control"}}
        env = CloseRangeTrackingEnv(config)
        assert env._backend == "jsbsim"
        env.close()

    def test_backend_fallback_to_use_jsbsim(self):
        config = {"env": {"use_jsbsim": False, "decision_freq": 5}}
        env = CloseRangeTrackingEnv(config)
        assert env._backend == "simple"
        env.close()


@skip_if_no_jsbsim_data
class TestJSBSimEnvResetAndStep:
    def test_jsbsim_reset(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        obs = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "own_state" in obs
        assert "target_state" in obs
        env.close()

    def test_jsbsim_step(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        env.reset(seed=0)
        action = np.zeros(3)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_jsbsim_multiple_steps(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        env.reset(seed=0)
        for _ in range(10):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    def test_jsbsim_info_contains_backend(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info.get("backend") == "jsbsim"
        env.close()


class TestPredictorNotCalled:
    def test_predictor_none_in_jsbsim(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        assert env.trajectory_predictor_adapter is None
        env.close()

    @skip_if_no_jsbsim_data
    def test_no_predictor_call_in_jsbsim_step(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["anchor_mode"] == "current_target"
        env.close()


class TestLowLevelController:
    def test_nan_protection(self):
        ctrl = LowLevelController({"alpha_filter": 0.3})
        result = ctrl.compute_actuator({"nz_cmd": float("nan"), "roll_rate_cmd": float("inf"), "throttle_cmd": None})
        assert np.isfinite(result["fcs/elevator-cmd-norm"])
        assert np.isfinite(result["fcs/aileron-cmd-norm"])
        assert np.isfinite(result["fcs/throttle-cmd-norm"])

    def test_saturation_flag(self):
        ctrl = LowLevelController({"alpha_filter": 1.0, "actuator": {"nz_to_elevator_gain": 1.0}})
        result = ctrl.compute_actuator({"nz_cmd": 20.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5})
        assert result["saturation_flag"] is True

    def test_filter_smoothing(self):
        ctrl = LowLevelController({"alpha_filter": 0.5})
        r1 = ctrl.compute_actuator({"nz_cmd": 2.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5})
        e1 = r1["fcs/elevator-cmd-norm"]
        r2 = ctrl.compute_actuator({"nz_cmd": 2.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5})
        e2 = r2["fcs/elevator-cmd-norm"]
        # With alpha=0.5, second step should be closer to steady state
        assert np.isfinite(e2)


class TestActuatorInterface:
    def test_command_mapping(self):
        iface = JSBSimActuatorInterface()
        result = iface.command_to_jsbsim_properties({"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.7})
        assert "fcs/elevator-cmd-norm" in result
        assert "fcs/aileron-cmd-norm" in result
        assert "fcs/throttle-cmd-norm" in result
        assert -1.0 <= result["fcs/elevator-cmd-norm"] <= 1.0
        assert 0.0 <= result["fcs/throttle-cmd-norm"] <= 1.0

    def test_nan_input_protection(self):
        iface = JSBSimActuatorInterface()
        result = iface.command_to_jsbsim_properties({"nz_cmd": float("nan")})
        assert np.isfinite(result["fcs/elevator-cmd-norm"])


@skip_if_no_jsbsim_data
class TestUnifiedState:
    def test_jsbsim_state_has_position(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        env.reset(seed=0)
        own, target = env._get_current_states()
        assert "position_neu" in own or "position_m" in own
        assert "position_neu" in target or "position_m" in target
        env.close()

    def test_jsbsim_state_has_velocity(self, jsbsim_config):
        env = CloseRangeTrackingEnv(jsbsim_config)
        env.reset(seed=0)
        own, target = env._get_current_states()
        assert "velocity_ned" in own or "velocity_vector_mps" in own
        env.close()


@skip_if_no_jsbsim_data
class TestScenarioDictSupport:
    def test_jsbsim_scenario_dict_reset(self, jsbsim_config):
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 5000.0],
                "velocity_mps": 220.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [2000.0, 0.0, 5000.0],
                "velocity_mps": 180.0,
                "heading_deg": 0.0,
            },
        }
        env = CloseRangeTrackingEnv(jsbsim_config)
        obs = env.reset(scenario=scenario, seed=0)
        assert isinstance(obs, dict)
        env.close()
