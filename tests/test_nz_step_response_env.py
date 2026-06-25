import numpy as np
import pytest

from uav_vpp_guidance.envs.nz_step_response_env import NzStepResponseEnv


def _base_config():
    return {
        "backend": "simple",
        "env": {
            "use_jsbsim": False,
            "strict_backend": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "high_level_dt": 0.2,
            "max_high_level_steps": 100,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 8000.0,
            "xy_limit_m": 20000.0,
            "target_mode": "constant_velocity",
        },
        "virtual_point": {
            "enabled": True,
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
            },
        },
        "task": {
            "name": "nz_step_response",
            "env_class": "NzStepResponseEnv",
            "nz_step_response": {
                "step_sequence": [
                    {"duration_s": 2.0, "nz_cmd": 1.0},
                    {"duration_s": 5.0, "nz_cmd": 5.0},
                    {"duration_s": 5.0, "nz_cmd": 1.0},
                ],
                "min_safe_speed_mps": 150.0,
                "base_altitude_m": 5000.0,
                "own_speed_mps": 250.0,
            },
        },
    }


@pytest.fixture
def env():
    environment = NzStepResponseEnv(_base_config())
    yield environment
    environment.close()


def test_task_post_step_prefers_nz_g_telemetry(env):
    env._task_reset(seed=0)
    env._nz_cmd_hist.append(5.0)
    _, _, _, info = env._task_post_step(
        own_state_post={
            "position_m": np.array([0.0, 0.0, 5000.0], dtype=float),
            "speed_mps": 250.0,
            "nz_g": 4.2,
            "nz": 1.0,
        },
        target_state_post={},
        info={"is_success": False},
        reward=0.0,
        terminated=False,
        truncated=False,
    )
    assert info["actual_nz"] == pytest.approx(4.2)


def test_step_injects_phase_command_and_sets_override_telemetry(env):
    env.reset(seed=0)
    info = None
    for _ in range(11):
        _, _, _, _, info = env.step()
    assert info is not None
    assert info["nz_cmd"] == pytest.approx(5.0)
    assert info["command_override_active"] is True
    assert info["raw_command"]["nz_cmd"] == pytest.approx(5.0)
