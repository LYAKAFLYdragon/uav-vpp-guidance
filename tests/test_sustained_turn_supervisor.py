from copy import deepcopy

import numpy as np
import pytest

from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv
from uav_vpp_guidance.flight_control.command_limiter import effective_throttle_limits


def _base_config():
    return {
        "backend": "simple",
        "env": {
            "use_jsbsim": False,
            "strict_backend": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "high_level_dt": 0.2,
            "max_high_level_steps": 128,
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
            "name": "sustained_turn",
            "env_class": "SustainedTurnEnv",
            "sustained_turn": {
                "target_pos_m": [2000.0, 0.0, 7000.0],
                "orbit_radius_m": 1200.0,
                "tangent_lead_deg": 90.0,
                "min_safe_speed_mps": 150.0,
                "max_planned_altitude_m": 5000.0,
                "supervisor": {
                    "enabled": True,
                    "speed_floor_mps": 175.0,
                    "caution_speed_mps": 195.0,
                    "recovery_speed_mps": 210.0,
                    "recovery_nz_min": 1.2,
                    "recovery_nz_max": 2.5,
                    "recovery_nz_target": 1.35,
                    "caution_nz_max": 4.0,
                    "recovery_roll_rate_cmd": 1.2,
                    "recovery_throttle_cmd": 0.95,
                    "caution_throttle_cmd": 0.8,
                    "recovery_roll_rate_scale": 0.5,
                    "caution_roll_rate_scale": 0.75,
                    "high_altitude_nz_cap": 3.0,
                },
            },
        },
    }


@pytest.fixture
def env():
    environment = SustainedTurnEnv(_base_config())
    yield environment
    environment.close()


def test_reset_clamps_default_plan_to_safe_altitude(env):
    scenario = {
        "name": "high_altitude_plan",
        "own_init": {
            "position_m": [0.0, 0.0, 6500.0],
            "velocity_mps": 250.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [2000.0, 0.0, 7000.0],
            "velocity_mps": 0.0,
            "heading_deg": 0.0,
        },
    }
    obs = env.reset(scenario=scenario, seed=0)
    assert env.target_pos[2] == pytest.approx(5000.0)
    assert obs["target_state"]["position_m"][2] == pytest.approx(5000.0)
    assert obs["own_state"]["altitude_m"] == pytest.approx(5000.0)


def test_task_supervisor_caps_nz_and_boosts_throttle(env):
    command = {"nz_cmd": 6.0, "roll_rate_cmd": 1.2, "throttle_cmd": 0.6}
    own_state = {
        "position_m": np.array([0.0, 0.0, 5200.0], dtype=float),
        "altitude_m": 5200.0,
        "roll_rad": 0.4,
        "speed_mps": 160.0,
    }
    adjusted, meta = env._task_adjust_command(
        command,
        own_state,
        target_state={},
        rel_state={},
        use_command_override=False,
    )
    assert adjusted["nz_cmd"] == pytest.approx(1.35)
    assert adjusted["throttle_cmd"] == pytest.approx(0.95)
    assert adjusted["roll_rate_cmd"] == pytest.approx(-1.2)
    assert meta["task_supervisor_state"] == "recovery"
    assert meta["high_altitude_protection_active"] is True
    assert meta["task_supervisor_pause_orbit_tracking"] is True


@pytest.mark.parametrize(
    ("roll_rad", "expected_sign"),
    [
        (0.6, -1.0),
        (-0.6, 1.0),
    ],
)
def test_task_supervisor_rolls_out_of_existing_bank(env, roll_rad, expected_sign):
    adjusted, meta = env._task_adjust_command(
        {"nz_cmd": 3.0, "roll_rate_cmd": 0.4, "throttle_cmd": 0.5},
        own_state={
            "position_m": np.array([0.0, 0.0, 5000.0], dtype=float),
            "altitude_m": 5000.0,
            "roll_rad": roll_rad,
            "speed_mps": 160.0,
        },
        target_state={},
        rel_state={},
        use_command_override=False,
    )
    assert meta["task_supervisor_state"] == "recovery"
    assert np.sign(adjusted["roll_rate_cmd"]) == expected_sign
    assert abs(adjusted["roll_rate_cmd"]) == pytest.approx(1.2)


def test_task_supervisor_pauses_and_resumes_orbit_tracking_with_hysteresis(env):
    own_state_recovery = {
        "position_m": np.array([800.0, 200.0, 5000.0], dtype=float),
        "altitude_m": 5000.0,
        "roll_rad": 0.3,
        "speed_mps": 160.0,
    }
    base_target = env._task_get_target_state({}, own_state_recovery)
    _, paused_target = env._task_pre_step(own_state_recovery, base_target)
    assert env._task_supervisor_pause_orbit_tracking is True
    assert np.allclose(paused_target["position_m"], env.target_pos)

    adjusted_mid, meta_mid = env._task_adjust_command(
        {"nz_cmd": 2.0, "roll_rate_cmd": 0.2, "throttle_cmd": 0.7},
        own_state={
            "position_m": np.array([800.0, 200.0, 5000.0], dtype=float),
            "altitude_m": 5000.0,
            "roll_rad": 0.2,
            "speed_mps": 200.0,
        },
        target_state={},
        rel_state={},
        use_command_override=False,
    )
    assert meta_mid["task_supervisor_state"] == "recovery"
    assert meta_mid["task_supervisor_pause_orbit_tracking"] is True
    assert adjusted_mid["throttle_cmd"] == pytest.approx(0.95)

    own_state_recovered = {
        "position_m": np.array([800.0, 200.0, 5200.0], dtype=float),
        "altitude_m": 5200.0,
        "roll_rad": 0.0,
        "speed_mps": 215.0,
    }
    adjusted_recovered, meta_recovered = env._task_adjust_command(
        {"nz_cmd": 2.0, "roll_rate_cmd": 0.2, "throttle_cmd": 0.7},
        own_state=own_state_recovered,
        target_state={},
        rel_state={},
        use_command_override=False,
    )
    assert meta_recovered["task_supervisor_pause_orbit_tracking"] is False
    assert meta_recovered["task_supervisor_state"] == "high_altitude"
    assert adjusted_recovered["roll_rate_cmd"] == pytest.approx(0.14)

    _, resumed_target = env._task_pre_step(own_state_recovered, base_target)
    assert env._task_supervisor_pause_orbit_tracking is False
    assert not np.allclose(resumed_target["position_m"], env.target_pos)


def test_recovery_can_bypass_high_level_command_filter(env):
    env.reset(seed=0)
    nominal = env._apply_command_filter(
        {"nz_cmd": 4.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    )
    assert nominal["nz_cmd"] == pytest.approx(4.0)

    recovery = env._apply_command_filter(
        {"nz_cmd": 1.35, "roll_rate_cmd": -1.2, "throttle_cmd": 0.95},
        reset=True,
    )
    assert recovery["nz_cmd"] == pytest.approx(1.35)
    assert recovery["roll_rate_cmd"] == pytest.approx(-1.2)
    assert recovery["throttle_cmd"] == pytest.approx(0.95)


def test_effective_throttle_limits_can_opt_in_to_full_recovery_throttle():
    throttle_min, throttle_max = effective_throttle_limits(
        {
            "throttle_min": 0.4,
            "throttle_max": 1.0,
            "throttle_effective_max": 1.0,
        }
    )
    assert throttle_min == pytest.approx(0.4)
    assert throttle_max == pytest.approx(1.0)


def test_task_supervisor_bypasses_command_override(env):
    command = {"nz_cmd": 5.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    adjusted, meta = env._task_adjust_command(
        command,
        own_state={"altitude_m": 5200.0, "speed_mps": 160.0},
        target_state={},
        rel_state={},
        use_command_override=True,
    )
    assert adjusted == command
    assert meta["task_supervisor_bypassed"] is True
