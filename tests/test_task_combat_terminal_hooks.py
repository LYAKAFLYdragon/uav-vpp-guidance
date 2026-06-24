import numpy as np

from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv
from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv


def _state():
    return {
        "position_m": np.array([1000.0, 0.0, 5000.0], dtype=float),
        "velocity_vector_mps": np.array([250.0, 0.0, 0.0], dtype=float),
        "speed_mps": 250.0,
        "altitude_m": 5000.0,
        "yaw_rad": 0.0,
    }


def test_break_turn_task_hook_preserves_combat_terminal_success():
    env = object.__new__(BreakTurnEnv)
    env.range_hist = []
    env.speed_hist = []
    env.nz_hist = []
    env.agg_hist = []
    env.maneuver_type = "break_turn"
    env.trajectory = []
    env.env_config = {"high_level_dt": 0.2}
    env.task_cfg = {"reward_shaping": {}, "min_safe_speed_mps": 150.0}
    env._sim_time_s = 4.0

    info = {
        "is_success": True,
        "combat_success": True,
        "combat_outcome": "win",
        "reason": "timeout_hp_advantage",
        "range_m": 1000.0,
        "nz_cmd": 1.0,
    }
    _reward, terminated, truncated, out = env._task_post_step(
        _state(), _state(), info, 0.0, False, True
    )

    assert terminated is False
    assert truncated is True
    assert out["is_success"] is True
    assert out["combat_outcome"] == "win"
    assert out["reason"] == "timeout_hp_advantage"


def test_sustained_turn_task_hook_preserves_combat_terminal_success():
    env = object.__new__(SustainedTurnEnv)
    env.target_pos = np.array([0.0, 0.0, 5000.0], dtype=float)
    env.orbit_direction = 1.0
    env.theta_unwrapped = []
    env.radius_hist = []
    env.speed_hist = []
    env.nz_hist = []
    env.agg_hist = []
    env._prev_yaw_rad = None
    env._orbit_bonus_count = 0
    env.env_config = {"high_level_dt": 0.2}
    env.task_cfg = {"reward_shaping": {}, "min_safe_speed_mps": 150.0}

    info = {
        "is_success": True,
        "combat_success": True,
        "combat_outcome": "win",
        "reason": "timeout_hp_advantage",
        "range_m": 1000.0,
        "nz_cmd": 1.0,
    }
    _reward, terminated, truncated, out = env._task_post_step(
        _state(), _state(), info, 0.0, False, True
    )

    assert terminated is False
    assert truncated is True
    assert out["is_success"] is True
    assert out["combat_outcome"] == "win"
    assert out["reason"] == "timeout_hp_advantage"
