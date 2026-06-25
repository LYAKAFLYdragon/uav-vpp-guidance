import copy

import numpy as np

from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
from uav_vpp_guidance.envs.opponent_policy import OpponentPolicy
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


class DummyDirectOpponent(OpponentPolicy):
    action_mode = "direct_command"

    def __init__(self):
        self.calls = 0

    def act(self, opponent_obs):
        self.calls += 1
        assert opponent_obs["observation_schema"]["role_reversed"] is True
        return np.array([0.0, 1.0, 1.0], dtype=np.float32)


def _config():
    return {
        "backend": "simple",
        "env": {
            "use_jsbsim": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "high_level_dt": 0.2,
            "max_high_level_steps": 5,
            "success_range_m": 300.0,
            "success_ata_deg": 10.0,
            "success_hold_time_s": 0.2,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 12000.0,
            "target_mode": "constant_velocity",
        },
        "virtual_point": {
            "enabled": True,
            "anchor_mode": "current_target",
            "d_long_range": [-1000.0, 1000.0],
            "d_lat_range": [-500.0, 500.0],
            "d_vert_range": [-300.0, 300.0],
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
        "reward": {"terminal_success": 10.0, "terminal_failure": -10.0, "terminal_crash": -20.0},
        "guidance": {
            "mode": "los_rate",
            "gains": {"k_los": 1.0, "k_pos": 0.5, "k_damp": 0.2, "k_roll": 1.0, "k_speed": 0.2},
        },
    }


def test_opponent_policy_is_called_and_controls_target():
    opponent = DummyDirectOpponent()
    env = CloseRangeTrackingEnv(
        _config(),
        opponent_policy=opponent,
        opponent_config={"stage": "unit_test", "type": "dummy"},
    )
    env.reset(seed=0)
    _obs, _reward, _terminated, _truncated, info = env.step(np.zeros(3))

    assert opponent.calls == 1
    assert info["opponent_stage"] == "unit_test"
    assert info["opponent_command"] is not None
    assert info["provenance"]["opponent_stage"] == "unit_test"
    env.close()


def test_get_opponent_obs_inverts_geometry_features():
    env = CloseRangeTrackingEnv(_config())
    ego_obs = env.reset(
        scenario={
            "own_init": {"position_m": [0, 0, 5000], "velocity_mps": 200, "heading_deg": 0},
            "target_init": {"position_m": [4000, 0, 5000], "velocity_mps": 200, "heading_deg": 180},
        },
        seed=0,
    )
    opponent_obs = env._get_opponent_obs(ego_obs["own_state"], ego_obs["target_state"])
    names = ego_obs["observation_schema"]["feature_names"]
    ego_values = dict(zip(names, ego_obs["observation_vector"]))
    opp_values = dict(zip(names, opponent_obs["observation_vector"]))

    assert opp_values["range_m"] == ego_values["range_m"]
    assert opp_values["range_rate_mps"] == -ego_values["range_rate_mps"]
    assert opp_values["altitude_diff_m"] == -ego_values["altitude_diff_m"]
    assert opp_values["speed_diff_mps"] == -ego_values["speed_diff_mps"]
    assert opp_values["ata_sin"] == ego_values["aa_sin"]
    assert opp_values["aa_sin"] == ego_values["ata_sin"]
    assert opp_values["own_speed"] == ego_values["target_speed"]
    assert opp_values["target_speed"] == ego_values["own_speed"]
    env.close()


def test_expert_opponent_returns_vpp_action_from_physical_state():
    env = CloseRangeTrackingEnv(_config())
    obs = env.reset(seed=0)
    opponent_obs = env._get_opponent_obs(obs["own_state"], obs["target_state"])
    action = ExpertOpponent({}).act(opponent_obs)
    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    env.close()


def test_combat_mode_suppresses_legacy_tracking_success_terminal():
    cfg = _config()
    cfg["env"]["success_range_m"] = 5000.0
    cfg["env"]["success_ata_deg"] = 180.0
    cfg["env"]["max_high_level_steps"] = 5
    cfg["attack_zone"] = {"enabled": True, "damage_per_step": 0.0}
    env = CloseRangeTrackingEnv(cfg)
    env.reset(seed=0)

    _obs, _reward, terminated, truncated, info = env.step(np.zeros(3))

    assert terminated is False
    assert truncated is False
    assert info["termination_info"]["tracking_success_suppressed_by_combat"] is True
    assert info["is_success"] is False
    env.close()


def test_combat_mode_timeout_after_suppressed_tracking_success():
    cfg = _config()
    cfg["env"]["success_range_m"] = 5000.0
    cfg["env"]["success_ata_deg"] = 180.0
    cfg["env"]["max_high_level_steps"] = 1
    cfg["attack_zone"] = {"enabled": True, "damage_per_step": 0.0}
    env = CloseRangeTrackingEnv(cfg)
    env.reset(seed=0)

    _obs, _reward, terminated, truncated, info = env.step(np.zeros(3))

    assert terminated is False
    assert truncated is True
    assert info["reason"] == "timeout_draw"
    assert info["combat_outcome"] == "draw"
    env.close()


def test_combat_mode_preserves_raw_termination_reason_after_overlay():
    cfg = _config()
    cfg["attack_zone"] = {"enabled": True, "damage_per_step": 0.0}
    env = CloseRangeTrackingEnv(cfg)
    env.reset(
        scenario={
            "own_init": {"position_m": [0, 0, 490], "velocity_mps": 200, "heading_deg": 0},
            "target_init": {"position_m": [1000, 0, 5000], "velocity_mps": 200, "heading_deg": 180},
        },
        seed=0,
    )

    _obs, _reward, terminated, truncated, info = env.step(
        command_override={"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
    )

    assert terminated is True
    assert truncated is False
    assert info["reason"] == "ego_crash_or_out_of_bounds"
    assert info["termination_info"]["raw_termination_reason"] == "crash"
    assert info["termination_info"]["raw_is_crash"] is True
    assert info["termination_info"]["raw_is_out_of_bounds"] is False
    env.close()
