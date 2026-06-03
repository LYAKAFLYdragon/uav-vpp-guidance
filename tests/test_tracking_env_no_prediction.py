"""
Tests for CloseRangeTrackingEnv in no-prediction mode.
"""

import pytest
import numpy as np
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


@pytest.fixture
def base_config():
    return {
        "experiment": {"name": "test_no_pred", "seed": 42, "output_root": "outputs"},
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
    }


class TestCloseRangeTrackingEnvNoPrediction:
    def test_reset_returns_observation(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "relative_state" in obs
        assert "own_state" in obs
        assert "target_state" in obs

    def test_step_returns_tuple(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_info_contains_backend(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "backend" in info
        assert info["backend"] == "simple"
        env.close()

    def test_info_contains_guidance_command(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "guidance_command" in info
        assert set(info["guidance_command"].keys()) == {
            "nz_cmd",
            "roll_rate_cmd",
            "throttle_cmd",
        }
        env.close()

    def test_episode_runs_to_completion(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 20:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()

    def test_multiple_episodes(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        for ep in range(3):
            env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < 10:
                _, _, terminated, truncated, info = env.step(np.zeros(3))
                done = terminated or truncated
                steps += 1
        env.close()

    def test_different_actions_produce_different_commands(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info1 = env.step(np.array([0.0, 0.0, 0.0]))
        _, _, _, _, info2 = env.step(np.array([1.0, 0.5, -0.3]))
        # Different actions should change the virtual point and guidance command
        assert not np.allclose(
            info1["guidance_command"]["nz_cmd"],
            info2["guidance_command"]["nz_cmd"],
            atol=1e-6,
        )
        env.close()

    def test_observation_vector_shape(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.zeros(3))
        assert "observation_vector" in obs
        assert obs["observation_vector"].ndim == 1
        assert obs["observation_vector"].shape[0] > 0
        env.close()

    def test_reward_is_finite(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        for _ in range(5):
            _, reward, _, _, _ = env.step(np.zeros(3))
            assert np.isfinite(reward)
        env.close()

    def test_commands_within_limits(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, _, info = env.step(np.zeros(3))
            cmd = info["guidance_command"]
            assert -2.0 <= cmd["nz_cmd"] <= 7.0
            assert -1.5 <= cmd["roll_rate_cmd"] <= 1.5
            assert 0.0 <= cmd["throttle_cmd"] <= 1.0
        env.close()

    def test_backend_simple_when_use_jsbsim_false(self, base_config):
        base_config["env"]["use_jsbsim"] = False
        env = CloseRangeTrackingEnv(base_config)
        assert env._backend == "simple"
        env.close()

    def test_backend_explicit_override(self, base_config):
        base_config["backend"] = "simple"
        env = CloseRangeTrackingEnv(base_config)
        assert env._backend == "simple"
        env.close()

    def test_virtual_point_in_info(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "virtual_point" in info
        assert "position" in info["virtual_point"]
        env.close()

    def test_termination_reason_in_info(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 600:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            done = terminated or truncated
            steps += 1
        assert "reason" in info["termination_info"]
        env.close()

    def test_prediction_enabled_false_by_default(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["prediction_enabled"] is False
        env.close()

    def test_anchor_mode_current_target(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_relative_state_fields(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.zeros(3))
        rel = obs["relative_state"]
        assert "range_m" in rel
        assert "ata_rad" in rel
        assert "aa_rad" in rel
        env.close()

    def test_reward_terms_structure(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        rt = info["reward_terms"]
        assert isinstance(rt, dict)
        assert len(rt) > 0
        env.close()

    def test_step_count_increments(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        assert env.current_step == 0
        env.step(np.zeros(3))
        assert env.current_step == 1
        env.step(np.zeros(3))
        assert env.current_step == 2
        env.close()

    def test_episode_count_increments(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        ep1 = env._episode_count
        env.reset(seed=1)
        ep2 = env._episode_count
        assert ep2 == ep1 + 1
        env.close()

    def test_close_does_not_raise(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        env.step(np.zeros(3))
        env.close()

    def test_terminal_reward_success(self, base_config):
        """成功到达时 terminal_reward 应为正数。"""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        # 构造一个接近目标的场景
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated:
                rt = info["reward_terms"]
                if info["termination_info"]["is_success"]:
                    assert rt["terminal_reward"] == pytest.approx(200.0, abs=1e-6)
                break
        env.close()

    def test_terminal_reward_crash(self, base_config):
        """碰撞时 terminal_reward 应为负数。"""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated:
                rt = info["reward_terms"]
                if info["termination_info"]["is_crash"]:
                    assert rt["terminal_reward"] == pytest.approx(-300.0, abs=1e-6)
                break
        env.close()

    def test_terminal_reward_timeout(self, base_config):
        """超时 truncation 时 terminal_reward 应为负数。"""
        base_config["env"]["max_high_level_steps"] = 1
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert truncated is True
        rt = info["reward_terms"]
        assert rt["terminal_reward"] == pytest.approx(-200.0, abs=1e-6)
        env.close()

    def test_timeout_info_reason_is_timeout(self, base_config):
        """timeout 时 info['termination_info']['reason'] 应为 'timeout'。"""
        base_config["env"]["max_high_level_steps"] = 1
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated is False
        assert truncated is True
        assert info["termination_info"]["reason"] == "timeout"
        assert info["termination_info"]["is_timeout"] is True
        assert info["is_timeout"] is True
        env.close()


class TestStrictBackend:
    """Tests for strict_backend config option."""

    def test_strict_backend_false_allows_simple_fallback(
        self, base_config, monkeypatch
    ):
        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = False

        def _fail_init(*args, **kwargs):
            raise RuntimeError("Simulated JSBSim failure")

        monkeypatch.setattr(
            "uav_vpp_guidance.envs.tracking_env.JSBSimEnv",
            _fail_init,
        )
        env = CloseRangeTrackingEnv(base_config)
        assert env._backend == "simple"
        env.close()

    def test_strict_backend_true_raises_on_jsbsim_failure(
        self, base_config, monkeypatch
    ):
        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = True

        def _fail_init(*args, **kwargs):
            raise RuntimeError("Simulated JSBSim failure")

        monkeypatch.setattr(
            "uav_vpp_guidance.envs.tracking_env.JSBSimEnv",
            _fail_init,
        )
        with pytest.raises(RuntimeError, match="strict_backend=True"):
            CloseRangeTrackingEnv(base_config)

    def test_backend_reflected_in_info(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["backend"] == env._backend
        env.close()
