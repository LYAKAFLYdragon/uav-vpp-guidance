"""
Tests for scenario-based no-prediction VPP evaluation.
"""

import pytest
import numpy as np
import os
import csv

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios import (
    evaluate_scenario, compute_ego_score, compute_target_score, save_metrics_csv
)
from uav_vpp_guidance.visualization.plot_no_prediction_results import (
    load_metrics_csv, load_trajectory, plot_2d_trajectory, plot_range_ata, plot_scores, plot_commands
)


@pytest.fixture
def scenario_config():
    return {
        "experiment": {"name": "test_scenarios", "seed": 42, "output_root": "outputs"},
        "env": {
            "use_jsbsim": False,
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


@pytest.fixture
def favorable_scenario():
    return {
        "name": "favorable",
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


class TestConfigLoading:
    def test_scenarios_yaml_loads(self):
        path = "config/experiment/no_prediction_vpp_scenarios.yaml"
        assert os.path.exists(path)
        config = load_yaml_config(path)
        assert "scenarios" in config
        scenarios = config["scenarios"]
        for name in ["favorable", "neutral", "disadvantage", "challenging"]:
            assert name in scenarios
            assert "own_init" in scenarios[name]
            assert "target_init" in scenarios[name]


class TestScenarioReset:
    def test_favorable_reset_state_legal(self, scenario_config, favorable_scenario):
        env = CloseRangeTrackingEnv(scenario_config)
        obs = env.reset(scenario=favorable_scenario, seed=0)
        assert isinstance(obs, dict)
        assert "own_state" in obs
        assert "target_state" in obs
        own = obs["own_state"]
        target = obs["target_state"]
        assert "position_m" in own
        assert "position_m" in target
        # Check that positions reflect scenario config
        assert np.allclose(own["position_m"], np.array([0.0, 0.0, 5000.0]))
        assert np.allclose(target["position_m"], np.array([2000.0, 0.0, 5000.0]))
        env.close()

    def test_scenario_can_step(self, scenario_config, favorable_scenario):
        env = CloseRangeTrackingEnv(scenario_config)
        env.reset(scenario=favorable_scenario, seed=0)
        for _ in range(10):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(reward, float)
            if terminated or truncated:
                break
        env.close()


class TestPredictorAdapter:
    def test_predictor_none_when_disabled(self, scenario_config):
        env = CloseRangeTrackingEnv(scenario_config)
        assert env.trajectory_predictor_adapter is None
        env.close()

    def test_no_predictor_call_during_step(self, scenario_config, favorable_scenario):
        env = CloseRangeTrackingEnv(scenario_config)
        env.reset(scenario=favorable_scenario, seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert info["anchor_mode"] == "current_target"
        env.close()


class TestScoreFunctions:
    def test_ego_score_range(self):
        rel = {"range_m": 900.0, "ata_rad": 0.0, "aa_rad": 0.0}
        score = compute_ego_score(rel)
        assert 0.0 <= score <= 1.0

    def test_target_score_range(self):
        rel = {"range_m": 5000.0, "ata_rad": np.pi, "aa_rad": 0.0}
        score = compute_target_score(rel)
        assert 0.0 <= score <= 1.0

    def test_ego_score_higher_when_close(self):
        rel_close = {"range_m": 900.0, "ata_rad": 0.0, "aa_rad": 0.0}
        rel_far = {"range_m": 5000.0, "ata_rad": np.pi, "aa_rad": np.pi}
        assert compute_ego_score(rel_close) > compute_ego_score(rel_far)


class TestEvaluateScenario:
    def test_evaluate_scenario_runs(self, scenario_config, favorable_scenario):
        env = CloseRangeTrackingEnv(scenario_config)
        metrics = evaluate_scenario(
            env, "favorable", favorable_scenario,
            num_episodes=2, seeds=[0], save_trajectories=False
        )
        assert isinstance(metrics, dict)
        assert metrics["scenario"] == "favorable"
        assert metrics["num_episodes"] == 2
        assert "success_rate" in metrics
        assert "crash_rate" in metrics
        assert "timeout_rate" in metrics
        assert "out_of_bounds_rate" in metrics
        assert "mean_score_ego" in metrics
        assert "mean_score_target" in metrics
        env.close()

    def test_evaluate_saves_trajectories(self, scenario_config, favorable_scenario):
        env = CloseRangeTrackingEnv(scenario_config)
        metrics = evaluate_scenario(
            env, "favorable", favorable_scenario,
            num_episodes=1, seeds=[0], save_trajectories=True,
            output_root="outputs"
        )
        traj_dir = os.path.join("outputs", "trajectories", "no_prediction_vpp", "simple", "favorable", "seed_0")
        assert os.path.exists(traj_dir), f"Trajectory directory not found: {traj_dir}"
        assert os.path.exists(os.path.join(traj_dir, "episode_0.csv"))
        env.close()


class TestMetricsCSV:
    def test_save_and_load_metrics_csv(self, tmp_path):
        metrics = [
            {"scenario": "fav", "success_rate": 0.5, "crash_rate": 0.0},
            {"scenario": "neu", "success_rate": 0.3, "crash_rate": 0.1},
        ]
        csv_path = os.path.join(str(tmp_path), "metrics.csv")
        save_metrics_csv(metrics, csv_path)
        assert os.path.exists(csv_path)
        loaded = load_metrics_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0]["scenario"] == "fav"


class TestTrajectoryPlotting:
    def test_load_trajectory_with_booleans(self, tmp_path):
        traj_path = os.path.join(str(tmp_path), "traj.csv")
        with open(traj_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time", "ego_x", "done"])
            writer.writerow(["0", "0.0", "100.0", "False"])
            writer.writerow(["1", "0.2", "110.0", "True"])
        rows = load_trajectory(traj_path)
        assert len(rows) == 2
        assert rows[0]["done"] is False
        assert rows[1]["done"] is True

    def test_plot_functions_run(self, tmp_path):
        traj = [
            {"step": 0, "time": 0.0, "ego_x": 0.0, "ego_y": 0.0, "ego_z": 5000.0,
             "target_x": 1000.0, "target_y": 0.0, "target_z": 5000.0,
             "virtual_x": 500.0, "virtual_y": 0.0, "virtual_z": 5000.0,
             "range_m": 1000.0, "ata_deg": 30.0, "aspect_deg": 20.0,
             "los_rate": 10.0, "nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.7,
             "ego_score": 0.5, "target_score": 0.3, "done": False, "termination_reason": ""},
            {"step": 1, "time": 0.2, "ego_x": 50.0, "ego_y": 0.0, "ego_z": 5000.0,
             "target_x": 1050.0, "target_y": 0.0, "target_z": 5000.0,
             "virtual_x": 550.0, "virtual_y": 0.0, "virtual_z": 5000.0,
             "range_m": 950.0, "ata_deg": 25.0, "aspect_deg": 18.0,
             "los_rate": 8.0, "nz_cmd": 1.2, "roll_rate_cmd": 0.1, "throttle_cmd": 0.75,
             "ego_score": 0.6, "target_score": 0.3, "done": True, "termination_reason": "success"},
        ]
        out_dir = str(tmp_path)
        plot_2d_trajectory(traj, out_dir, label="test")
        assert os.path.exists(os.path.join(out_dir, "trajectory_2d_test.png"))
        plot_range_ata(traj, out_dir, label="test")
        assert os.path.exists(os.path.join(out_dir, "range_ata_test.png"))
        plot_scores(traj, out_dir, label="test")
        assert os.path.exists(os.path.join(out_dir, "scores_test.png"))
        plot_commands(traj, out_dir, label="test")
        assert os.path.exists(os.path.join(out_dir, "commands_test.png"))


class TestSmokeModes:
    def test_evaluation_smoke_runs(self, scenario_config, favorable_scenario):
        env = CloseRangeTrackingEnv(scenario_config)
        metrics = evaluate_scenario(
            env, "favorable", favorable_scenario,
            num_episodes=1, seeds=[0], policy=None, save_trajectories=False
        )
        assert metrics["num_episodes"] == 1
        env.close()
