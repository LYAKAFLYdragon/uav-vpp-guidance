"""
Integration tests for PPO training pipeline.

Covers:
- Smoke training run
- Checkpoint save/load
- Policy evaluation
- Trajectory prediction disabled enforcement
- Output directory structure
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


CONFIG_PATH = "config/experiment/train_no_prediction_vpp_ppo.yaml"


class TestPPOSmokeTraining:
    @pytest.fixture(scope="class")
    def smoke_output_dir(self, tmp_path_factory):
        """Run smoke training once and return output directory."""
        output_dir = str(tmp_path_factory.mktemp("ppo_smoke"))
        result = subprocess.run(
            [
                sys.executable, "-m",
                "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
                "--config", CONFIG_PATH,
                "--smoke",
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Print stdout/stderr on failure for debugging
        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
        assert result.returncode == 0, f"Smoke training failed with code {result.returncode}"
        return output_dir

    def test_smoke_summary_json_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "logs", "smoke_summary.json")
        assert os.path.exists(path), f"smoke_summary.json not found at {path}"

    def test_smoke_summary_content(self, smoke_output_dir):
        import json
        path = os.path.join(smoke_output_dir, "logs", "smoke_summary.json")
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary.get("smoke") is True
        assert summary.get("total_timesteps", 0) > 0
        assert summary.get("episodes", 0) > 0
        assert "checkpoint_dir" in summary

    def test_episode_train_log_csv_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "logs", "episode_train_log.csv")
        assert os.path.exists(path), f"episode_train_log.csv not found at {path}"

    def test_update_train_log_csv_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "logs", "update_train_log.csv")
        assert os.path.exists(path), f"update_train_log.csv not found at {path}"

    def test_eval_log_csv_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "logs", "eval_log.csv")
        assert os.path.exists(path)

    def test_best_checkpoint_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "checkpoints", "best.pt")
        assert os.path.exists(path), "best.pt not saved"

    def test_last_checkpoint_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "checkpoints", "last.pt")
        assert os.path.exists(path), "last.pt not saved"

    def test_config_snapshot_exists(self, smoke_output_dir):
        path = os.path.join(smoke_output_dir, "config_snapshot.yaml")
        assert os.path.exists(path)

    def test_no_files_in_root(self, smoke_output_dir):
        """Ensure no experiment files are created in project root."""
        root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        forbidden = ["train_log.csv", "eval_log.csv", "best.pt", "last.pt", "smoke_summary.json", "episode_train_log.csv", "update_train_log.csv"]
        for name in forbidden:
            assert not os.path.exists(os.path.join(root, name)), f"Forbidden file in root: {name}"


class TestPolicyEvaluation:
    @pytest.fixture(scope="class")
    def smoke_output_dir(self, tmp_path_factory):
        output_dir = str(tmp_path_factory.mktemp("ppo_smoke_eval"))
        result = subprocess.run(
            [
                sys.executable, "-m",
                "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
                "--config", CONFIG_PATH,
                "--smoke",
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        return output_dir

    def test_evaluate_policy_simple(self, smoke_output_dir):
        checkpoint = os.path.join(smoke_output_dir, "checkpoints", "best.pt")
        output_dir = os.path.join(smoke_output_dir, "eval_test")
        result = subprocess.run(
            [
                sys.executable, "-m",
                "uav_vpp_guidance.evaluation.evaluate_policy",
                "--config", CONFIG_PATH,
                "--checkpoint", checkpoint,
                "--backend", "simple",
                "--episodes", "2",
                "--seeds", "0",
                "--save-trajectories",
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
        assert result.returncode == 0

        # Check outputs
        assert os.path.exists(os.path.join(output_dir, "policy_metrics.json"))
        assert os.path.exists(os.path.join(output_dir, "policy_metrics.csv"))
        assert os.path.exists(os.path.join(output_dir, "trajectories", "seed0_ep0.csv"))


class TestNoPredictionEnforcement:
    def test_trajectory_prediction_disabled_in_config(self):
        config = load_yaml_config(CONFIG_PATH)
        enabled = config.get("trajectory_prediction", {}).get("enabled", True)
        assert enabled is False, "trajectory_prediction.enabled must be false in baseline config"

    def test_predictor_adapter_is_none(self):
        config = load_yaml_config(CONFIG_PATH)
        env = CloseRangeTrackingEnv(config)
        assert env.trajectory_predictor_adapter is None
        env.close()

    def test_predictor_not_called_during_steps(self):
        config = load_yaml_config(CONFIG_PATH)
        env = CloseRangeTrackingEnv(config)
        obs = env.reset(seed=0)
        for _ in range(5):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        # If predictor was called, it would fail because adapter is None
        # and anchor_mode is current_target (which doesn't use predictor)
        env.close()


class TestCheckpointConsistency:
    def test_deterministic_action_consistency_after_save_load(self):
        config = load_yaml_config(CONFIG_PATH)
        env = CloseRangeTrackingEnv(config)
        obs = env.reset(seed=0)
        obs_dim = obs["observation_vector"].shape[0]
        action_dim = 3

        agent1 = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")
        obs_vec = obs["observation_vector"]
        a1 = agent1.get_deterministic_action(obs_vec)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent1.save(path)
            agent2 = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")
            agent2.load(path)
            a2 = agent2.get_deterministic_action(obs_vec)
            assert np.allclose(a1, a2, atol=1e-6)
        finally:
            os.unlink(path)
        env.close()
