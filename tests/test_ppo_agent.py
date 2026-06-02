"""
Tests for PPO agent modules: policy network, replay buffer, and PPO agent.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from uav_vpp_guidance.agents.policy_network import MLPActorCritic
from uav_vpp_guidance.agents.replay_buffer import PPORolloutBuffer
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Policy Network Tests
# ---------------------------------------------------------------------------

class TestPolicyNetwork:
    def test_forward_dimension(self):
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32], activation="tanh"
        )
        obs = torch.randn(4, 16)
        mean, value = net.forward(obs)
        assert mean.shape == (4, 3), f"Expected mean shape (4, 3), got {mean.shape}"
        assert value.shape == (4,), f"Expected value shape (4,), got {value.shape}"

    def test_single_observation(self):
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32], activation="tanh"
        )
        obs = torch.randn(16)
        mean, value = net.forward(obs)
        assert mean.shape == (1, 3)
        assert value.shape == (1,)

    def test_actor_output_action_dim(self):
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32], activation="tanh"
        )
        obs = torch.randn(1, 16)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape == (1, 3), f"Expected action shape (1, 3), got {action.shape}"
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_critic_output_value_dim(self):
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32], activation="tanh"
        )
        obs = torch.randn(5, 16)
        value = net.get_value(obs)
        assert value.shape == (5,), f"Expected value shape (5,), got {value.shape}"

    def test_action_scaling_within_config_range(self):
        action_low = np.array([-1500.0, -1500.0, -300.0], dtype=np.float32)
        action_high = np.array([1500.0, 1500.0, 300.0], dtype=np.float32)
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32],
            activation="tanh", action_low=action_low, action_high=action_high,
        )
        # Sample many actions to check bounds
        obs = torch.randn(100, 16)
        actions = []
        for _ in range(10):
            action, _, _, _ = net.get_action_and_value(obs)
            actions.append(action.detach().cpu().numpy())
        actions = np.concatenate(actions, axis=0)
        assert actions.shape[1] == 3
        assert np.all(actions >= action_low - 1e-3), f"Actions below low bound: min={actions.min(axis=0)}"
        assert np.all(actions <= action_high + 1e-3), f"Actions above high bound: max={actions.max(axis=0)}"

    def test_deterministic_action_consistency(self):
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32], activation="tanh"
        )
        obs = torch.randn(1, 16)
        a1 = net.get_deterministic_action(obs)
        a2 = net.get_deterministic_action(obs)
        assert np.allclose(a1.cpu().numpy(), a2.cpu().numpy())

    def test_evaluate_action_returns_log_prob_entropy_value(self):
        net = MLPActorCritic(
            obs_dim=16, action_dim=3, hidden_sizes=[32, 32], activation="tanh"
        )
        obs = torch.randn(4, 16)
        action, log_prob1, entropy1, value1 = net.get_action_and_value(obs)
        log_prob2, entropy2, value2 = net.get_action_and_value(obs, action=action)
        assert log_prob2.shape == (4,)
        assert entropy2.shape == (4,)
        assert value2.shape == (4,)

    def test_invalid_obs_dim_raises(self):
        with pytest.raises(ValueError):
            MLPActorCritic(obs_dim=0, action_dim=3, hidden_sizes=[32])
        with pytest.raises(ValueError):
            MLPActorCritic(obs_dim=-1, action_dim=3, hidden_sizes=[32])

    def test_invalid_action_dim_raises(self):
        with pytest.raises(ValueError):
            MLPActorCritic(obs_dim=16, action_dim=0, hidden_sizes=[32])


# ---------------------------------------------------------------------------
# Rollout Buffer Tests
# ---------------------------------------------------------------------------

class TestPPORolloutBuffer:
    def test_add_and_length(self):
        buf = PPORolloutBuffer(capacity=10, obs_dim=16, action_dim=3)
        for i in range(5):
            buf.add(
                obs=np.ones(16) * i,
                action=np.ones(3) * i,
                log_prob=-0.5 * i,
                reward=float(i),
                done=False,
                value=float(i),
            )
        assert len(buf) == 5

    def test_capacity_full_raises(self):
        buf = PPORolloutBuffer(capacity=3, obs_dim=16, action_dim=3)
        for i in range(3):
            buf.add(
                obs=np.ones(16),
                action=np.ones(3),
                log_prob=-0.5,
                reward=1.0,
                done=False,
                value=1.0,
            )
        assert buf.full
        with pytest.raises(RuntimeError):
            buf.add(
                obs=np.ones(16),
                action=np.ones(3),
                log_prob=-0.5,
                reward=1.0,
                done=False,
                value=1.0,
            )

    def test_compute_gae_no_nan(self):
        buf = PPORolloutBuffer(capacity=10, obs_dim=16, action_dim=3)
        for i in range(8):
            buf.add(
                obs=np.ones(16),
                action=np.ones(3),
                log_prob=-0.5,
                reward=1.0,
                done=(i == 7),
                value=1.0,
            )
        advantages, returns = buf.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert len(advantages) == 8
        assert len(returns) == 8
        assert np.all(np.isfinite(advantages)), "Advantages contain NaN or inf"
        assert np.all(np.isfinite(returns)), "Returns contain NaN or inf"

    def test_compute_gae_empty_buffer(self):
        buf = PPORolloutBuffer(capacity=10, obs_dim=16, action_dim=3)
        advantages, returns = buf.compute_gae(next_value=0.0)
        assert len(advantages) == 0
        assert len(returns) == 0

    def test_get_data_shapes(self):
        buf = PPORolloutBuffer(capacity=10, obs_dim=16, action_dim=3, device="cpu")
        for i in range(10):
            buf.add(
                obs=np.random.randn(16),
                action=np.random.randn(3),
                log_prob=-0.5,
                reward=1.0,
                done=False,
                value=1.0,
            )
        buf.compute_gae(next_value=0.0)
        data = buf.get_data()
        assert data["obs"].shape == (10, 16)
        assert data["actions"].shape == (10, 3)
        assert data["log_probs"].shape == (10,)
        assert data["advantages"].shape == (10,)
        assert data["returns"].shape == (10,)

    def test_minibatch_iteration(self):
        buf = PPORolloutBuffer(capacity=10, obs_dim=16, action_dim=3, device="cpu")
        for i in range(10):
            buf.add(
                obs=np.random.randn(16),
                action=np.random.randn(3),
                log_prob=-0.5,
                reward=1.0,
                done=False,
                value=1.0,
            )
        buf.compute_gae(next_value=0.0)
        batches = list(buf.get_minibatches(batch_size=4))
        assert len(batches) > 0
        for batch in batches:
            assert batch["obs"].shape[0] <= 4

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            PPORolloutBuffer(capacity=0, obs_dim=16, action_dim=3)

    def test_invalid_obs_dim_raises(self):
        with pytest.raises(ValueError):
            PPORolloutBuffer(capacity=10, obs_dim=-1, action_dim=3)


# ---------------------------------------------------------------------------
# PPO Agent Tests
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def test_select_action_returns_correct_shapes(self):
        config = {
            "ppo": {
                "rollout_steps": 64,
                "minibatch_size": 32,
                "update_epochs": 2,
                "learning_rate": 3e-4,
            },
            "policy": {
                "hidden_sizes": [32, 32],
                "activation": "tanh",
                "action_dim": 3,
                "action_low": [-1500.0, -1500.0, -300.0],
                "action_high": [1500.0, 1500.0, 300.0],
            },
        }
        agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        obs = np.random.randn(16).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert action.shape == (3,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert np.isfinite(action).all()
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_deterministic_action(self):
        config = {
            "ppo": {"rollout_steps": 64, "minibatch_size": 32, "update_epochs": 2},
            "policy": {"hidden_sizes": [32, 32], "activation": "tanh", "action_dim": 3},
        }
        agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        obs = np.random.randn(16).astype(np.float32)
        a1 = agent.get_deterministic_action(obs)
        a2 = agent.get_deterministic_action(obs)
        assert np.allclose(a1, a2)

    def test_update_completes(self):
        config = {
            "ppo": {
                "rollout_steps": 32,
                "minibatch_size": 16,
                "update_epochs": 2,
                "learning_rate": 3e-4,
                "clip_coef": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "normalize_advantage": True,
            },
            "policy": {"hidden_sizes": [32, 32], "activation": "tanh", "action_dim": 3},
        }
        agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        # Fill buffer
        for i in range(32):
            obs = np.random.randn(16).astype(np.float32)
            action, log_prob, value = agent.select_action(obs)
            agent.store_transition(obs, action, log_prob, float(i % 5), False, value)

        next_obs = np.random.randn(16).astype(np.float32)
        stats = agent.update(next_obs=next_obs)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert "explained_variance" in stats
        assert np.isfinite(stats["policy_loss"])
        assert np.isfinite(stats["value_loss"])

    def test_checkpoint_save_load(self):
        config = {
            "ppo": {"rollout_steps": 64, "minibatch_size": 32, "update_epochs": 2},
            "policy": {"hidden_sizes": [32, 32], "activation": "tanh", "action_dim": 3},
        }
        agent1 = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        obs = np.random.randn(16).astype(np.float32)
        a1 = agent1.get_deterministic_action(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            agent1.save(path)
            assert os.path.exists(path)

            agent2 = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
            agent2.load(path)
            a2 = agent2.get_deterministic_action(obs)
            assert np.allclose(a1, a2, atol=1e-6)

    def test_load_nonexistent_raises(self):
        config = {
            "ppo": {"rollout_steps": 64, "minibatch_size": 32, "update_epochs": 2},
            "policy": {"hidden_sizes": [32, 32], "activation": "tanh", "action_dim": 3},
        }
        agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        with pytest.raises(FileNotFoundError):
            agent.load("/nonexistent/path/checkpoint.pt")

    def test_total_timesteps_counter(self):
        config = {
            "ppo": {"rollout_steps": 64, "minibatch_size": 32, "update_epochs": 2},
            "policy": {"hidden_sizes": [32, 32], "activation": "tanh", "action_dim": 3},
        }
        agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        assert agent.total_timesteps == 0
        obs = np.random.randn(16).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        agent.store_transition(obs, action, log_prob, 1.0, False, value)
        assert agent.total_timesteps == 1
