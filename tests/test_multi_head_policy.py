import numpy as np
import pytest
import torch

from uav_vpp_guidance.agents.policy_network import MLPActorCritic
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


class TestMultiHeadPolicyNetwork:
    """Test task-conditioned multi-head actor network."""

    def test_single_head_backward_compatible(self):
        """num_tasks=1 should behave exactly like the old single-head network."""
        net = MLPActorCritic(
            obs_dim=10,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=1,
        )
        obs = torch.randn(5, 10)
        mean, value = net.forward(obs)
        assert mean.shape == (5, 3)
        assert value.shape == (5,)
        # Should have actor_mean and actor_log_std
        assert hasattr(net, "actor_mean")
        assert hasattr(net, "actor_log_std")
        # Should NOT have actor_means or actor_log_stds
        assert not hasattr(net, "actor_means")
        assert not hasattr(net, "actor_log_stds")

    def test_multi_head_forward_shape(self):
        """num_tasks=2 should produce correct output shapes."""
        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        obs = torch.randn(5, 11)
        # Set last dim to 0.0 (head_on) and 1.0 (crossing)
        obs[0:2, -1] = 0.0
        obs[2:5, -1] = 1.0
        mean, value = net.forward(obs)
        assert mean.shape == (5, 3)
        assert value.shape == (5,)
        # Should have actor_means and actor_log_stds
        assert hasattr(net, "actor_means")
        assert hasattr(net, "actor_log_stds")
        assert len(net.actor_means) == 2
        assert len(net.actor_log_stds) == 2

    def test_multi_head_task_selection(self):
        """Different tasks should produce different means."""
        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        # Fix seed for reproducibility
        torch.manual_seed(42)
        obs = torch.randn(1, 11)
        obs[:, -1] = 0.0
        mean_0, _ = net.forward(obs)
        obs[:, -1] = 1.0
        mean_1, _ = net.forward(obs)
        # Different tasks should produce different means (with high probability)
        assert not torch.allclose(mean_0, mean_1)

    def test_multi_head_get_action_and_value(self):
        """get_action_and_value should work with multi-head network."""
        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        obs = torch.randn(5, 11)
        obs[:, -1] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape == (5, 3)
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5,)

    def test_multi_head_evaluate_action(self):
        """Evaluating an existing action should work with multi-head."""
        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        obs = torch.randn(5, 11)
        obs[:, -1] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        action = torch.randn(5, 3)
        log_prob, entropy, value = net.get_action_and_value(obs, action=action)
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5,)

    def test_multi_head_deterministic_action(self):
        """get_deterministic_action should work with multi-head."""
        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        obs = torch.randn(1, 11)
        obs[:, -1] = 0.0
        action_0 = net.get_deterministic_action(obs)
        obs[:, -1] = 1.0
        action_1 = net.get_deterministic_action(obs)
        assert action_0.shape == (1, 3)
        assert action_1.shape == (1, 3)
        assert not torch.allclose(action_0, action_1)

    def test_multi_head_parameter_count(self):
        """Multi-head should have more parameters than single-head."""
        net_single = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=1,
        )
        net_multi = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        assert net_multi.count_parameters() > net_single.count_parameters()

    def test_ppo_agent_with_multi_head(self):
        """PPOAgent should correctly pass num_tasks to network."""
        config = {
            "policy": {
                "hidden_sizes": [8, 8],
                "num_tasks": 2,
            },
            "ppo": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_coef": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "update_epochs": 1,
                "minibatch_size": 32,
                "normalize_advantage": True,
                "rollout_steps": 64,
            },
        }
        agent = PPOAgent(
            obs_dim=11,
            action_dim=3,
            config=config,
            device="cpu",
        )
        assert agent.network.num_tasks == 2
        assert hasattr(agent.network, "actor_means")
        assert len(agent.network.actor_means) == 2

        # Test select_action
        obs = np.random.randn(11).astype(np.float32)
        obs[-1] = 0.0
        action, log_prob, value = agent.select_action(obs, deterministic=False)
        assert action.shape == (3,)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

        # Test get_deterministic_action
        obs[-1] = 1.0
        action = agent.get_deterministic_action(obs)
        assert action.shape == (3,)

    def test_multi_head_save_load(self):
        """Save and load should preserve multi-head parameters."""
        import tempfile
        import os

        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        state = net.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        torch.save(state, path)
        try:
            loaded_state = torch.load(path, map_location="cpu")
            net2 = MLPActorCritic(
                obs_dim=11,
                action_dim=3,
                hidden_sizes=[8, 8],
                num_tasks=2,
            )
            net2.load_state_dict(loaded_state, strict=True)
            # Verify parameters match
            for p1, p2 in zip(net.parameters(), net2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_extract_task_id_edge_cases(self):
        """Task ID extraction should handle edge cases."""
        net = MLPActorCritic(
            obs_dim=11,
            action_dim=3,
            hidden_sizes=[8, 8],
            num_tasks=2,
        )
        # Test batch with mixed values
        obs = torch.randn(5, 11)
        obs[:, -1] = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        task_id = net._extract_task_id(obs)
        assert task_id.shape == (5,)
        # Values > 0.5 should be task 1, <= 0.5 should be task 0
        expected = torch.tensor([0, 0, 0, 1, 1])
        assert torch.all(task_id == expected)
