"""
Tests for the Intentional Updates PPO agent.
"""

import numpy as np
import pytest

from uav_vpp_guidance.ablations.intentional.intentional_ppo_agent import IntentionalPPOAgent


class TestIntentionalPPOAgent:
    def _make_config(self, **overrides):
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
                "use_intentional_critic": True,
                "use_intentional_actor": True,
                "use_combat_aware_eta": False,
                "eta_critic": 0.1,
                "eta_actor": 0.01,
                "iu_eps": 1e-8,
                "beta_adv": 0.999,
            },
            "policy": {
                "hidden_sizes": [32, 32],
                "activation": "tanh",
                "action_dim": 3,
                "action_low": [-1.0, -1.0, -1.0],
                "action_high": [1.0, 1.0, 1.0],
            },
        }
        config["ppo"].update(overrides)
        return config

    def _fill_buffer(self, agent, n=32, info=None):
        for i in range(n):
            obs = np.random.randn(16).astype(np.float32)
            action, log_prob, value = agent.select_action(obs)
            agent.store_transition(obs, action, log_prob, float(i % 5), False, value, info=info)

    def test_standard_ppo_fallback(self):
        config = self._make_config(
            use_intentional_critic=False,
            use_intentional_actor=False,
        )
        agent = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        self._fill_buffer(agent)
        stats = agent.update(next_obs=np.random.randn(16).astype(np.float32))
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert stats["scale_actor"] == pytest.approx(1.0)
        assert stats["scale_critic"] == pytest.approx(1.0)

    def test_intentional_critic_only(self):
        config = self._make_config(
            use_intentional_critic=True,
            use_intentional_actor=False,
        )
        agent = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        self._fill_buffer(agent)
        stats = agent.update(next_obs=np.random.randn(16).astype(np.float32))
        assert stats["scale_critic"] != pytest.approx(1.0)
        assert stats["scale_actor"] == pytest.approx(1.0)

    def test_intentional_actor_only(self):
        config = self._make_config(
            use_intentional_critic=False,
            use_intentional_actor=True,
        )
        agent = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        self._fill_buffer(agent)
        stats = agent.update(next_obs=np.random.randn(16).astype(np.float32))
        assert stats["scale_actor"] != pytest.approx(1.0)
        assert stats["scale_critic"] == pytest.approx(1.0)

    def test_intentional_actor_and_critic(self):
        config = self._make_config(
            use_intentional_critic=True,
            use_intentional_actor=True,
        )
        agent = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        self._fill_buffer(agent)
        stats = agent.update(next_obs=np.random.randn(16).astype(np.float32))
        assert stats["scale_actor"] != pytest.approx(1.0)
        assert stats["scale_critic"] != pytest.approx(1.0)

    def test_combat_aware_buffer_populated(self):
        config = self._make_config(use_combat_aware_eta=True)
        agent = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        assert agent.combat_schedule is not None

        rel_state = {
            "range_m": 7000.0,
            "aa_rad": 0.0,
            "ata_rad": 0.0,
            "altitude_diff_m": 0.0,
            "speed_diff_mps": 0.0,
            "range_rate_mps": -100.0,
            "missile_threat": 0.0,
        }
        info = {"relative_state": rel_state}
        self._fill_buffer(agent, info=info)
        assert len(agent.phase_buffer) == 32

        stats = agent.update(next_obs=np.random.randn(16).astype(np.float32))
        assert len(agent.phase_buffer) == 0  # cleared after update
        assert "scale_actor" in stats

    def test_checkpoint_save_load(self):
        import os
        import tempfile

        config = self._make_config()
        agent1 = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
        obs = np.random.randn(16).astype(np.float32)
        a1 = agent1.get_deterministic_action(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            agent1.save(path)

            agent2 = IntentionalPPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
            agent2.load(path)
            a2 = agent2.get_deterministic_action(obs)
            assert np.allclose(a1, a2, atol=1e-6)
