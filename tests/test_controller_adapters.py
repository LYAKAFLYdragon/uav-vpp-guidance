"""Unit tests for controller adapters."""

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.controller_adapters import (
    ControllerAdapter,
    PPOAdapter,
    ZeroOffsetAdapter,
)


class DummyAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def get_deterministic_action(self, obs):
        return self.action


def test_ppo_adapter_returns_action():
    agent = DummyAgent([0.1, -0.2, 0.3])
    adapter = PPOAdapter(agent, name="ppo")
    obs = {"observation_vector": np.zeros(10)}
    action = adapter.act(obs)
    np.testing.assert_allclose(action, [0.1, -0.2, 0.3], rtol=1e-6)
    assert adapter.name == "ppo"


def test_zero_offset_adapter_returns_zeros():
    adapter = ZeroOffsetAdapter(name="baseline_pid", action_dim=3)
    obs = {"observation_vector": np.zeros(10)}
    action = adapter.act(obs)
    assert action.shape == (3,)
    np.testing.assert_array_equal(action, np.zeros(3, dtype=np.float32))


def test_zero_offset_adapter_action_dim():
    adapter = ZeroOffsetAdapter(name="enhanced_pid", action_dim=4)
    action = adapter.act({"observation_vector": np.zeros(5)})
    assert action.shape == (4,)


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        ControllerAdapter()
