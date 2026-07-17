from __future__ import annotations

import numpy as np

from uav_vpp_guidance.evaluation import temporal_schedule_specialist_policy as module


class _FakeSpecialist:
    def __init__(self, checkpoint_path: str, config_path: str, device: str) -> None:
        self.label = checkpoint_path

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([1.0 if self.label == "head.pt" else -1.0, 0.0, 0.0], dtype=np.float32)


class _FakeEnv:
    def __init__(self) -> None:
        self.contexts = []

    def set_runtime_specialist_context(self, **kwargs) -> None:
        self.contexts.append(kwargs)


def test_temporal_schedule_switches_at_predeclared_step(monkeypatch):
    monkeypatch.setattr(module, "FrozenSpecialistPolicy", _FakeSpecialist)
    policy = module.TemporalScheduleSpecialistPolicy(
        specialists_config={
            "head": {"checkpoint": "head.pt", "config_path": "head.yaml", "specialist_key": "head_on"},
            "cross": {"checkpoint": "cross.pt", "config_path": "cross.yaml", "specialist_key": "crossing_feasible"},
        },
        initial_specialist="head",
        later_specialist="cross",
        initial_high_level_steps=2,
    )
    env = _FakeEnv()
    policy.set_env(env)

    np.testing.assert_array_equal(policy.get_deterministic_action(np.zeros(3)), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(policy.get_deterministic_action(np.zeros(3)), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(policy.get_deterministic_action(np.zeros(3)), np.array([-1.0, 0.0, 0.0]))
    metadata = policy.get_last_step_metadata()
    assert metadata["temporal_schedule_phase"] == "later"
    assert metadata["temporal_schedule_switch_count"] == 1
    assert metadata["temporal_schedule_first_switch_step"] == 3
    assert metadata["temporal_schedule_uses_future_state"] is False
