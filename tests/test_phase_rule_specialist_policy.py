from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

from uav_vpp_guidance.evaluation import phase_rule_specialist_policy as module


ROOT = Path(__file__).resolve().parent.parent


class _FakeSpecialist:
    def __init__(self, checkpoint_path: str, config_path: str, device: str) -> None:
        self.label = checkpoint_path

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([1.0 if self.label == "head.pt" else -1.0, 0.0, 0.0], dtype=np.float32)


class _FakeEnv:
    def __init__(self) -> None:
        self.first_pass_complete = False
        self.contexts = []

    def get_runtime_phase_state(self) -> dict:
        return {"first_pass_complete": self.first_pass_complete, "current_step": 0}

    def set_runtime_specialist_context(self, **kwargs) -> None:
        self.contexts.append(kwargs)


def _policy(monkeypatch):
    monkeypatch.setattr(module, "FrozenSpecialistPolicy", _FakeSpecialist)
    return module.PhaseRuleSpecialistPolicy(
        specialists_config={
            "head_on": {"checkpoint": "head.pt", "config_path": "head.yaml", "specialist_key": "head_on"},
            "crossing": {"checkpoint": "cross.pt", "config_path": "cross.yaml", "specialist_key": "crossing_feasible"},
        },
        pre_merge_specialist="head_on",
        post_merge_specialist="crossing",
    )


def test_phase_rule_switches_only_on_observed_first_pass(monkeypatch):
    policy = _policy(monkeypatch)
    env = _FakeEnv()
    policy.set_env(env)
    policy.set_task_name("head_on")

    np.testing.assert_array_equal(policy.get_deterministic_action(np.zeros(3)), np.array([1.0, 0.0, 0.0]))
    assert policy.get_last_step_metadata()["phase_rule_schedule_phase"] == "pre_merge"
    assert policy.get_last_step_metadata()["phase_rule_uses_future_state"] is False

    env.first_pass_complete = True
    np.testing.assert_array_equal(policy.get_deterministic_action(np.zeros(3)), np.array([-1.0, 0.0, 0.0]))
    metadata = policy.get_last_step_metadata()
    assert metadata["phase_rule_schedule_phase"] == "post_merge"
    assert metadata["phase_rule_selected_specialist"] == "crossing_feasible"
    assert metadata["phase_rule_switch_count"] == 1
    assert metadata["phase_rule_first_switch_step"] == 2
    assert env.contexts[-1]["specialist_reason"] == "phase_rule_observed_first_pass"


def test_phase_rule_reset_clears_switch_history(monkeypatch):
    policy = _policy(monkeypatch)
    env = _FakeEnv()
    policy.set_env(env)
    policy.get_deterministic_action(np.zeros(3))
    env.first_pass_complete = True
    policy.get_deterministic_action(np.zeros(3))

    policy.reset_episode()
    env.first_pass_complete = False
    policy.get_deterministic_action(np.zeros(3))
    assert policy.get_last_step_metadata()["phase_rule_switch_count"] == 0
    assert policy.get_last_step_metadata()["phase_rule_first_switch_step"] is None


def test_runner_preflight_checks_phase_rule_specialist_assets(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_jsbsim_hrl_comparison as runner

    checkpoint = tmp_path / "head.pt"
    config = tmp_path / "head.yaml"
    checkpoint.write_bytes(b"frozen")
    config.write_text("policy: {}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_repo_path", lambda value: None if value is None else tmp_path / str(value))
    comparison = {
        "methods": {
            "phase": {
                "agent_type": "phase_rule_specialist",
                "specialists": {
                    "head": {"checkpoint": "head.pt", "config_path": "head.yaml"},
                    "cross": {"checkpoint": "missing.pt", "config_path": "missing.yaml"},
                },
            }
        },
        "opponent_registry": {},
    }

    checks, errors = runner._checkpoint_checks(comparison, ["phase"], "none", allow_missing=False)

    assert len(checks) == 4
    assert any("specialist cross checkpoint missing" in error for error in errors)
    assert any("specialist cross config missing" in error for error in errors)
