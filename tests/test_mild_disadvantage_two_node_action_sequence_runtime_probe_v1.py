from pathlib import Path
import importlib.util

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py"
VERIFY = ROOT / "scripts/verify_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py"
CONFIG = ROOT / "config/experiment/mild_disadvantage_two_node_action_sequence_runtime_probe_v1.yaml"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def get_deterministic_action(self, _observation):
        return self.action.copy()


def test_preregistered_sequence_and_gate_are_fixed():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["node0_delta"] == [0.0, 0.08, 0.0]
    assert config["protocol"]["node1_delta"] == [-0.08, 0.0, 0.0]
    assert config["preregistered_gate"]["minimum_mean_range_delta_m"] == 0.001
    assert config["preregistered_gate"]["minimum_mean_range_rate_delta_mps"] == 0.002


def test_two_node_policy_applies_only_at_call_zero_and_eight():
    runner = load(RUNNER, "two_node_runtime_runner")
    policy = runner.TwoNodePolicy(FixedPolicy([0.2, -0.1, 0.3]), True, np.array([0.0, 0.08, 0.0]), np.array([-0.08, 0.0, 0.0]))
    actions = [policy.get_deterministic_action(None) for _ in range(10)]
    np.testing.assert_allclose(actions[0], [0.2, -0.02, 0.3])
    np.testing.assert_allclose(actions[8], [0.12, -0.1, 0.3])
    np.testing.assert_allclose(actions[1], [0.2, -0.1, 0.3])
    assert [record["node"] for record in policy.records].count("node0") == 1
    assert [record["node"] for record in policy.records].count("node1") == 1


def test_verifier_rejects_missing_probe():
    verifier = load(VERIFY, "two_node_runtime_verifier")
    result = verifier.verify(ROOT / "outputs" / "__missing_two_node_runtime_probe__")
    assert result["ok"] is False
    assert result["errors"]
