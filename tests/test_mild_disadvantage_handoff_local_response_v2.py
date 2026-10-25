from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mild_disadvantage_handoff_local_response_collector_v1.py"
CONFIG = ROOT / "config/experiment/mild_disadvantage_handoff_local_response_collector_v2.yaml"


def load_runner():
    spec = importlib.util.spec_from_file_location("local_response_collector_v2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)

    def get_deterministic_action(self, _observation):
        return self.action.copy()


def test_signed_pulses_are_opposite_and_axis_local():
    runner = load_runner()
    base = FixedPolicy([0.20, -0.10, 0.30])
    plus = runner.HandoffPulsePolicy(base, "axis1_plus", 0.08)
    minus = runner.HandoffPulsePolicy(base, "axis1_minus", 0.08)

    plus_action = plus.get_deterministic_action(None)
    minus_action = minus.get_deterministic_action(None)

    np.testing.assert_allclose(plus_action - plus.base_action, [0.0, 0.08, 0.0])
    np.testing.assert_allclose(minus_action - minus.base_action, [0.0, -0.08, 0.0])
    assert plus.pulse_sign == 1
    assert minus.pulse_sign == -1
    assert not plus.pulse_clipped
    assert not minus.pulse_clipped


def test_v2_config_has_independent_source_and_output():
    import yaml

    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert spec["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V2-20260904"
    assert spec["experiment"]["output_root"] == "outputs/mild_disadvantage_handoff_local_response_v2"
    assert spec["protocol"]["pulse_contract"] == "signed_axis_delta"


def test_analyzer_locates_opponent_scoped_trace_layout():
    analyzer_path = ROOT / "scripts/analyze_mild_disadvantage_handoff_local_response_v1.py"
    spec = importlib.util.spec_from_file_location("local_response_analyzer_v2", analyzer_path)
    analyzer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(analyzer)
    root = ROOT / "outputs/mild_disadvantage_handoff_local_response_v2_smoke"
    located = analyzer.locate_trace_file(root, "expert/x625_y200_01.csv", "axis0_minus")
    assert located.is_file()
