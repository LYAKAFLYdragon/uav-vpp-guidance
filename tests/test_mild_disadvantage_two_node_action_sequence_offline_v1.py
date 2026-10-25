from pathlib import Path
import importlib.util

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/screen_mild_disadvantage_two_node_action_sequence_offline_v1.py"
CONFIG = ROOT / "config/experiment/mild_disadvantage_two_node_action_sequence_offline_screen_v1.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("two_node_sequence_screen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_node_grid_has_zero_and_nonzero_vectors():
    module = load_module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vectors = module.action_vectors(config["sequence"]["action_levels"])
    assert len(vectors) == 27
    assert (0.0, 0.0, 0.0) in vectors
    assert (0.08, -0.08, 0.0) in vectors


def test_delayed_node_uses_h4_sensitivity_at_h12():
    module = load_module()
    rows = [
        {"axis": "0", "response_offset": "4", "range_m_sensitivity_per_action": "1.0"},
        {"axis": "1", "response_offset": "4", "range_m_sensitivity_per_action": "2.0"},
        {"axis": "2", "response_offset": "4", "range_m_sensitivity_per_action": "3.0"},
        {"axis": "0", "response_offset": "12", "range_m_sensitivity_per_action": "4.0"},
        {"axis": "1", "response_offset": "12", "range_m_sensitivity_per_action": "5.0"},
        {"axis": "2", "response_offset": "12", "range_m_sensitivity_per_action": "6.0"},
    ]
    node0 = (0.08, 0.0, 0.0)
    node1 = (0.0, 0.08, 0.0)
    assert module.sensitivity(rows, "range_m", 12, node0) == pytest.approx(0.32)
    assert module.sensitivity(rows, "range_m", 4, node1) == pytest.approx(0.16)
