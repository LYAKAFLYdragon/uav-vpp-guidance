from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("two_node_runtime_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_is_scenario_conditioned_not_outlier_based():
    module = load_module()
    summary = module.summarize(
        [
            {"opponent": "end_to_end", "candidate": "x625_y200", "response_offset": "12", "delta_range_m": "0.1", "delta_range_rate_mps": "-0.2"},
            {"opponent": "end_to_end", "candidate": "x625_y200", "response_offset": "12", "delta_range_m": "0.1", "delta_range_rate_mps": "-0.2"},
            {"opponent": "end_to_end", "candidate": "x625_y200", "response_offset": "12", "delta_range_m": "0.1", "delta_range_rate_mps": "-0.2"},
            {"opponent": "end_to_end", "candidate": "x750_y250", "response_offset": "12", "delta_range_m": "0.1", "delta_range_rate_mps": "0.2"},
            {"opponent": "end_to_end", "candidate": "x750_y250", "response_offset": "12", "delta_range_m": "0.1", "delta_range_rate_mps": "0.2"},
            {"opponent": "end_to_end", "candidate": "x750_y250", "response_offset": "12", "delta_range_m": "0.1", "delta_range_rate_mps": "0.2"},
        ],
        [],
    )
    assert summary["attribution"] == "scenario_conditioned_closed_loop_response"
    assert summary["scenario_summary"]["x625_y200"]["range_rate_positive_count"] == 0
    assert summary["scenario_summary"]["x750_y250"]["range_rate_positive_count"] == 3


def test_audit_requires_eighteen_end_to_end_pair_rows(tmp_path):
    module = load_module()
    (tmp_path / "paired_response_records.csv").write_text("opponent,candidate,response_offset\nend_to_end,x625_y200,12\n", encoding="utf-8")
    try:
        module.load_pair_audit(tmp_path)
    except RuntimeError as exc:
        assert "expected 18" in str(exc)
    else:
        raise AssertionError("incomplete audit input was accepted")
