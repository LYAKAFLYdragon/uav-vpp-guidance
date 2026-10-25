from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/analyze_mild_disadvantage_envelope_recovery_matched_chain_v1.py"


def _load():
    spec = spec_from_file_location("matched_chain_analysis", SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classifies_available_and_missing_chain_fields():
    module = _load()
    classification = module.classify_columns(
        {
            "action_0",
            "action_1",
            "action_2",
            "range_m",
            "range_rate_mps",
            "ata_deg",
            "aa_deg",
            "altitude_m",
        }
    )
    assert classification["compiled_action"]["present"] is True
    assert classification["geometry"]["present"] is True
    assert classification["vpp_actual"]["present"] is False
    assert classification["guidance_command"]["present"] is False
    assert classification["roll_response"]["present"] is False


def test_direction_summary_handles_mixed_signs():
    module = _load()
    summary = module._direction([1.0, -1.0, 0.0])
    assert summary["n"] == 3
    assert summary["dominant_direction"] == "mixed_or_zero"
    assert summary["directional_consistency"] == 1 / 3
