from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_thesis_dualskill_taxonomy60_discriminability.py"


def _module():
    spec = importlib.util.spec_from_file_location("taxonomy60_discriminability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(method: str, mode: int, action: list[float]) -> dict[str, object]:
    return {
        "scenario": "scenario_a",
        "opponent_stage": "expert",
        "controller": method,
        "scenario_metadata": {"initial_class": "head_on"},
        "trajectory": [{"commander_mode_id": mode, "vpp_action": action}],
    }


def test_pair_row_marks_first_action_difference_without_guard() -> None:
    module = _module()
    row = module.pair_row(
        _record("static_oracle_gate", 0, [0.1, 0.0, 0.0]),
        _record("canonical_ppo", 0, [0.1, 0.2, 0.0]),
        comparison="M2_static_gate_vs_M3_canonical_ppo",
        left_guarded=False,
        right_guarded=False,
    )
    assert row["first_mode_different"] is False
    assert row["first_vpp_action_different"] is True
    assert row["first_decision_distinguishable"] is True
    assert row["routing_attribution_eligible"] is True


def test_pair_row_excludes_treatment_identical_m3_m4() -> None:
    module = _module()
    row = module.pair_row(
        _record("canonical_ppo", 1, [0.1, 0.0, 0.0]),
        _record("ppo_unbootstrapped_crossing", 1, [0.1, 0.0, 0.0]),
        comparison="M3_canonical_ppo_vs_M4_no_bootstrap",
        left_guarded=False,
        right_guarded=False,
    )
    assert row["first_decision_distinguishable"] is False
    assert row["paper_matrix_disposition"] == "excluded_treatment_identical"


def test_pair_row_marks_guarded_ppo_comparison_as_noncausal() -> None:
    module = _module()
    row = module.pair_row(
        _record("static_oracle_gate", 0, [0.1, 0.0, 0.0]),
        _record("canonical_ppo", 1, [0.2, 0.0, 0.0]),
        comparison="M2_static_gate_vs_M3_canonical_ppo",
        left_guarded=False,
        right_guarded=True,
    )
    assert row["first_decision_distinguishable"] is True
    assert row["routing_attribution_eligible"] is False
    assert row["paper_matrix_disposition"] == "deployed_interface_only"


def test_effective_mode_normalizes_fixed_and_runtime_head_on_receipts() -> None:
    module = _module()
    fixed = _record("fixed_head_on", None, [0.1, 0.0, 0.0])
    static = _record("static_oracle_gate", None, [0.1, 0.0, 0.0])
    static["trajectory"][0]["runtime_specialist_key"] = "head_on"
    row = module.pair_row(
        fixed,
        static,
        comparison="M0_fixed_head_on_vs_M2_static_gate",
        left_guarded=False,
        right_guarded=False,
    )
    assert row["left_first_effective_mode"] == "head_on"
    assert row["right_first_effective_mode"] == "head_on"
    assert row["first_mode_different"] is False


def test_pair_row_records_a_later_dynamic_decision_difference() -> None:
    module = _module()
    static = _record("static_oracle_gate", 0, [0.1, 0.0, 0.0])
    ppo = _record("canonical_ppo", 0, [0.1, 0.0, 0.0])
    static["trajectory"].append({"commander_mode_id": 0, "vpp_action": [0.1, 0.0, 0.0]})
    ppo["trajectory"].append({"commander_mode_id": 1, "vpp_action": [0.2, 0.0, 0.0]})
    row = module.pair_row(
        static,
        ppo,
        comparison="M2_static_gate_vs_M3_canonical_ppo",
        left_guarded=False,
        right_guarded=False,
    )
    assert row["first_decision_distinguishable"] is False
    assert row["episode_decision_distinguishable"] is True
    assert row["first_episode_decision_difference_step"] == 2
    assert row["routing_attribution_eligible"] is True
