import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_formal_engagement_gate import analyze_engagement_gate


def test_engagement_gate_flags_crossing_without_damaging_wins():
    payload = {
        "summary": {
            "head_on::adversarial::expert::dNone::rNone": {
                "task": "head_on",
                "mode": "adversarial",
                "opponent_stage": "expert",
                "episodes": 20,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "damage_exchange_rate": 1.0,
            },
            "crossing_feasible::adversarial::end_to_end::dNone::rNone": {
                "task": "crossing_feasible",
                "mode": "adversarial",
                "opponent_stage": "end_to_end",
                "episodes": 20,
                "win_rate": 1.0,
                "effective_engagement_rate": 0.0,
                "damaging_win_rate": 0.0,
                "damage_exchange_rate": 1.0,
            },
        }
    }

    report = analyze_engagement_gate(payload)

    head_on = report["groups"]["head_on::adversarial::expert::dNone::rNone"]
    crossing = report["groups"]["crossing_feasible::adversarial::end_to_end::dNone::rNone"]

    assert head_on["ready_for_scale"] is True
    assert head_on["recommended_action"] == "formal_expand"

    assert crossing["ready_for_scale"] is False
    assert crossing["recommended_action"] == "hold_fix_engagement_quality"
    assert "effective_engagement_rate_below_gate" in crossing["gate_issues"]
    assert "damaging_win_rate_below_gate" in crossing["gate_issues"]
