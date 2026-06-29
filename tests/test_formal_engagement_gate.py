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
                "mean_damage_dealt": 100.0,
                "mean_damage_taken": 5.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 0,
                "crashes": 0,
                "timeouts": 0,
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
                "mean_damage_dealt": 0.0,
                "mean_damage_taken": 80.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 0,
                "crashes": 0,
                "timeouts": 20,
            },
        }
    }

    report = analyze_engagement_gate(payload)

    head_on = report["groups"]["head_on::adversarial::expert::dNone::rNone"]
    crossing = report["groups"]["crossing_feasible::adversarial::end_to_end::dNone::rNone"]

    assert head_on["ready_for_scale"] is True
    assert head_on["recommended_action"] == "formal_expand"
    assert head_on["mean_damage_dealt"] == 100.0
    assert head_on["mean_damage_taken"] == 5.0
    assert head_on["ego_crashes"] == 0
    assert head_on["target_crash_or_oob"] == 0
    assert head_on["crashes"] == 0
    assert head_on["timeouts"] == 0

    assert crossing["ready_for_scale"] is False
    assert crossing["recommended_action"] == "hold_fix_engagement_quality"
    assert "effective_engagement_rate_below_gate" in crossing["gate_issues"]
    assert "damaging_win_rate_below_gate" in crossing["gate_issues"]
    assert crossing["mean_damage_dealt"] == 0.0
    assert crossing["mean_damage_taken"] == 80.0
    assert crossing["ego_crashes"] == 0
    assert crossing["target_crash_or_oob"] == 0
    assert crossing["timeouts"] == 20


def test_engagement_gate_accepts_runner_rows_payload():
    payload = {
        "rows": [
            {
                "method": "no_prediction_vpp",
                "task": "crossing_feasible",
                "opponent_stage": "expert",
                "episodes": 3,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "damage_exchange_rate": 1.0,
                "mean_damage_dealt": 70.0,
                "mean_damage_taken": 20.0,
                "ego_crashes": 1,
                "target_crash_or_oob": 2,
                "crashes": 3,
                "timeouts": 0,
            }
        ]
    }

    report = analyze_engagement_gate(payload)
    group = report["groups"]["crossing_feasible::no_prediction_vpp::expert"]

    assert group["ready_for_scale"] is True
    assert group["mean_damage_dealt"] == 70.0
    assert group["mean_damage_taken"] == 20.0
    assert group["ego_crashes"] == 1
    assert group["target_crash_or_oob"] == 2
    assert group["crashes"] == 3
