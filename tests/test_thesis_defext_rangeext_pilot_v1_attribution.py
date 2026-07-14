from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from uav_vpp_guidance.evaluation import thesis_defext_rangeext_attribution as audit


ROOT = Path(__file__).resolve().parent.parent
CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_defext_rangeext_feasibility_pilot_v1.yaml"
)


def _record(
    opponent: str,
    method: str,
    signature: str,
    *,
    handoff_step: int | None = 40,
    qualifying: bool = True,
    loss: float | None = 0.7,
    ego_failure: bool = False,
) -> dict:
    reached = handoff_step is not None
    return {
        "opponent": opponent,
        "method": method,
        "scenario": signature.replace("|", "_"),
        "scenario_signature": signature,
        "mirror_sign": "positive",
        "scenario_seed": 1,
        "handoff_reached": reached,
        "handoff_step": handoff_step,
        "valid_target_steps": 20 if qualifying else 4 if reached else 0,
        "qualifying": qualifying,
        "intent_loss_auc20": loss if qualifying else None,
        "range_opening_fraction_auc20": 0.5 if reached else None,
        "specific_energy_height_delta_m_at_20": 100.0 if qualifying else None,
        "target_attack_zone_exposure_fraction_auc20": 0.1 if reached else None,
        "ego_attack_zone_exposure_fraction_auc20": 0.2 if reached else None,
        "terminal_reason": (
            "ego_crash_or_out_of_bounds"
            if ego_failure
            else "timeout_hp_disadvantage"
        ),
        "ego_failure": ego_failure,
        "win": False,
        "loss": True,
        "draw": False,
        "ego_hp": 60.0,
        "target_hp": 90.0,
    }


def test_attribution_separates_safety_signal_from_handoff_mismatch():
    opponent = "expert"
    safety = "defext_hold_d|co_altitude|positive"
    mismatch = "defext_hold_a|own_below|positive"
    records = {
        opponent: {
            audit.CANDIDATE: [
                _record(opponent, audit.CANDIDATE, safety, ego_failure=True),
                _record(opponent, audit.CANDIDATE, mismatch, handoff_step=50),
            ],
            audit.HEAD_ON: [
                _record(opponent, audit.HEAD_ON, safety, loss=0.75),
                _record(opponent, audit.HEAD_ON, mismatch, handoff_step=51),
            ],
            audit.CROSSING: [
                _record(opponent, audit.CROSSING, safety, loss=0.72),
                _record(opponent, audit.CROSSING, mismatch, handoff_step=49),
            ],
        }
    }
    result = audit.analyze_records(records)
    rows = {row["scenario_signature"]: row for row in result["rows"]}
    assert rows[safety]["candidate_specific_failure_same_handoff"] is True
    assert "candidate_specific_safety_signal" in rows[safety]["attribution_tags"]
    assert rows[mismatch]["all_method_handoff_metadata_equivalent"] is False
    assert rows[mismatch]["candidate_head_strict_pairable"] is False
    assert result["summary"]["posthoc_attribution_verdict"] == (
        "causal_mechanism_not_identifiable_from_frozen_artifacts"
    )
    assert result["summary"]["step_telemetry_available"] is False


def _frozen_result_root() -> Path:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return Path(
        os.environ.get("DEFEXT_PILOT_RESULT_ROOT", config["outputs"]["root"])
    )


@pytest.mark.skipif(
    not (_frozen_result_root() / "heldout" / "records.json").is_file(),
    reason="external frozen pilot evidence is not available",
)
def test_frozen_pilot_attribution_facts():
    records = json.loads(
        (_frozen_result_root() / "heldout" / "records.json").read_text(
            encoding="utf-8"
        )
    )
    result = audit.analyze_records(records)
    summary = result["summary"]
    assert summary["row_count"] == 72
    expected = {
        "expert": (10, 9, 5, 11, 6, 1),
        "end_to_end": (15, 11, 7, 12, 6, 0),
        "independent_ppo_vpp": (7, 6, 3, 6, 4, 2),
    }
    for opponent, values in expected.items():
        item = summary["per_opponent"][opponent]
        assert (
            item["pre_handoff_mismatch_scenarios"],
            item["candidate_head_loose_pairs"],
            item["candidate_head_strict_pairs"],
            item["candidate_crossing_loose_pairs"],
            item["candidate_crossing_strict_pairs"],
            item["candidate_specific_failures_same_handoff"],
        ) == values
    safety_signatures = {
        (row["opponent"], row["scenario_signature"])
        for row in summary["candidate_specific_safety_cases"]
    }
    assert safety_signatures == {
        ("expert", "defext_hold_d|co_altitude|positive"),
        ("independent_ppo_vpp", "defext_hold_c|own_above|positive"),
        ("independent_ppo_vpp", "defext_hold_d|co_altitude|positive"),
    }
    assert summary["formal_verdict_unchanged"] == "safety_or_contract_no_go"
    assert summary["new_skill_gap_supported"] is False
