from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as pilot


ROOT = Path(__file__).resolve().parent.parent
DESIGN_CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_pilot_v1.yaml"
TRAIN_DISTRIBUTION = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_train_distribution.yaml"


def _config() -> dict:
    return yaml.safe_load(DESIGN_CONFIG.read_text(encoding="utf-8"))


def _record(method: str, index: int, loss: float, *, handoff: str = "same") -> dict:
    return {
        "method": method,
        "scenario_signature": f"s{index}",
        "mirror_sign": "negative" if index % 2 == 0 else "positive",
        "handoff_state_sha256": handoff,
        "handoff_reached": True,
        "valid_target_steps": 20,
        "qualifying": True,
        "intent_loss_auc20": loss,
        "ego_failure": False,
        "telemetry_complete": True,
    }


def _records() -> dict:
    records = {}
    for opponent in pilot.OPPONENTS:
        records[opponent] = {
            "candidate_fixed_reentry_recovery": [_record("candidate", index, 0.70, handoff=f"h{index}") for index in range(24)],
            "frozen_fixed_head_on": [_record("head", index, 0.80, handoff=f"h{index}") for index in range(24)],
            "frozen_fixed_crossing": [_record("cross", index, 0.75, handoff=f"h{index}") for index in range(24)],
        }
    return records


def test_neutral_sampler_uses_only_the_frozen_train_support():
    distribution = yaml.safe_load(TRAIN_DISTRIBUTION.read_text(encoding="utf-8"))
    contract = distribution["sampling_contract"]
    for index in range(20):
        scenario = pilot.sample_train_scenario(distribution, index)
        own, target = scenario["own_init"], scenario["target_init"]
        distance = sum((target["position_m"][axis] - own["position_m"][axis]) ** 2 for axis in range(3)) ** 0.5
        assert contract["initial_range_m"][0] <= distance <= contract["initial_range_m"][1]
        assert target["velocity_mps"] - own["velocity_mps"] >= contract["require_target_faster_than_own_by_mps"]
        assert scenario["metadata"]["initial_class"] == "neutral"
        assert scenario["metadata"]["scenario_signature"].startswith("neutral_train_")
        assert contract["own_to_target_los_angle_deg"][1] <= 165.0
        assert contract["target_velocity_to_own_los_angle_deg"][1] <= 165.0


def test_target_contract_changes_only_inside_the_explicit_context():
    from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy

    before = (legacy.TARGET_SKILL, legacy.TARGET_PROFILE)
    with pilot._legacy_target_contract():
        assert (legacy.TARGET_SKILL, legacy.TARGET_PROFILE) == ("reentry_recovery", "reentry_preparation")
    assert (legacy.TARGET_SKILL, legacy.TARGET_PROFILE) == before


def test_heldout_decision_requires_paired_handoff_hashes_and_all_three_opponents():
    decision = pilot.analyze_heldout(_config(), _records())
    assert decision["new_skill_gap_supported"] is True
    assert decision["verdict"] == "new_reentry_recovery_skill_gap_preliminarily_supported"

    mismatched = copy.deepcopy(_records())
    mismatched["expert"]["frozen_fixed_head_on"][0]["handoff_state_sha256"] = "different"
    with pytest.raises(pilot.NeutralPostMergePilotError, match="handoff state hashes"):
        pilot.analyze_heldout(_config(), mismatched)


def test_design_config_is_not_an_execution_config():
    config = _config()
    assert config["authorization"]["execution_permitted"] is False
    assert config["methods"]["candidate_fixed_reentry_recovery"]["checkpoint"] is None
