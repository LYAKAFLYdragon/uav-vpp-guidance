from __future__ import annotations

import copy
from pathlib import Path

import yaml

from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as pilot


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_defext_rangeext_feasibility_pilot_v1.yaml"
TRAIN = ROOT / "config" / "experiment" / "thesis_defext_rangeext_feasibility_pilot_v1_train_distribution.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _record(opponent: str, method: str, index: int, loss: float, *, ego_failure: bool = False):
    return {
        "opponent": opponent,
        "method": method,
        "scenario": f"scenario_{index}",
        "scenario_signature": f"sig_{index}",
        "mirror_sign": "negative" if index % 2 == 0 else "positive",
        "scenario_seed": 1000 + index,
        "handoff_reached": True,
        "handoff_step": 20,
        "valid_target_steps": 20,
        "qualifying": True,
        "intent_loss_auc20": loss,
        "range_opening_fraction_auc20": 0.5,
        "specific_energy_height_delta_m_at_20": 100.0,
        "target_attack_zone_exposure_fraction_auc20": 0.1,
        "ego_attack_zone_exposure_fraction_auc20": 0.2,
        "terminal_reason": "timeout_hp_advantage",
        "ego_failure": ego_failure,
        "win": True,
        "loss": False,
        "draw": False,
        "ego_hp": 90.0,
        "target_hp": 80.0,
    }


def test_train_sampler_stays_inside_preregistered_continuous_support():
    distribution = yaml.safe_load(TRAIN.read_text(encoding="utf-8"))
    contract = distribution["sampling_contract"]
    for index in range(20):
        scenario = pilot.sample_train_scenario(distribution, index)
        own = scenario["own_init"]
        target = scenario["target_init"]
        dx = target["position_m"][0] - own["position_m"][0]
        dy = target["position_m"][1] - own["position_m"][1]
        dz = target["position_m"][2] - own["position_m"][2]
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        assert contract["initial_range_m"][0] <= distance <= contract["initial_range_m"][1]
        assert target["velocity_mps"] - own["velocity_mps"] >= contract["require_target_faster_than_own_by_mps"]
        assert scenario["metadata"]["initial_class"] == "disadvantage"


def test_authorization_requires_exact_clean_sha_and_fresh_output(monkeypatch, tmp_path: Path):
    config = _config()
    authorization = config["authorization"]
    authorization.update(
        {
            "training_permitted": True,
            "pilot_execution_permitted": True,
            "baseline_evaluation_permitted": True,
            "heldout_evaluation_permitted": True,
            "scope": "single_defensive_extension_feasibility_pilot_only",
            "required_implementation_git_sha": "test-sha",
            "authorized_code_files": [
                {
                    "path": "src/uav_vpp_guidance/training/thesis_defext_rangeext_pilot.py",
                    "sha256": pilot.sha256_file(
                        ROOT / "src/uav_vpp_guidance/training/thesis_defext_rangeext_pilot.py"
                    ),
                }
            ],
        }
    )
    config["outputs"]["root"] = str(tmp_path / "fresh")
    monkeypatch.setattr(pilot, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(pilot, "_git_clean", lambda: True)
    monkeypatch.setattr(pilot, "_git_is_ancestor", lambda _commit: True)
    result = pilot.validate_authorization(config)
    assert result["git_sha"] == "test-sha"
    assert result["git_clean"] is True
    monkeypatch.setattr(pilot, "_git_clean", lambda: False)
    try:
        pilot.validate_authorization(config)
    except pilot.PilotContractError as exc:
        assert "clean HEAD" in str(exc)
    else:
        raise AssertionError("dirty authorization unexpectedly passed")


def test_heldout_decision_supports_new_skill_only_when_better_than_best_existing():
    config = _config()
    records = {}
    for opponent in pilot.OPPONENTS:
        records[opponent] = {
            "candidate_fixed_defensive_extension": [
                _record(opponent, "candidate_fixed_defensive_extension", index, 0.70)
                for index in range(24)
            ],
            "frozen_fixed_head_on": [
                _record(opponent, "frozen_fixed_head_on", index, 0.80)
                for index in range(24)
            ],
            "frozen_fixed_crossing": [
                _record(opponent, "frozen_fixed_crossing", index, 0.75)
                for index in range(24)
            ],
        }
    decision = pilot.analyze_heldout(config, records)
    assert decision["new_skill_gap_supported"] is True
    assert decision["verdict"] == "new_defensive_extension_skill_gap_preliminarily_supported"

    crossing_already_sufficient = copy.deepcopy(records)
    for opponent in pilot.OPPONENTS:
        for item in crossing_already_sufficient[opponent]["frozen_fixed_crossing"]:
            item["intent_loss_auc20"] = 0.68
    decision = pilot.analyze_heldout(config, crossing_already_sufficient)
    assert decision["new_skill_gap_supported"] is False
    assert decision["verdict"] == "existing_library_routing_or_composition_gap"
