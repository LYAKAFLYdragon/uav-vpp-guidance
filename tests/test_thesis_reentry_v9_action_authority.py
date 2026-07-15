from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from uav_vpp_guidance.evaluation.thesis_reentry_v9_action_authority import (
    ActionAuthorityContractError,
    OPPONENTS,
    apply_perturbation,
    is_ego_failure,
    summarize_records,
)


def test_apply_perturbation_records_clipping_and_actual_delta():
    result = apply_perturbation([0.9, 0.0, 0.0], [0.25, 0.0, 0.0])

    assert result["applied_action"] == pytest.approx([1.0, 0.0, 0.0])
    assert result["applied_delta"] == pytest.approx([0.1, 0.0, 0.0])
    assert result["clipped"] is True


def test_ego_failure_never_mislabels_target_failure():
    assert is_ego_failure("ego_crash_or_out_of_bounds") is True
    assert is_ego_failure("target_crash_or_out_of_bounds") is False


def _record(method: str, opponent: str, pair_key: str, loss: float, *, failure: bool = False) -> dict:
    return {
        "method": method,
        "opponent": opponent,
        "pair_key": pair_key,
        "handoff_reached": True,
        "window_steps": 20,
        "contract_passed": True,
        "intent_loss_auc20": loss,
        "mean_applied_delta_norm": 0.25 if method != "frozen_fixed_head_on" else 0.0,
        "mirror_sign": "negative" if pair_key.endswith("0") else "positive",
        "height_condition": "own_below" if pair_key.endswith("0") else "co_altitude",
        "handoff_state_hash": "same",
        "ego_failure": failure,
    }


def test_summary_requires_two_opponents_with_practical_safe_delta():
    records = []
    for opponent in OPPONENTS:
        for index in range(12):
            key = f"{opponent}-{index}"
            records.append(_record("frozen_fixed_head_on", opponent, key, 1.0))
            records.append(_record("forward_positive", opponent, key, 0.97 if opponent != "independent_ppo_vpp" else 1.01))
    result = summarize_records(
        records,
        method_deltas={"frozen_fixed_head_on": [0, 0, 0], "forward_positive": [0.25, 0, 0]},
        min_window_steps=20,
        min_pairs=4,
        min_applied_delta_norm=0.15,
        practical_delta=-0.02,
        safety_delta_max=0.05,
    )

    assert result["authority_candidates"] == ["forward_positive"]
    assert result["passes"] is True


def test_summary_rejects_nonmatching_handoff_hashes():
    records = []
    for opponent in OPPONENTS:
        for index in range(12):
            key = f"{opponent}-{index}"
            reference = _record("frozen_fixed_head_on", opponent, key, 1.0)
            candidate = _record("forward_positive", opponent, key, 0.95)
            candidate["handoff_state_hash"] = "drift"
            records.extend((reference, candidate))
    result = summarize_records(
        records,
        method_deltas={"frozen_fixed_head_on": [0, 0, 0], "forward_positive": [0.25, 0, 0]},
        min_window_steps=20,
        min_pairs=4,
        min_applied_delta_norm=0.15,
        practical_delta=-0.02,
        safety_delta_max=0.05,
    )

    assert result["authority_candidates"] == []


def test_v9_design_remains_nonlearning_and_has_the_complete_axis_grid():
    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "config/experiment/thesis_five_state_reentry_v9_action_authority_design.yaml").read_text(encoding="utf-8"))

    assert config["authorization"]["execution_permitted"] is False
    assert config["authorization"]["training_permitted"] is False
    assert config["authorization"]["reward_change_permitted"] is False
    assert set(config["methods"]) == {
        "frozen_fixed_head_on", "forward_positive", "forward_negative",
        "lateral_positive", "lateral_negative", "vertical_positive", "vertical_negative",
    }
