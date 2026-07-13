from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.envs.observation import build_observation
from uav_vpp_guidance.hierarchy.five_state_observation_contract import (
    BASE_GEOMETRY_FEATURES,
    CONTRACT_VERSION,
    EXPLICIT_HISTORY_FEATURES,
    FiveStateObservationContract,
    INTENT_TARGET_FEATURES,
    INTENT_WEIGHT_FEATURES,
    build_feature_name_transplant_map,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _states():
    return (
        {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        },
        {
            "position_m": np.array([2000.0, 100.0, 5200.0]),
            "velocity_vector_mps": np.array([-180.0, 0.0, 0.0]),
        },
    )


def test_base_geometry_names_match_the_unextended_environment_observation():
    own, target = _states()
    vector, names = build_observation(own, target, return_feature_names=True)

    assert tuple(names) == BASE_GEOMETRY_FEATURES
    assert vector.shape == (16,)


def test_contract_composes_only_exact_finite_segments():
    contract = FiveStateObservationContract()
    observation = contract.compose(
        base_geometry=np.zeros(16),
        explicit_history=np.zeros(len(EXPLICIT_HISTORY_FEATURES)),
        intent_targets=np.zeros(len(INTENT_TARGET_FEATURES)),
        intent_weights=np.ones(len(INTENT_WEIGHT_FEATURES)),
        temporal_embedding=np.zeros(32),
    )

    assert CONTRACT_VERSION == "five_state_shared_intent_v1"
    assert observation.shape == (66,)
    with pytest.raises(ValueError, match="base_geometry shape mismatch"):
        contract.compose(
            base_geometry=np.zeros(19),
            explicit_history=np.zeros(6),
            intent_targets=np.zeros(6),
            intent_weights=np.ones(6),
            temporal_embedding=np.zeros(32),
        )
    with pytest.raises(ValueError, match="non-finite"):
        contract.compose(
            base_geometry=np.zeros(16),
            explicit_history=np.array([np.nan] * 6),
            intent_targets=np.zeros(6),
            intent_weights=np.ones(6),
            temporal_embedding=np.zeros(32),
        )


def test_feature_transplant_map_is_name_based_not_position_based():
    mapping = build_feature_name_transplant_map(
        ("legacy_extra", "range_rate_mps", "range_m", "task_bit"),
        BASE_GEOMETRY_FEATURES,
    )

    assert mapping == {1: 1, 2: 0}


def test_contract_config_matches_the_code_contract_and_bans_legacy_shortcuts():
    path = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_observation_contract.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = FiveStateObservationContract(
        temporal_embedding_dim=config["temporal_embedding"]["dimension"],
        history_window_steps=config["explicit_history"]["window_steps"],
    )

    assert config["contract_version"] == CONTRACT_VERSION
    assert tuple(config["base_geometry_feature_names"]) == BASE_GEOMETRY_FEATURES
    assert tuple(config["explicit_history"]["feature_names"]) == EXPLICIT_HISTORY_FEATURES
    assert contract.obs_dim == 66
    assert config["contract_rules"] == {
        "task_id_input": "prohibited",
        "opponent_stage_input": "prohibited",
        "implicit_padding_or_truncation": "prohibited",
        "future_target_state_input": "prohibited",
        "legacy_weight_transfer": "feature_name_match_only",
    }
