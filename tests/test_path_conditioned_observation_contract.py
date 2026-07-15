from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path
import torch

from uav_vpp_guidance.hierarchy.path_conditioned_observation_contract import (
    CONTRACT_VERSION,
    PATH_CONTEXT_DIM,
    PATH_CONTEXT_FEATURES,
    PathConditionedObservationContract,
    PathConditionedObservationError,
    build_path_context,
)
from uav_vpp_guidance.training.path_conditioned_shared_skill_policy import (
    CHECKPOINT_SCHEMA,
    PathConditionedSharedSkillPolicy,
)


ROOT = Path(__file__).resolve().parent.parent


def test_v2_path_contract_preserves_the_exact_v1_prefix_and_appends_eight_features():
    contract = PathConditionedObservationContract()
    path = build_path_context(
        requested_path_world_delta_m=[3000.0, -400.0, 0.0],
        path_progress=0.75,
        selected_anchor_index=2,
    )
    observation = contract.compose(
        base_geometry=np.zeros(16),
        explicit_history=np.zeros(6),
        intent_targets=np.zeros(6),
        intent_weights=np.ones(6),
        temporal_embedding=np.zeros(32),
        path_context=path,
    )
    assert contract.obs_dim == 74
    assert observation.shape == (74,)
    assert tuple(contract.feature_names[-PATH_CONTEXT_DIM:]) == PATH_CONTEXT_FEATURES
    assert np.allclose(observation[:66], [0.0] * 22 + [0.0] * 6 + [1.0] * 6 + [0.0] * 32)


def test_path_context_keeps_direction_reports_envelope_excess_and_uses_no_action_input():
    value = build_path_context(
        requested_path_world_delta_m=[4500.0, -1600.0, 250.0],
        path_progress=0.5,
        selected_anchor_index=1,
    )
    assert np.allclose(value[:3], [2.0, -2.0, 0.5])
    assert value[3] == pytest.approx(0.5)
    assert np.allclose(value[4:7], [0.0, 1.0, 0.0])
    assert value[7] == 1.0


@pytest.mark.parametrize(
    ("progress", "index"),
    ((-0.1, 0), (1.1, 0), (0.5, -1), (0.5, 3)),
)
def test_path_context_rejects_invalid_current_geometry_contract(progress: float, index: int):
    with pytest.raises(PathConditionedObservationError):
        build_path_context(
            requested_path_world_delta_m=[0.0, 0.0, 0.0],
            path_progress=progress,
            selected_anchor_index=index,
        )


def test_v2_yaml_contract_is_explicitly_separate_from_legacy_66d_checkpointing():
    config = yaml.safe_load(
        (ROOT / "config/experiment/thesis_five_state_shared_intent_v2_path_context_observation_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["contract_version"] == CONTRACT_VERSION
    assert config["observation_dim"] == 74
    assert config["path_context"]["future_target_state_input"] == "prohibited"
    assert config["checkpoint_contract"]["legacy_66d_checkpoint_transplant"] == "prohibited"


def test_v17_policy_accepts_only_the_new_exact_path_conditioned_schema():
    policy = PathConditionedSharedSkillPolicy(hidden_dim=16)
    action = policy(torch.zeros(2, 74))
    assert action.shape == (2, 3)
    assert torch.all(torch.abs(action) <= 1.0)
    assert CHECKPOINT_SCHEMA == "thesis_path_conditioned_shared_skill_ppo_v1"
    with pytest.raises(ValueError, match="exact"):
        policy(torch.zeros(1, 66))
