from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
GEOMETRY = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_shared_skills_geometry_v1.yaml"
COMBAT = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_shared_skills_combat_v1.yaml"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_common_readiness_contract(config):
    assert config["experiment"]["lane"] == "noncanonical_thesis_extension"
    assert config["experiment"]["execution_mode"] == "readiness_only"
    assert config["experiment"]["training_permitted"] is False
    assert config["experiment"]["high_level_ppo_training"] == "prohibited"
    assert config["encoder"]["sha256"] == "385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59"
    assert config["encoder"]["trainable"] is False
    assert config["encoder"]["fallback"] == "prohibited"
    assert config["observation_action_contract"] == {
        "observation_dim": 66,
        "profile_conditioned_segments": ["intent_targets", "intent_weights"],
        "action_dim": 3,
        "action_components": [
            "forward_bias_normalized",
            "lateral_bias_normalized",
            "vertical_bias_normalized",
        ],
        "implicit_padding_or_truncation": "prohibited",
        "checkpoint_fallback": "prohibited",
    }
    assert config["runtime"]["simulator"] == "JSBSim"
    assert config["runtime"]["strict_backend"] is True
    assert config["runtime"]["predicted_target_vpp_interface"] == "frozen"
    assert config["runtime"]["guidance_pid"] == "frozen_existing_chain"
    assert config["runtime"]["prediction_reward"] == "prohibited"
    assert config["runtime"]["trajectory_prediction"]["enabled"] is True
    assert config["runtime"]["trajectory_prediction"]["checkpoint_strict"] is True
    assert config["runtime"]["trajectory_prediction"]["freeze_predictor_during_rl"] is True
    assert config["runtime"]["trajectory_prediction"]["integration"]["anchor_mode"] == "predicted_target"
    assert config["runtime"]["virtual_point"]["anchor_mode"] == "predicted_target"
    assert config["outputs"]["must_be_fresh_and_empty"] is True
    assert config["outputs"]["checkpoint_retention_top_k"] == 3
    assert config["outputs"]["train_full_raw_telemetry"] == "prohibited"
    assert "p3_temporal_encoder" not in config["outputs"]["root"]


def test_geometry_pretrain_config_is_readiness_only_and_covers_all_four_skills():
    config = _load(GEOMETRY)
    _assert_common_readiness_contract(config)

    assert config["experiment"]["stage"] == "p4_geometry_pretrain"
    assert config["skills"]["train_all"] == [
        "pursuit_conversion",
        "lead_intercept",
        "defensive_extension",
        "reentry_recovery",
    ]
    assert config["skills"]["sampling"] == "geometry_phase_profile_balanced"
    assert config["skills"]["require_declared_geometry_phase_coverage"] is True
    assert config["skills"]["require_profile_conditioning"] is True
    assert config["history_bootstrap"]["policy_action_start_after_observed_steps"] == 10
    assert config["history_bootstrap"]["bootstrap_action"] == [0.0, 0.0, 0.0]
    assert config["history_bootstrap"]["bootstrap_transitions_stored_for_ppo"] is False
    assert config["history_bootstrap"]["implicit_padding_or_truncation"] == "prohibited"
    assert config["runtime"]["trajectory_prediction"]["checkpoint_path"].endswith("frozen_trajectory_predictor_best_model.pt")
    assert config["runtime"]["trajectory_prediction"]["freeze_predictor_during_rl"] is True
    assert config["runtime"]["virtual_point"]["anchor_mode"] == "predicted_target"
    assert config["training"]["total_timesteps_per_skill"] == 200000
    assert config["training"]["train_opponent"] == "expert_rule_based"
    assert config["training"]["raw_train_trajectory_persistence"] == "prohibited"
    assert config["selection"]["per_skill_gate"]["minimum_intent_progress_fraction"] == 0.55
    assert config["selection"]["per_skill_gate"]["max_ego_crash_oob_delta_vs_zero_vpp_reference"] == 0.05


def test_combat_finetune_config_is_readiness_only_and_three_opponent_balanced():
    config = _load(COMBAT)
    _assert_common_readiness_contract(config)

    assert config["experiment"]["stage"] == "p4_combat_finetune"
    assert config["combat_sampling"] == {
        "opponent_registry_source": "thesis_five_state_shared_intent_v1_opponents_v1",
        "strategy": "balanced_round_robin",
        "opponents": ["expert_rule_based", "end_to_end_neural", "independent_ppo_vpp"],
        "missing_or_failed_opponent": "fail_closed",
        "report_each_opponent_separately": True,
        "aggregate_opponents": "prohibited",
    }
    assert config["skills"]["require_geometry_pretrain_checkpoint_for_each_skill"] is True
    assert config["selection"]["heldout60_use"] == "prohibited"
