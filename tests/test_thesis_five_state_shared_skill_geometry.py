from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.training.thesis_shared_skill_geometry import (
    _ensure_training_authorized,
    build_geometry_training_plan,
    evaluate_skill_gate,
    intent_loss,
    shaped_geometry_reward,
)
from uav_vpp_guidance.training.thesis_shared_skill_policy import ThesisSharedSkillPPOAgent


REPO_ROOT = Path(__file__).resolve().parent.parent
GEOMETRY_CONFIG = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_shared_skills_geometry_v1.yaml"
REGISTRY_PATH = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_skill_registry_v1.yaml"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _observation():
    return {
        "relative_state": {
            "aa_rad": np.deg2rad(15.0),
            "ata_rad": np.deg2rad(20.0),
            "range_m": 900.0,
            "range_rate_mps": -100.0,
        },
        "own_state": {"position_m": [0.0, 0.0, 5000.0], "velocity_vector_mps": [220.0, 0.0, 0.0]},
        "target_state": {"position_m": [900.0, 0.0, 5000.0], "velocity_vector_mps": [200.0, 0.0, 0.0]},
    }


def _aggregate(skill_definition):
    aggregate = {
        "episodes": 30,
        "profile_episodes": Counter({profile: 4 for profile in skill_definition["allowed_profiles"]}),
        "phase_steps": Counter(),
        "state_phase_steps": Counter(),
        "intent_progress_steps": 18,
        "policy_steps": 30,
        "terminal_reasons": Counter({"timeout": 30}),
        "response_samples": defaultdict(list),
    }
    for state in skill_definition["geometry_phase_coverage"]["geometry_states"]:
        for phase in skill_definition["geometry_phase_coverage"]["phases"]:
            aggregate["state_phase_steps"][f"{state}:{phase}"] = 2
    return aggregate


def test_geometry_reward_rewards_profile_progress_and_penalizes_ego_terminal():
    config = _load(GEOMETRY_CONFIG)
    baseline = intent_loss(_observation(), "front_intercept")
    assert baseline >= 0.0

    progress = shaped_geometry_reward(1.0, 0.5, {"reason": "timeout"}, config)
    failure = shaped_geometry_reward(1.0, 0.5, {"reason": "crash"}, config)

    assert progress["intent_progress"] > 0.0
    assert progress["reward"] > failure["reward"]
    assert failure["ego_terminal_penalty"] == 1.0


def test_geometry_gate_requires_each_skill_profile_phase_and_per_opponent_safety():
    config = _load(GEOMETRY_CONFIG)
    registry = _load(REGISTRY_PATH)
    skill = registry["skills"]["pursuit_conversion"]
    aggregate_by_opponent = {
        opponent: _aggregate(skill)
        for opponent in config["selection"]["dev30_opponents"]
    }
    zero = {opponent: Counter({"timeout": 30}) for opponent in aggregate_by_opponent}

    passed = evaluate_skill_gate(aggregate_by_opponent, skill, config, zero)
    assert passed["passed"] is True
    assert passed["status"] == "ready_for_combat_finetune"

    aggregate_by_opponent["expert_rule_based"]["terminal_reasons"] = Counter({"crash": 30})
    failed = evaluate_skill_gate(aggregate_by_opponent, skill, config, zero)
    assert failed["passed"] is False
    assert failed["checks"]["safety"] is False


def test_geometry_plan_covers_all_registered_skills_without_training_or_output_writes():
    config = _load(GEOMETRY_CONFIG)
    registry = _load(REGISTRY_PATH)
    plan = build_geometry_training_plan(config, registry)

    assert plan["training_started"] is False
    assert set(plan["skills"]) == set(registry["skills"])
    assert all(item["eligible_train_scenarios"] > 0 for item in plan["skills"].values())
    assert all(item["total_timesteps"] == 200000 for item in plan["skills"].values())


def test_geometry_ppo_checkpoint_is_strict_and_current_config_refuses_training(tmp_path):
    config = _load(GEOMETRY_CONFIG)
    agent = ThesisSharedSkillPPOAgent(config)
    observation = np.zeros(66, dtype=np.float32)
    action, log_prob, value = agent.select_action(observation)
    agent.store_transition(observation, action, log_prob, 0.25, False, value)
    agent.store_transition(observation, action, log_prob, 0.0, True, value)
    stats = agent.update(None)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl"}

    checkpoint = tmp_path / "strict.pt"
    metadata = {"skill_name": "pursuit_conversion", "p3_encoder_sha256": config["encoder"]["sha256"]}
    agent.save(checkpoint, metadata)
    restored = ThesisSharedSkillPPOAgent(config)
    restored.load_strict(checkpoint, metadata)
    with pytest.raises(ValueError, match="metadata mismatch"):
        restored.load_strict(checkpoint, {"skill_name": "lead_intercept"})

    with pytest.raises(PermissionError, match="training_permitted: false"):
        _ensure_training_authorized(config, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
