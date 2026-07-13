from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_vpp_guidance.hierarchy.shared_intent_validity import SKILL_NAMES, base_allowed_profiles
from uav_vpp_guidance.training.thesis_shared_skill_policy import (
    OBSERVATION_DIM,
    VPP_ACTION_DIM,
    ThesisSharedSkillPolicy,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_skill_registry_v1.yaml"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_thesis_five_state_shared_skills.py"


def _verifier_module():
    spec = importlib.util.spec_from_file_location("p4_skill_verifier", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_freezes_p3_and_declares_four_untrained_skills_without_fallback():
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    p3 = registry["frozen_inputs"]["p3_temporal_encoder"]

    assert p3["sha256"] == "385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59"
    assert p3["history_window_steps"] == 10
    assert p3["base_geometry_dim"] == 16
    assert p3["embedding_dim"] == 32
    assert p3["frozen"] is True
    assert p3["finetune_in_p4"] == "prohibited"
    assert tuple(registry["skills"]) == SKILL_NAMES

    for skill_id, skill_name in enumerate(SKILL_NAMES):
        entry = registry["skills"][skill_name]
        assert entry["skill_id"] == skill_id
        assert entry["status"] == "untrained_not_ready"
        assert entry["checkpoint"] is None
        assert entry["checkpoint_fallback"] == "prohibited"
        assert entry["observation_dim"] == 66
        assert entry["action_dim"] == 3
        assert tuple(entry["allowed_profiles"]) == base_allowed_profiles(skill_name)
        assert entry["readiness_gate"]["status_before_training"] == "not_evaluated"
        assert entry["readiness_gate"]["strict_interface"] == "66d_profile_conditioned_to_3d_normalized_vpp"
        assert entry["readiness_gate"]["checkpoint_fallback"] == "prohibited"
        assert entry["readiness_gate"]["simulator"] == "JSBSim_strict_backend_only"


def test_shared_skill_interface_is_strict_profile_conditioned_and_deterministic():
    torch.manual_seed(7)
    policy = ThesisSharedSkillPolicy().eval()
    observation = torch.linspace(-1.0, 1.0, OBSERVATION_DIM).reshape(1, OBSERVATION_DIM)
    with torch.no_grad():
        first = policy(observation)
        second = policy(observation)

    assert first.shape == (1, VPP_ACTION_DIM)
    assert torch.isfinite(first).all()
    assert (first.abs() <= 1.0).all()
    assert torch.allclose(first, second, atol=1e-7)
    with torch.no_grad():
        try:
            policy(torch.zeros(1, OBSERVATION_DIM - 1))
        except ValueError as exc:
            assert "66" in str(exc)
        else:  # pragma: no cover - guard against a silent input adaptation
            raise AssertionError("P4 policy accepted a non-66-D observation")


def test_readiness_verifier_passes_static_contract_without_launching_training():
    module = _verifier_module()
    report = module.build_readiness_report(run_jsbsim_probe=False)

    assert report["static_readiness_pass"] is True
    assert report["training_launched"] is False
    assert report["training_launch_allowed_by_configs"] is False
    assert set(report["skill_readiness_status"].values()) == {"not_evaluated_untrained"}
    assert any(item["name"] == "p3_encoder:frozen_load_and_determinism" and item["passed"] for item in report["checks"])
    assert any(item["name"] == "opponents:three_way_balanced_loadable" and item["passed"] for item in report["checks"])
