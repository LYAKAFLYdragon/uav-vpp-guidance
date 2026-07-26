"""Deterministic regression tests for the implemented F07 VPP action contract."""
import json
from pathlib import Path

import os
from types import SimpleNamespace

import numpy as np
import pytest

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
import uav_vpp_guidance.agents.ppo_agent as ppo_agent_module
from uav_vpp_guidance.evaluation.f07_action_contract_evidence import (
    CanonicalVPPActionContract,
    construction_matrix,
)
from uav_vpp_guidance.utils.vpp_action_contract import validate_vpp_checkpoint_metadata
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"
VPP_CONFIG = {
    "action_dim": 3,
    "d_long_range": [-1500.0, 1500.0],
    "d_lat_range": [-800.0, 800.0],
    "d_vert_range": [-500.0, 500.0],
}


def test_compatible_boundaries_report_one_canonical_3d_contract():
    result = CanonicalVPPActionContract().validate(
        config_dim=3, generator_dim=3, action_space_dim=3, checkpoint_dim=3
    )
    assert result["dimension"] == 3


def test_missing_mismatched_and_stale_contracts_fail_descriptively():
    matrix = construction_matrix()
    assert matrix["cases"]["compatible_3d"]["accepted"]
    for name in ("missing_config", "generator_mismatch", "stale_checkpoint"):
        assert not matrix["cases"][name]["accepted"]
        assert "F07" in matrix["cases"][name]["error"]


def test_generator_requires_explicit_canonical_dimension_and_rejects_implicit_5d():
    with pytest.raises(ValueError, match="action_dim is required"):
        VirtualPointGenerator({})
    with pytest.raises(ValueError, match="canonical VPP dimension 3"):
        VirtualPointGenerator({**VPP_CONFIG, "action_dim": 5})
    generator = VirtualPointGenerator(VPP_CONFIG)
    own = {"position_neu": np.zeros(3)}
    target = {"position_neu": np.array([1000.0, 0.0, 0.0])}
    generator.action_to_virtual_point(np.zeros(3), own, target)
    with pytest.raises(ValueError, match="VPP action dimension 5"):
        generator.action_to_virtual_point(np.zeros(5), own, target)


def test_legacy_5d_input_requires_an_explicit_compatibility_mode():
    generator = VirtualPointGenerator(
        {**VPP_CONFIG, "legacy_compatibility_mode": "vpp_5d_legacy"}
    )
    own = {"position_neu": np.zeros(3)}
    target = {"position_neu": np.array([1000.0, 0.0, 0.0])}
    result = generator.action_to_virtual_point(np.zeros(5), own, target)
    assert result["position"].shape == (3,)


def test_checkpoint_metadata_rejects_missing_or_stale_vpp_dimension(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="missing action_dim"):
        validate_vpp_checkpoint_metadata({}, expected_dim=3)
    with pytest.raises(ValueError, match="does not match"):
        validate_vpp_checkpoint_metadata({"action_dim": 5}, expected_dim=3)

    path = tmp_path / "stale.pt"
    path.touch()
    fake_agent = SimpleNamespace(
        device="cpu", action_dim=3, config={"virtual_point": VPP_CONFIG}
    )
    monkeypatch.setattr(
        ppo_agent_module.torch,
        "load",
        lambda *_args, **_kwargs: {"action_dim": 5},
    )
    with pytest.raises(ValueError, match="Checkpoint action-dim mismatch"):
        PPOAgent.load(fake_agent, os.fspath(path))


def test_f07_gate_authorizes_only_the_implemented_allowlist():
    gate = json.loads(
        (SPEC / "evidence/f07_action_contract_bundle/f07_evidence_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["status"] == "approved"
    assert set(gate["protected_path_allowlist"]) == {
        "src/uav_vpp_guidance/virtual_point/generator.py",
        "src/uav_vpp_guidance/envs/tracking_env.py",
        "src/uav_vpp_guidance/agents/ppo_agent.py",
        "src/uav_vpp_guidance/training/train_prediction_vpp_ppo.py",
        "src/uav_vpp_guidance/hierarchy/specialist_policy.py",
    }
