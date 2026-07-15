from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

from uav_vpp_guidance.evaluation import thesis_neutral_postmerge_reentry_recovery_pilot as pilot


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_pilot_v1.yaml"
BUILDER = ROOT / "scripts" / "build_thesis_neutral_postmerge_reentry_recovery_pilot_v1_manifests.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_neutral_postmerge_reentry_recovery_pilot_v1.py"
DEV = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_dev12.yaml"
HELDOUT = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_pilot_v1_heldout24.yaml"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_manifests_are_neutral_premerge_continuous_and_disjoint():
    builder = _load_module(BUILDER, "neutral_postmerge_builder")
    dev = yaml.safe_load(DEV.read_text(encoding="utf-8"))
    heldout = yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))
    assert len(dev["scenarios"]) == 12
    assert len(heldout["scenarios"]) == 24
    assert builder.payload_sha256(dev) == dev["integrity"]["payload_sha256"]
    assert builder.payload_sha256(heldout) == heldout["integrity"]["payload_sha256"]
    for manifest in (dev, heldout):
        assert all(item["physical_intersection_count"] == 0 for item in manifest["disjointness"].values())
        assert all(item["seed_intersection_count"] == 0 for item in manifest["disjointness"].values())
        assert all(
            scenario["metadata"]["initial_class"] == "neutral"
            and scenario["metadata"]["taxonomy_geometry_state"] == "neutral"
            and scenario["metadata"]["phase_at_reset"] == "pre_merge"
            and scenario["metadata"]["continuous_run_in"] is True
            and scenario["metadata"]["routing_enabled"] is False
            for scenario in manifest["scenarios"]
        )


def test_design_locks_the_single_skill_profile_and_disables_execution():
    config = _config()
    assert config["source_id"] == pilot.SOURCE_ID
    assert all(value is False for value in config["authorization"].values())
    assert config["fixed_contract"]["skill"] == "reentry_recovery"
    assert config["fixed_contract"]["profile"] == "reentry_preparation"
    assert config["fixed_contract"]["profile_targets"] == [30.0, 130.0, 2200.0, -30.0, 100.0, 0.0]
    assert config["fixed_contract"]["profile_weights"] == [0.8, 0.8, 1.0, 0.8, 0.7, 0.4]
    assert config["fixed_contract"]["routing_enabled"] is False
    assert config["methods"]["candidate_fixed_reentry_recovery"]["checkpoint"] is None
    assert config["opponents"]["pooled_gate"] == "prohibited"


def test_neutral_postmerge_validity_mask_allows_the_preregistered_action(tmp_path: Path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["outputs"]["root"] = str(tmp_path / "v1_design_only_output")
    config_path = tmp_path / "v1_design.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = pilot.validate_design(config_path)
    assert result["execution_permitted"] is False
    assert result["training_permitted"] is False
    assert result["target_window"] == "neutral->post_merge"
    assert result["planned_future_evaluation_records"] == 324
    assert result["finite_profile_targets"] is True


def test_preflight_module_never_exposes_an_execute_argument():
    preflight = _load_module(PREFLIGHT, "neutral_postmerge_preflight")
    assert preflight.main is not None
    assert "execute" not in PREFLIGHT.read_text(encoding="utf-8")


def test_any_authorization_expansion_fails_closed(tmp_path: Path):
    config = copy.deepcopy(_config())
    config["authorization"]["execution_permitted"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(pilot.NeutralPostMergePilotContractError, match="must remain false"):
        pilot.validate_design(path)
