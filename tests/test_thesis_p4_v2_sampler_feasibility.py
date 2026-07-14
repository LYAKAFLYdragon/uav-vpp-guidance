from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import (
    PreflightContractError,
    build_dynamic_taxonomy_sidecar,
    build_sampler_feasibility_plan,
    classify_dynamic_state,
    derive_declared_cells,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_skills_geometry_v2_sampler_feasibility.yaml"
REGISTRY_PATH = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_skill_registry_v1.yaml"
TRAIN_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_p4_v2_sampler_train_candidates.yaml"
DEV_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_p4_v2_sampler_dev_candidates.yaml"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_thesis_p4_v2_sampler_feasibility.py"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_runner():
    spec = importlib.util.spec_from_file_location("p4_v2_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("own_angle", "target_angle", "expected"),
    [
        (20.0, 160.0, "advantage"),
        (20.0, 20.0, "head_on"),
        (160.0, 20.0, "disadvantage"),
        (160.0, 160.0, "neutral"),
        (90.0, 20.0, "crossing_entry"),
    ],
)
def test_dynamic_taxonomy_preserves_the_five_state_contract(own_angle, target_angle, expected):
    assert classify_dynamic_state(own_angle, target_angle) == expected


def test_dynamic_sidecar_is_explicit_and_not_an_observation_extension():
    sidecar = build_dynamic_taxonomy_sidecar(
        {
            "initial_class": "advantage",
            "phase": "post_merge",
            "phase_entry_step": 3,
            "own_to_target_los_angle_deg": 160.0,
            "target_velocity_to_own_los_angle_deg": 20.0,
        }
    )
    assert sidecar.initial_class == "advantage"
    assert sidecar.dynamic_class == "disadvantage"
    assert sidecar.phase == "post_merge"
    assert build_dynamic_taxonomy_sidecar(
        {
            "initial_class": "head_on",
            "phase": "pre_merge",
            "phase_entry_step": 0,
            "own_to_target_los_angle_deg": 20.0,
            "target_velocity_to_own_los_angle_deg": 20.0,
        }
    ).phase_entry_step == 0
    with pytest.raises(PreflightContractError, match="unsupported phase"):
        build_dynamic_taxonomy_sidecar(
            {
                "initial_class": "advantage",
                "phase": "invalid",
                "phase_entry_step": 1,
                "own_to_target_los_angle_deg": 20.0,
                "target_velocity_to_own_los_angle_deg": 20.0,
            }
        )


def test_preflight_design_preserves_26_cell_contract_and_disjoint_splits():
    config = _load(CONFIG_PATH)
    registry = _load(REGISTRY_PATH)
    train = _load(TRAIN_PATH)
    dev = _load(DEV_PATH)
    plan = build_sampler_feasibility_plan(config, registry, train, dev)
    assert len(derive_declared_cells(registry)) == 26
    assert len(plan.declared_cells) == 26
    assert plan.train_signature_namespace != plan.dev_signature_namespace
    assert plan.min_episodes_per_cell_per_opponent == 2
    assert plan.min_policy_steps_per_cell_per_opponent == 20
    assert config["authorization"] == {
        "execution_permitted": False,
        "jsbsim_probe_permitted": False,
        "training_permitted": False,
        "combat_finetune_permitted": False,
        "high_level_ppo_permitted": False,
        "require_explicit_future_authorization": True,
    }


def test_preflight_rejects_overlapping_train_dev_packages():
    config = _load(CONFIG_PATH)
    registry = _load(REGISTRY_PATH)
    train = _load(TRAIN_PATH)
    dev = _load(DEV_PATH)
    dev["distance_speed_package_ids"] = list(train["distance_speed_package_ids"])
    with pytest.raises(PreflightContractError, match="distance-speed packages must be disjoint"):
        build_sampler_feasibility_plan(config, registry, train, dev)


def test_runner_is_design_only_and_refuses_execution():
    runner = _load_runner()
    plan = runner.validate_design(CONFIG_PATH)
    assert plan["declared_cell_count"] == 26
    assert plan["p4_evidence_manifest"] == "reports/p4_evidence_bundle_20260714/evidence_manifest.json"
    assert plan["next_action"] == "explicit_authorization_required_before_any_jsbsim_probe"
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--execute"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not authorized" in result.stderr
