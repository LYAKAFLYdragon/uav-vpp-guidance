from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b5_phase_feasible_sampler.yaml"
BUILDER = ROOT / "scripts" / "build_thesis_global_advantage_p2_b5_phase_feasible_sampler_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_global_advantage_p2_b5_phase_feasible_sampler.py"
RUNNER = ROOT / "scripts" / "run_thesis_global_advantage_p2_b5_phase_feasible_sampler.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan():
    from uav_vpp_guidance.evaluation.global_advantage_p2_b5_phase_feasible_sampler import (
        build_phase_feasible_sampler_plan,
    )

    return build_phase_feasible_sampler_plan(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def _ledger(opponent: str, index: int):
    return {
        "header": {
            "opponent": opponent,
            "scenario": f"scenario_{opponent}_{index}",
            "scenario_signature": f"package_{index}|height_{index}|mirror_{index}",
            "mirror_sign": "negative" if index == 0 else "positive",
        },
        "summary": {
            "valid_target_steps": 10,
            "all_ready_structural_contract_valid": True,
            "all_actions_identity_preserved": True,
            "strict_jsbsim_lineage": True,
            "no_reset": True,
            "no_padding": True,
            "continuity_valid": True,
            "scenario_application_passed": True,
            "raw_si_phase_replay_passed": True,
            "initial_pre_merge_semantics_passed": True,
        },
    }


class _Encoder:
    def encode(self, history: np.ndarray) -> np.ndarray:
        assert history.shape == (10, 16)
        return np.zeros(32, dtype=np.float32)


def _observation(raw_range_m: float, raw_range_rate_mps: float) -> dict:
    from uav_vpp_guidance.hierarchy.five_state_observation_contract import BASE_GEOMETRY_FEATURES

    ata = math.radians(160.0)
    aa = math.radians(20.0)
    values = {
        "range_m": raw_range_m / 5000.0,
        "range_rate_mps": raw_range_rate_mps / 200.0,
        "altitude_diff_m": 0.0,
        "speed_diff_mps": 0.0,
        "los_azimuth_sin": 0.0,
        "los_azimuth_cos": 1.0,
        "los_elevation_sin": 0.0,
        "los_elevation_cos": 1.0,
        "ata_sin": math.sin(ata),
        "ata_cos": math.cos(ata),
        "aa_sin": math.sin(aa),
        "aa_cos": math.cos(aa),
        "own_speed": 0.5,
        "target_speed": 0.5,
        "own_altitude": 0.5,
        "target_altitude": 0.5,
    }
    return {
        "relative_state": {"range_m": raw_range_m, "range_rate_mps": raw_range_rate_mps},
        "observation_schema": {"feature_names": list(BASE_GEOMETRY_FEATURES)},
        "observation_vector": [values[name] for name in BASE_GEOMETRY_FEATURES],
    }


def test_b5_manifest_is_unique_disadvantage_only_and_disjoint():
    builder = _load(BUILDER, "p2_b5_builder")
    sources = {label: builder._load_yaml(path) for label, path in builder.DISJOINT_MANIFESTS}
    manifest = builder.build_manifest(sources)
    scenarios = manifest["scenarios"]

    assert manifest["source_id"] == builder.SOURCE_ID
    assert len(scenarios) == 12
    assert len({scenario["name"] for scenario in scenarios}) == 12
    assert {scenario["metadata"]["taxonomy_geometry_state"] for scenario in scenarios} == {"disadvantage"}
    assert {scenario["metadata"]["phase_at_reset"] for scenario in scenarios} == {"pre_merge"}
    assert {scenario["metadata"]["mirror_sign"] for scenario in scenarios} == {"negative", "positive"}
    assert all(entry["physical_intersection_count"] == 0 for entry in manifest["disjointness"].values())


def test_b5_raw_si_phase_input_cannot_use_normalized_policy_vector():
    from uav_vpp_guidance.evaluation.global_advantage_p2_b5_phase_feasible_sampler import (
        normalization_is_consistent,
        policy_normalization_diagnostic,
        raw_si_phase_input,
    )
    from uav_vpp_guidance.training.thesis_shared_skill_geometry import PhaseTracker

    plan = _plan()
    raw = raw_si_phase_input(
        {
            "relative_state": {"range_m": 4000.0, "range_rate_mps": -100.0},
            "observation_vector": [0.8, -0.5],
        }
    )
    diagnostic = policy_normalization_diagnostic({"range_m": 0.8, "range_rate_mps": -0.5}, plan)

    assert raw["range_m"] == 4000.0
    assert raw["range_rate_mps"] == -100.0
    assert normalization_is_consistent(raw, diagnostic, plan)
    assert PhaseTracker().update(raw["range_m"], raw["range_rate_mps"]) == "pre_merge"


def test_b5_collector_preserves_first_pass_continuity_and_builds_66d_from_real_history():
    from uav_vpp_guidance.evaluation.global_advantage_p2_b5_phase_feasible_sampler import (
        PhaseFeasibleSamplerCollector,
    )

    plan = _plan()
    collector = PhaseFeasibleSamplerCollector(
        plan=plan,
        opponent="expert",
        scenario_name="b5_collector_contract",
        scenario_metadata={
            "distance_speed_package": "unit_a",
            "height_condition": "co_altitude",
            "mirror_sign": "negative",
        },
        seed=15101,
        environment_episode=7,
        observation_schema=_observation(4000.0, -100.0)["observation_schema"],
        encoder=_Encoder(),
    )
    action = np.asarray([0.2, -0.1, 0.3], dtype=np.float32)
    for step in range(1, 11):
        raw_range = 4000.0 if step == 1 else 900.0
        collector.begin_step(
            step=step,
            observation=_observation(raw_range, -50.0),
            action=action,
        )
        collector.finish_step(
            step=step,
            action=action.copy(),
            info={
                "backend": "jsbsim",
                "backend_fallback_occurred": False,
                "prediction_valid": True,
                "prediction_fallback": False,
                "first_pass_complete": True,
            },
        )
    ledger = collector.ledger()
    last = ledger["steps"][-1]

    assert ledger["summary"]["continuity_valid"] is True
    assert ledger["first_pass_boundary"]["step"] == 1
    assert ledger["continuity_receipt"]["sampler_step"] == 2
    assert last["raw_si_phase_input"]["range_m"] == 900.0
    assert last["policy_normalization_diagnostic"]["range_m_normalized"] == pytest.approx(0.18)
    assert last["ata_deg"] == pytest.approx(160.0)
    assert last["aa_deg"] == pytest.approx(20.0)
    assert last["dynamic_state"] == "disadvantage"
    assert last["phase"] == "re_entry"
    assert last["target_motif_step"] is True
    assert last["valid_target_step"] is True
    assert len(last["observed_history_10x16"]) == 10
    assert len(last["observation_66d"]) == 66
    assert last["action_identity_preserved"] is True


def test_b5_design_preflight_is_nonexecuting_and_freezes_all_inputs():
    preflight = _load(PREFLIGHT, "p2_b5_preflight")
    result = preflight.validate_design(CONFIG)

    assert result["execution_permitted"] is False
    assert result["training_permitted"] is False
    assert result["raw_si_phase_contract"] is True
    assert result["scenario_count"] == 12
    assert result["output_root"].endswith("global_advantage_v1_p2_b5_phase_feasible_sampler_r1")
    assert isinstance(result["output_root_absent"], bool)
    assert isinstance(result["disk_gate_would_pass"], bool)


def test_b5_gate_requires_each_opponent_and_never_unlocks_training():
    from uav_vpp_guidance.evaluation.global_advantage_p2_b5_phase_feasible_sampler import evaluate_gate

    plan = _plan()
    ledgers = [_ledger(opponent, index) for opponent in plan.opponents for index in range(2)]
    passed = evaluate_gate(ledgers, plan)

    assert passed["gate_passed"] is True
    assert passed["training_unlocked"] is False
    assert passed["pooling_prohibited"] is True
    assert passed["pilot_preregistration_may_be_drafted"] is True

    missing_independent = evaluate_gate(
        [ledger for ledger in ledgers if ledger["header"]["opponent"] != "independent_ppo_vpp"],
        plan,
    )
    assert missing_independent["gate_passed"] is False
    assert missing_independent["opponents"]["independent_ppo_vpp"]["passed"] is False
    assert missing_independent["training_unlocked"] is False


def test_b5_runner_refuses_execution_before_a_separate_authorization():
    runner = _load(RUNNER, "p2_b5_runner")

    with pytest.raises(runner.PhaseFeasibleSamplerError, match="not authorised"):
        runner.run(CONFIG)
