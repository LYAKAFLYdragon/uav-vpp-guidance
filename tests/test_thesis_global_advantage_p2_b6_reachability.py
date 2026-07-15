from __future__ import annotations

import copy
import math
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b6_opponent_conditional_reachability.yaml"
RUNNER = ROOT / "scripts" / "run_thesis_global_advantage_p2_b6_reachability.py"


class _Encoder:
    def encode(self, history: np.ndarray) -> np.ndarray:
        assert history.shape == (10, 16)
        return np.zeros(32, dtype=np.float32)


def _plan():
    from uav_vpp_guidance.evaluation.global_advantage_p2_b6_reachability import build_b6_plan

    return build_b6_plan(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def _runner():
    spec = importlib.util.spec_from_file_location("b6_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _ledger(opponent: str, index: int):
    from uav_vpp_guidance.evaluation.global_advantage_p2_b6_reachability import ProfileFreeReachabilityCollector

    plan = _plan()
    collector = ProfileFreeReachabilityCollector(
        plan=plan,
        opponent=opponent,
        scenario_name=f"{opponent}_{index}",
        scenario_metadata={
            "distance_speed_package": f"package_{index}",
            "height_condition": "co_altitude",
            "mirror_sign": "negative" if index == 0 else "positive",
        },
        seed=16201 + index,
        environment_episode=index,
        observation_schema=_observation(4000.0, -100.0)["observation_schema"],
        encoder=_Encoder(),
    )
    action = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    for step in range(1, 20):
        raw_range = 4000.0 if step == 1 else 900.0
        collector.begin_step(step=step, observation=_observation(raw_range, -50.0), action=action)
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
    ledger["summary"].update(
        {
            "scenario_application_passed": True,
            "raw_si_phase_replay_passed": True,
            "initial_pre_merge_semantics_passed": True,
        }
    )
    return ledger


def test_b6_collector_is_profile_free_but_checks_all_profile_66d_vectors():
    ledger = _ledger("expert", 0)
    last = ledger["steps"][-1]

    assert last["dynamic_state"] == "disadvantage"
    assert last["phase"] == "re_entry"
    assert last["profile_free_eligible_step"] is True
    assert len(last["profile_free_observation_54d"]) == 54
    assert last["all_profiles_66d_finite"] is True
    assert set(last["profiles_66d"]) == {
        "front_intercept",
        "rear_quarter_alignment",
        "lateral_displacement",
        "defensive_break",
        "range_extension",
        "energy_altitude_recovery",
        "reentry_preparation",
    }
    assert ledger["summary"]["continuity_valid"] is True


def test_b6_candidate_gate_requires_each_opponent_independently():
    from uav_vpp_guidance.evaluation.global_advantage_p2_b6_reachability import evaluate_candidate_atlas

    plan = _plan()
    ledgers = [_ledger(opponent, index) for opponent in plan.opponents for index in range(2)]
    result = evaluate_candidate_atlas(ledgers, plan)

    assert result["selected_candidate"]["state"] == "disadvantage"
    assert result["selected_candidate"]["phase"] == "re_entry"
    assert result["pilot_authorized"] is False
    no_independent = evaluate_candidate_atlas(
        [ledger for ledger in ledgers if ledger["header"]["opponent"] != "independent_ppo_vpp"],
        plan,
    )
    assert no_independent["selected_candidate"] is None


def test_b6_runner_validates_design_but_refuses_unauthorised_execution():
    runner = _runner()
    config, plan, _runtime, _registry, scenarios = runner.validate_sources(CONFIG)

    assert config["authorization"]["execution_permitted"] is False
    assert plan.execution_permitted is False
    assert len(scenarios) == 60
    with pytest.raises(runner.B6ReachabilityError, match="not authorised"):
        runner.run(CONFIG)
