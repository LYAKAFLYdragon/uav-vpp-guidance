from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from uav_vpp_guidance.evaluation.single_motif_continuous_66d import (
    REQUIRED_OPPONENTS,
    SOURCE_ID,
    SingleMotifContinuous66DCollector,
    build_single_motif_plan,
    project_base_observation,
    write_single_motif_artifacts,
)
from uav_vpp_guidance.evaluation.single_motif_continuous_66d_analysis import (
    evaluate_gate,
    load_collector_runs,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import (
    BASE_GEOMETRY_FEATURES,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_single_motif_continuous_66d_r1.yaml"


class _Encoder:
    def encode(self, history: np.ndarray) -> np.ndarray:
        assert history.shape == (10, 16)
        return np.linspace(-0.5, 0.5, 32, dtype=np.float32)


def _plan():
    return build_single_motif_plan(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def _schema_and_vector(range_m: float = 900.0, range_rate_mps: float = -30.0):
    values = {
        "range_m": range_m,
        "range_rate_mps": range_rate_mps,
        "altitude_diff_m": 0.0,
        "speed_diff_mps": -160.0,
        "los_azimuth_sin": 0.0,
        "los_azimuth_cos": 1.0,
        "los_elevation_sin": 0.0,
        "los_elevation_cos": 1.0,
        "ata_sin": float(np.sin(np.deg2rad(20.0))),
        "ata_cos": float(np.cos(np.deg2rad(20.0))),
        "aa_sin": float(np.sin(np.deg2rad(160.0))),
        "aa_cos": float(np.cos(np.deg2rad(160.0))),
        "own_speed": 200.0,
        "target_speed": 360.0,
        "own_altitude": 5200.0,
        "target_altitude": 5200.0,
        "gain_k_los": 1.2,
        "gain_k_pos": 0.8,
        "is_crossing": 0.0,
    }
    names = (
        "gain_k_pos",
        "is_crossing",
        *reversed(BASE_GEOMETRY_FEATURES),
        "gain_k_los",
    )
    return {"dim": len(names), "feature_names": list(names)}, np.asarray(
        [values[name] for name in names], dtype=np.float32
    )


def _collector(opponent: str = "expert", scenario: str = "scenario_neg"):
    schema, _ = _schema_and_vector()
    return SingleMotifContinuous66DCollector(
        plan=_plan(),
        run_id="run",
        method_name="legacy_static_oracle_task_gate",
        task_name="head_on",
        opponent_id=opponent,
        seed=7,
        episode=0,
        scenario_name=scenario,
        scenario_metadata={
            "distance_speed_package": "run_in_a",
            "height_condition": "co_altitude",
            "mirror_sign": "negative" if "neg" in scenario else "positive",
        },
        observation_schema=schema,
        environment_episode=1,
        encoder=_Encoder(),
    )


def _rollout(collector: SingleMotifContinuous66DCollector, steps: int = 12):
    schema, vector = _schema_and_vector()
    action = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    for step in range(1, steps + 1):
        observation = {"observation_vector": vector.copy(), "observation_schema": schema}
        collector.begin_step(
            step=step,
            observation=observation,
            action=action,
            environment_episode=1,
        )
        collector.finish_step(
            step=step,
            action=action,
            observation=observation,
            environment_episode=1,
            info={
                "backend": "jsbsim",
                "backend_fallback_occurred": False,
                "first_pass_complete": True,
                "prediction_valid": True,
                "prediction_fallback": False,
                "vp_position_neu": [100.0, 20.0, 5200.0],
                "vp_offset": [10.0, 20.0, 30.0],
                "vp_world_offset": [10.0, 20.0, 30.0],
                "vp_forward_bias_m": 100.0,
                "vp_lateral_bias_m": 20.0,
                "nz_cmd": 2.0,
                "roll_rate_cmd": 0.1,
                "throttle_cmd": 0.8,
                "own_state": {"nz_g": 1.9, "speed_mps": 200.0, "altitude_m": 5200.0},
            },
        )
    return action


def test_config_preregisters_nonlearning_three_opponent_gate():
    plan = _plan()
    assert plan.source_id == SOURCE_ID
    assert plan.opponents == REQUIRED_OPPONENTS
    assert plan.target_skill == "defensive_extension"
    assert plan.target_profile == "range_extension"
    assert plan.min_episodes_per_opponent == 2
    assert plan.min_target_steps_per_opponent == 20


def test_base_projection_is_by_feature_name_not_position():
    schema, vector = _schema_and_vector()
    projected = project_base_observation(vector, schema)
    assert projected.shape == (16,)
    assert projected[0] == 900.0
    assert projected[1] == -30.0
    assert projected[12] == 200.0


def test_collector_builds_exact_66d_after_ten_real_frames_without_mutating_action():
    collector = _collector()
    action = _rollout(collector)
    ledger = collector.ledger()
    assert action.tolist() == [np.float32(0.1), np.float32(-0.2), np.float32(0.3)]
    assert ledger["summary"]["warmup_steps_without_padding"] == 9
    assert ledger["summary"]["contract_ready_steps"] == 3
    assert ledger["summary"]["valid_target_steps"] == 3
    ready = ledger["steps"][9]
    assert len(ready["observed_history_10x16"]) == 10
    assert len(ready["observation_66d"]) == 66
    assert len(ready["normalized_vpp_action_3d"]) == 3
    assert ready["target_pair_allowed"] is True
    assert ready["action_identity_preserved"] is True
    assert ready["history_padding_used"] is False


def test_artifact_writer_and_hash_verifier_preserve_large_ledger(tmp_path: Path):
    collector = _collector()
    _rollout(collector)
    records = [
        {
            "single_motif_continuous_66d": collector.ledger(),
            "phase_reachability_handoff_ledger": {"continuity_valid": True},
        }
    ]
    index_path = write_single_motif_artifacts(tmp_path, records)
    assert index_path is not None and index_path.is_file()
    assert "single_motif_continuous_66d" not in records[0]
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "git_info": {"commit": "test", "dirty": False},
                "paper_safe": True,
                "invalid_for_paper_reasons": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_collector_runs([tmp_path])
    assert len(loaded) == 1
    assert loaded[0]["ledger"]["summary"]["valid_target_steps"] == 3


def test_gate_is_independent_per_opponent_and_never_authorizes_training():
    plan = _plan()
    episodes = []
    for opponent in REQUIRED_OPPONENTS:
        for mirror in ("negative", "positive"):
            collector = _collector(opponent, f"scenario_{mirror}")
            _rollout(collector, steps=20)
            ledger = collector.ledger()
            ledger["header"]["scenario_signature"] = f"run_in_a|co_altitude|{mirror}"
            ledger["header"]["mirror_sign"] = mirror
            episodes.append(
                {
                    "run_dir": f"run_{opponent}_{mirror}",
                    "artifact_path": f"{opponent}_{mirror}.json",
                    "artifact_sha256": "a" * 64,
                    "run_manifest": {
                        "git_info": {"commit": "test", "dirty": False},
                        "paper_safe": True,
                        "invalid_for_paper_reasons": [],
                    },
                    "index_entry": {"v2_continuity_valid": True},
                    "ledger": copy.deepcopy(ledger),
                }
            )
    result = evaluate_gate(episodes, plan)
    assert result["gate_passed"] is True
    assert result["training_authorized"] is False
    assert result["pilot_preregistration_draft_permitted"] is True

    failed = evaluate_gate(
        [episode for episode in episodes if episode["ledger"]["header"]["opponent"] != "expert"],
        plan,
    )
    assert failed["gate_passed"] is False
    assert failed["interpretation"] == "cannot_claim_missing_skill_input_output_contract_still_missing"
