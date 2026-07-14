from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from uav_vpp_guidance.evaluation.global_advantage_p1_analysis import audit_p1_manifest
from uav_vpp_guidance.evaluation import global_advantage_runin_contract as contract


ROOT = Path(__file__).resolve().parent.parent
CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_global_advantage_v1_p1_runin_preflight.yaml"
)


def _base_observation() -> dict:
    names = list(contract.BASE_GEOMETRY_FEATURES)
    values = [1000.0, -20.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    values += [0.0, 1.0, 0.0, 1.0, 250.0, 240.0, 5200.0, 5200.0]
    return {
        "observation_vector": values,
        "observation_schema": {"feature_names": names},
    }


def _episode(repeat: int, *, trajectory: str = "trajectory") -> dict:
    return {
        "opponent": "expert",
        "scenario_signature": "scenario_a",
        "repeat_index": repeat,
        "reset_state_sha256": "reset",
        "trajectory_sha256": trajectory,
        "boundary_state_sha256": "boundary",
        "terminal_reason": "timeout",
        "telemetry_complete": True,
        "no_backend_or_prediction_fallback": True,
        "step_hashes": ["s1", trajectory],
    }


def test_p1_config_preserves_nonlearning_contract():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    plan = contract.build_p1_plan(config)
    assert plan.source_id == contract.SOURCE_ID
    assert plan.scenario_count == 30
    assert plan.opponents == contract.OPPONENTS
    assert plan.order_variants == contract.ORDER_VARIANTS
    assert plan.max_high_level_steps == 260


def test_quantised_hash_is_stable_and_base_projection_is_name_bound():
    assert contract.stable_hash({"x": 1.00000001}, 6) == contract.stable_hash(
        {"x": 1.00000002}, 6
    )
    base = contract.base_observation(_base_observation())
    assert tuple(base) == contract.BASE_GEOMETRY_FEATURES
    assert base["range_m"] == 1000.0
    assert contract.classify_step(base, "pre_merge") == "head_on"


def test_repeat_comparison_requires_all_hashes_and_telemetry_to_agree():
    passed = contract.compare_repeats([_episode(0), _episode(1), _episode(2)])
    assert passed["passed"] is True
    failed = contract.compare_repeats(
        [_episode(0), _episode(1, trajectory="changed"), _episode(2)]
    )
    row = failed["comparison_rows"][0]
    assert failed["passed"] is False
    assert row["trajectory_equal"] is False
    assert row["first_trajectory_difference_step"] == 2


def test_telemetry_record_preserves_core_control_and_physical_fields():
    base = contract.base_observation(_base_observation())
    record = contract.telemetry_record(
        step=1,
        base=base,
        phase="pre_merge",
        dynamic_state="head_on",
        history=[base],
        temporal_embedding=None,
        action=[0.1, -0.2, 0.3],
        info={
            "backend": "jsbsim",
            "vp_forward_bias_m": 100.0,
            "vp_lateral_bias_m": -50.0,
            "vp_vertical_bias_m": 20.0,
            "nz_cmd": 2.5,
            "ego_attack_aoa_deg": 8.0,
            "saturation_flag": False,
            "own_state": {"nz_g": 2.4},
        },
        decimal_places=6,
    )
    assert record["normalized_vpp_action"] == [0.1, -0.2, 0.3]
    assert record["vpp_vertical_component"] == 0.3
    assert record["actual_nz_g"] == 2.4
    assert record["physical_state_sha256"]
    assert record["step_sha256"]


def test_prediction_warmup_is_recorded_without_becoming_runtime_fallback():
    base = contract.base_observation(_base_observation())
    common = {
        "backend": "jsbsim",
        "vp_forward_bias_m": 100.0,
        "vp_lateral_bias_m": -50.0,
        "nz_cmd": 2.5,
        "ego_attack_aoa_deg": 8.0,
        "saturation_flag": False,
        "own_state": {"nz_g": 2.4},
    }
    reset = contract.telemetry_record(
        step=0,
        base=base,
        phase="pre_merge",
        dynamic_state="head_on",
        history=[base],
        temporal_embedding=None,
        action=None,
        info={"backend": "jsbsim"},
        decimal_places=6,
    )
    warmup = contract.telemetry_record(
        step=1,
        base=base,
        phase="pre_merge",
        dynamic_state="head_on",
        history=[base],
        temporal_embedding=None,
        action=[0.0, 0.0, 0.0],
        info={
            **common,
            "prediction_fallback": True,
            "prediction_fallback_phase": "warmup",
        },
        decimal_places=6,
    )
    header = {
        "opponent": "expert",
        "scenario_signature": "scenario_a",
        "scenario_name": "scenario_a",
        "repeat_index": 0,
        "order_variant": "forward",
    }
    summary = contract.summarize_episode(
        header=header,
        reset=reset,
        steps=[warmup],
        terminal_reason="timeout",
        decimal_places=6,
    )
    assert summary["no_backend_or_prediction_fallback"] is True
    assert summary["prediction_warmup_steps"] == 1


def test_read_only_audit_freezes_a_nonreproducible_manifest(tmp_path: Path):
    """The post-hoc audit must fail closed without changing frozen artifacts."""

    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    episodes = []
    artifacts = []
    for repeat, trajectory in enumerate(("a", "b", "c")):
        path = episode_dir / f"repeat_{repeat}.json"
        path.write_text(
            json.dumps(
                {
                    "header": {
                        "scenario_name": "scenario_a",
                        "scenario_metadata": {
                            "initial_class": "head_on",
                            "height_condition": "co_altitude",
                            "mirror_sign": "positive",
                        },
                    },
                    "reset": {},
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        episode = _episode(repeat, trajectory=trajectory) | {
            "scenario_name": "scenario_a",
            "order_variant": contract.ORDER_VARIANTS[repeat],
            "artifact_path": str(path),
            "artifact_sha256": digest,
            "step_count": 2,
            "prediction_warmup_steps": 0,
        }
        episodes.append(episode)
        artifacts.append({"path": str(path), "sha256": digest})
    manifest = {
        "source_id": contract.SOURCE_ID,
        "episodes": episodes,
        "artifacts": artifacts,
        "gate": contract.compare_repeats(episodes),
    }

    audit = audit_p1_manifest(tmp_path, manifest)

    assert audit["artifact_count"] == 3
    assert audit["gate_passed"] is False
    assert audit["findings"]["full_trajectory_reproducible"] is False
    assert audit["findings"]["full_reset_observation_payload_persisted"] is False
    assert audit["verdict"] == "runin_protocol_not_reproducible_do_not_train"
