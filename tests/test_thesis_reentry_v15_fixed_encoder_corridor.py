from __future__ import annotations

from collections import deque
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_reentry_v15_fixed_encoder_corridor import (
    METHODS,
    OPPONENTS,
    SOURCE_ID,
    compose_fixed_encoder_observation,
    objective_profile,
    scheduled_action,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile


ROOT = Path(__file__).resolve().parent.parent


def _load_runner():
    path = ROOT / "scripts" / "run_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor.py"
    spec = importlib.util.spec_from_file_location("v15_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_encoder_profile_composes_the_existing_66d_schema():
    contract = FiveStateObservationContract()
    value = compose_fixed_encoder_observation(
        contract=contract,
        base_geometry=np.zeros(16),
        explicit_history=np.zeros(6),
        temporal_embedding=np.zeros(32),
        encoder_profile="range_extension",
    )
    profile = compile_profile("range_extension")
    assert value.shape == (66,)
    assert np.allclose(value[22:28], profile["target_vector"])
    assert np.allclose(value[28:34], profile["weight_vector"])


def test_v15_episode_does_not_read_a_dynamic_semantic_profile():
    runner = _load_runner()
    episode = object.__new__(runner.FixedEncoderTemporalCorridorEpisode)
    episode.base = np.zeros(16, dtype=np.float32)
    episode.history = deque([np.zeros(16, dtype=np.float32) for _ in range(10)], maxlen=10)
    episode.history_features = SimpleNamespace(features=lambda: np.zeros(6, dtype=np.float32))
    episode.encoder = SimpleNamespace(encode=lambda _history: np.zeros(32, dtype=np.float32))
    episode.contract = FiveStateObservationContract()
    episode.config = {"observation_contract": {"encoder_profile": "range_extension"}, "semantic_compiler": {}}
    assert episode._observation66d().shape == (66,)


def test_v15_design_requires_a_fixed_encoder_profile_and_rejects_semantic_profile(tmp_path: Path):
    runner = _load_runner()
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor_design.yaml").read_text(encoding="utf-8"))
    config["outputs"]["root"] = str(tmp_path / "v15_output")
    valid_path = tmp_path / "v15_valid.yaml"
    valid_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = runner.validate_design(valid_path)
    assert result["encoder_profile"] == "range_extension"
    assert result["fixed_encoder_dynamic_corridor_separation_passed"] is True
    config["semantic_compiler"]["profile"] = "range_extension"
    invalid_path = tmp_path / "v15_invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(runner.V15RuntimeError, match="must not define"):
        runner.validate_design(invalid_path)


def _synthetic_record(scenario: dict, method: str, opponent: str) -> dict:
    profile = compile_profile("range_extension")
    window = []
    for index in range(20):
        action_profile, axes = scheduled_action(method, index, escape_steps=5, corridor_steps=10)
        stage = "escape" if index < 5 else "reentry"
        improved = method == "escape_then_reentry_corridor" and opponent in {"expert", "end_to_end"}
        window.append(
            {
                "local_step": index,
                "objective_stage": stage,
                "objective_profile": objective_profile(index, escape_steps=5),
                "action_profile": action_profile,
                "semantic_axes": axes,
                "actual_semantic_delta_norm_m": 0.0 if method == "frozen_fixed_head_on" else 1.0,
                "observation_66d": [0.0] * 22 + profile["target_vector"].tolist() + profile["weight_vector"].tolist() + [0.0] * 32,
                "after_component_errors": [0.1] * 6,
                "after_weighted_error": 0.98 if improved else 1.0,
                "env_vpp_world_offset_m": [0.0, 0.0, 0.0],
            }
        )
    return {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "method": method,
        "pair_key": scenario["metadata"]["pair_key"],
        "handoff_reached": True,
        "contract_passed": True,
        "window_steps": 20,
        "handoff_state_hash": f"{opponent}:{scenario['metadata']['pair_key']}",
        "ego_failure": False,
        "window": window,
    }


def test_v15_independent_verifier_recomputes_fixed_encoder_gate(tmp_path: Path):
    runner = _load_runner()
    verifier_path = ROOT / "scripts" / "verify_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor.py"
    spec = importlib.util.spec_from_file_location("v15_verifier", verifier_path)
    assert spec and spec.loader
    verifier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = verifier
    spec.loader.exec_module(verifier)
    config_path = ROOT / "config" / "experiment" / "thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor_design.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["inputs"]["manifest"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    output = tmp_path / "v15_output"
    records = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                row = _synthetic_record(scenario, method, opponent)
                records.append(row)
                path = output / "episodes" / method / opponent / f"{scenario['name']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"record": row}), encoding="utf-8")
    summary = summarize_records(
        records,
        scenarios_per_opponent=12,
        min_window_steps=20,
        min_pairs=4,
        min_total_weighted_improvement=0.01,
        min_reentry_weighted_improvement=0.01,
        max_escape_range_rate_worsening=0.02,
        safety_delta_max=0.05,
    )
    (output / "fixed_encoder_temporal_corridor_gate.json").write_text(json.dumps({"source_id": SOURCE_ID, **summary}), encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "records": 144,
                "config_sha256": runner._hash(config_path),
                "manifest_sha256": runner._hash(manifest_path),
                "encoder_profile": "range_extension",
                "training": "prohibited",
                "reward_recording": "prohibited",
                "authorized_code_files": [{"path": "synthetic", "sha256": "synthetic"}],
            }
        ),
        encoding="utf-8",
    )
    result = verifier.verify(config_path, output)
    assert result["passes"] is True
    assert result["fixed_encoder_profile"] == "range_extension"
