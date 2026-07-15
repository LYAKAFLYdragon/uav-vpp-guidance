from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_reentry_v16_dynamic_reference_path import (
    METHODS,
    OPPONENTS,
    SOURCE_ID,
    semantic_axes_for_method,
    semantic_profile_for_method,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.virtual_point.dynamic_reference_path import (
    DynamicReferencePathConfig,
    build_three_anchor_path,
    compile_path_anchor_action,
    path_progress_from_range_rate,
    select_path_anchor,
    semantic_profile_for_anchor,
)


ROOT = Path(__file__).resolve().parent.parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v16_design_locks_a_three_method_current_information_probe(tmp_path: Path):
    runner = _load_module(
        ROOT / "scripts" / "run_thesis_five_state_reentry_v16_dynamic_reference_path.py",
        "v16_runner_design",
    )
    config = yaml.safe_load(
        (ROOT / "config/experiment/thesis_five_state_reentry_v16_dynamic_reference_path_design.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["outputs"]["root"] = str(tmp_path / "v16_output")
    path = tmp_path / "v16_design.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = runner.validate_design(path)
    assert result["planned_records"] == 108
    assert result["path_progress_source"] == "observed_range_rate_only"
    assert config["authorization"]["training_permitted"] is False
    assert config["contract"]["future_simulator_state"] == "prohibited"


def test_v16_semantic_profile_only_varies_at_the_path_adapter_boundary():
    assert semantic_profile_for_method("frozen_fixed_head_on", "reentry_preparation") == "reentry_preparation"
    assert semantic_profile_for_method("static_range_extension", "reentry_preparation") == "range_extension"
    assert semantic_profile_for_method("progress_triggered_reference_path", "reentry_preparation") == "reentry_preparation"
    assert not any(semantic_axes_for_method("frozen_fixed_head_on").values())
    assert all(semantic_axes_for_method("progress_triggered_reference_path").values())


def _path_step(method: str, *, local_step: int, dynamic_delta: bool) -> dict:
    own = np.asarray([0.0, 0.0, 5000.0])
    target = np.asarray([1000.0, 0.0, 5000.0])
    predicted = np.asarray([1200.0, 0.0, 5000.0])
    path_config = DynamicReferencePathConfig()
    path = build_three_anchor_path(
        own_position_neu=own,
        target_position_neu=target,
        predicted_target_position_neu=predicted,
        config=path_config,
    )
    progress = path_progress_from_range_rate(40.0, path_config)
    selected_index, selected_anchor = select_path_anchor(path, progress)
    selected_profile = semantic_profile_for_anchor(selected_index)
    action_profile = semantic_profile_for_method(method, selected_profile)
    semantic_receipt = {
        "profile": action_profile,
        "enabled_axes": semantic_axes_for_method(method),
        "applied_action": [0.0, 0.0, 0.0],
        "actual_semantic_delta_norm_m": 0.0,
    }
    path_receipt = None
    if method == "progress_triggered_reference_path":
        path_receipt = compile_path_anchor_action(
            semantic_action=semantic_receipt["applied_action"],
            selected_anchor_neu=selected_anchor,
            generator_anchor_neu=predicted,
            raw_action_ranges=[[-1500.0, 1500.0], [-800.0, 800.0], [-500.0, 500.0]],
            raw_action_offset_frame="world_neu",
            own_state={"position_neu": own},
            target_state={"position_neu": target},
        )
    profile = compile_profile("range_extension")
    return {
        "local_step": local_step,
        "objective_stage": "escape",
        "objective_profile": selected_profile,
        "action_profile": action_profile,
        "semantic_axes": semantic_axes_for_method(method),
        "own_position_neu": own.tolist(),
        "target_position_neu": target.tolist(),
        "predicted_target_position_neu": predicted.tolist(),
        "prediction_valid": True,
        "prediction_fallback": False,
        "range_rate_mps": 40.0,
        "path_progress": progress,
        "anchors_neu": path.tolist(),
        "selected_anchor_index": selected_index,
        "selected_anchor_profile": selected_profile,
        "semantic_receipt": semantic_receipt,
        "path_action_receipt": path_receipt,
        "phase_before": "re_entry",
        "observation_66d": [0.0] * 22 + profile["target_vector"].tolist() + profile["weight_vector"].tolist() + [0.0] * 32,
        "before_intent_vector": [0.0] * 6,
        "after_intent_vector": [0.0] * 6,
        "before_component_errors": [0.1] * 6,
        "after_component_errors": [0.1] * 6,
        "before_weighted_error": 0.1,
        "after_weighted_error": 0.1,
        "env_vpp_world_offset_m": [2.0, 0.0, 0.0] if dynamic_delta else [0.0, 0.0, 0.0],
        "guidance_command_normalized": [0.01, 0.0, 0.0] if dynamic_delta else [0.0, 0.0, 0.0],
        "actual_nz_g": 1.0,
        "saturated": False,
    }


def _record(scenario: dict, method: str, opponent: str) -> dict:
    return {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "method": method,
        "pair_key": scenario["metadata"]["pair_key"],
        "handoff_reached": True,
        "contract_passed": True,
        "window_steps": 20,
        "handoff_state_hash": f"{opponent}:{scenario['metadata']['pair_key']}",
        "terminal_reason": "timeout",
        "ego_failure": False,
        "window": [
            _path_step(method, local_step=index, dynamic_delta=method == "progress_triggered_reference_path")
            for index in range(20)
        ],
    }


def test_v16_summary_requires_two_unpooled_mechanism_signals():
    scenario = {"metadata": {"pair_key": "synthetic"}}
    records = [
        _record(scenario, method, opponent)
        for method in METHODS
        for opponent in OPPONENTS
    ]
    summary = summarize_records(
        records,
        scenarios_per_opponent=1,
        min_window_steps=20,
        min_pairs=1,
        min_anchor_path_delta_norm_m=1.0,
        min_paired_vpp_delta_norm_m=1.0,
        min_paired_guidance_delta_norm=0.001,
        max_escape_range_rate_worsening=0.02,
        safety_delta_max=0.05,
    )
    assert summary["passes"] is True
    assert set(summary["mechanism_candidate_opponents"]) == set(OPPONENTS)


def test_v16_independent_verifier_replays_path_and_rejects_gate_drift(tmp_path: Path):
    runner = _load_module(
        ROOT / "scripts" / "run_thesis_five_state_reentry_v16_dynamic_reference_path.py",
        "v16_runner_for_verifier",
    )
    verifier = _load_module(
        ROOT / "scripts" / "verify_thesis_five_state_reentry_v16_dynamic_reference_path.py",
        "v16_verifier",
    )
    config_path = ROOT / "config/experiment/thesis_five_state_reentry_v16_dynamic_reference_path_design.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load((ROOT / config["inputs"]["manifest"]).read_text(encoding="utf-8"))
    output = tmp_path / "v16_output"
    records = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                row = _record(scenario, method, opponent)
                records.append(row)
                path = output / "episodes" / method / opponent / f"{scenario['name']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"record": row}), encoding="utf-8")
    summary = summarize_records(
        records,
        scenarios_per_opponent=12,
        min_window_steps=20,
        min_pairs=4,
        min_anchor_path_delta_norm_m=1.0,
        min_paired_vpp_delta_norm_m=1.0,
        min_paired_guidance_delta_norm=0.001,
        max_escape_range_rate_worsening=0.02,
        safety_delta_max=0.05,
    )
    (output / "dynamic_reference_path_gate.json").write_text(
        json.dumps({"source_id": SOURCE_ID, **summary}), encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "records": 108,
                "config_sha256": runner._hash(config_path),
                "manifest_sha256": runner._hash(ROOT / config["inputs"]["manifest"]),
                "training": "prohibited",
                "reward_recording": "prohibited",
                "path_input": "current_own_target_and_frozen_prediction_only",
                "authorized_code_files": [{"path": "synthetic", "sha256": "synthetic"}],
            }
        ),
        encoding="utf-8",
    )
    assert verifier.verify(config_path, output)["passes"] is True
    (output / "dynamic_reference_path_gate.json").write_text(
        json.dumps({"source_id": SOURCE_ID, "passes": False}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="reported gate"):
        verifier.verify(config_path, output)
