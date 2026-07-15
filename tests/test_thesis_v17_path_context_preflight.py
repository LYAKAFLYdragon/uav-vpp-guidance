from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_v17_path_context_preflight import OPPONENTS, SOURCE_ID, summarize_records
from uav_vpp_guidance.hierarchy.path_conditioned_observation_contract import build_path_context


ROOT = Path(__file__).resolve().parent.parent


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(scenario: dict, opponent: str) -> dict:
    window = []
    for index in range(20):
        anchor = 0 if index == 0 else 2
        context = build_path_context(
            requested_path_world_delta_m=[0.0, 0.0, 0.0],
            path_progress=0.0 if anchor == 0 else 1.0,
            selected_anchor_index=anchor,
        )
        window.append(
            {
                "local_step": index,
                "observation_74d": [0.0] * 66 + context.tolist(),
                "path_context": context.tolist(),
                "path_context_finite": True,
                "raw_action": [0.0, 0.0, 0.0],
                "raw_action_finite": True,
                "requested_path_world_delta_m": [0.0, 0.0, 0.0],
                "path_progress": 0.0 if anchor == 0 else 1.0,
                "selected_anchor_index": anchor,
                "prediction_valid": True,
                "prediction_fallback": False,
                "env_vpp_world_offset_m": [0.0, 0.0, 0.0],
                "guidance_command_normalized": [0.0, 0.0, 0.0],
            }
        )
    return {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "pair_key": scenario["metadata"]["pair_key"],
        "mirror_sign": scenario["metadata"]["mirror_sign"],
        "scenario_signature": scenario["metadata"]["scenario_signature"],
        "handoff_reached": True,
        "contract_passed": True,
        "window_steps": 20,
        "terminal_reason": "timeout_hp_advantage",
        "ego_failure": False,
        "window": window,
    }


def test_v17_design_preflight_locks_72_records_and_new_74d_contract(tmp_path: Path):
    runner = _module(ROOT / "scripts/run_thesis_five_state_v17_path_context_preflight.py", "v17_runner_design")
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_v17_path_context_preflight_design.yaml").read_text(encoding="utf-8"))
    config["outputs"]["root"] = str(tmp_path / "v17_output")
    path = tmp_path / "v17_design.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = runner.validate_design(path)
    assert result["planned_records"] == 72
    assert result["observation_dim"] == 74


def test_v17_unpooled_sampler_gate_requires_each_opponent_to_cover_escape_and_reentry():
    scenario = {"metadata": {"pair_key": "p", "mirror_sign": "negative", "scenario_signature": "s"}}
    records = [_record(scenario, opponent) for opponent in OPPONENTS]
    summary = summarize_records(records, scenarios_per_opponent=1, min_qualifying_episodes=1, min_valid_steps=20, min_mirrors=1, min_signatures=1)
    assert summary["passes"] is True
    for step in records[-1]["window"]:
        step["selected_anchor_index"] = 0
    summary = summarize_records(records, scenarios_per_opponent=1, min_qualifying_episodes=1, min_valid_steps=20, min_mirrors=1, min_signatures=1)
    assert summary["matrix"]["independent_ppo_vpp"]["passes"] is False


def test_v17_independent_verifier_recomputes_the_full_frozen_universe(tmp_path: Path):
    runner = _module(ROOT / "scripts/run_thesis_five_state_v17_path_context_preflight.py", "v17_runner_for_verifier")
    verifier = _module(ROOT / "scripts/verify_thesis_five_state_v17_path_context_preflight.py", "v17_verifier")
    config_path = ROOT / "config/experiment/thesis_five_state_v17_path_context_preflight_design.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load((ROOT / config["inputs"]["manifest"]).read_text(encoding="utf-8"))
    output = tmp_path / "v17_output"
    records = []
    for opponent in OPPONENTS:
        for scenario in manifest["scenarios"]:
            row = _record(scenario, opponent)
            records.append(row)
            path = output / "episodes" / opponent / f"{scenario['name']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"record": row}), encoding="utf-8")
    summary = summarize_records(records, scenarios_per_opponent=24, min_qualifying_episodes=8, min_valid_steps=160, min_mirrors=2, min_signatures=2)
    (output / "path_context_preflight_gate.json").write_text(json.dumps({"source_id": SOURCE_ID, **summary}), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({"source_id": SOURCE_ID, "records": 72, "config_sha256": runner._hash(config_path), "manifest_sha256": runner._hash(ROOT / config["inputs"]["manifest"]), "training": "prohibited", "reward_recording": "prohibited"}), encoding="utf-8")
    assert verifier.verify(config_path, output)["passes"] is True
    (output / "path_context_preflight_gate.json").write_text(json.dumps({"source_id": SOURCE_ID, "passes": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reported gate"):
        verifier.verify(config_path, output)
