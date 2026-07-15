from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_reentry_v14_temporal_corridor import (
    METHODS,
    objective_profile,
    scheduled_action,
    summarize_records,
)


ROOT = Path(__file__).resolve().parent.parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_corridor_uses_a_time_indexed_profile_without_changing_window_length():
    assert objective_profile(0, escape_steps=5) == "range_extension"
    assert objective_profile(5, escape_steps=5) == "reentry_preparation"
    profile, axes = scheduled_action("escape_then_reentry_corridor", 0, escape_steps=5, corridor_steps=10)
    assert profile == "range_extension" and all(axes.values())
    profile, axes = scheduled_action("escape_then_reentry_corridor", 5, escape_steps=5, corridor_steps=10)
    assert profile == "reentry_preparation" and all(axes.values())
    profile, axes = scheduled_action("escape_then_reentry_corridor", 10, escape_steps=5, corridor_steps=10)
    assert profile == "reentry_preparation" and not any(axes.values())


def _record(method: str, opponent: str, key: str, escape_error: float, reentry_error: float, escape_rate_error: float) -> dict:
    window = []
    for index in range(20):
        stage = "escape" if index < 5 else "reentry"
        error = escape_error if stage == "escape" else reentry_error
        component_errors = [0.1] * 6
        if stage == "escape":
            component_errors[3] = escape_rate_error
        window.append({"objective_stage": stage, "after_weighted_error": error, "after_component_errors": component_errors})
    return {"method": method, "opponent": opponent, "pair_key": key, "handoff_reached": True, "contract_passed": True, "window_steps": 20, "handoff_state_hash": "same", "ego_failure": False, "window": window}


def test_corridor_summary_requires_escape_and_reentry_improvements_in_two_opponents():
    records = []
    for opponent in ("expert", "end_to_end", "independent_ppo_vpp"):
        for index in range(2):
            key = f"{opponent}-{index}"
            records.append(_record("frozen_fixed_head_on", opponent, key, 1.0, 1.0, 0.1))
            for method in METHODS[1:]:
                if opponent == "independent_ppo_vpp":
                    records.append(_record(method, opponent, key, 1.0, 1.0, 0.2))
                else:
                    records.append(_record(method, opponent, key, 0.98, 0.98, 0.1))
    summary = summarize_records(records, scenarios_per_opponent=2, min_window_steps=20, min_pairs=2, min_total_weighted_improvement=0.01, min_reentry_weighted_improvement=0.01, max_escape_range_rate_worsening=0.02, safety_delta_max=0.05)

    assert summary["passes"] is True
    assert "escape_then_reentry_corridor" in summary["corridor_candidates"]


def test_v14_design_locks_corridor_to_the_existing_twenty_step_window(tmp_path: Path):
    module = _load_module(ROOT / "scripts" / "run_thesis_five_state_reentry_v14_temporal_corridor.py", "v14_runner_design")
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v14_temporal_corridor_design.yaml").read_text(encoding="utf-8"))
    assert config["observability"]["local_window_steps"] == 20
    assert config["corridor"]["corridor_steps"] == 10
    assert config["authorization"]["training_permitted"] is False
    config["outputs"]["root"] = str(tmp_path / "v14_output")
    path = tmp_path / "v14_design.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert module.validate_design(path)["planned_records"] == 144


def test_v14_refuses_execution_before_creating_output_when_disk_gate_fails(monkeypatch):
    runner = _load_module(ROOT / "scripts" / "run_thesis_five_state_reentry_v14_temporal_corridor.py", "v14_runner_disk_gate")
    monkeypatch.setattr(runner, "_load_runtime_config", lambda _path: {})
    monkeypatch.setattr(runner, "validate_design", lambda _path: {"disk_gate_would_pass": False})
    monkeypatch.setattr(sys, "argv", ["v14_runner", "--config", "ignored.yaml", "--execute"])
    with pytest.raises(runner.V14RuntimeError, match="free-disk gate"):
        runner.main()


def _full_record(runner, scenario: dict, method: str, opponent: str) -> dict:
    window = []
    for index in range(20):
        profile, axes = scheduled_action(method, index, escape_steps=5, corridor_steps=10)
        stage = "escape" if index < 5 else "reentry"
        improved = method == "escape_then_reentry_corridor" and opponent in {"expert", "end_to_end"}
        error = 0.98 if improved else 1.0
        window.append(
            {
                "local_step": index,
                "objective_stage": stage,
                "objective_profile": objective_profile(index, escape_steps=5),
                "action_profile": profile,
                "semantic_axes": axes,
                "actual_semantic_delta_norm_m": 0.0 if method == "frozen_fixed_head_on" else 1.0,
                "observation_66d": [0.0] * 66,
                "after_component_errors": [0.1] * 6,
                "after_weighted_error": error,
                "env_vpp_world_offset_m": [0.0, 0.0, 0.0],
            }
        )
    return {
        "source_id": runner.SOURCE_ID,
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


def test_v14_independent_verifier_recomputes_gate_and_rejects_gate_drift(tmp_path: Path):
    runner = _load_module(ROOT / "scripts" / "run_thesis_five_state_reentry_v14_temporal_corridor.py", "v14_runner_for_verifier")
    verifier = _load_module(ROOT / "scripts" / "verify_thesis_five_state_reentry_v14_temporal_corridor.py", "v14_verifier")
    config_path = ROOT / "config" / "experiment" / "thesis_five_state_reentry_v14_temporal_corridor_design.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load((ROOT / config["inputs"]["manifest"]).read_text(encoding="utf-8"))
    output = tmp_path / "v14_output"
    records = []
    for method in METHODS:
        for opponent in ("expert", "end_to_end", "independent_ppo_vpp"):
            for scenario in manifest["scenarios"]:
                row = _full_record(runner, scenario, method, opponent)
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
    (output / "temporal_corridor_gate.json").write_text(json.dumps({"source_id": runner.SOURCE_ID, **summary}), encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_id": runner.SOURCE_ID,
                "records": 144,
                "config_sha256": runner._hash(config_path),
                "manifest_sha256": runner._hash(ROOT / config["inputs"]["manifest"]),
                "training": "prohibited",
                "reward_recording": "prohibited",
                "authorized_code_files": [{"path": "synthetic", "sha256": "synthetic"}],
            }
        ),
        encoding="utf-8",
    )
    assert verifier.verify(config_path, output)["passes"] is True
    (output / "temporal_corridor_gate.json").write_text(json.dumps({"source_id": runner.SOURCE_ID, "passes": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reported gate"):
        verifier.verify(config_path, output)
