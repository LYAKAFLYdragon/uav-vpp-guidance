from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_five_state_phase_feasible_disadvantage_manifest.py"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_phase_feasible_disadvantage_v1.yaml"
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_five_state_phase_feasible_disadvantage_v1.yaml"
ANALYZER = ROOT / "scripts" / "analyze_thesis_five_state_phase_feasible_disadvantage.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_feasible_manifest_preserves_disadvantage_and_nontraining_gate():
    module = _load(BUILDER, "phase_feasible_builder")
    manifest = module.build_manifest()
    assert manifest["source_id"] == module.SOURCE_ID
    assert len(manifest["scenarios"]) == 12
    assert manifest["protocol"]["training_permitted"] is False
    assert manifest["phase_gate"]["re_entry_step_minimum_per_episode"] == 10
    assert all(item["metadata"]["taxonomy_geometry_state"] == "disadvantage" for item in manifest["scenarios"])
    assert all(item["target_init"]["velocity_mps"] > item["own_init"]["velocity_mps"] for item in manifest["scenarios"])


def test_phase_feasible_config_locks_manifest_and_reference_only_scope():
    module = _load(BUILDER, "phase_feasible_builder_config")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["phase_feasible_disadvantage_protocol"]["training_permitted"] is False
    assert config["phase_feasible_disadvantage_protocol"]["reference_controller"] == "legacy_static_oracle_task_gate"
    assert config["run_defaults"]["main_methods"] == ["legacy_static_oracle_task_gate"]
    assert config["scenario_manifest"]["active_task_groups"] == ["head_on"]
    assert config["scenario_manifest"]["payload_sha256"] == manifest["integrity"]["payload_sha256"]
    assert module._payload_sha256(manifest) == manifest["integrity"]["payload_sha256"]


def _write_phase_run(path: Path, opponent: str, scenarios: list[dict], *, reentry: bool) -> None:
    episodes = []
    for scenario in scenarios:
        trajectory = [
            {"pre_merge": True, "range_rate_mps": -80.0},
            *[{"pre_merge": False, "range_rate_mps": -30.0 if reentry else 30.0} for _ in range(16)],
        ]
        episodes.append(
            {
                "controller": "legacy_static_oracle_task_gate",
                "opponent_stage": opponent,
                "backend": "jsbsim",
                "strict_backend": True,
                "scenario": scenario["name"],
                "seed": scenario["metadata"]["scenario_seed"],
                "combat_reason": "timeout_hp_disadvantage",
                "trajectory": trajectory,
            }
        )
    aggregate = path / "aggregate"
    aggregate.mkdir(parents=True)
    (aggregate / "episode_records.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")


def test_phase_gate_requires_each_opponent_to_reach_reentry(tmp_path):
    analyzer = _load(ANALYZER, "phase_feasible_analyzer")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    run_dirs = {}
    for opponent in analyzer.OPPONENTS:
        path = tmp_path / opponent
        _write_phase_run(path, opponent, manifest["scenarios"], reentry=opponent != "end_to_end")
        run_dirs[opponent] = path
    result = analyzer.analyze(manifest_path=MANIFEST, run_dirs=run_dirs, output_dir=tmp_path / "out")
    assert result["gate"]["expert"]["passed"] is True
    assert result["gate"]["end_to_end"]["passed"] is False
    assert result["verdict"] == "phase_feasible_envelope_not_established"
