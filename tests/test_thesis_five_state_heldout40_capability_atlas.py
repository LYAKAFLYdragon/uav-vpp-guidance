from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
ATLAS_PATH = ROOT / "scripts" / "analyze_thesis_five_state_heldout40_capability_atlas.py"
MANIFEST_PATH = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_heldout40_v1.yaml"


def _atlas():
    spec = importlib.util.spec_from_file_location("heldout40_atlas", ATLAS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory(method: str) -> list[dict]:
    is_canonical = method == "canonical_ppo_high_level_policy"
    return [
        {
            "time_s": 0.2,
            "pre_merge": True,
            "range_rate_mps": -350.0,
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
            "vp_forward_bias_m": 10.0,
            "vp_lateral_bias_m": 20.0,
            "vp_pos_z": 5204.0,
            "ego_pos_z": 5200.0,
            "commander_requested_mode_name": "head_on_specialist" if is_canonical else None,
            "commander_mode_name": "head_on_specialist" if is_canonical else None,
            "commander_mode_constraint_triggered": False,
            "commander_mode_constraint_reason": None,
            "commander_first_switch_step": None,
        },
        {
            "time_s": 0.4,
            "pre_merge": False,
            "range_rate_mps": -80.0,
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
            "vp_forward_bias_m": 20.0,
            "vp_lateral_bias_m": 30.0,
            "vp_pos_z": 5208.0,
            "ego_pos_z": 5200.0,
            "commander_requested_mode_name": "crossing_specialist" if is_canonical else None,
            "commander_mode_name": "crossing_specialist" if is_canonical else None,
            "commander_mode_constraint_triggered": is_canonical,
            "commander_mode_constraint_reason": "synthetic_guard" if is_canonical else None,
            "commander_first_switch_step": 2 if is_canonical else None,
        },
    ]


def _outcome_for(opponent: str, state: str, method: str, scenario_index: int) -> str:
    if state == "disadvantage":
        return "loss"
    if state == "head_on":
        if opponent == "end_to_end" and method == "canonical_ppo_high_level_policy" and scenario_index < 2:
            return "loss"
        if opponent == "end_to_end" and method == "crossing_vpp_specialist_no_routing" and scenario_index == 0:
            return "win"
        if opponent == "end_to_end" and method == "head_on_vpp_specialist_no_routing" and scenario_index == 1:
            return "win"
        if opponent == "expert" and method == "canonical_ppo_high_level_policy" and scenario_index == 0:
            return "loss"
    return "win"


def _write_run(run_dir: Path, opponent: str, scenarios: list[dict]) -> None:
    records = []
    task_episode = {"head_on": 0, "crossing_feasible": 0}
    state_index = {state: 0 for state in ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry")}
    for scenario in scenarios:
        metadata = scenario["metadata"]
        state = metadata["initial_class"]
        index = state_index[state]
        state_index[state] += 1
        task = metadata["task_registry_key"]
        for method in (
            "canonical_ppo_high_level_policy",
            "legacy_static_oracle_task_gate",
            "head_on_vpp_specialist_no_routing",
            "crossing_vpp_specialist_no_routing",
        ):
            outcome = _outcome_for(opponent, state, method, index)
            records.append(
                {
                    "opponent_stage": opponent,
                    "controller": method,
                    "task": task,
                    "scenario": scenario["name"],
                    "seed": metadata["scenario_seed"],
                    "episode": task_episode[task],
                    "backend": "jsbsim",
                    "strict_backend": True,
                    "scenario_metadata": metadata,
                    "win": outcome == "win",
                    "loss": outcome == "loss",
                    "draw": False,
                    "combat_reason": "timeout_hp_advantage" if outcome == "win" else "timeout_hp_disadvantage",
                    "ego_hp": 100.0 if outcome == "win" else 90.0,
                    "target_hp": 90.0 if outcome == "win" else 100.0,
                    "hp_advantage": 10.0 if outcome == "win" else -10.0,
                    "total_time_s": 0.4,
                    "commander_first_switch_step": 2 if method == "canonical_ppo_high_level_policy" else None,
                    "trajectory": _trajectory(method),
                }
            )
        task_episode[task] += 1
    aggregate = run_dir / "aggregate"
    aggregate.mkdir(parents=True)
    (aggregate / "episode_records.json").write_text(json.dumps({"episodes": records}), encoding="utf-8")


def _write_analysis_r2(path: Path) -> None:
    path.mkdir()
    source = "THESIS-FIVE-STATE-HELDOUT40-V1"
    (path / "heldout40_analysis.json").write_text(json.dumps({"source_id": source}), encoding="utf-8")
    (path / "heldout40_pairing_verification.json").write_text(
        json.dumps({"source_id": source, "actual_cells": 120, "all_cells_passed": True}),
        encoding="utf-8",
    )
    (path / "heldout40_terminal_discriminability_gate.json").write_text(
        json.dumps({"source_id": source, "opponents": {name: {"passed": True} for name in ("expert", "end_to_end", "independent_ppo_vpp")}}),
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_capability_atlas_is_read_only_and_emits_all_required_evidence(tmp_path):
    atlas = _atlas()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dirs = {}
    source_hashes = {}
    for opponent in atlas.OPPONENTS:
        run = tmp_path / opponent
        _write_run(run, opponent, manifest["scenarios"])
        run_dirs[opponent] = run
        source_hashes[opponent] = _hash(run / "aggregate" / "episode_records.json")
    analysis_r2 = tmp_path / "analysis_r2"
    _write_analysis_r2(analysis_r2)

    output = tmp_path / "atlas"
    report = atlas.analyze(
        run_dirs=run_dirs,
        manifest_path=MANIFEST_PATH,
        analysis_dir=analysis_r2,
        output_dir=output,
    )

    assert report["mode"] == "read_only"
    assert report["episode_count"] == 480
    assert report["input_artifacts"]["manifest"]["sha256"] == _hash(MANIFEST_PATH)
    assert report["labels"]["within_opponent"]["head_on/end_to_end"] == atlas.LABEL_ROUTING
    assert report["labels"]["within_opponent"]["disadvantage/expert"] == atlas.LABEL_LIBRARY
    assert report["labels"]["within_opponent"]["advantage/expert"] == atlas.LABEL_CEILING
    assert any(
        item["opponent_dependent"]
        for item in report["labels"]["opponent_sensitivity"]
        if item["initial_class"] == "head_on" and item["method"] == atlas.CANONICAL_METHOD
    )
    for name in report["artifact_names"]:
        assert (output / name).is_file(), name
    assert len(list(csv_rows(output / "capability_atlas_headon_end_to_end_routing_audit.csv"))) == 2
    assert len(list(csv_rows(output / "capability_atlas_disadvantage_audit.csv"))) == 24
    assert "does not assign a global strength" in (
        output / "capability_atlas_report.md"
    ).read_text(encoding="utf-8")
    assert "first evidence-backed diagnostic motif" in (
        output / "capability_atlas_report.md"
    ).read_text(encoding="utf-8")
    for opponent, run in run_dirs.items():
        assert _hash(run / "aggregate" / "episode_records.json") == source_hashes[opponent]
        assert report["input_artifacts"]["formal_episode_records"][opponent]["sha256"] == source_hashes[opponent]


def csv_rows(path: Path):
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def test_capability_atlas_rejects_unverified_pairing_input(tmp_path):
    atlas = _atlas()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dirs = {}
    for opponent in atlas.OPPONENTS:
        run = tmp_path / opponent
        _write_run(run, opponent, manifest["scenarios"])
        run_dirs[opponent] = run
    analysis_r2 = tmp_path / "analysis_r2"
    _write_analysis_r2(analysis_r2)
    (analysis_r2 / "heldout40_pairing_verification.json").write_text(
        json.dumps({"source_id": atlas.SOURCE_ID, "actual_cells": 120, "all_cells_passed": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pairing verification"):
        atlas.analyze(
            run_dirs=run_dirs,
            manifest_path=MANIFEST_PATH,
            analysis_dir=analysis_r2,
            output_dir=tmp_path / "atlas",
        )
