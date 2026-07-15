from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_contract import (
    FiveStateRawSIV5ContractError,
    build_plan,
    evaluate_observability,
    expected_pairs_by_state,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_five_state_raw_si_v5_observability.yaml"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_raw_si_v5_observability60.yaml"
BUILDER = ROOT / "scripts" / "build_thesis_five_state_raw_si_v5_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_five_state_raw_si_v5_observability.py"
RUNNER = ROOT / "scripts" / "run_thesis_five_state_raw_si_v5_observability.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _complete_records(config: dict, manifest: dict) -> list[dict]:
    records = []
    for opponent in config["opponents"]["order"]:
        for scenario in manifest["scenarios"]:
            records.append(
                {
                    "source_id": config["source_id"],
                    "opponent": opponent,
                    "initial_state": scenario["metadata"]["initial_class"],
                    "pair_key": scenario["metadata"]["pair_key"],
                    "metadata": scenario["metadata"],
                    "result": {
                        "raw_event_reached": True,
                        "post_event_66d_steps": 20,
                        "telemetry_complete": True,
                        "no_backend_fallback": True,
                        "no_reset_after_initialization": True,
                        "no_history_padding": True,
                        "no_checkpoint_fallback": True,
                    },
                }
            )
    return records


def test_v5_manifest_is_rebuildable_disjoint_and_preserves_the_60_cell_contract():
    builder = _load(BUILDER, "v5_manifest_builder")
    sources = {label: builder._load_yaml(path) for label, path in builder.DISJOINT_MANIFESTS}
    rebuilt = builder.build_manifest(sources)
    tracked = _yaml(MANIFEST)

    assert rebuilt["integrity"]["payload_sha256"] == tracked["integrity"]["payload_sha256"]
    assert len(validate_manifest(tracked)) == 60
    assert {item["metadata"]["initial_class"] for item in tracked["scenarios"]} == {
        "advantage", "head_on", "disadvantage", "neutral", "crossing_entry"
    }
    assert all(item["physical_intersection_count"] == 0 for item in tracked["disjointness"].values())
    assert all(item["seed_intersection_count"] == 0 for item in tracked["disjointness"].values())


def test_v5_design_preflight_is_nonexecuting_and_prohibits_training(tmp_path: Path):
    config = _yaml(CONFIG)
    preflight = _load(PREFLIGHT, "v5_preflight")

    config["outputs"]["root"] = str(tmp_path / "fresh_v5_preflight_output")
    design_path = tmp_path / "v5_design.yaml"
    design_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = preflight.validate_design(design_path)
    assert result["planned_records"] == 180
    assert result["reference_controller"] == "frozen_fixed_head_on"
    assert config["authorization"]["execution_permitted"] is False
    assert all(value is False for value in config["authorization"].values())
    assert "proposed_training" not in config


def test_v5_gate_is_stratified_by_initial_state_and_opponent_without_pooling():
    config, manifest = _yaml(CONFIG), _yaml(MANIFEST)
    plan = build_plan(config)
    gate = evaluate_observability(_complete_records(config, manifest), plan, expected_pairs_by_state(manifest))

    assert gate["pooling_prohibited"] is True
    assert gate["all_initial_state_opponent_cells_pass"] is True
    assert gate["verdict"] == "v5_five_state_observability_established"
    assert set(gate["matrix"]) == set(config["opponents"]["order"])
    assert all(set(row) == set(config["observability"]["initial_state_order"]) for row in gate["matrix"].values())


def test_v5_gate_rejects_duplicate_pair_and_contract_breaks():
    config, manifest = _yaml(CONFIG), _yaml(MANIFEST)
    plan = build_plan(config)
    records = _complete_records(config, manifest)
    records[0] = copy.deepcopy(records[1])
    with pytest.raises(FiveStateRawSIV5ContractError, match="extra or duplicate"):
        evaluate_observability(records, plan, expected_pairs_by_state(manifest))

    changed = copy.deepcopy(config)
    changed["observability"]["reference_profile"] = "range_extension"
    with pytest.raises(FiveStateRawSIV5ContractError, match="front-intercept"):
        build_plan(changed)


def test_v5_runner_requires_a_separate_authorized_overlay():
    runner = _load(RUNNER, "v5_runner")
    with pytest.raises(RuntimeError, match="separate authorized overlay"):
        runner.authorized_preflight(CONFIG)


def test_v5_step_hash_canonicalizes_ndarray_telemetry_without_losing_the_record():
    recorder = _load(
        ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "thesis_five_state_raw_si_v5_record_only.py",
        "v5_record_only",
    )
    first = recorder._stable_hash({"guidance_command": recorder.np.asarray([1.0, float("nan")])})
    second = recorder._stable_hash({"guidance_command": [1.0, float("nan")]})

    assert isinstance(first, str) and len(first) == 64
    assert first == second
