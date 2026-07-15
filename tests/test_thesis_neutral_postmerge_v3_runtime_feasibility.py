from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys

import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v3_contract import (
    NeutralPostMergeV3ContractError,
    build_plan,
    episode_identity,
    evaluate_family_coverage,
    runtime_identity_index,
    validate_feasibility_manifest,
)
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v3_runtime import runtime_record


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.yaml"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility48.yaml"
TRAIN = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_v3_train_distribution.yaml"
BUILDER = ROOT / "scripts" / "build_thesis_neutral_postmerge_reentry_recovery_v3_feasibility_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py"
RUNNER = ROOT / "scripts" / "run_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v3_manifest_has_four_disjoint_families_and_round_trip_evaluation_identities():
    builder = _load(BUILDER, "v3_builder")
    sources = {label: builder._load_yaml(path) for label, path in builder.DISJOINT_MANIFESTS}
    manifest = builder.build_manifest(sources)

    keys = validate_feasibility_manifest(manifest)
    assert len(keys) == 48
    assert len(set(keys)) == 48
    assert {item["metadata"]["feasibility_family_id"] for item in manifest["scenarios"]} == set(builder.FAMILY_ORDER)
    assert all(item["physical_intersection_count"] == 0 for item in manifest["disjointness"].values())
    assert all(item["seed_intersection_count"] == 0 for item in manifest["disjointness"].values())
    assert all(episode_identity(item)["paired_delta_eligible"] for item in manifest["scenarios"])


def test_v3_train_identity_is_explicitly_unpaired_and_rejects_evaluation_metadata():
    train = _payload(TRAIN)
    scenario = {
        "metadata": {
            "identity_kind": train["identity_contract"]["identity_kind"],
            "train_stream_id": train["identity_contract"]["train_stream_id"],
            "scenario_seed": train["sampling_contract"]["seed"],
        }
    }
    identity = episode_identity(scenario)
    assert identity == {
        "kind": "train",
        "key": "neutral_postmerge_reentry_v3_train::seed=2026071701",
        "paired_delta_eligible": False,
    }
    scenario["metadata"]["pair_key"] = "not-allowed"
    with pytest.raises(NeutralPostMergeV3ContractError, match="must not carry evaluation pairing metadata"):
        episode_identity(scenario)


def test_v3_runtime_serializer_keeps_only_frozen_baseline_feasibility_fields():
    scenario = _payload(MANIFEST)["scenarios"][0]
    result = {
        "handoff_reached": True,
        "valid_target_steps": 20,
        "qualifying": True,
        "telemetry_complete": True,
        "terminal_reason": "timeout",
        "handoff_state_sha256": "handoff",
    }
    ledger = {
        "summary": {
            "no_backend_fallback": True,
            "full_post_handoff_telemetry": True,
        }
    }
    evidence = {
        "scenario_application_passed": True,
        "raw_si_phase_replay_passed": True,
        "initial_pre_merge_semantics_passed": True,
    }
    record = runtime_record(
        opponent="expert",
        scenario=scenario,
        result=result,
        ledger=ledger,
        runtime_evidence=evidence,
    )

    assert record["paired_delta_eligible"] is False
    assert record["role"] == "nonlearning_frozen_baseline_runtime_feasibility"
    assert record["pair_key"] == scenario["metadata"]["pair_key"]
    invalid_evidence = dict(evidence)
    invalid_evidence["raw_si_phase_replay_passed"] = False
    with pytest.raises(NeutralPostMergeV3ContractError, match="raw_si_phase_replay_passed"):
        runtime_record(
            opponent="expert",
            scenario=scenario,
            result=result,
            ledger=ledger,
            runtime_evidence=invalid_evidence,
        )
    ledger["summary"]["no_backend_fallback"] = False
    with pytest.raises(NeutralPostMergeV3ContractError, match="backend fallback"):
        runtime_record(
            opponent="expert",
            scenario=scenario,
            result=result,
            ledger=ledger,
            runtime_evidence=evidence,
        )


def test_v3_family_gate_selects_only_a_family_that_passes_every_opponent():
    config, manifest = _payload(CONFIG), _payload(MANIFEST)
    plan = build_plan(config)
    records = []
    for scenario in manifest["scenarios"]:
        metadata = scenario["metadata"]
        passing = metadata["feasibility_family_id"] == "midrange_balanced"
        for opponent in plan.opponents:
            records.append(
                {
                    "opponent": opponent,
                    "family_id": metadata["feasibility_family_id"],
                    "pair_key": metadata["pair_key"],
                    "metadata": metadata,
                    "result": {
                        "handoff_reached": passing,
                        "qualifying": passing,
                        "telemetry_complete": True,
                        "valid_target_steps": 20 if passing else 0,
                    },
                }
            )
    expected = runtime_identity_index(manifest)
    result = evaluate_family_coverage(records, plan, expected)

    assert result["selected_family"]["family_id"] == "midrange_balanced"
    assert result["future_pilot_authorized"] is False
    end_to_end = result["selected_family"]["per_opponent"]["end_to_end"]
    assert end_to_end["qualifying_episodes"] == 12
    assert end_to_end["valid_target_steps"] == 240

    without_end_to_end = [item for item in records if item["opponent"] != "end_to_end"]
    with pytest.raises(NeutralPostMergeV3ContractError, match="runtime record universe is incomplete"):
        evaluate_family_coverage(without_end_to_end, plan, expected)


def test_v3_design_preflight_is_nonexecuting_and_runner_rejects_execute(monkeypatch: pytest.MonkeyPatch):
    preflight = _load(PREFLIGHT, "v3_preflight")
    runner = _load(RUNNER, "v3_runner")
    result = preflight.validate_design()

    assert result["execution_permitted"] is False
    assert result["training_permitted"] is False
    assert result["planned_records"] == 144
    assert result["output_root_absent"] is True
    monkeypatch.setattr(sys, "argv", ["v3-runner", "--preflight"])
    assert runner.main() == 0
    monkeypatch.setattr(sys, "argv", ["v3-runner", "--execute"])
    assert runner.main() == 2


def test_v3_future_authorized_overlay_must_hash_and_merge_a_design_base(tmp_path: Path):
    runner = _load(RUNNER, "v3_runner_overlay")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"source_id": "base", "nested": {"frozen": 1}}), encoding="utf-8")
    overlay = tmp_path / "authorized.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "status": "authorized_one_time_execution",
                "nested": {"authorization_only": 2},
                "execution": {"base_config_sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )

    merged, resolved_base, loaded_overlay = runner._load_authorized_config(overlay)
    assert resolved_base == base
    assert merged["nested"] == {"frozen": 1, "authorization_only": 2}
    assert loaded_overlay["status"] == "authorized_one_time_execution"


def test_v3_authorized_overlay_scope_cannot_change_frozen_method_fields(tmp_path: Path):
    runner = _load(RUNNER, "v3_runner_scope")
    base = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.yaml"
    permitted = {
        "base_config": str(base.relative_to(ROOT)).replace("\\", "/"),
        "status": "authorized_one_time_execution",
        "authorization": {},
        "execution": {},
        "outputs": {"creation_permitted_by_this_config": True},
    }
    runner._validate_overlay_scope(permitted, base)
    permitted["fixed_contract"] = {"action_dim": 9}
    with pytest.raises(RuntimeError, match="cannot alter frozen method fields"):
        runner._validate_overlay_scope(permitted, base)
