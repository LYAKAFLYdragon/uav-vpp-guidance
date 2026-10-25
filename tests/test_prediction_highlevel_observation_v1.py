import json
from pathlib import Path

import pytest

from scripts.analyze_prediction_highlevel_observation_v1 import analyze
from scripts.run_prediction_highlevel_observation_v1 import (
    _assert_new_output_root,
    _write_pilot_outputs,
    main,
)
from scripts.preflight_prediction_highlevel_observation_v1 import audit_source_contract
from scripts.verify_prediction_highlevel_observation_v1 import verify
from uav_vpp_guidance.evaluation.prediction_highlevel_observation_v1 import (
    SOURCE_ID,
    aggregate_episodes,
    load_resolved_config,
    method_config,
    paired_deltas,
    wilson_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiment/prediction_highlevel_observation_v1.yaml"


def test_protocol_config_has_locked_h0_h1_h2_contract():
    config = load_resolved_config(CONFIG)
    assert config["protocol"]["source_id"] == SOURCE_ID
    assert config["protocol"]["methods"] == [
        "h0_current_geometry",
        "h1_frozen_predicted_geometry",
        "h2_retrained_predicted_geometry",
    ]
    assert len(config["protocol"]["scenarios"]) == 5
    assert config["protocol"]["predictor_future_state_access"] is False
    assert config["protocol"]["h1_updates_ppo"] is False
    assert config["protocol"]["h2_updates_only_high_level_ppo"] is True

    h0 = method_config(config, "h0_current_geometry")
    h1 = method_config(config, "h1_frozen_predicted_geometry")
    h2 = method_config(config, "h2_retrained_predicted_geometry")
    assert h0["trajectory_prediction"]["integration"]["replace_base_geometry"] is False
    assert h1["trajectory_prediction"]["integration"]["replace_base_geometry"] is True
    assert h2["trajectory_prediction"]["integration"]["replace_base_geometry"] is True


def test_preflight_binds_source_contract_to_current_code():
    config = load_resolved_config(CONFIG)
    errors, source_files, checks = audit_source_contract(config)
    assert errors == []
    assert all(receipt["exists"] for receipt in source_files.values())
    assert checks["predicted_geometry_does_not_read_target_future_velocity"] is True
    assert checks["predicted_geometry_does_not_reference_oracle_anchor"] is True
    assert checks["h0_h1_h2_shared_execution_sections"] is True


def test_runner_is_locked_without_execute(capsys):
    assert main(["--config", str(CONFIG)]) == 0
    captured = capsys.readouterr()
    assert "LOCKED" in captured.out


def test_pilot_never_self_authorizes_formal_execution(tmp_path):
    root_manifest = {"source_id": SOURCE_ID, "runs": []}
    _write_pilot_outputs(tmp_path, root_manifest, None)
    status = json.loads((tmp_path / "pilot_status.json").read_text(encoding="utf-8"))
    assert status["formal_authorized"] is False
    assert status["status"] == "pending_independent_verification"


def test_formal_runner_requires_independent_pilot_verification(tmp_path):
    with pytest.raises(SystemExit, match="pilot verification"):
        main([
            "--execute",
            "--phase",
            "formal",
            "--output-dir",
            str(tmp_path / "formal"),
            "--h2-checkpoint",
            str(tmp_path / "h2.pt"),
        ])


def test_runner_refuses_to_overwrite_existing_output(tmp_path):
    output_root = tmp_path / "existing"
    output_root.mkdir()
    (output_root / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _assert_new_output_root(output_root)


def test_wilson_interval_and_paired_delta_are_deterministic():
    low, high = wilson_interval(3, 4)
    assert 0.3 < low < 0.5
    assert 0.7 < high < 1.0
    base = [{"opponent": "expert", "scenario": "head_on", "evaluation_seed": 1, "episode_seed": 2, "return": 1.0, "final_range_m": 10.0, "final_ata_deg": 20.0, "is_success": False}]
    candidate = [{"opponent": "expert", "scenario": "head_on", "evaluation_seed": 1, "episode_seed": 2, "return": 2.0, "final_range_m": 8.0, "final_ata_deg": 18.0, "is_success": True}]
    rows = paired_deltas({"h0_current_geometry": base, "h1_frozen_predicted_geometry": candidate})
    assert len(rows) == 1
    assert rows[0]["delta_return"] == 1.0
    assert rows[0]["delta_final_range_m"] == -2.0
    assert rows[0]["success_delta"] == 1


def test_paired_deltas_do_not_collapse_training_seed_replicates():
    base = [
        {
            "opponent": "expert",
            "scenario": "head_on",
            "training_seed": seed,
            "evaluation_seed": 1,
            "episode_seed": 2,
            "return": 1.0,
            "final_range_m": 10.0,
            "final_ata_deg": 20.0,
            "is_success": False,
        }
        for seed in (42, 43)
    ]
    candidate = [
        {
            "opponent": "expert",
            "scenario": "head_on",
            "training_seed": seed,
            "evaluation_seed": 1,
            "episode_seed": 2,
            "return": 2.0,
            "final_range_m": 8.0,
            "final_ata_deg": 18.0,
            "is_success": True,
        }
        for seed in (42, 43)
    ]

    rows = paired_deltas({
        "h0_current_geometry": base,
        "h1_frozen_predicted_geometry": candidate,
    })

    assert {row["training_seed"] for row in rows} == {42, 43}
    assert len(rows) == 2


def test_analyzer_groups_completed_episode_records():
    rows = [
        {"method": "h0_current_geometry", "opponent": "expert", "scenario": "head_on", "return": 1.0, "is_success": True, "reason": "success"},
        {"method": "h0_current_geometry", "opponent": "expert", "scenario": "head_on", "return": 0.0, "is_success": False, "reason": "timeout"},
    ]
    result = analyze(rows)
    assert result["source_id"] == SOURCE_ID
    assert result["episode_count"] == 2
    assert result["summaries"][0]["episodes"] == 2
    assert result["summaries"][0]["terminal_counts"] == {"success": 1, "timeout": 1}


def test_aggregate_prefers_predicted_geometry_fallback_rate():
    result = aggregate_episodes([
        {
            "is_success": False,
            "reason": "timeout",
            "return": 0.0,
            "predicted_geometry_fallback_rate": 0.75,
            "prediction_fallback_rate": 0.0,
        }
    ])
    assert result["mean_prediction_fallback_rate"] == pytest.approx(0.75)


def test_analysis_group_tables_retain_group_keys(tmp_path):
    from scripts.analyze_prediction_highlevel_observation_v1 import _write_required_artifacts

    rows = [{
        "method": "h0_current_geometry",
        "training_seed": 42,
        "evaluation_seed": 7,
        "opponent": "expert",
        "scenario": "head_on",
        "return": 1.0,
        "is_success": True,
        "reason": "success",
    }]
    result = analyze(rows)
    _write_required_artifacts(tmp_path, result, rows)

    per_seed = (tmp_path / "metrics_per_seed.csv").read_text(encoding="utf-8")
    by_opponent = (tmp_path / "metrics_by_opponent.csv").read_text(encoding="utf-8")
    by_attitude = (tmp_path / "metrics_by_attitude.csv").read_text(encoding="utf-8")
    assert "h0_current_geometry" in per_seed
    assert "expert" in by_opponent
    assert "head_on" in by_attitude


def test_analyzer_preserves_formal_execution_manifest(tmp_path):
    root = tmp_path / "formal"
    source_run = root / "h0_current_geometry" / "seed42" / "expert"
    source_run.mkdir(parents=True)
    (source_run / "metrics.json").write_text(
        json.dumps({
            "raw_episodes": [{
                "method": "h0_current_geometry",
                "opponent": "expert",
                "scenario": "head_on",
                "return": 0.0,
                "is_success": False,
                "reason": "timeout",
            }]
        }),
        encoding="utf-8",
    )
    original = {"source_id": SOURCE_ID, "execution": "immutable"}
    (root / "formal_manifest.json").write_text(
        json.dumps(original), encoding="utf-8"
    )

    from scripts.analyze_prediction_highlevel_observation_v1 import _write_required_artifacts

    rows = [{
        "method": "h0_current_geometry",
        "opponent": "expert",
        "scenario": "head_on",
        "return": 0.0,
        "is_success": False,
        "reason": "timeout",
    }]
    result = analyze(rows)
    _write_required_artifacts(root, result, rows)

    assert json.loads((root / "formal_manifest.json").read_text(encoding="utf-8")) == original
    analysis_manifest = root / "formal_analysis_manifest.json"
    assert analysis_manifest.is_file()
    assert json.loads(analysis_manifest.read_text(encoding="utf-8"))["execution_manifest_preserved"] is True
    assert (root / "metrics_per_seed.csv").is_file()


def test_verifier_rejects_missing_artifacts(tmp_path):
    result = verify(tmp_path)
    assert result["ok"] is False
    assert "no method/opponent run artifacts found" in result["errors"]
