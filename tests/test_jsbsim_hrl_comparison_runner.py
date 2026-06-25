"""Smoke tests for the JSBSim HRL/VPP comparison runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"


def _run_dry_run(tmp_path: Path, *extra_args: str) -> Path:
    run_id = "test_hrl_compare"
    output_root = tmp_path / "outputs"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--backend",
        "simple",
        "--dry-run",
        "--allow-missing-checkpoints",
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return output_root / run_id


def test_main_preset_dry_run_writes_manifest_contract_and_snapshots(tmp_path):
    run_dir = _run_dry_run(tmp_path)

    manifest_path = run_dir / "run_manifest.json"
    contract_path = run_dir / "artifact_contract.json"
    assert manifest_path.exists()
    assert contract_path.exists()
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "dry_run_checks.json").exists()
    assert (run_dir / "design_notes.json").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["paper_safe"] is False
    assert manifest["extra"]["artifact_validation"]["valid"] is True
    assert manifest["extra"]["methods"] == [
        "prediction_vpp",
        "no_prediction_vpp",
        "end_to_end_rl",
    ]
    assert manifest["extra"]["tasks"] == [
        "head_on",
        "crossing_feasible",
        "break_turn",
        "sustained_turn",
    ]
    assert "artifact_contract.json" in contract["required_files"]
    assert "design_notes.json" in contract["required_files"]

    snapshots = manifest["extra"]["config_snapshots"]
    assert len(snapshots) == 12
    pred_head_on = yaml.safe_load(
        Path(snapshots["prediction_vpp/head_on"]["path"]).read_text(encoding="utf-8")
    )
    assert pred_head_on["backend"] == "simple"
    assert pred_head_on["trajectory_prediction"]["enabled"] is True
    assert pred_head_on["virtual_point"]["anchor_mode"] == "predicted_target"
    assert pred_head_on["provenance"]["config_overrides"]


def test_ablation_noisy_prediction_records_noise_override(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--preset",
        "ablation",
        "--methods",
        "noisy_prediction_vpp",
        "--tasks",
        "head_on",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["noisy_prediction_vpp/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    integration = cfg["trajectory_prediction"]["integration"]
    assert integration["prediction_noise_std_m"] == 120.0
    overrides = cfg["provenance"]["config_overrides"]
    assert any(
        item["key"] == "trajectory_prediction.integration.prediction_noise_std_m"
        and item["new_value"] == 120.0
        for item in overrides
    )


def test_jsbsim_root_defaults_to_repo_root_when_backend_is_jsbsim(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--backend",
        "jsbsim",
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "head_on",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["jsbsim_root"] == str(REPO_ROOT)
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["env"]["legacy_project_root"] == str(REPO_ROOT)


def test_jsbsim_root_cli_is_recorded_as_config_override(tmp_path):
    external_root = "E:\\CloseAirCombat_control"
    run_dir = _run_dry_run(
        tmp_path,
        "--backend",
        "jsbsim",
        "--jsbsim-root",
        external_root,
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "head_on",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["jsbsim_root"] == external_root
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["env"]["legacy_project_root"] == external_root
    assert any(
        item["key"] == "env.legacy_project_root"
        and item["new_value"] == external_root
        for item in cfg["provenance"]["config_overrides"]
    )


def test_opponent_stage_dry_run_applies_attack_zone_parameters(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "head_on",
        "--opponent-stage",
        "expert",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert cfg["attack_zone"]["enabled"] is True
    assert cfg["attack_zone"]["damage_per_step"] == 0.5
    assert cfg["attack_zone"]["close_range_max_km"] == 3.0
    overrides = cfg["provenance"]["config_overrides"]
    assert any(item["key"] == "attack_zone" for item in overrides)
    assert any(
        item["key"] == "attack_zone.enabled" and item["new_value"] is True
        for item in overrides
    )


def test_single_episode_smoke_writes_tracking_metrics_and_observation_audit(tmp_path):
    run_id = "test_hrl_smoke"
    output_root = tmp_path / "outputs"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--backend",
        "simple",
        "--run-status",
        "smoke",
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "head_on",
        "--seeds",
        "0",
        "--n-episodes",
        "1",
        "--no-trajectory",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    run_dir = output_root / run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["artifact_validation"]["valid"] is True
    assert (run_dir / "observation_audit.json").exists()
    aggregate = json.loads(
        (run_dir / "aggregate" / "method_task_summary.json").read_text(encoding="utf-8")
    )
    assert aggregate["rows"]
    row = aggregate["rows"][0]
    for field in (
        "damage_exchange_rate",
        "effective_engagement_rate",
        "damaging_win_rate",
        "mean_damage_dealt",
        "mean_damage_taken",
    ):
        assert field in row

    header = (run_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    for field in (
        "final_range_m",
        "mean_range_m",
        "time_in_envelope_s",
        "envelope_fraction",
        "observation_dim",
        "checkpoint_obs_dim",
        "obs_dim_matches_checkpoint",
    ):
        assert field in header

    audit = json.loads((run_dir / "observation_audit.json").read_text(encoding="utf-8"))
    item = audit["audits"]["no_prediction_vpp/head_on"]
    assert item["schema_dim_matches_vector"] is True
    assert item["obs_dim_matches_checkpoint"] is True
