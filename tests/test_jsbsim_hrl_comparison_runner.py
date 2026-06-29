"""Smoke tests for the JSBSim HRL/VPP comparison runner."""

from __future__ import annotations

import json
import math
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


def test_prediction_jsbsim_compare_lookahead_override_preserves_fair_vpp_config(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--prediction-lookahead-time-s",
        "0.2",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["prediction_overrides"] == {"lookahead_time_s": 0.2}
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert cfg["trajectory_prediction"]["enabled"] is True
    assert cfg["trajectory_prediction"]["prediction"]["lookahead_time_s"] == 0.2
    assert cfg["virtual_point"]["anchor_mode"] == "predicted_target"
    assert cfg["virtual_point"]["dynamics_aware"] is True
    assert cfg["virtual_point"]["lookahead_steps"] == 5
    assert any(
        item["key"] == "trajectory_prediction.prediction.lookahead_time_s"
        and item["new_value"] == 0.2
        and item["source"] == "run_jsbsim_hrl_comparison.py:prediction_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offset_frame_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "head_on",
        "--vpp-offset-frame",
        "target_velocity",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {"offset_frame": "target_velocity"}
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offset_frame"] == "target_velocity"
    assert any(
        item["key"] == "virtual_point.offset_frame"
        and item["new_value"] == "target_velocity"
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offset_frame_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "head_on",
        "--vpp-offset-frame-by-task",
        "head_on=target_velocity,crossing_feasible=world_neu",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offset_frame_by_task": {
            "head_on": "target_velocity",
            "crossing_feasible": "world_neu",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offset_frame_by_task"] == {
        "head_on": "target_velocity",
        "crossing_feasible": "world_neu",
    }
    assert any(
        item["key"] == "virtual_point.offset_frame_by_task"
        and item["new_value"]
        == {
            "head_on": "target_velocity",
            "crossing_feasible": "world_neu",
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_lateral_latch_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-lateral-world-offset-latch-on-activation-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
        )
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_release_lateral_only_cli_overrides_are_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation-by-task",
        "head_on=true,crossing_feasible=false",
        "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        },
        "post_merge_offensive_anchor_blend_release_lateral_only_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        },
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        )
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
        )
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_release_lateral_only_hold_steps_cli_overrides_are_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-hold-steps-by-task",
        "head_on=60,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task": {
            "head_on": 60,
            "crossing_feasible": 0,
        },
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
    ] == {
        "head_on": 60,
        "crossing_feasible": 0,
    }
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
        )
        and item["new_value"]
        == {
            "head_on": 60,
            "crossing_feasible": 0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_release_recovery_cli_overrides_are_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m-by-task",
        "head_on=4500,crossing_feasible=0",
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-longitudinal-blend-by-task",
        "head_on=0.0,crossing_feasible=0.0",
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-lateral-blend-by-task",
        "head_on=0.25,crossing_feasible=0.0",
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max-by-task",
        "head_on=-4500",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task": {
            "head_on": 4500,
            "crossing_feasible": 0,
        },
        "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 0.0,
        },
        "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task": {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        },
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task": {
            "head_on": -4500.0,
        },
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
    ] == {
        "head_on": 4500,
        "crossing_feasible": 0,
    }
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
    ] == {
        "head_on": 0.0,
        "crossing_feasible": 0.0,
    }
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
    ] == {
        "head_on": 0.25,
        "crossing_feasible": 0.0,
    }
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
    ] == {
        "head_on": -4500.0,
    }
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
        )
        and item["new_value"]
        == {
            "head_on": 4500,
            "crossing_feasible": 0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
        )
        and item["new_value"]
        == {
            "head_on": 0.0,
            "crossing_feasible": 0.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
        )
        and item["new_value"]
        == {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
        )
        and item["new_value"]
        == {
            "head_on": -4500.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_release_forward_bias_clamp_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-m-min-by-task",
        "head_on=-4500",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task": {
            "head_on": -4500.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
    ] == {
        "head_on": -4500.0,
    }
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
        )
        and item["new_value"] == {"head_on": -4500.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_release_forward_bias_clamp_negative_lateral_altitude_gate_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-clamp-negative-lateral-below-altitude-m-by-task",
        "head_on=4500",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task": {
            "head_on": 4500.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
    ] == {
        "head_on": 4500.0,
    }
    assert any(
        item["key"]
        == (
            "virtual_point."
            "post_merge_offensive_anchor_blend_release_"
            "vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
        )
        and item["new_value"] == {"head_on": 4500.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_predicted_target_blend_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-predicted-target-blend-by-task",
        "head_on=0.35,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "predicted_target_blend_by_task": {
            "head_on": 0.35,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["predicted_target_blend_by_task"] == {
        "head_on": 0.35,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.predicted_target_blend_by_task"
        and item["new_value"]
        == {
            "head_on": 0.35,
            "crossing_feasible": 1.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_predicted_target_forward_scale_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-predicted-target-forward-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "predicted_target_forward_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["predicted_target_forward_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.predicted_target_forward_scale_by_task"
        and item["new_value"]
        == {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_longitudinal_scale_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-longitudinal-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "longitudinal_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["longitudinal_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.longitudinal_scale_by_task"
        and item["new_value"]
        == {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-blend-by-task",
        "head_on=0.5,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_blend_by_task": {
            "head_on": 0.5,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_predicted_target_blend_by_task"] == {
        "head_on": 0.5,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_predicted_target_blend_by_task"
        and item["new_value"]
        == {
            "head_on": 0.5,
            "crossing_feasible": 1.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_blend_release_on_attack_zone_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-blend-release-on-attack-zone-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_blend_release_on_attack_zone_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_predicted_target_blend_release_on_attack_zone_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_blend_release_on_attack_zone_by_task"
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-blend-release-requires-target-attack-zone-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_blend_release_below_altitude_m_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-blend-release-below-altitude-m-by-task",
        "head_on=2000,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_blend_release_below_altitude_m_by_task": {
            "head_on": 2000.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_predicted_target_blend_release_below_altitude_m_by_task"
    ] == {
        "head_on": 2000.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_blend_release_below_altitude_m_by_task"
        and item["new_value"]
        == {
            "head_on": 2000.0,
            "crossing_feasible": 0.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_blend_hold_steps_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-blend-hold-steps-by-task",
        "head_on=200,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_blend_hold_steps_by_task": {
            "head_on": 200,
            "crossing_feasible": 0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_predicted_target_blend_hold_steps_by_task"] == {
        "head_on": 200,
        "crossing_feasible": 0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_predicted_target_blend_hold_steps_by_task"
        and item["new_value"] == {"head_on": 200, "crossing_feasible": 0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_blend_release_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-blend-release-blend-by-task",
        "head_on=0.75,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_blend_release_blend_by_task": {
            "head_on": 0.75,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_predicted_target_blend_release_blend_by_task"] == {
        "head_on": 0.75,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_predicted_target_blend_release_blend_by_task"
        and item["new_value"] == {"head_on": 0.75, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_lateral_scale_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-lateral-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "lateral_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["lateral_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.lateral_scale_by_task"
        and item["new_value"]
        == {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-by-task",
        "head_on=offensive_position,crossing_feasible=predicted_target",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_by_task": {
            "head_on": "offensive_position",
            "crossing_feasible": "predicted_target",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_anchor_mode_by_task"] == {
        "head_on": "offensive_position",
        "crossing_feasible": "predicted_target",
    }
    assert any(
        item["key"] == "virtual_point.post_merge_anchor_mode_by_task"
        and item["new_value"]
        == {
            "head_on": "offensive_position",
            "crossing_feasible": "predicted_target",
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_requires_geometry_disadvantage_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-requires-geometry-disadvantage-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_requires_geometry_disadvantage_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_release_ego_only_streak_steps_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-release-ego-only-streak-steps-by-task",
        "head_on=1,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_release_ego_only_streak_steps_by_task": {
            "head_on": 1,
            "crossing_feasible": 0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_release_ego_only_streak_steps_by_task"
    ] == {
        "head_on": 1,
        "crossing_feasible": 0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_release_ego_only_streak_steps_by_task"
        and item["new_value"] == {"head_on": 1, "crossing_feasible": 0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_release_reset_on_streak_break_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-release-reset-on-streak-break-by-task",
        "head_on=false,crossing_feasible=true",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_release_reset_on_streak_break_by_task": {
            "head_on": False,
            "crossing_feasible": True,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_release_reset_on_streak_break_by_task"
    ] == {
        "head_on": False,
        "crossing_feasible": True,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_release_reset_on_streak_break_by_task"
        and item["new_value"]
        == {"head_on": False, "crossing_feasible": True}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_longitudinal_scale_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-longitudinal-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_longitudinal_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_anchor_mode_longitudinal_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_anchor_mode_longitudinal_scale_by_task"
        and item["new_value"] == {"head_on": 0.0, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_lateral_scale_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-lateral-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_lateral_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_anchor_mode_lateral_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_anchor_mode_lateral_scale_by_task"
        and item["new_value"] == {"head_on": 0.0, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_offensive_anchor_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-offensive-anchor-blend-by-task",
        "head_on=0.25,crossing_feasible=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_offensive_anchor_blend_by_task": {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_offensive_anchor_blend_by_task"
    ] == {
        "head_on": 0.25,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_offensive_anchor_blend_by_task"
        and item["new_value"] == {"head_on": 0.25, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-offensive-anchor-longitudinal-blend-by-task",
        "head_on=1.0,crossing_feasible=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task": {
            "head_on": 1.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"
    ] == {
        "head_on": 1.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"
        and item["new_value"] == {"head_on": 1.0, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-offensive-anchor-lateral-blend-by-task",
        "head_on=0.25,crossing_feasible=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task": {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"
    ] == {
        "head_on": 0.25,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"
        and item["new_value"] == {"head_on": 0.25, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_longitudinal_m_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-longitudinal-m-by-task",
        "head_on=700,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_longitudinal_m_by_task": {
            "head_on": 700.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_longitudinal_m_by_task"] == {
        "head_on": 700.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_longitudinal_m_by_task"
        and item["new_value"] == {"head_on": 700.0, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_frame_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-frame-by-task",
        "head_on=encounter,crossing_feasible=target_velocity",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_frame_by_task": {
            "head_on": "encounter",
            "crossing_feasible": "target_velocity",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_frame_by_task"] == {
        "head_on": "encounter",
        "crossing_feasible": "target_velocity",
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_frame_by_task"
        and item["new_value"]
        == {
            "head_on": "encounter",
            "crossing_feasible": "target_velocity",
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_frame_by_task_accepts_encounter_stable(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-frame-by-task",
        "head_on=encounter_stable,crossing_feasible=target_velocity",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_frame_by_task": {
            "head_on": "encounter_stable",
            "crossing_feasible": "target_velocity",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_frame_by_task"] == {
        "head_on": "encounter_stable",
        "crossing_feasible": "target_velocity",
    }


def test_vpp_offensive_anchor_encounter_stable_max_heading_delta_deg_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-encounter-stable-max-heading-delta-deg-by-task",
        "head_on=20,crossing_feasible=45",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task": {
            "head_on": 20.0,
            "crossing_feasible": 45.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
    ] == {
        "head_on": 20.0,
        "crossing_feasible": 45.0,
    }
    assert any(
        item["key"]
        == "virtual_point.offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
        and item["new_value"] == {"head_on": 20.0, "crossing_feasible": 45.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_lateral_m_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-lateral-m-by-task",
        "head_on=300,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_lateral_m_by_task": {
            "head_on": 300.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_lateral_m_by_task"] == {
        "head_on": 300.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_lateral_m_by_task"
        and item["new_value"] == {"head_on": 300.0, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_lateral_frame_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-lateral-frame-by-task",
        "head_on=encounter,crossing_feasible=target_velocity",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_lateral_frame_by_task": {
            "head_on": "encounter",
            "crossing_feasible": "target_velocity",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_lateral_frame_by_task"] == {
        "head_on": "encounter",
        "crossing_feasible": "target_velocity",
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_lateral_frame_by_task"
        and item["new_value"]
        == {"head_on": "encounter", "crossing_feasible": "target_velocity"}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_lateral_sign_mode_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-lateral-sign-mode-by-task",
        "head_on=fixed_positive,crossing_feasible=same_side",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_lateral_sign_mode_by_task": {
            "head_on": "fixed_positive",
            "crossing_feasible": "same_side",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_lateral_sign_mode_by_task"] == {
        "head_on": "fixed_positive",
        "crossing_feasible": "same_side",
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_lateral_sign_mode_by_task"
        and item["new_value"]
        == {"head_on": "fixed_positive", "crossing_feasible": "same_side"}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_vertical_m_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-vertical-m-by-task",
        "head_on=500,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_vertical_m_by_task": {
            "head_on": 500.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_vertical_m_by_task"] == {
        "head_on": 500.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_vertical_m_by_task"
        and item["new_value"] == {"head_on": 500.0, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_offensive_anchor_blend_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-offensive-anchor-blend-by-task",
        "head_on=0.25,crossing_feasible=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "offensive_anchor_blend_by_task": {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["offensive_anchor_blend_by_task"] == {
        "head_on": 0.25,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"] == "virtual_point.offensive_anchor_blend_by_task"
        and item["new_value"] == {"head_on": 0.25, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-by-task",
        "head_on=0.25,crossing_feasible=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_by_task": {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] == {
        "head_on": 0.25,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_offensive_anchor_blend_by_task"
        and item["new_value"] == {"head_on": 0.25, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_longitudinal_scale_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-longitudinal-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_longitudinal_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_longitudinal_scale_by_task"
    ] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_longitudinal_scale_by_task"
        and item["new_value"] == {"head_on": 0.0, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_lateral_scale_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-lateral-scale-by-task",
        "head_on=0.5,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_lateral_scale_by_task": {
            "head_on": 0.5,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_lateral_scale_by_task"
    ] == {
        "head_on": 0.5,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_lateral_scale_by_task"
        and item["new_value"] == {"head_on": 0.5, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
        "head_on=170.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task": {
            "head_on": 170.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"
    ] == {
        "head_on": 170.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"
        and item["new_value"] == {"head_on": 170.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_blend_release_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-blend-by-task",
        "head_on=0.0,crossing_feasible=0.25",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_blend_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 0.25,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_blend_by_task"
    ] == {
        "head_on": 0.0,
        "crossing_feasible": 0.25,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_blend_release_blend_by_task"
        and item["new_value"] == {"head_on": 0.0, "crossing_feasible": 0.25}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task",
        "head_on=5,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task": {
            "head_on": 5,
            "crossing_feasible": 0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
    ] == {
        "head_on": 5,
        "crossing_feasible": 0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        and item["new_value"] == {"head_on": 5, "crossing_feasible": 0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-reset-on-streak-break-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
        and item["new_value"] == {"head_on": True, "crossing_feasible": False}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task",
        "head_on=800,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task": {
            "head_on": 800.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
    ] == {
        "head_on": 800.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        and item["new_value"] == {"head_on": 800.0, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_anchor_mode_recovery_below_altitude_m_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-anchor-mode-recovery-below-altitude-m-by-task",
        "head_on=4500,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_anchor_mode_recovery_below_altitude_m_by_task": {
            "head_on": 4500.0,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "post_merge_anchor_mode_recovery_below_altitude_m_by_task"
    ] == {
        "head_on": 4500.0,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"]
        == "virtual_point.post_merge_anchor_mode_recovery_below_altitude_m_by_task"
        and item["new_value"] == {"head_on": 4500.0, "crossing_feasible": 0.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_forward_scale_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-forward-scale-by-task",
        "head_on=0.0,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_forward_scale_by_task": {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["post_merge_predicted_target_forward_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert any(
        item["key"] == "virtual_point.post_merge_predicted_target_forward_scale_by_task"
        and item["new_value"] == {"head_on": 0.0, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_forward_scale_hold_steps_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-forward-scale-hold-steps-by-task",
        "head_on=60,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_forward_scale_hold_steps_by_task": {
            "head_on": 60,
            "crossing_feasible": 0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert (
        cfg["virtual_point"][
            "post_merge_predicted_target_forward_scale_hold_steps_by_task"
        ]
        == {
            "head_on": 60,
            "crossing_feasible": 0,
        }
    )
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_forward_scale_hold_steps_by_task"
        and item["new_value"] == {"head_on": 60, "crossing_feasible": 0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-forward-scale-release-ego-only-streak-steps-by-task",
        "head_on=35,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task": {
            "head_on": 35,
            "crossing_feasible": 0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert (
        cfg["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        ]
        == {
            "head_on": 35,
            "crossing_feasible": 0,
        }
    )
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        and item["new_value"] == {"head_on": 35, "crossing_feasible": 0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_forward_scale_release_scale_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-forward-scale-release-scale-by-task",
        "head_on=0.5,crossing_feasible=1.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_forward_scale_release_scale_by_task": {
            "head_on": 0.5,
            "crossing_feasible": 1.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert (
        cfg["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_scale_by_task"
        ]
        == {
            "head_on": 0.5,
            "crossing_feasible": 1.0,
        }
    )
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_forward_scale_release_scale_by_task"
        and item["new_value"] == {"head_on": 0.5, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_forward_scale_release_scale_by_task_cli_override_merges_existing_task_mapping(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--config",
        "config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_heldout.yaml",
        "--methods",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-forward-scale-release-scale-by-task",
        "head_on=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_forward_scale_release_scale_by_task": {
            "head_on": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert (
        cfg["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_scale_by_task"
        ]
        == {
            "head_on": 0.0,
            "crossing_feasible": 1.0,
        }
    )
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_forward_scale_release_scale_by_task"
        and item["new_value"] == {"head_on": 0.0, "crossing_feasible": 1.0}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-post-merge-predicted-target-forward-scale-release-reset-on-streak-break-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert (
        cfg["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
        ]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
    )
    assert any(
        item["key"]
        == "virtual_point.post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
        and item["new_value"] == {"head_on": True, "crossing_feasible": False}
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_mode_by_task_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-mode-by-task",
        "head_on=current_target,crossing_feasible=predicted_target",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_mode_by_task": {
            "head_on": "current_target",
            "crossing_feasible": "predicted_target",
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_mode_by_task"] == {
        "head_on": "current_target",
        "crossing_feasible": "predicted_target",
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_mode_by_task"
        and item["new_value"]
        == {
            "head_on": "current_target",
            "crossing_feasible": "predicted_target",
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_trigger_range_m_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-trigger-range-m-by-task",
        "head_on=150,crossing_feasible=1000",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_trigger_range_m_by_task": {
            "head_on": 150.0,
            "crossing_feasible": 1000.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_trigger_range_m_by_task"] == {
        "head_on": 150.0,
        "crossing_feasible": 1000.0,
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_trigger_range_m_by_task"
        and item["new_value"]
        == {
            "head_on": 150.0,
            "crossing_feasible": 1000.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_alignment_angle_deg_max_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-alignment-angle-deg-max-by-task",
        "head_on=10,crossing_feasible=180",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_alignment_angle_deg_max_by_task": {
            "head_on": 10.0,
            "crossing_feasible": 180.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_alignment_angle_deg_max_by_task"] == {
        "head_on": 10.0,
        "crossing_feasible": 180.0,
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_alignment_angle_deg_max_by_task"
        and item["new_value"]
        == {
            "head_on": 10.0,
            "crossing_feasible": 180.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_release_on_post_merge_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-release-on-post-merge-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_release_on_post_merge_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_release_on_post_merge_by_task"] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_release_on_post_merge_by_task"
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_release_alignment_angle_deg_max_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-release-alignment-angle-deg-max-by-task",
        "head_on=120,crossing_feasible=180",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_release_alignment_angle_deg_max_by_task": {
            "head_on": 120.0,
            "crossing_feasible": 180.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"][
        "close_range_anchor_release_alignment_angle_deg_max_by_task"
    ] == {
        "head_on": 120.0,
        "crossing_feasible": 180.0,
    }
    assert any(
        item["key"]
        == "virtual_point.close_range_anchor_release_alignment_angle_deg_max_by_task"
        and item["new_value"]
        == {
            "head_on": 120.0,
            "crossing_feasible": 180.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_requires_first_pass_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-requires-first-pass-by-task",
        "head_on=true,crossing_feasible=false",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_requires_first_pass_by_task": {
            "head_on": True,
            "crossing_feasible": False,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_requires_first_pass_by_task"] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_requires_first_pass_by_task"
        and item["new_value"]
        == {
            "head_on": True,
            "crossing_feasible": False,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_offensive_anchor_blend_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-offensive-anchor-blend-by-task",
        "head_on=0.25,crossing_feasible=0.0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_offensive_anchor_blend_by_task": {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_offensive_anchor_blend_by_task"] == {
        "head_on": 0.25,
        "crossing_feasible": 0.0,
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_offensive_anchor_blend_by_task"
        and item["new_value"]
        == {
            "head_on": 0.25,
            "crossing_feasible": 0.0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_vpp_close_range_anchor_post_merge_hold_steps_by_task_cli_override_is_recorded(
    tmp_path,
):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "prediction_vpp_jsbsim_compare",
        "--tasks",
        "head_on",
        "--vpp-close-range-anchor-post-merge-hold-steps-by-task",
        "head_on=1,crossing_feasible=0",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vpp_overrides"] == {
        "close_range_anchor_post_merge_hold_steps_by_task": {
            "head_on": 1,
            "crossing_feasible": 0,
        }
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["prediction_vpp_jsbsim_compare/head_on"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert cfg["virtual_point"]["close_range_anchor_post_merge_hold_steps_by_task"] == {
        "head_on": 1,
        "crossing_feasible": 0,
    }
    assert any(
        item["key"] == "virtual_point.close_range_anchor_post_merge_hold_steps_by_task"
        and item["new_value"]
        == {
            "head_on": 1,
            "crossing_feasible": 0,
        }
        and item["source"] == "run_jsbsim_hrl_comparison.py:vpp_cli"
        for item in cfg["provenance"]["config_overrides"]
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


def test_crossing_attack_zone_aoa_cli_override_is_recorded(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "crossing_feasible",
        "--opponent-stage",
        "expert",
        "--attack-zone-close-range-max-aoa-deg",
        "60",
        "--attack-zone-damage-per-step",
        "0.75",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["attack_zone_overrides"] == {
        "damage_per_step": 0.75,
        "close_range_max_aoa_deg": 60.0,
    }
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/crossing_feasible"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert cfg["attack_zone"]["enabled"] is True
    assert cfg["attack_zone"]["damage_per_step"] == 0.75
    assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    overrides = cfg["provenance"]["config_overrides"]
    assert any(
        item["key"] == "attack_zone.close_range_max_aoa_deg"
        and item["new_value"] == 60.0
        and item["source"] == "run_jsbsim_hrl_comparison.py:attack_zone_cli"
        for item in overrides
    )
    assert any(
        item["key"] == "attack_zone.damage_per_step"
        and item["new_value"] == 0.75
        for item in overrides
    )


def test_reset075_heldout_config_dry_run_selects_formal_scope_and_snapshots(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--config",
        "config/experiment/jsbsim_hrl_reset075_heldout.yaml",
        "--opponent-stage",
        "expert",
        "--attack-zone-close-range-max-aoa-deg",
        "60",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["methods"] == [
        "no_prediction_vpp",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075",
    ]
    assert manifest["extra"]["tasks"] == ["head_on", "crossing_feasible"]
    assert manifest["extra"]["attack_zone_overrides"] == {"close_range_max_aoa_deg": 60.0}

    snapshot_path = Path(
        manifest["extra"]["config_snapshots"][
            "prediction_vpp_jsbsim_compare_long_lh1p0_reset075/head_on"
        ]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert cfg["virtual_point"]["offset_frame_by_task"] == {
        "head_on": "target_velocity",
        "crossing_feasible": "world_neu",
    }
    assert cfg["virtual_point"]["post_merge_predicted_target_forward_scale_release_scale_by_task"] == {
        "head_on": 0.75,
        "crossing_feasible": 1.0,
    }
    assert cfg["virtual_point"][
        "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
    ] == {
        "head_on": True,
        "crossing_feasible": False,
    }
    assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert any(
        item["key"] == "attack_zone.close_range_max_aoa_deg"
        and item["new_value"] == 60.0
        and item["source"] == "run_jsbsim_hrl_comparison.py:attack_zone_cli"
        for item in cfg["provenance"]["config_overrides"]
    )


def test_sustained_turn_uses_formal_bank80_task_config(tmp_path):
    run_dir = _run_dry_run(
        tmp_path,
        "--methods",
        "no_prediction_vpp",
        "--tasks",
        "sustained_turn",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    snapshot_path = Path(
        manifest["extra"]["config_snapshots"]["no_prediction_vpp/sustained_turn"]["path"]
    )
    cfg = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    llc = cfg["low_level_controller"]
    assert abs(llc["max_bank_rad"] - 1.3962634015954636) < 1e-12
    assert llc["altitude_hold_gain"] == 0.03
    assert llc["bank_protection_nz_increment"] == 0.7
    assert llc["bank_protection_nz_increment_max"] == 0.5
    assert cfg["task"]["sustained_turn"]["supervisor"]["enabled"] is True
    assert any(
        item["key"] == "low_level_controller"
        and item["source"] == "run_jsbsim_hrl_comparison.py:sustained_turn_eval_config"
        for item in cfg["provenance"]["config_overrides"]
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
        "--opponent-stage",
        "expert",
        "--attack-zone-close-range-max-aoa-deg",
        "60",
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
    combat_diag_path = run_dir / "aggregate" / "combat_geometry_diagnostics.json"
    assert combat_diag_path.exists()
    combat_diag = json.loads(combat_diag_path.read_text(encoding="utf-8"))
    assert set(
        [
            "first_ego_attack_time_s",
            "first_target_attack_time_s",
            "post_merge_attack_zone_advantage_s",
            "pre_merge_vp_forward_bias_m",
            "vp_forward_bias_m",
            "vp_lateral_bias_m",
            "merge_min_range_m",
            "damage_margin",
        ]
    ).issubset(set(combat_diag["metrics"]))
    aggregate = json.loads(
        (run_dir / "aggregate" / "method_task_summary.json").read_text(encoding="utf-8")
    )
    assert aggregate["rows"]
    row = aggregate["rows"][0]
    for field in (
        "group",
        "run_status",
        "mode",
        "attack_zone_enabled",
        "combat_initial_hp",
        "damage_per_step",
        "close_range_max_km",
        "close_range_max_aoa_deg",
        "damage_exchange_rate",
        "effective_engagement_rate",
        "damaging_win_rate",
        "mean_damage_dealt",
        "mean_damage_taken",
        "ego_crashes",
        "target_crash_or_oob",
        "crashes",
        "timeouts",
    ):
        assert field in row
    assert row["run_status"] == "smoke"
    assert row["mode"] == "smoke"
    assert row["attack_zone_enabled"] is True
    assert row["combat_initial_hp"] == 100.0
    assert row["damage_per_step"] == 0.5
    assert row["close_range_max_km"] == 3.0
    assert row["close_range_max_aoa_deg"] == 60.0
    assert row["group"] == "head_on::no_prediction_vpp::expert::d0.5::r3.0::a60.0"

    header = (run_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    for field in (
        "run_status",
        "mode",
        "attack_zone_enabled",
        "combat_initial_hp",
        "damage_per_step",
        "close_range_max_km",
        "close_range_max_aoa_deg",
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


def test_crash_split_distinguishes_target_crash_from_ego_crash():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_jsbsim_hrl_comparison import (  # noqa: WPS433
        _record_has_ego_crash,
        _record_has_target_crash_or_oob,
    )

    target_crash = {
        "termination_reason": "target_crash_or_out_of_bounds",
        "combat_reason": "target_crash_or_out_of_bounds",
    }
    ego_crash = {
        "termination_reason": "out_of_bounds",
        "combat_reason": "out_of_bounds",
    }

    assert _record_has_target_crash_or_oob(target_crash) is True
    assert _record_has_ego_crash(target_crash) is False
    assert _record_has_target_crash_or_oob(ego_crash) is False
    assert _record_has_ego_crash(ego_crash) is True


def test_combat_geometry_summary_builder_outputs_required_metrics(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_jsbsim_hrl_comparison import (  # noqa: WPS433
        _write_combat_geometry_diagnostics,
        summarize_episode_combat_geometry,
    )

    record = {
        "controller": "prediction_vpp",
        "task": "head_on",
        "opponent_stage": "expert",
        "run_status": "smoke",
        "mode": "smoke",
        "seed": 10,
        "episode": 0,
        "scenario": "head_on_minimal",
        "steps": 3,
        "total_time_s": 0.6,
        "combat_initial_hp": 100.0,
        "ego_hp": 98.0,
        "target_hp": 95.0,
        "trajectory": [
            {
                "time_s": 0.2,
                "range_m": 1200.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": True,
                "post_merge": False,
                "vp_forward_bias_m": 20.0,
                "vp_lateral_bias_m": -5.0,
                "offset_frame": "world_neu",
                "configured_offset_frame": "target_velocity",
                "virtual_point_source": "direct_track",
                "anchor_mode": "predicted_target",
                "anchor_mode_requested": "predicted_target",
                "close_range_anchor_mode": "offensive_position",
                "close_range_anchor_trigger_range_m": 1000.0,
                "close_range_anchor_release_on_post_merge": True,
                "close_range_anchor_requires_first_pass": True,
                "close_range_anchor_offensive_anchor_blend": 0.25,
                "close_range_anchor_offensive_anchor_blend_active": False,
                "close_range_anchor_release_alignment_angle_deg_max": 12.0,
                "close_range_anchor_post_merge_hold_steps": 20.0,
                "close_range_anchor_alignment_angle_deg_max": 10.0,
                "close_range_anchor_mode_active": False,
                "offensive_anchor_frame": "encounter",
                "offensive_anchor_lateral_frame": "target_velocity",
                "offensive_anchor_longitudinal_m": 800.0,
                "offensive_anchor_lateral_m": 300.0,
                "offensive_anchor_vertical_m": 500.0,
                "longitudinal_scale": 1.0,
                "lateral_scale": 1.0,
                "post_merge_offensive_anchor_longitudinal_scale": 0.75,
                "post_merge_offensive_anchor_lateral_scale": 0.25,
                "post_merge_predicted_target_forward_scale": 0.0,
                "post_merge_predicted_target_forward_scale_release_scale": 1.0,
                "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": 1.0,
                "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": True,
                "post_merge_predicted_target_forward_scale_hold_steps": 20.0,
                "post_merge_predicted_target_forward_scale_active": False,
                "post_merge_predicted_target_forward_scale_release_scale_active": False,
                "post_merge_predicted_target_forward_scale_released": False,
                "post_merge_anchor_mode": "offensive_position",
                "post_merge_anchor_mode_requires_geometry_disadvantage": True,
                "post_merge_anchor_mode_condition_met": False,
                "post_merge_anchor_mode_active": False,
                "post_merge_anchor_mode_release_ego_only_streak_steps": 1.0,
                "post_merge_anchor_mode_recovery_below_altitude_m": 4500.0,
                "post_merge_anchor_mode_recovery_active": False,
                "post_merge_anchor_mode_release_reset_on_streak_break": False,
                "post_merge_anchor_mode_ego_only_streak_steps": 0.0,
                "post_merge_anchor_mode_release_triggered": False,
                "post_merge_anchor_mode_release_reset_triggered": False,
                "post_merge_anchor_mode_released": False,
                "post_merge_offensive_anchor_condition_met": False,
                "post_merge_offensive_anchor_alignment_disadvantage": False,
                "post_merge_offensive_anchor_blend_active": False,
                "post_merge_offensive_anchor_blend_release_blend": 0.1,
                "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": 5.0,
                "post_merge_offensive_anchor_blend_release_reset_on_streak_break": True,
                "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": 800.0,
                "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": 20.0,
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": True,
                "post_merge_offensive_anchor_lateral_world_offset_latch_active": True,
                "post_merge_offensive_anchor_blend_release_lateral_only": True,
                "post_merge_offensive_anchor_blend_release_blend_active": False,
                "post_merge_offensive_anchor_blend_released": False,
                "post_merge_offensive_anchor_blend_release_direct_track_active": False,
                "post_merge_offensive_anchor_blend_release_lateral_only_active": False,
                "effective_guidance_mode": "proportional_navigation",
                "mode_switch_effective": True,
                "direct_track_mode_effective": True,
            },
            {
                "time_s": 0.4,
                "range_m": 800.0,
                "ego_in_attack_zone": True,
                "target_in_attack_zone": True,
                "post_merge": True,
                "vp_forward_bias_m": 40.0,
                "vp_lateral_bias_m": -15.0,
                "offset_frame": "world_neu",
                "configured_offset_frame": "target_velocity",
                "virtual_point_source": "direct_track",
                "anchor_mode": "offensive_position",
                "anchor_mode_requested": "predicted_target",
                "close_range_anchor_mode": "offensive_position",
                "close_range_anchor_trigger_range_m": 1000.0,
                "close_range_anchor_release_on_post_merge": True,
                "close_range_anchor_requires_first_pass": True,
                "close_range_anchor_offensive_anchor_blend": 0.25,
                "close_range_anchor_offensive_anchor_blend_active": True,
                "close_range_anchor_release_alignment_angle_deg_max": 12.0,
                "close_range_anchor_post_merge_hold_steps": 20.0,
                "close_range_anchor_alignment_angle_deg_max": 10.0,
                "close_range_anchor_mode_active": True,
                "offensive_anchor_frame": "encounter",
                "offensive_anchor_lateral_frame": "target_velocity",
                "offensive_anchor_longitudinal_m": 800.0,
                "offensive_anchor_lateral_m": 300.0,
                "offensive_anchor_vertical_m": 500.0,
                "longitudinal_scale": 0.0,
                "lateral_scale": 0.5,
                "post_merge_offensive_anchor_longitudinal_scale": 0.75,
                "post_merge_offensive_anchor_lateral_scale": 0.25,
                "post_merge_predicted_target_forward_scale": 0.0,
                "post_merge_predicted_target_forward_scale_release_scale": 1.0,
                "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": 1.0,
                "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": True,
                "post_merge_predicted_target_forward_scale_hold_steps": 20.0,
                "post_merge_predicted_target_forward_scale_active": True,
                "post_merge_predicted_target_forward_scale_release_scale_active": False,
                "post_merge_predicted_target_forward_scale_released": False,
                "post_merge_anchor_mode": "offensive_position",
                "post_merge_anchor_mode_requires_geometry_disadvantage": True,
                "post_merge_anchor_mode_condition_met": True,
                "post_merge_anchor_mode_active": True,
                "post_merge_anchor_mode_release_ego_only_streak_steps": 1.0,
                "post_merge_anchor_mode_recovery_below_altitude_m": 4500.0,
                "post_merge_anchor_mode_recovery_active": True,
                "post_merge_anchor_mode_release_reset_on_streak_break": False,
                "post_merge_anchor_mode_ego_only_streak_steps": 1.0,
                "post_merge_anchor_mode_release_triggered": True,
                "post_merge_anchor_mode_release_reset_triggered": False,
                "post_merge_anchor_mode_released": True,
                "post_merge_offensive_anchor_condition_met": True,
                "post_merge_offensive_anchor_alignment_disadvantage": True,
                "post_merge_offensive_anchor_blend_active": True,
                "post_merge_offensive_anchor_blend_release_blend": 0.0,
                "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": 5.0,
                "post_merge_offensive_anchor_blend_release_reset_on_streak_break": True,
                "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": 800.0,
                "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": 20.0,
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": True,
                "post_merge_offensive_anchor_lateral_world_offset_latch_active": True,
                "post_merge_offensive_anchor_blend_release_lateral_only": True,
                "post_merge_offensive_anchor_blend_release_blend_active": False,
                "post_merge_offensive_anchor_blend_released": False,
                "post_merge_offensive_anchor_blend_release_direct_track_active": False,
                "post_merge_offensive_anchor_blend_release_lateral_only_active": False,
                "effective_guidance_mode": "proportional_navigation",
                "mode_switch_effective": True,
                "direct_track_mode_effective": True,
            },
            {
                "time_s": 0.6,
                "range_m": 900.0,
                "ego_in_attack_zone": True,
                "target_in_attack_zone": False,
                "post_merge": True,
                "vp_forward_bias_m": 60.0,
                "vp_lateral_bias_m": -25.0,
                "offset_frame": "world_neu",
                "configured_offset_frame": "target_velocity",
                "virtual_point_source": "direct_track",
                "anchor_mode": "predicted_target",
                "anchor_mode_requested": "predicted_target",
                "close_range_anchor_mode": "offensive_position",
                "close_range_anchor_trigger_range_m": 1000.0,
                "close_range_anchor_release_on_post_merge": True,
                "close_range_anchor_requires_first_pass": True,
                "close_range_anchor_offensive_anchor_blend": 0.25,
                "close_range_anchor_offensive_anchor_blend_active": False,
                "close_range_anchor_release_alignment_angle_deg_max": 12.0,
                "close_range_anchor_post_merge_hold_steps": 20.0,
                "close_range_anchor_alignment_angle_deg_max": 10.0,
                "close_range_anchor_mode_active": False,
                "offensive_anchor_frame": "encounter",
                "offensive_anchor_lateral_frame": "target_velocity",
                "offensive_anchor_longitudinal_m": 800.0,
                "offensive_anchor_lateral_m": 300.0,
                "offensive_anchor_vertical_m": 500.0,
                "longitudinal_scale": 1.0,
                "lateral_scale": 1.0,
                "post_merge_offensive_anchor_longitudinal_scale": 0.75,
                "post_merge_offensive_anchor_lateral_scale": 0.25,
                "post_merge_predicted_target_forward_scale": 0.0,
                "post_merge_predicted_target_forward_scale_release_scale": 1.0,
                "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": 1.0,
                "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": True,
                "post_merge_predicted_target_forward_scale_hold_steps": 20.0,
                "post_merge_predicted_target_forward_scale_active": False,
                "post_merge_predicted_target_forward_scale_release_scale_active": True,
                "post_merge_predicted_target_forward_scale_released": True,
                "post_merge_anchor_mode": "offensive_position",
                "post_merge_anchor_mode_requires_geometry_disadvantage": True,
                "post_merge_anchor_mode_condition_met": True,
                "post_merge_anchor_mode_active": False,
                "post_merge_anchor_mode_release_ego_only_streak_steps": 1.0,
                "post_merge_anchor_mode_recovery_below_altitude_m": 4500.0,
                "post_merge_anchor_mode_recovery_active": False,
                "post_merge_anchor_mode_release_reset_on_streak_break": False,
                "post_merge_anchor_mode_ego_only_streak_steps": 1.0,
                "post_merge_anchor_mode_release_triggered": False,
                "post_merge_anchor_mode_release_reset_triggered": False,
                "post_merge_anchor_mode_released": True,
                "post_merge_offensive_anchor_condition_met": True,
                "post_merge_offensive_anchor_alignment_disadvantage": False,
                "post_merge_offensive_anchor_blend_active": False,
                "post_merge_offensive_anchor_blend_release_blend": 0.0,
                "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": 5.0,
                "post_merge_offensive_anchor_blend_release_reset_on_streak_break": True,
                "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": 800.0,
                "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": 20.0,
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": True,
                "post_merge_offensive_anchor_lateral_world_offset_latch_active": True,
                "post_merge_offensive_anchor_blend_release_lateral_only": True,
                "post_merge_offensive_anchor_blend_release_blend_active": True,
                "post_merge_offensive_anchor_blend_released": True,
                "post_merge_offensive_anchor_blend_release_direct_track_active": True,
                "post_merge_offensive_anchor_blend_release_lateral_only_active": True,
                "effective_guidance_mode": "proportional_navigation",
                "mode_switch_effective": True,
                "direct_track_mode_effective": True,
            },
        ],
    }

    diagnostics = summarize_episode_combat_geometry(record)
    assert diagnostics["first_ego_attack_time_s"] == 0.4
    assert diagnostics["first_target_attack_time_s"] == 0.2
    assert diagnostics["first_pass_step"] == 2.0
    assert math.isclose(diagnostics["post_merge_attack_zone_advantage_s"], 0.2)
    assert diagnostics["close_range_anchor_mode_active_fraction"] == 0.5
    assert diagnostics["close_range_anchor_mode_first_active_step"] == 2.0
    assert diagnostics["post_merge_anchor_mode_active_fraction"] == 0.5
    assert diagnostics["post_merge_anchor_mode_condition_met_fraction"] == 1.0
    assert diagnostics["post_merge_anchor_mode_first_active_step"] == 2.0
    assert diagnostics["post_merge_anchor_mode_recovery_active_fraction"] == 0.5
    assert diagnostics["post_merge_anchor_mode_recovery_first_step"] == 2.0
    assert diagnostics["post_merge_anchor_mode_released_fraction"] == 1.0
    assert diagnostics["post_merge_anchor_mode_release_first_step"] == 2.0
    assert diagnostics["post_merge_offensive_anchor_condition_met_fraction"] == 1.0
    assert diagnostics["post_merge_offensive_anchor_alignment_disadvantage_fraction"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_blend_active_fraction"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_first_active_step"] == 2.0
    assert diagnostics["post_merge_offensive_anchor_blend_release_blend_active_fraction"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_blend_released_fraction"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_blend_release_first_step"] == 3.0
    assert diagnostics["post_merge_offensive_anchor_blend_release_direct_track_active_fraction"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_blend_release_direct_track_first_step"] == 3.0
    assert diagnostics["post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction"] == 1.0
    assert diagnostics["post_merge_offensive_anchor_lateral_world_offset_latch_first_active_step"] == 2.0
    assert diagnostics["post_merge_offensive_anchor_blend_release_lateral_only_active_fraction"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_blend_release_lateral_only_first_step"] == 3.0
    assert diagnostics["post_merge_predicted_target_forward_scale_active_fraction"] == 0.5
    assert diagnostics["post_merge_predicted_target_forward_scale_first_active_step"] == 2.0
    assert diagnostics["post_merge_predicted_target_forward_scale_release_scale_active_fraction"] == 0.5
    assert diagnostics["post_merge_predicted_target_forward_scale_released_fraction"] == 0.5
    assert diagnostics["post_merge_predicted_target_forward_scale_release_first_step"] == 3.0
    assert diagnostics["post_merge_offensive_anchor_active_longitudinal_scale"] == 0.0
    assert diagnostics["post_merge_offensive_anchor_active_lateral_scale"] == 0.5
    assert diagnostics["post_merge_active_vp_longitudinal_scale"] == 0.0
    assert diagnostics["post_merge_active_vp_lateral_scale"] == 0.5
    assert diagnostics["post_merge_offensive_anchor_active_override_longitudinal_scale"] == 0.75
    assert diagnostics["post_merge_offensive_anchor_active_override_lateral_scale"] == 0.25
    assert diagnostics["pre_merge_vp_forward_bias_m"] == 20.0
    assert diagnostics["post_merge_anchor_mode_active_vp_forward_bias_m"] == 40.0
    assert diagnostics["post_merge_anchor_mode_active_vp_lateral_bias_m"] == -15.0
    assert diagnostics["post_merge_offensive_anchor_active_vp_forward_bias_m"] == 40.0
    assert diagnostics["post_merge_offensive_anchor_active_vp_lateral_bias_m"] == -15.0
    assert diagnostics["post_merge_predicted_target_forward_scale_active_vp_forward_bias_m"] == 40.0
    assert diagnostics["post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m"] == -15.0
    assert diagnostics["vp_forward_bias_m"] == 40.0
    assert diagnostics["vp_lateral_bias_m"] == -15.0
    assert diagnostics["merge_min_range_m"] == 800.0
    assert diagnostics["damage_margin"] == 3.0

    path = _write_combat_geometry_diagnostics(tmp_path, [record])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["episode_rows"][0]["damage_margin"] == 3.0
    assert payload["episode_rows"][0]["offset_frame"] == "world_neu"
    assert payload["episode_rows"][0]["configured_offset_frame"] == "target_velocity"
    assert payload["episode_rows"][0]["virtual_point_source"] == "direct_track"
    assert payload["episode_rows"][0]["anchor_mode_requested"] == "predicted_target"
    assert payload["episode_rows"][0]["close_range_anchor_mode"] == "offensive_position"
    assert payload["episode_rows"][0]["close_range_anchor_trigger_range_m"] == 1000.0
    assert payload["episode_rows"][0]["close_range_anchor_release_on_post_merge"] is True
    assert payload["episode_rows"][0]["close_range_anchor_requires_first_pass"] is True
    assert payload["episode_rows"][0]["close_range_anchor_offensive_anchor_blend"] == 0.25
    assert payload["episode_rows"][0]["close_range_anchor_offensive_anchor_blend_active"] is False
    assert payload["episode_rows"][0]["close_range_anchor_post_merge_hold_steps"] == 20.0
    assert payload["episode_rows"][0]["close_range_anchor_alignment_angle_deg_max"] == 10.0
    assert (
        payload["episode_rows"][0]["close_range_anchor_release_alignment_angle_deg_max"]
        == 12.0
    )
    assert payload["episode_rows"][0]["offensive_anchor_frame"] == "encounter"
    assert payload["episode_rows"][0]["offensive_anchor_lateral_frame"] == "target_velocity"
    assert payload["episode_rows"][0]["offensive_anchor_longitudinal_m"] == 800.0
    assert payload["episode_rows"][0]["offensive_anchor_lateral_m"] == 300.0
    assert payload["episode_rows"][0]["offensive_anchor_vertical_m"] == 500.0
    assert (
        payload["episode_rows"][0][
            "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction"
        ]
        == 1.0
    )
    assert (
        payload["episode_rows"][0][
            "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction"
        ]
        == 0.5
    )
    assert payload["episode_rows"][0]["post_merge_predicted_target_forward_scale"] == 0.0
    assert payload["episode_rows"][0]["post_merge_predicted_target_forward_scale_release_scale"] == 1.0
    assert (
        payload["episode_rows"][0][
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
        ]
        == 1.0
    )
    assert (
        payload["episode_rows"][0][
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
        ]
        is True
    )
    assert payload["episode_rows"][0]["post_merge_predicted_target_forward_scale_hold_steps"] == 20.0
    assert payload["episode_rows"][0]["post_merge_anchor_mode"] == "offensive_position"
    assert (
        payload["episode_rows"][0]["post_merge_anchor_mode_requires_geometry_disadvantage"]
        is True
    )
    assert (
        payload["episode_rows"][0]["post_merge_anchor_mode_release_ego_only_streak_steps"]
        == 1.0
    )
    assert (
        payload["episode_rows"][0]["post_merge_anchor_mode_recovery_below_altitude_m"]
        == 4500.0
    )
    assert (
        payload["episode_rows"][0]["post_merge_anchor_mode_release_reset_on_streak_break"]
        is False
    )
    assert payload["episode_rows"][0]["post_merge_offensive_anchor_blend_release_blend"] == 0.1
    assert (
        payload["episode_rows"][0][
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
        ]
        == 5.0
    )
    assert (
        payload["episode_rows"][0][
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break"
        ]
        is True
    )
    assert (
        payload["episode_rows"][0][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
        ]
        == 800.0
    )
    assert (
        payload["episode_rows"][0][
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"
        ]
        == 20.0
    )
    assert payload["episode_rows"][0]["effective_guidance_mode"] == "proportional_navigation"
    assert payload["episode_rows"][0]["mode_switch_effective"] is True
    assert payload["episode_rows"][0]["direct_track_mode_effective"] is True
    row = payload["rows"][0]
    assert row["post_merge_anchor_mode"] == "offensive_position"
    assert row["post_merge_anchor_mode_requires_geometry_disadvantage"] is True
    assert row["post_merge_anchor_mode_release_ego_only_streak_steps"] == 1.0
    assert row["post_merge_anchor_mode_recovery_below_altitude_m"] == 4500.0
    assert row["post_merge_anchor_mode_release_reset_on_streak_break"] is False
    assert row["close_range_anchor_mode"] == "offensive_position"
    assert row["close_range_anchor_trigger_range_m"] == 1000.0
    assert row["close_range_anchor_release_on_post_merge"] is True
    assert row["close_range_anchor_requires_first_pass"] is True
    assert row["close_range_anchor_offensive_anchor_blend"] == 0.25
    assert row["close_range_anchor_offensive_anchor_blend_active"] is False
    assert row["close_range_anchor_post_merge_hold_steps"] == 20.0
    assert row["close_range_anchor_alignment_angle_deg_max"] == 10.0
    assert row["close_range_anchor_release_alignment_angle_deg_max"] == 12.0
    assert row["offensive_anchor_frame"] == "encounter"
    assert row["offensive_anchor_lateral_frame"] == "target_velocity"
    assert row["offensive_anchor_longitudinal_m"] == 800.0
    assert row["offensive_anchor_lateral_m"] == 300.0
    assert row["offensive_anchor_vertical_m"] == 500.0
    assert row["post_merge_predicted_target_forward_scale"] == 0.0
    assert row["post_merge_predicted_target_forward_scale_release_scale"] == 1.0
    assert row["post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"] == 1.0
    assert row["post_merge_predicted_target_forward_scale_release_reset_on_streak_break"] is True
    assert row["post_merge_predicted_target_forward_scale_hold_steps"] == 20.0
    assert row["post_merge_offensive_anchor_blend_release_blend"] == 0.1
    assert row["post_merge_offensive_anchor_blend_release_ego_only_streak_steps"] == 5.0
    assert row["post_merge_offensive_anchor_blend_release_reset_on_streak_break"] is True
    assert row["post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"] == 800.0
    assert row["post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"] == 20.0
    for field in (
        "first_ego_attack_time_s",
        "first_target_attack_time_s",
        "first_pass_step",
        "post_merge_attack_zone_advantage_s",
        "close_range_anchor_mode_active_fraction",
        "close_range_anchor_mode_first_active_step",
        "post_merge_anchor_mode_active_fraction",
        "post_merge_anchor_mode_condition_met_fraction",
        "post_merge_anchor_mode_first_active_step",
        "post_merge_anchor_mode_recovery_active_fraction",
        "post_merge_anchor_mode_recovery_first_step",
        "post_merge_anchor_mode_released_fraction",
        "post_merge_anchor_mode_release_first_step",
        "post_merge_offensive_anchor_condition_met_fraction",
        "post_merge_offensive_anchor_alignment_disadvantage_fraction",
        "post_merge_offensive_anchor_blend_active_fraction",
        "post_merge_offensive_anchor_first_active_step",
        "post_merge_offensive_anchor_blend_release_blend_active_fraction",
        "post_merge_offensive_anchor_blend_released_fraction",
        "post_merge_offensive_anchor_blend_release_first_step",
        "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
        "post_merge_offensive_anchor_blend_release_direct_track_first_step",
        "post_merge_predicted_target_forward_scale_active_fraction",
        "post_merge_predicted_target_forward_scale_first_active_step",
        "post_merge_predicted_target_forward_scale_release_scale_active_fraction",
        "post_merge_predicted_target_forward_scale_released_fraction",
        "post_merge_predicted_target_forward_scale_release_first_step",
        "post_merge_offensive_anchor_active_longitudinal_scale",
        "post_merge_offensive_anchor_active_lateral_scale",
        "post_merge_active_vp_longitudinal_scale",
        "post_merge_active_vp_lateral_scale",
        "post_merge_offensive_anchor_active_override_longitudinal_scale",
        "post_merge_offensive_anchor_active_override_lateral_scale",
        "pre_merge_vp_forward_bias_m",
        "post_merge_anchor_mode_active_vp_forward_bias_m",
        "post_merge_anchor_mode_active_vp_lateral_bias_m",
        "post_merge_offensive_anchor_active_vp_forward_bias_m",
        "post_merge_offensive_anchor_active_vp_lateral_bias_m",
        "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m",
        "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m",
        "vp_forward_bias_m",
        "vp_lateral_bias_m",
        "merge_min_range_m",
        "damage_margin",
    ):
        assert field in row


def test_combat_geometry_summary_builder_captures_release_recovery_metrics(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_jsbsim_hrl_comparison import (  # noqa: WPS433
        _write_combat_geometry_diagnostics,
        summarize_episode_combat_geometry,
    )

    record = {
        "controller": "prediction_vpp",
        "task": "head_on",
        "opponent_stage": "expert",
        "run_status": "smoke",
        "mode": "smoke",
        "seed": 11,
        "episode": 0,
        "scenario": "head_on_recovery",
        "steps": 2,
        "total_time_s": 0.4,
        "combat_initial_hp": 100.0,
        "ego_hp": 100.0,
        "target_hp": 100.0,
        "trajectory": [
            {
                "step": 1,
                "time_s": 0.2,
                "range_m": 1000.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "post_merge": True,
                "vp_forward_bias_m": -1200.0,
                "vp_lateral_bias_m": 300.0,
                "offset_frame": "world_neu",
                "configured_offset_frame": "target_velocity",
                "virtual_point_source": "vpp_policy",
                "anchor_mode": "predicted_target",
                "anchor_mode_requested": "predicted_target",
                "offensive_anchor_frame": "encounter",
                "longitudinal_scale": 1.0,
                "lateral_scale": 1.0,
                "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": 4500.0,
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": -4500.0,
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": -4500.0,
                "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": 0.0,
                "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": 0.25,
                "post_merge_offensive_anchor_blend_release_recovery_active": False,
                "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active": False,
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active": False,
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active": False,
            },
            {
                "step": 2,
                "time_s": 0.4,
                "range_m": 1200.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "post_merge": True,
                "vp_forward_bias_m": -600.0,
                "vp_lateral_bias_m": 180.0,
                "offset_frame": "world_neu",
                "configured_offset_frame": "target_velocity",
                "virtual_point_source": "vpp_policy",
                "anchor_mode": "predicted_target",
                "anchor_mode_requested": "predicted_target",
                "offensive_anchor_frame": "encounter",
                "longitudinal_scale": 1.0,
                "lateral_scale": 1.0,
                "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": 4500.0,
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": -4500.0,
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": -4500.0,
                "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": 0.0,
                "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": 0.25,
                "post_merge_offensive_anchor_blend_release_recovery_active": True,
                "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active": False,
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active": True,
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active": True,
            },
        ],
    }

    diagnostics = summarize_episode_combat_geometry(record)
    assert (
        diagnostics[
            "post_merge_offensive_anchor_blend_release_recovery_active_fraction"
        ]
        == 0.5
    )
    assert (
        diagnostics["post_merge_offensive_anchor_blend_release_recovery_first_step"]
        == 2.0
    )
    assert (
        diagnostics[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction"
        ]
        == 0.5
    )
    assert (
        diagnostics[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step"
        ]
        == 2.0
    )
    assert (
        diagnostics[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active_fraction"
        ]
        == 0.5
    )
    assert (
        diagnostics[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_first_step"
        ]
        == 2.0
    )

    path = _write_combat_geometry_diagnostics(tmp_path, [record])
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_row = payload["episode_rows"][0]
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
        ]
        == 4500.0
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
        ]
        == 0.0
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
        ]
        == -4500.0
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
        ]
        == -4500.0
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
        ]
        == 0.25
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_active_fraction"
        ]
        == 0.5
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_first_step"
        ]
        == 2.0
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction"
        ]
        == 0.5
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step"
        ]
        == 2.0
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active_fraction"
        ]
        == 0.5
    )
    assert (
        episode_row[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_first_step"
        ]
        == 2.0
    )
    row = payload["rows"][0]
    assert (
        row["post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"]
        == 4500.0
    )
    assert (
        row["post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"]
        == 0.0
    )
    assert (
        row["post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"]
        == -4500.0
    )
    assert (
        row["post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"]
        == -4500.0
    )
    assert (
        row["post_merge_offensive_anchor_blend_release_recovery_lateral_blend"]
        == 0.25
    )
    assert (
        row["post_merge_offensive_anchor_blend_release_recovery_active_fraction"]
        == 0.5
    )
    assert (
        row[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction"
        ]
        == 0.5
    )
    assert (
        row[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active_fraction"
        ]
        == 0.5
    )


def _make_tactical_basis_geometry_record():
    return {
        "controller": "prediction_vpp",
        "task": "head_on",
        "opponent_stage": "expert",
        "run_status": "pilot",
        "mode": "pilot",
        "seed": 0,
        "episode": 0,
        "trajectory": [
            {
                "step": 1,
                "time_s": 0.0,
                "pre_merge": True,
                "post_merge": False,
                "range_m": 1500.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "tactical_basis_action_ll": 0.5,
                "tactical_basis_action_io": 1.0,
                "tactical_basis_action_cd": 0.25,
                "vp_forward_bias_m": 10.0,
                "vp_lateral_bias_m": 1.0,
            },
            {
                "step": 2,
                "time_s": 0.2,
                "pre_merge": True,
                "post_merge": False,
                "range_m": 1200.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "tactical_basis_action_ll": 0.0,
                "tactical_basis_action_io": -1.0,
                "tactical_basis_action_cd": -0.25,
                "vp_forward_bias_m": 20.0,
                "vp_lateral_bias_m": 2.0,
            },
            {
                "step": 3,
                "time_s": 0.4,
                "pre_merge": False,
                "post_merge": True,
                "range_m": 900.0,
                "ego_in_attack_zone": True,
                "target_in_attack_zone": False,
                "tactical_basis_action_ll": -0.5,
                "tactical_basis_action_io": 0.5,
                "tactical_basis_action_cd": 0.0,
                "vp_forward_bias_m": 30.0,
                "vp_lateral_bias_m": 3.0,
            },
            {
                "step": 4,
                "time_s": 0.6,
                "pre_merge": False,
                "post_merge": True,
                "range_m": 800.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": True,
                "tactical_basis_action_ll": 0.5,
                "tactical_basis_action_io": -0.5,
                "tactical_basis_action_cd": 0.0,
                "vp_forward_bias_m": 40.0,
                "vp_lateral_bias_m": 4.0,
            },
        ],
    }


def test_combat_geometry_diagnostics_include_tactical_basis_means(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_jsbsim_hrl_comparison import (  # noqa: WPS433
        _write_combat_geometry_diagnostics,
        summarize_episode_combat_geometry,
    )

    record = _make_tactical_basis_geometry_record()
    diagnostics = summarize_episode_combat_geometry(record)

    assert diagnostics["pre_merge_mean_tactical_basis_action_ll"] == 0.25
    assert diagnostics["pre_merge_mean_tactical_basis_action_io"] == 0.0
    assert diagnostics["pre_merge_mean_tactical_basis_action_cd"] == 0.0
    assert diagnostics["post_merge_mean_tactical_basis_action_ll"] == 0.0
    assert diagnostics["post_merge_mean_tactical_basis_action_io"] == 0.0

    payload = json.loads(
        _write_combat_geometry_diagnostics(tmp_path, [record]).read_text(
            encoding="utf-8"
        )
    )
    row = payload["rows"][0]
    assert row["pre_merge_mean_tactical_basis_action_ll"] == 0.25
    assert row["post_merge_mean_tactical_basis_action_io"] == 0.0


def test_combat_geometry_diagnostics_compute_pre_merge_positive_io_fraction():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_jsbsim_hrl_comparison import summarize_episode_combat_geometry  # noqa: WPS433

    diagnostics = summarize_episode_combat_geometry(
        _make_tactical_basis_geometry_record()
    )

    assert diagnostics["pre_merge_positive_tactical_basis_io_fraction"] == 0.5


def test_combat_geometry_diagnostics_preserve_existing_vp_bias_metrics():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_jsbsim_hrl_comparison import summarize_episode_combat_geometry  # noqa: WPS433

    diagnostics = summarize_episode_combat_geometry(
        _make_tactical_basis_geometry_record()
    )

    assert diagnostics["vp_forward_bias_m"] == 25.0
    assert diagnostics["vp_lateral_bias_m"] == 2.5
