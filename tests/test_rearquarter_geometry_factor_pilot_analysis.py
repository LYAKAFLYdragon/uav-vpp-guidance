from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_rearquarter_geometry_factor_pilot import (  # noqa: E402
    build_report,
    load_run_record,
    render_markdown,
)


def _write_run(
    run_dir: Path,
    *,
    opponent_stage: str,
    task_to_values: dict,
) -> None:
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    diagnostic_rows = []
    for task, values in task_to_values.items():
        summary_rows.append(
            {
                "method": "prediction_vpp",
                "task": task,
                "opponent_stage": opponent_stage,
                "win_rate": values["win_rate"],
                "hp_advantage": values["damage_margin"],
                "damaging_win_rate": values["win_rate"],
            }
        )
        diagnostic_rows.append(
            {
                "method": "prediction_vpp",
                "task": task,
                "opponent_stage": opponent_stage,
                "damage_margin": values["damage_margin"],
                "post_merge_attack_zone_advantage_s": values["post_adv"],
                "post_merge_offensive_anchor_blend_active_fraction": values["active_frac"],
                "post_merge_offensive_anchor_first_active_step": values["active_step"],
                "post_merge_offensive_anchor_active_longitudinal_scale": values["active_long_scale"],
                "post_merge_offensive_anchor_active_lateral_scale": values["active_lat_scale"],
                "offensive_anchor_frame": values.get("anchor_frame"),
                "offensive_anchor_encounter_stable_max_heading_delta_deg": values.get(
                    "anchor_stable_deg"
                ),
                "post_merge_offensive_anchor_active_vp_forward_bias_m": values["active_forward"],
                "post_merge_offensive_anchor_active_vp_lateral_bias_m": values["active_lateral"],
                "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m": values["pred_forward"],
                "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m": values["pred_lateral"],
                "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": values["direct_frac"],
                "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction": values.get(
                    "latch_frac"
                ),
                "post_merge_offensive_anchor_lateral_world_offset_latch_first_active_step": values.get(
                    "latch_step"
                ),
                "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction": values.get(
                    "lateral_only_frac"
                ),
                "post_merge_offensive_anchor_blend_release_lateral_only_first_step": values.get(
                    "lateral_only_step"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_active_fraction": values.get(
                    "recovery_frac"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_first_step": values.get(
                    "recovery_step"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": values.get(
                    "recovery_below_altitude_m"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": values.get(
                    "recovery_longitudinal_blend"
                ),
                "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": values.get(
                    "recovery_lateral_blend"
                ),
                "merge_min_range_m": values["merge_min"],
            }
        )
    config_overrides = []
    frame_by_task = {
        task: values["anchor_frame"]
        for task, values in task_to_values.items()
        if values.get("anchor_frame") is not None
    }
    if frame_by_task:
        config_overrides.append(
            {
                "key": "virtual_point.offensive_anchor_frame_by_task",
                "new_value": frame_by_task,
            }
        )
    stable_deg_by_task = {
        task: values["anchor_stable_deg"]
        for task, values in task_to_values.items()
        if values.get("anchor_stable_deg") is not None
    }
    if stable_deg_by_task:
        config_overrides.append(
            {
                "key": (
                    "virtual_point."
                    "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
                ),
                "new_value": stable_deg_by_task,
            }
        )
    anchor_long_scale_by_task = {
        task: values["anchor_cfg_long_scale"]
        for task, values in task_to_values.items()
        if values.get("anchor_cfg_long_scale") is not None
    }
    if anchor_long_scale_by_task:
        config_overrides.append(
            {
                "key": (
                    "virtual_point.post_merge_offensive_anchor_longitudinal_scale_by_task"
                ),
                "new_value": anchor_long_scale_by_task,
            }
        )
    anchor_lat_scale_by_task = {
        task: values["anchor_cfg_lat_scale"]
        for task, values in task_to_values.items()
        if values.get("anchor_cfg_lat_scale") is not None
    }
    if anchor_lat_scale_by_task:
        config_overrides.append(
            {
                "key": "virtual_point.post_merge_offensive_anchor_lateral_scale_by_task",
                "new_value": anchor_lat_scale_by_task,
            }
        )
    latch_enabled_by_task = {
        task: values["latch_enabled"]
        for task, values in task_to_values.items()
        if values.get("latch_enabled") is not None
    }
    if latch_enabled_by_task:
        config_overrides.append(
            {
                "key": (
                    "virtual_point."
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
                ),
                "new_value": latch_enabled_by_task,
            }
        )
    lateral_only_enabled_by_task = {
        task: values["release_lateral_only_enabled"]
        for task, values in task_to_values.items()
        if values.get("release_lateral_only_enabled") is not None
    }
    if lateral_only_enabled_by_task:
        config_overrides.append(
            {
                "key": (
                    "virtual_point."
                    "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
                ),
                "new_value": lateral_only_enabled_by_task,
            }
        )
    (aggregate_dir / "method_task_summary.json").write_text(
        json.dumps({"rows": summary_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (aggregate_dir / "combat_geometry_diagnostics.json").write_text(
        json.dumps({"rows": diagnostic_rows, "episode_rows": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"config_overrides": config_overrides}, indent=2) + "\n",
        encoding="utf-8",
    )


def _report_row(
    *,
    label: str,
    opponent_stage: str,
    task: str,
    win_rate: float,
    damage_margin: float,
    post_adv: float,
    active_long_scale: float,
    active_lat_scale: float,
    active_forward: float,
    active_lateral: float,
    direct_frac: float,
    anchor_frame: str,
    anchor_stable_deg: float | None = None,
    anchor_cfg_long_scale: float | None = None,
    anchor_cfg_lat_scale: float | None = None,
    latch_frac: float | None = None,
    latch_step: float | None = None,
    lateral_only_frac: float | None = None,
    lateral_only_step: float | None = None,
    latch_enabled: bool | None = None,
    release_lateral_only_enabled: bool | None = None,
    recovery_frac: float | None = None,
    recovery_step: float | None = None,
) -> dict:
    return {
        "label": label,
        "opponent_stage": opponent_stage,
        "task": task,
        "win_rate": win_rate,
        "damage_margin": damage_margin,
        "post_merge_attack_zone_advantage_s": post_adv,
        "post_merge_offensive_anchor_blend_active_fraction": 0.09 if task == "head_on" else 0.0,
        "post_merge_offensive_anchor_first_active_step": 26.0 if task == "head_on" else float("nan"),
        "offensive_anchor_frame": anchor_frame,
        "offensive_anchor_encounter_stable_max_heading_delta_deg": anchor_stable_deg,
        "configured_post_merge_offensive_anchor_longitudinal_scale": anchor_cfg_long_scale,
        "configured_post_merge_offensive_anchor_lateral_scale": anchor_cfg_lat_scale,
        "post_merge_active_vp_longitudinal_scale": active_long_scale,
        "post_merge_active_vp_lateral_scale": active_lat_scale,
        "post_merge_offensive_anchor_active_vp_forward_bias_m": active_forward,
        "post_merge_offensive_anchor_active_vp_lateral_bias_m": active_lateral,
        "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m": (
            active_forward - 500.0 if task == "head_on" else -900.0
        ),
        "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": direct_frac,
        "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction": latch_frac,
        "post_merge_offensive_anchor_lateral_world_offset_latch_first_active_step": latch_step,
        "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction": lateral_only_frac,
        "post_merge_offensive_anchor_blend_release_lateral_only_first_step": lateral_only_step,
        "configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": (
            latch_enabled
        ),
        "configured_post_merge_offensive_anchor_blend_release_lateral_only": (
            release_lateral_only_enabled
        ),
        "post_merge_offensive_anchor_blend_release_recovery_active_fraction": recovery_frac,
        "post_merge_offensive_anchor_blend_release_recovery_first_step": recovery_step,
        "merge_min_range_m": 36.0 if task == "head_on" else 560.0,
    }


def test_load_run_record_and_report_compute_deltas(tmp_path):
    baseline_dir = tmp_path / "baseline_expert"
    lag_dir = tmp_path / "lag_only_expert"
    _write_run(
        baseline_dir,
        opponent_stage="expert",
        task_to_values={
            "head_on": {
                "win_rate": 0.33,
                "damage_margin": 8.0,
                "post_adv": 3.2,
                "active_frac": 0.09,
                "active_step": 26.0,
                "active_long_scale": 1.0,
                "active_lat_scale": 1.0,
                "anchor_frame": "encounter",
                "anchor_stable_deg": None,
                "anchor_cfg_long_scale": None,
                "anchor_cfg_lat_scale": None,
                "active_forward": -3200.0,
                "active_lateral": 600.0,
                "pred_forward": -4100.0,
                "pred_lateral": 500.0,
                "direct_frac": 0.04,
                "merge_min": 40.0,
            },
            "crossing_feasible": {
                "win_rate": 0.33,
                "damage_margin": 10.0,
                "post_adv": 1.0,
                "active_frac": 0.0,
                "active_step": float("nan"),
                "active_long_scale": float("nan"),
                "active_lat_scale": float("nan"),
                "anchor_frame": "target_velocity",
                "anchor_stable_deg": 45.0,
                "anchor_cfg_long_scale": None,
                "anchor_cfg_lat_scale": None,
                "active_forward": float("nan"),
                "active_lateral": float("nan"),
                "pred_forward": -300.0,
                "pred_lateral": 80.0,
                "direct_frac": 0.0,
                "merge_min": 550.0,
            },
        },
    )
    _write_run(
        lag_dir,
        opponent_stage="expert",
        task_to_values={
            "head_on": {
                "win_rate": 0.67,
                "damage_margin": 12.5,
                "post_adv": 4.8,
                "active_frac": 0.06,
                "active_step": 28.0,
                "active_long_scale": 0.0,
                "active_lat_scale": 1.0,
                "anchor_frame": "encounter_stable",
                "anchor_stable_deg": 20.0,
                "anchor_cfg_long_scale": 0.25,
                "anchor_cfg_lat_scale": 0.5,
                "active_forward": -1800.0,
                "active_lateral": 50.0,
                "pred_forward": -2400.0,
                "pred_lateral": 40.0,
                "direct_frac": 0.01,
                "latch_frac": 0.67,
                "latch_step": 27.0,
                "lateral_only_frac": 0.33,
                "lateral_only_step": 29.0,
                "latch_enabled": True,
                "release_lateral_only_enabled": True,
                "recovery_frac": 0.25,
                "recovery_step": 140.0,
                "recovery_below_altitude_m": 4500.0,
                "recovery_longitudinal_blend": 0.0,
                "recovery_lateral_blend": 0.25,
                "merge_min": 60.0,
            },
            "crossing_feasible": {
                "win_rate": 0.33,
                "damage_margin": 9.8,
                "post_adv": 0.9,
                "active_frac": 0.0,
                "active_step": float("nan"),
                "active_long_scale": float("nan"),
                "active_lat_scale": float("nan"),
                "anchor_frame": "target_velocity",
                "anchor_stable_deg": 45.0,
                "anchor_cfg_long_scale": None,
                "anchor_cfg_lat_scale": None,
                "active_forward": float("nan"),
                "active_lateral": float("nan"),
                "pred_forward": -280.0,
                "pred_lateral": 75.0,
                "direct_frac": 0.0,
                "merge_min": 560.0,
            },
        },
    )

    rows = []
    rows.extend(
        load_run_record(
            label="baseline",
            run_dir=baseline_dir,
            method="prediction_vpp",
            tasks=["head_on", "crossing_feasible"],
        )
    )
    rows.extend(
        load_run_record(
            label="lag_only",
            run_dir=lag_dir,
            method="prediction_vpp",
            tasks=["head_on", "crossing_feasible"],
        )
    )
    report = build_report(rows, baseline_label="baseline")

    assert len(report["rows"]) == 4
    lag_head_on = next(
        row
        for row in report["rows"]
        if row["label"] == "lag_only" and row["task"] == "head_on"
    )
    assert lag_head_on["win_rate_delta_vs_baseline"] == 0.34
    assert lag_head_on["damage_margin_delta_vs_baseline"] == 4.5
    assert lag_head_on["post_merge_active_vp_longitudinal_scale"] == 0.0
    assert lag_head_on["post_merge_active_vp_lateral_scale"] == 1.0
    assert lag_head_on["configured_post_merge_offensive_anchor_longitudinal_scale"] == 0.25
    assert lag_head_on["configured_post_merge_offensive_anchor_lateral_scale"] == 0.5
    assert (
        lag_head_on["configured_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"]
        is True
    )
    assert (
        lag_head_on["configured_post_merge_offensive_anchor_blend_release_lateral_only"]
        is True
    )
    assert (
        lag_head_on["post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction"]
        == 0.67
    )
    assert (
        lag_head_on["post_merge_offensive_anchor_blend_release_lateral_only_active_fraction"]
        == 0.33
    )
    assert (
        lag_head_on["post_merge_offensive_anchor_blend_release_recovery_active_fraction"]
        == 0.25
    )
    assert (
        lag_head_on["post_merge_offensive_anchor_blend_release_recovery_first_step"]
        == 140.0
    )
    assert lag_head_on["offensive_anchor_frame"] == "encounter_stable"
    assert lag_head_on["offensive_anchor_encounter_stable_max_heading_delta_deg"] == 20.0
    assert report["expert_head_on_ranking"][0]["label"] == "lag_only"
    assert (
        report["expert_head_on_ranking"][0][
            "configured_post_merge_offensive_anchor_longitudinal_scale"
        ]
        == 0.25
    )


def test_build_report_surfaces_release_signal_and_geometry_equivalence():
    rows = [
        _report_row(
            label="baseline",
            opponent_stage="expert",
            task="head_on",
            win_rate=1.0,
            damage_margin=15.833,
            post_adv=6.333,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=-4317.592,
            active_lateral=213.684,
            direct_frac=0.046,
            anchor_frame="encounter",
        ),
        _report_row(
            label="baseline",
            opponent_stage="expert",
            task="crossing_feasible",
            win_rate=0.333,
            damage_margin=2.667,
            post_adv=2.667,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
        ),
        _report_row(
            label="baseline",
            opponent_stage="end_to_end",
            task="crossing_feasible",
            win_rate=1.0,
            damage_margin=0.333,
            post_adv=1.533,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
        ),
        _report_row(
            label="scaleguard_long0",
            opponent_stage="expert",
            task="head_on",
            win_rate=0.5,
            damage_margin=11.167,
            post_adv=4.467,
            active_long_scale=0.0,
            active_lat_scale=1.0,
            active_forward=-4326.640,
            active_lateral=195.296,
            direct_frac=0.030,
            anchor_frame="encounter",
            anchor_cfg_long_scale=0.0,
            anchor_cfg_lat_scale=1.0,
            recovery_frac=0.42,
            recovery_step=180.0,
        ),
        _report_row(
            label="scaleguard_long0",
            opponent_stage="expert",
            task="crossing_feasible",
            win_rate=0.333,
            damage_margin=2.667,
            post_adv=2.667,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
            anchor_cfg_long_scale=1.0,
            anchor_cfg_lat_scale=1.0,
        ),
        _report_row(
            label="scaleguard_long0",
            opponent_stage="end_to_end",
            task="crossing_feasible",
            win_rate=1.0,
            damage_margin=0.333,
            post_adv=1.533,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
            anchor_cfg_long_scale=1.0,
            anchor_cfg_lat_scale=1.0,
        ),
        _report_row(
            label="stable20",
            opponent_stage="expert",
            task="head_on",
            win_rate=0.5,
            damage_margin=11.167,
            post_adv=4.467,
            active_long_scale=0.0,
            active_lat_scale=1.0,
            active_forward=-4326.640,
            active_lateral=195.296,
            direct_frac=0.030,
            anchor_frame="encounter_stable",
            anchor_stable_deg=20.0,
        ),
        _report_row(
            label="stable20",
            opponent_stage="expert",
            task="crossing_feasible",
            win_rate=0.333,
            damage_margin=2.667,
            post_adv=2.667,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
            anchor_stable_deg=45.0,
        ),
        _report_row(
            label="stable20",
            opponent_stage="end_to_end",
            task="crossing_feasible",
            win_rate=1.0,
            damage_margin=0.333,
            post_adv=1.533,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
            anchor_stable_deg=45.0,
        ),
        _report_row(
            label="nodirecttrack_release",
            opponent_stage="expert",
            task="head_on",
            win_rate=0.667,
            damage_margin=16.333,
            post_adv=6.533,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=-4282.872,
            active_lateral=221.869,
            direct_frac=0.0,
            anchor_frame="encounter",
        ),
        _report_row(
            label="nodirecttrack_release",
            opponent_stage="expert",
            task="crossing_feasible",
            win_rate=0.333,
            damage_margin=2.667,
            post_adv=2.667,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
        ),
        _report_row(
            label="nodirecttrack_release",
            opponent_stage="end_to_end",
            task="crossing_feasible",
            win_rate=1.0,
            damage_margin=0.333,
            post_adv=1.533,
            active_long_scale=float("nan"),
            active_lat_scale=float("nan"),
            active_forward=float("nan"),
            active_lateral=float("nan"),
            direct_frac=0.0,
            anchor_frame="target_velocity",
        ),
    ]

    report = build_report(rows, baseline_label="baseline")

    diagnosis = report["executive_diagnosis"]
    assert diagnosis["best_nonbaseline_expert_head_on"]["label"] == "nodirecttrack_release"
    assert diagnosis["lowest_directtrack_expert_head_on"]["label"] == "nodirecttrack_release"
    assert diagnosis["strongest_recovery_signal"]["label"] == "scaleguard_long0"
    assert diagnosis["strongest_recovery_signal"]["recovery_active_fraction"] == 0.42
    assert diagnosis["crossing_neutral_labels"] == [
        "nodirecttrack_release",
        "scaleguard_long0",
        "stable20",
    ]
    assert len(diagnosis["geometry_equivalence_pairs"]) == 1
    pair = diagnosis["geometry_equivalence_pairs"][0]
    assert {pair["lhs"]["label"], pair["rhs"]["label"]} == {
        "scaleguard_long0",
        "stable20",
    }
    assert pair["lhs"]["damage_margin"] == 11.167
    assert pair["rhs"]["post_merge_active_vp_longitudinal_scale"] == 0.0
    markdown = render_markdown(report)
    assert "recovery_frac" in markdown
    assert "strongest recovery-stage signal" in markdown
