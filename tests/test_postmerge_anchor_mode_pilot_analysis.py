from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_postmerge_anchor_mode_pilot import build_report, load_run_rows  # noqa: E402


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
                "post_merge_anchor_mode_active_fraction": values["anchor_frac"],
                "post_merge_anchor_mode_first_active_step": values["anchor_step"],
                "post_merge_anchor_mode_offensive_anchor_blend": values["anchor_blend"],
                "post_merge_anchor_mode_offensive_anchor_longitudinal_blend": values["anchor_long_blend"],
                "post_merge_anchor_mode_offensive_anchor_lateral_blend": values["anchor_lat_blend"],
                "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": values.get(
                    "latch_on_activation",
                    False,
                ),
                "post_merge_anchor_mode_lateral_world_offset_latch_active_fraction": values.get(
                    "latch_active_frac",
                    float("nan"),
                ),
                "post_merge_anchor_mode_lateral_world_offset_latch_first_active_step": values.get(
                    "latch_first_step",
                    float("nan"),
                ),
                "post_merge_anchor_mode_active_vp_forward_bias_m": values["anchor_forward"],
                "post_merge_anchor_mode_active_vp_lateral_bias_m": values["anchor_lateral"],
                "post_merge_offensive_anchor_blend_active_fraction": values["blend_frac"],
                "post_merge_offensive_anchor_active_vp_forward_bias_m": values["blend_forward"],
                "post_merge_offensive_anchor_active_vp_lateral_bias_m": values["blend_lateral"],
                "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m": values["pred_forward"],
                "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m": values["pred_lateral"],
                "merge_min_range_m": values["merge_min"],
                "offensive_anchor_lateral_sign_mode": values.get("lateral_sign_mode"),
                "offensive_anchor_lateral_frame": values.get("lateral_frame"),
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


def test_build_report_compares_baseline_and_candidate(tmp_path):
    baseline_expert = tmp_path / "baseline_expert"
    baseline_e2e = tmp_path / "baseline_e2e"
    candidate_expert = tmp_path / "candidate_expert"
    candidate_e2e = tmp_path / "candidate_e2e"
    _write_run(
        baseline_expert,
        opponent_stage="expert",
        task_to_values={
            "head_on": {
                "win_rate": 0.33,
                "damage_margin": 7.0,
                "post_adv": 2.5,
                "anchor_frac": 0.0,
                "anchor_step": float("nan"),
                "anchor_blend": float("nan"),
                "anchor_long_blend": float("nan"),
                "anchor_lat_blend": float("nan"),
                "anchor_forward": float("nan"),
                "anchor_lateral": float("nan"),
                "blend_frac": 0.09,
                "blend_forward": -4300.0,
                "blend_lateral": 210.0,
                "pred_forward": -3200.0,
                "pred_lateral": 500.0,
                "merge_min": 37.0,
                "lateral_sign_mode": None,
                "lateral_frame": None,
            },
            "crossing_feasible": {
                "win_rate": 0.33,
                "damage_margin": 2.0,
                "post_adv": 1.0,
                "anchor_frac": 0.0,
                "anchor_step": float("nan"),
                "anchor_blend": float("nan"),
                "anchor_long_blend": float("nan"),
                "anchor_lat_blend": float("nan"),
                "anchor_forward": float("nan"),
                "anchor_lateral": float("nan"),
                "blend_frac": 0.0,
                "blend_forward": float("nan"),
                "blend_lateral": float("nan"),
                "pred_forward": -5400.0,
                "pred_lateral": 1800.0,
                "merge_min": 550.0,
                "lateral_sign_mode": None,
                "lateral_frame": None,
            },
        },
    )
    _write_run(
        baseline_e2e,
        opponent_stage="end_to_end",
        task_to_values={
            "head_on": {
                "win_rate": 1.0,
                "damage_margin": 10.0,
                "post_adv": 4.0,
                "anchor_frac": 0.0,
                "anchor_step": float("nan"),
                "anchor_blend": float("nan"),
                "anchor_long_blend": float("nan"),
                "anchor_lat_blend": float("nan"),
                "anchor_forward": float("nan"),
                "anchor_lateral": float("nan"),
                "blend_frac": 0.01,
                "blend_forward": 2800.0,
                "blend_lateral": -4100.0,
                "pred_forward": -900.0,
                "pred_lateral": 800.0,
                "merge_min": 194.0,
                "lateral_sign_mode": None,
                "lateral_frame": None,
            },
            "crossing_feasible": {
                "win_rate": 1.0,
                "damage_margin": 0.3,
                "post_adv": 1.5,
                "anchor_frac": 0.0,
                "anchor_step": float("nan"),
                "anchor_blend": float("nan"),
                "anchor_long_blend": float("nan"),
                "anchor_lat_blend": float("nan"),
                "anchor_forward": float("nan"),
                "anchor_lateral": float("nan"),
                "blend_frac": 0.0,
                "blend_forward": float("nan"),
                "blend_lateral": float("nan"),
                "pred_forward": 1070.0,
                "pred_lateral": 1710.0,
                "merge_min": 598.0,
                "lateral_sign_mode": None,
                "lateral_frame": None,
            },
        },
    )
    _write_run(
        candidate_expert,
        opponent_stage="expert",
        task_to_values={
            "head_on": {
                "win_rate": 0.67,
                "damage_margin": 12.0,
                "post_adv": 5.0,
                "anchor_frac": 0.18,
                "anchor_step": 27.0,
                "anchor_blend": 0.25,
                "anchor_long_blend": 1.0,
                "anchor_lat_blend": 0.25,
                "anchor_forward": -1200.0,
                "anchor_lateral": 300.0,
                "blend_frac": 0.0,
                "blend_forward": float("nan"),
                "blend_lateral": float("nan"),
                "pred_forward": -3400.0,
                "pred_lateral": 400.0,
                "merge_min": 45.0,
                "latch_on_activation": True,
                "latch_active_frac": 0.18,
                "latch_first_step": 27.0,
                "lateral_sign_mode": "fixed_positive",
                "lateral_frame": "encounter",
            },
            "crossing_feasible": {
                "win_rate": 0.33,
                "damage_margin": 2.0,
                "post_adv": 1.0,
                "anchor_frac": 0.0,
                "anchor_step": float("nan"),
                "anchor_blend": float("nan"),
                "anchor_long_blend": float("nan"),
                "anchor_lat_blend": float("nan"),
                "anchor_forward": float("nan"),
                "anchor_lateral": float("nan"),
                "blend_frac": 0.0,
                "blend_forward": float("nan"),
                "blend_lateral": float("nan"),
                "pred_forward": -5400.0,
                "pred_lateral": 1800.0,
                "merge_min": 550.0,
                "lateral_sign_mode": "same_side",
                "lateral_frame": "target_velocity",
            },
        },
    )
    _write_run(
        candidate_e2e,
        opponent_stage="end_to_end",
        task_to_values={
            "head_on": {
                "win_rate": 1.0,
                "damage_margin": 9.5,
                "post_adv": 3.8,
                "anchor_frac": 0.02,
                "anchor_step": 106.0,
                "anchor_blend": float("nan"),
                "anchor_long_blend": 1.0,
                "anchor_lat_blend": 0.25,
                "anchor_forward": 1500.0,
                "anchor_lateral": -2500.0,
                "blend_frac": 0.0,
                "blend_forward": float("nan"),
                "blend_lateral": float("nan"),
                "pred_forward": -950.0,
                "pred_lateral": 790.0,
                "merge_min": 194.0,
                "latch_on_activation": True,
                "latch_active_frac": 0.02,
                "latch_first_step": 106.0,
                "lateral_sign_mode": "fixed_positive",
                "lateral_frame": "encounter",
            },
            "crossing_feasible": {
                "win_rate": 1.0,
                "damage_margin": 0.3,
                "post_adv": 1.5,
                "anchor_frac": 0.0,
                "anchor_step": float("nan"),
                "anchor_blend": float("nan"),
                "anchor_long_blend": float("nan"),
                "anchor_lat_blend": float("nan"),
                "anchor_forward": float("nan"),
                "anchor_lateral": float("nan"),
                "blend_frac": 0.0,
                "blend_forward": float("nan"),
                "blend_lateral": float("nan"),
                "pred_forward": 1070.0,
                "pred_lateral": 1710.0,
                "merge_min": 598.0,
                "lateral_sign_mode": "same_side",
                "lateral_frame": "target_velocity",
            },
        },
    )

    baseline_rows = []
    baseline_rows.extend(
        load_run_rows(label="baseline", run_dir=baseline_expert, method="prediction_vpp", tasks=["head_on", "crossing_feasible"])
    )
    baseline_rows.extend(
        load_run_rows(label="baseline", run_dir=baseline_e2e, method="prediction_vpp", tasks=["head_on", "crossing_feasible"])
    )
    candidate_rows = []
    candidate_rows.extend(
        load_run_rows(label="candidate", run_dir=candidate_expert, method="prediction_vpp", tasks=["head_on", "crossing_feasible"])
    )
    candidate_rows.extend(
        load_run_rows(label="candidate", run_dir=candidate_e2e, method="prediction_vpp", tasks=["head_on", "crossing_feasible"])
    )
    report = build_report(baseline_rows, candidate_rows)

    expert_head_on = report["headline"]["expert_head_on"]
    assert expert_head_on["delta"]["win_rate"] == 0.34
    assert expert_head_on["delta"]["damage_margin"] == 5.0
    assert expert_head_on["candidate"]["post_merge_anchor_mode_offensive_anchor_blend"] == 0.25
    assert (
        expert_head_on["candidate"][
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend"
        ]
        == 1.0
    )
    assert (
        expert_head_on["candidate"][
            "post_merge_anchor_mode_offensive_anchor_lateral_blend"
        ]
        == 0.25
    )
    assert expert_head_on["candidate"]["offensive_anchor_lateral_frame"] == "encounter"
    assert expert_head_on["candidate"]["post_merge_anchor_mode_active_fraction"] == 0.18
    assert (
        expert_head_on["candidate"][
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation"
        ]
        is True
    )
    assert (
        expert_head_on["candidate"][
            "post_merge_anchor_mode_lateral_world_offset_latch_active_fraction"
        ]
        == 0.18
    )
    assert report["headline"]["end_to_end_head_on"]["delta"]["damage_margin"] == -0.5
