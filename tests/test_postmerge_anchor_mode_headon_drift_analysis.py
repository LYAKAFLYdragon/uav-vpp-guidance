from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_postmerge_anchor_mode_headon_drift import (  # noqa: E402
    build_comparison_report,
    summarize_episode_anchor_mode_drift,
)


def _make_episode(
    *,
    seed: int,
    damage_margin: float,
    active_steps: list[int],
    forward_values: list[float],
    lateral_values: list[float],
    range_values: list[float],
    latch_world_offset_xy: tuple[float, float] | None = None,
) -> dict:
    ego_hp = 100.0
    target_hp = 100.0 - damage_margin
    trajectory = []
    for step in range(1, 7):
        idx = step - 1
        point = {
            "step": step,
            "post_merge": step >= 2,
            "post_merge_anchor_mode_active": step in active_steps,
            "post_merge_anchor_mode_released": step >= 6,
            "vp_forward_bias_m": float("nan"),
            "vp_lateral_bias_m": float("nan"),
            "range_m": float("nan"),
        }
        if step in active_steps:
            active_idx = active_steps.index(step)
            point["vp_forward_bias_m"] = forward_values[active_idx]
            point["vp_lateral_bias_m"] = lateral_values[active_idx]
            point["range_m"] = range_values[active_idx]
            if latch_world_offset_xy is not None:
                point["offensive_anchor_lateral_world_offset_x"] = latch_world_offset_xy[0]
                point["offensive_anchor_lateral_world_offset_y"] = latch_world_offset_xy[1]
                point["offensive_anchor_lateral_world_offset_z"] = 0.0
        trajectory.append(point)
    return {
        "seed": seed,
        "episode": 0,
        "task": "head_on",
        "scenario": "head_on_minimal",
        "success": damage_margin >= 0.0,
        "termination_reason": "timeout_hp_advantage",
        "ego_hp": ego_hp,
        "target_hp": target_hp,
        "trajectory": trajectory,
    }


def test_summarize_episode_anchor_mode_drift_captures_segment_statistics():
    episode = _make_episode(
        seed=1,
        damage_margin=5.0,
        active_steps=[3, 4, 5],
        forward_values=[-500.0, 100.0, 2500.0],
        lateral_values=[0.0, 800.0, 2000.0],
        range_values=[5000.0, 3000.0, 9000.0],
        latch_world_offset_xy=(120.0, -45.0),
    )

    summary = summarize_episode_anchor_mode_drift(episode)

    assert summary["damage_margin"] == 5.0
    assert summary["first_pass_step"] == 2.0
    assert summary["active_segments"] == [(3, 5)]
    assert summary["total_active_steps"] == 3
    assert summary["first_active_step"] == 3.0
    assert summary["last_active_step"] == 5.0
    assert summary["first_active_vp_forward_bias_m"] == -500.0
    assert summary["last_active_vp_forward_bias_m"] == 2500.0
    assert summary["mean_active_vp_forward_bias_m"] == 700.0
    assert summary["max_active_range_m"] == 9000.0
    assert summary["latch_world_offset_x"] == 120.0
    assert summary["latch_world_offset_y"] == -45.0


def test_build_comparison_report_pairs_seed_and_computes_deltas():
    baseline = [
        summarize_episode_anchor_mode_drift(
            _make_episode(
                seed=42,
                damage_margin=10.0,
                active_steps=[3, 4],
                forward_values=[-200.0, 100.0],
                lateral_values=[0.0, 400.0],
                range_values=[4000.0, 5000.0],
            )
        )
    ]
    candidate = [
        summarize_episode_anchor_mode_drift(
            _make_episode(
                seed=42,
                damage_margin=-5.0,
                active_steps=[3, 4, 5],
                forward_values=[-200.0, 1200.0, 3000.0],
                lateral_values=[0.0, 800.0, 1600.0],
                range_values=[4000.0, 7000.0, 12000.0],
                latch_world_offset_xy=(150.0, -250.0),
            )
        )
    ]

    report = build_comparison_report(baseline, candidate)
    row = report["rows"][0]

    assert row["seed"] == 42
    assert row["delta"]["damage_margin"] == -15.0
    assert row["delta"]["total_active_steps"] == 1.0
    assert row["delta"]["mean_active_vp_forward_bias_m"] > 0.0
    assert row["candidate"]["latch_world_offset_x"] == 150.0
