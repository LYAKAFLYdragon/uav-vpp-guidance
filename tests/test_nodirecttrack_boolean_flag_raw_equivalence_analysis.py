from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_nodirecttrack_boolean_flag_raw_equivalence import (  # noqa: E402
    build_seed_comparison,
    extract_step_window,
    first_vp_bias_divergence_step,
)


def _episode(trajectory, *, termination_reason="timeout_hp_advantage", success=True):
    return {
        "termination_reason": termination_reason,
        "success": success,
        "steps": len(trajectory),
        "trajectory": trajectory,
    }


def _row(
    step: int,
    *,
    released: bool = False,
    latch: bool = False,
    recovery: bool = False,
    direct_track: bool = False,
    vp_forward_bias_m: float = 0.0,
    recovery_preview_vp_forward_bias_m: float | None = None,
):
    return {
        "step": step,
        "post_merge_offensive_anchor_blend_released": released,
        "post_merge_offensive_anchor_lateral_world_offset_latch_active": latch,
        "post_merge_offensive_anchor_blend_release_recovery_active": recovery,
        "direct_track_mode_effective": direct_track,
        "vp_forward_bias_m": vp_forward_bias_m,
        "post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m": recovery_preview_vp_forward_bias_m,
    }


def test_first_vp_bias_divergence_step_uses_threshold():
    lhs = [_row(1, vp_forward_bias_m=0.0), _row(2, vp_forward_bias_m=-10.0)]
    rhs = [_row(1, vp_forward_bias_m=0.5), _row(2, vp_forward_bias_m=-12.0)]

    assert first_vp_bias_divergence_step(lhs, rhs, threshold_m=1.0) == 2
    assert first_vp_bias_divergence_step(lhs, rhs, threshold_m=5.0) is None


def test_extract_step_window_returns_requested_slice():
    trajectory = [_row(step, vp_forward_bias_m=-100.0 * step) for step in range(1, 8)]

    window = extract_step_window(trajectory, center_step=4, pre_steps=1, post_steps=2)

    assert [point["step"] for point in window] == [3, 4, 5, 6]


def test_build_seed_comparison_identifies_oldflag_latch_and_recovery():
    oldhack = _episode(
        [
            _row(1, vp_forward_bias_m=10.0),
            _row(2, vp_forward_bias_m=5.0),
            _row(3, released=True, vp_forward_bias_m=-100.0),
            _row(4, released=True, vp_forward_bias_m=-110.0),
        ]
    )
    oldflag = _episode(
        [
            _row(1, latch=True, vp_forward_bias_m=10.0),
            _row(2, latch=True, vp_forward_bias_m=5.0),
            _row(3, released=True, latch=True, vp_forward_bias_m=-100.0),
            _row(
                4,
                released=True,
                latch=True,
                recovery=True,
                vp_forward_bias_m=-130.0,
                recovery_preview_vp_forward_bias_m=-125.0,
            ),
        ],
        termination_reason="ego_crash_or_out_of_bounds",
        success=False,
    )
    fixedflag = _episode(
        [
            _row(1, vp_forward_bias_m=10.0),
            _row(2, vp_forward_bias_m=5.0),
            _row(3, released=True, vp_forward_bias_m=-100.0),
            _row(4, released=True, vp_forward_bias_m=-111.0),
        ],
        termination_reason="target_crash_or_out_of_bounds",
        success=True,
    )

    report = build_seed_comparison(
        seed=48000000,
        oldhack_episode=oldhack,
        oldflag_episode=oldflag,
        fixedflag_episode=fixedflag,
    )

    assert report["oldhack"]["first_release_step"] == 3
    assert report["oldhack"]["first_latch_step"] is None
    assert report["oldflag"]["first_latch_step"] == 1
    assert report["oldflag"]["first_recovery_step"] == 4
    assert report["fixedflag"]["first_latch_step"] is None
    assert report["fixedflag"]["first_recovery_step"] is None
    assert (
        report["oldflag"]["recovery_preview_vp_forward_bias_m_at_first_recovery"]
        == -125.0
    )
    assert (
        report["pairwise_divergence"][
            "oldflag_vs_fixedflag_first_vp_bias_divergence_gt_10m_step"
        ]
        == 4
    )
    assert [point["step"] for point in report["release_window"]["oldhack"]] == [1, 2, 3, 4]
