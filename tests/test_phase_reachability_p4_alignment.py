from __future__ import annotations

from uav_vpp_guidance.evaluation.phase_reachability_p4_alignment import (
    P4PhaseTracker,
    audit_episode,
    decide_primary_attribution,
)


def _frame(step: int, *, range_m: float, rate: float, first_pass: bool = False) -> dict:
    return {
        "step": step,
        "range_m": range_m,
        "range_rate_mps": rate,
        # aa=160 / ata=20 is the frozen disadvantage convention.
        "aa_deg": 160.0,
        "ata_deg": 20.0,
        "prediction_valid": True,
        "prediction_fallback": False,
        "vp_forward_bias_m": 10.0,
        "vp_lateral_bias_m": 20.0,
        "nz_cmd": 2.0,
        "roll_rate_cmd": 0.1,
        "throttle_cmd": 0.8,
        "saturation_flag": False,
        "first_pass_complete": first_pass,
    }


def test_p4_phase_tracker_matches_merge_then_reentry_semantics():
    tracker = P4PhaseTracker()
    assert tracker.update(1100.0, -100.0) == "pre_merge"
    assert tracker.update(900.0, -100.0) == "post_merge"
    assert tracker.update(850.0, -100.0) == "post_merge"
    assert tracker.update(800.0, -100.0) == "post_merge"
    assert tracker.update(750.0, -100.0) == "post_merge"
    assert tracker.update(700.0, -100.0) == "re_entry"


def test_raw_v2_record_is_taxonomy_auditable_but_not_exact_66d_reconstructable():
    record = {
        "opponent_stage": "expert",
        "scenario": "case",
        "seed": 1,
        "scenario_metadata": {"taxonomy_geometry_state": "disadvantage"},
        "trajectory": [
            _frame(1, range_m=1100.0, rate=-100.0),
            _frame(2, range_m=900.0, rate=-80.0, first_pass=True),
            _frame(3, range_m=800.0, rate=-70.0),
            _frame(4, range_m=700.0, rate=-60.0),
            _frame(5, range_m=600.0, rate=-50.0),
            _frame(6, range_m=500.0, rate=-40.0),
        ],
    }
    row = audit_episode(record)
    assert row["initial_dynamic_state"] == "disadvantage"
    assert row["p4_re_entry_steps"] == 1
    assert row["v2_first_pass_step"] == 2
    assert row["exact_66d_to_3d_pair_constructable"] is False


def test_primary_attribution_prefers_sampler_mismatch_over_raw_observability_limit():
    decision = decide_primary_attribution(
        all_v2_episodes_reach_first_pass=True,
        p4_v2_all_cells_supported=False,
        exact_66d_pair_count=0,
        episode_count=36,
    )
    assert decision.primary_attribution == "sampler_mismatch"
    assert decision.pilot_preregistration_eligible is False
