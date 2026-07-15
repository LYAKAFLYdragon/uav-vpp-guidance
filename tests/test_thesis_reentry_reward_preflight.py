from __future__ import annotations

import numpy as np

from uav_vpp_guidance.evaluation.thesis_reentry_reward_preflight import annotate_ledger


def _step(range_normalized: float, rate_normalized: float, first_pass: bool, has_66d: bool) -> dict:
    base = np.zeros(16, dtype=float)
    base[0], base[1] = range_normalized, rate_normalized
    return {
        "base_observation_16d": base.tolist(),
        "raw_si_first_pass_latched": first_pass,
        "observation_66d": [0.0] * 66 if has_66d else None,
    }


def test_annotation_uses_v8_phase_and_final_step_only_terminal_reward():
    ledger = {"steps": [
        _step(0.24, -1.0, False, False),
        _step(0.18, -1.0, False, False),
        _step(0.14, -0.5, False, False),
        _step(0.12, -0.2, False, False),
        _step(0.14, 0.4, False, True),
        _step(0.18, 0.5, True, True),
        _step(0.26, -0.1, True, True),
        _step(0.24, -0.3, True, True),
    ]}
    summary = annotate_ledger(
        ledger,
        terminal_reason="ego_crash_or_out_of_bounds",
        first_pass_event={
            "closing_rate_threshold_mps": 0.0,
            "opening_rate_threshold_mps": 0.0,
            "minimum_closing_steps": 3,
            "minimum_opening_steps": 2,
        },
    )

    assert ledger["steps"][2]["pilot_v1_phase"] == "post_merge"
    assert ledger["steps"][7]["pilot_v1_phase"] == "re_entry"
    assert [step["pilot_v1_terminal_reward"] for step in ledger["steps"][:-1]] == [0.0] * 7
    assert ledger["steps"][-1]["pilot_v1_terminal_reward"] == -2.0
    assert summary == {"reentry_66d_steps": 1, "terminal_reward_total": -2.0}
