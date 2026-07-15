from __future__ import annotations

import pytest

from uav_vpp_guidance.evaluation.thesis_v17_phase_occupancy_audit import (
    SOURCE_ID,
    V17PhaseOccupancyAuditError,
    summarize_phase_occupancy,
)


def _record(*, pair_key: str, phases: list[str], first_pass: bool, handoff: bool) -> dict:
    return {
        "source_id": SOURCE_ID,
        "opponent": "end_to_end",
        "pair_key": pair_key,
        "handoff_reached": handoff,
        "window_steps": 20 if handoff else 0,
        "terminal_reason": "timeout_hp_advantage",
        "phase_trace": [
            {"phase": phase, "first_pass_complete": first_pass}
            for phase in phases
        ],
    }


def test_phase_occupancy_distinguishes_first_pass_from_post_merge_coverage():
    result = summarize_phase_occupancy(
        [
            _record(pair_key="pre_only", phases=["pre_merge", "pre_merge"], first_pass=True, handoff=False),
            _record(pair_key="reentry", phases=["pre_merge", "post_merge", "re_entry"], first_pass=True, handoff=True),
        ]
    )

    cell = result["matrix"]["end_to_end"]
    assert cell["records"] == 2
    assert cell["first_pass_latched"] == 2
    assert cell["episodes_with_post_merge"] == 1
    assert cell["episodes_with_re_entry"] == 1
    assert cell["handoff_reached"] == 1
    assert cell["first_pass_without_post_merge"] == 1


def test_phase_occupancy_rejects_unknown_phase_and_duplicate_pair():
    with pytest.raises(V17PhaseOccupancyAuditError, match="unknown phase"):
        summarize_phase_occupancy([_record(pair_key="bad", phases=["unknown"], first_pass=False, handoff=False)])
    duplicate = _record(pair_key="same", phases=["pre_merge"], first_pass=False, handoff=False)
    with pytest.raises(V17PhaseOccupancyAuditError, match="duplicate"):
        summarize_phase_occupancy([duplicate, duplicate])
