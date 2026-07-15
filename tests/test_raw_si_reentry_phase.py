from __future__ import annotations

import pytest

from uav_vpp_guidance.evaluation.raw_si_reentry_phase import (
    RawSIReentryPhaseError,
    RawSIReentryPhaseTracker,
)


def test_reentry_requires_a_physical_merge_and_first_pass_latch():
    tracker = RawSIReentryPhaseTracker()

    assert tracker.update(1200.0, -220.0, step_index=0) == "pre_merge"
    assert tracker.update(900.0, -220.0, step_index=1) == "post_merge"
    assert tracker.update(700.0, -120.0, step_index=2) == "post_merge"
    assert tracker.update(600.0, -40.0, step_index=3) == "post_merge"
    assert tracker.update(700.0, 80.0, step_index=4) == "post_merge"
    assert tracker.update(900.0, 100.0, step_index=5) == "post_merge"

    assert tracker.first_pass_complete is True
    assert tracker.update(1400.0, -10.0, step_index=6) == "post_merge"
    assert tracker.update(1300.0, -30.0, step_index=7) == "re_entry"


def test_reentry_tracker_rejects_nonphysical_or_nonfinite_inputs():
    tracker = RawSIReentryPhaseTracker()

    with pytest.raises(RawSIReentryPhaseError, match="non-negative"):
        tracker.update(-1.0, -10.0, step_index=0)
    with pytest.raises(RawSIReentryPhaseError, match="finite"):
        tracker.update(1000.0, float("nan"), step_index=0)
