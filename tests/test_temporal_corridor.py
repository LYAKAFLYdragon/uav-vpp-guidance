from __future__ import annotations

import numpy as np

from uav_vpp_guidance.virtual_point.temporal_corridor import TemporalCorridorConfig, compile_temporal_corridor_action


RANGES = [[-1500.0, 1500.0], [-800.0, 800.0], [-500.0, 500.0]]


def test_temporal_corridor_stays_inside_the_existing_action_box_and_slew_limit():
    config = TemporalCorridorConfig(longitudinal_extent_m=450.0, lateral_extent_m=300.0, vertical_extent_m=100.0)
    previous = np.zeros(3)
    actions = []
    for step in range(config.horizon_steps):
        receipt = compile_temporal_corridor_action(
            own_position_neu=[0.0, 100.0, 5000.0],
            predicted_target_position_neu=[1000.0, 0.0, 5000.0],
            phase_step=step,
            previous_action=previous,
            raw_action_ranges=RANGES,
            config=config,
        )
        previous = np.asarray(receipt["applied_action"])
        actions.append(previous)
        assert np.max(np.abs(previous)) <= 1.0
    assert np.max(np.abs(np.diff(np.asarray(actions), axis=0))) <= config.max_action_delta + 1e-12


def test_temporal_corridor_reports_residual_when_a_requested_path_exceeds_the_raw_box():
    receipt = compile_temporal_corridor_action(
        own_position_neu=[0.0, 100.0, 5000.0],
        predicted_target_position_neu=[1000.0, 0.0, 5000.0],
        phase_step=10,
        previous_action=[0.0, 0.0, 0.0],
        raw_action_ranges=RANGES,
        config=TemporalCorridorConfig(longitudinal_extent_m=3000.0, lateral_extent_m=2000.0, vertical_extent_m=1000.0, max_action_delta=2.0),
    )
    assert receipt["hard_bound_active"] is True
    assert receipt["projection_residual_norm_m"] > 0.0
