from __future__ import annotations

import numpy as np

from uav_vpp_guidance.evaluation.legacy_hierarchical_policy import (
    LegacyStrategyBridgeConfig,
    body_to_inertial_rotation,
    build_legacy_high_level_obs,
    build_legacy_strategy_reference_point,
)


def _make_state(
    *,
    position,
    velocity,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
):
    return {
        "position_neu": np.asarray(position, dtype=np.float64),
        "velocity_vector_mps": np.asarray(velocity, dtype=np.float64),
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
        "yaw_rad": float(yaw),
        "vt_mps": float(np.linalg.norm(velocity)),
        "altitude_m": float(position[2]),
    }


def test_build_legacy_high_level_obs_returns_expected_shape():
    own_state = _make_state(position=[0.0, 0.0, 1000.0], velocity=[200.0, 0.0, 0.0])
    target_state = _make_state(
        position=[1000.0, 0.0, 1200.0],
        velocity=[180.0, 0.0, 0.0],
    )

    obs = build_legacy_high_level_obs(own_state, target_state)

    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs[0] > 0.0


def test_build_legacy_high_level_obs_marks_left_side_target_with_negative_phi():
    own_state = _make_state(position=[0.0, 0.0, 1000.0], velocity=[200.0, 0.0, 0.0])
    target_state = _make_state(
        position=[1000.0, -500.0, 1000.0],
        velocity=[180.0, 0.0, 0.0],
    )

    obs = build_legacy_high_level_obs(own_state, target_state)

    assert obs[1] < 0.0


def test_lag_strategy_reference_point_places_point_behind_target_body_axis():
    target_state = _make_state(
        position=[1000.0, 200.0, 1500.0],
        velocity=[200.0, 0.0, 0.0],
        yaw=np.pi / 2.0,
    )
    own_state = _make_state(position=[0.0, 0.0, 1500.0], velocity=[200.0, 0.0, 0.0])
    cfg = LegacyStrategyBridgeConfig(lag_distance=500.0)

    point = build_legacy_strategy_reference_point(0, own_state, target_state, cfg)
    rotation = body_to_inertial_rotation(0.0, 0.0, np.pi / 2.0)
    expected = target_state["position_neu"] + rotation @ np.array(
        [-500.0, 0.0, 0.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(point, expected)


def test_pure_strategy_reference_point_matches_target_position():
    target_state = _make_state(
        position=[1000.0, 200.0, 1500.0],
        velocity=[200.0, 0.0, 0.0],
    )
    own_state = _make_state(position=[0.0, 0.0, 1500.0], velocity=[200.0, 0.0, 0.0])

    point = build_legacy_strategy_reference_point(
        2,
        own_state,
        target_state,
        LegacyStrategyBridgeConfig(),
    )

    np.testing.assert_allclose(point, target_state["position_neu"])
