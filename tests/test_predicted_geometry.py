import numpy as np

from uav_vpp_guidance.envs.observation import build_observation
from uav_vpp_guidance.trajectory_prediction.predicted_geometry import (
    compute_predicted_geometry,
    normalize_for_base_observation,
)


def _state(position, velocity):
    return {
        "position_m": np.asarray(position, dtype=np.float64),
        "velocity_vector_mps": np.asarray(velocity, dtype=np.float64),
    }


def test_predicted_geometry_uses_estimated_velocity_from_predicted_position():
    own = _state([0, 0, 0], [100, 0, 0])
    target = _state([1000, 0, 0], [100, 0, 0])
    predicted = np.array([1200, 200, 0], dtype=np.float64)

    geometry = compute_predicted_geometry(own, target, predicted, 1.0)

    assert geometry["predicted_range_m"] == np.sqrt(1200**2 + 200**2)
    assert geometry["predicted_range_rate_mps"] > 0.0
    assert geometry["predicted_target_velocity_mps"] == np.sqrt(200**2 + 200**2)


def test_predicted_geometry_does_not_depend_on_target_future_state():
    own = _state([0, 0, 0], [100, 0, 0])
    target_a = _state([1000, 0, 0], [100, 0, 0])
    target_b = _state([1000, 0, 0], [-9000, 7000, 0])
    predicted = [1200, 0, 0]

    first = compute_predicted_geometry(own, target_a, predicted, 1.0)
    second = compute_predicted_geometry(own, target_b, predicted, 1.0)

    assert first == second


def test_replacement_preserves_base_observation_dimension_and_names():
    own = _state([0, 0, 5000], [200, 0, 0])
    target = _state([2000, 0, 5000], [-200, 0, 0])
    current, names = build_observation(own, target, return_feature_names=True)
    geometry = compute_predicted_geometry(own, target, [1800, 300, 5000], 1.0)
    replacement = normalize_for_base_observation(geometry)
    predicted, predicted_names = build_observation(
        own,
        target,
        predicted_geometry=replacement,
        replace_geometry_with_prediction=True,
        return_feature_names=True,
    )

    assert current.shape == (16,)
    assert predicted.shape == (16,)
    assert names == predicted_names
    assert not np.array_equal(current, predicted)
    assert np.isclose(predicted[0], replacement["range_m"])
    assert np.isclose(predicted[1], replacement["range_rate_mps"])
