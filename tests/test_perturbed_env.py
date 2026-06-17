"""Tests for PerturbedJSBSimEnv wind, sensor noise and actuator delay."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from uav_vpp_guidance.envs.perturbed_jsbsim_env import PerturbedJSBSimEnv


def _make_mock_jsbsim_env(num_aircraft: int = 1) -> MagicMock:
    """Build a mock that exposes the attributes PerturbedJSBSimEnv expects."""
    aircraft = {}
    for i in range(num_aircraft):
        ac = MagicMock()
        ac.get_property_value.return_value = 0.0
        ac.set_property_value = MagicMock()
        aircraft[f"ac{i}"] = ac

    jsbsim_env = MagicMock()
    jsbsim_env._aircraft = aircraft
    return jsbsim_env


class _MockBaseEnv:
    """Minimal stand-in for a wrapped environment."""

    def __init__(self, num_aircraft: int = 1):
        self.jsbsim_env = _make_mock_jsbsim_env(num_aircraft)
        self.max_steps = 32
        self._obs = {"observation_vector": np.zeros(16)}
        if num_aircraft == 2:
            self._target_uid = "ac1"
            self.reset = MagicMock(return_value=(self._obs, self._obs))
            self.step = MagicMock(
                return_value=(self._obs, self._obs, 0.0, 0.0, False, False, {})
            )
        else:
            self.reset = MagicMock(return_value=self._obs)
            self.step = MagicMock(
                return_value=(self._obs, 0.0, False, False, {})
            )


def _make_base_env(num_aircraft: int = 1):
    """Build a mock base env with a nested jsbsim_env."""
    return _MockBaseEnv(num_aircraft)


def _make_perturbation_config() -> Dict[str, Any]:
    return {
        "wind": {
            "enabled": True,
            "speed_std_mps": 5.0,
            "direction_std_deg": 30.0,
        },
        "sensor_noise": {
            "enabled": True,
            "position_noise_std_m": 10.0,
            "velocity_noise_std_mps": 5.0,
            "range_noise_std_m": 20.0,
            "angle_noise_std_deg": 1.0,
        },
        "actuator_delay": {
            "enabled": True,
            "min_delay_steps": 1,
            "max_delay_steps": 1,
        },
    }


def test_wind_properties_set_on_each_aircraft():
    base = _make_base_env(num_aircraft=2)
    pert_env = PerturbedJSBSimEnv(
        base, perturbation_config=_make_perturbation_config(), scale=1.0
    )
    pert_env.reset(seed=0)

    # Step with a dummy adversarial action tuple.
    pert_env.step((np.zeros(3), np.zeros(3)))

    for uid, ac in base.jsbsim_env._aircraft.items():
        calls = {c.args[0]: c.args[1] for c in ac.set_property_value.call_args_list}
        assert "atmosphere/total-wind-north-fps" in calls
        assert "atmosphere/total-wind-east-fps" in calls
        assert "atmosphere/total-wind-down-fps" in calls


def test_sensor_noise_is_component_wise():
    """Noise must change the observation vector and flag it."""
    base = _make_base_env(num_aircraft=1)
    base.reset.return_value = {
        "observation_vector": np.array(
            [
                0.5,   # range
                0.0,   # range_rate
                0.0,   # altitude_diff
                0.0,   # speed_diff
                0.0,   # los_azimuth_sin
                1.0,   # los_azimuth_cos
                0.0,   # los_elevation_sin
                1.0,   # los_elevation_cos
                0.0,   # ata_sin
                1.0,   # ata_cos
                0.0,   # aa_sin
                1.0,   # aa_cos
                0.5,   # own_speed
                0.5,   # target_speed
                0.5,   # own_altitude
                0.5,   # target_altitude
            ]
        )
    }
    pert_env = PerturbedJSBSimEnv(
        base, perturbation_config=_make_perturbation_config(), scale=1.0
    )
    obs = pert_env.reset(seed=42)

    assert obs["noise_applied"] is True
    assert not np.allclose(obs["observation_vector"], base.reset.return_value["observation_vector"])
    # Sin/cos pairs should remain roughly normalised (small noise).
    vec = obs["observation_vector"]
    for sin_idx, cos_idx in [(4, 5), (6, 7), (8, 9), (10, 11)]:
        norm = float(np.sqrt(vec[sin_idx] ** 2 + vec[cos_idx] ** 2))
        assert norm == pytest.approx(1.0, abs=0.05)


def test_delay_buffer_holds_action():
    """With a fixed 1-step delay, the first stepped action should be a neutral action."""
    base = _make_base_env(num_aircraft=1)
    pert_env = PerturbedJSBSimEnv(
        base, perturbation_config=_make_perturbation_config(), scale=1.0
    )
    pert_env.reset(seed=0)

    action = np.array([1.0, 2.0, 3.0])
    pert_env.step(action)
    # The base env should receive a neutral zero action on the first step.
    passed_action = base.step.call_args[0][0]
    np.testing.assert_allclose(passed_action, np.zeros(3), atol=1e-9)

    pert_env.step(action)
    # The second step should finally pass the original action.
    passed_action = base.step.call_args[0][0]
    np.testing.assert_allclose(passed_action, action)


def test_scale_zero_disables_perturbations():
    base = _make_base_env(num_aircraft=2)
    pert_env = PerturbedJSBSimEnv(
        base, perturbation_config=_make_perturbation_config(), scale=0.0
    )
    p_obs, t_obs = pert_env.reset(seed=0)
    # No noise should be applied to either agent's observation.
    assert p_obs.get("noise_applied") is not True
    assert t_obs.get("noise_applied") is not True

    pert_env.step((np.zeros(3), np.zeros(3)))
    # Wind should not be set when scale is 0.
    for ac in base.jsbsim_env._aircraft.values():
        ac.set_property_value.assert_not_called()
