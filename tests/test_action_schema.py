"""Tests for the unified action-space schema validator."""

import numpy as np
import pytest

from uav_vpp_guidance.utils.action_schema import (
    ActionSchema,
    infer_action_schema,
    split_action,
    validate_action,
)


def test_infer_3d():
    schema = infer_action_schema(np.zeros(3))
    assert schema == ActionSchema.PPO_3D


def test_infer_4d():
    schema = infer_action_schema(np.zeros(4))
    assert schema == ActionSchema.PPO_PID_4D
    assert schema.has_aggressiveness


def test_infer_6d():
    schema = infer_action_schema(np.zeros(6))
    assert schema == ActionSchema.APIC_PID_6D
    assert schema.is_apic


def test_reject_5d():
    with pytest.raises(ValueError):
        infer_action_schema(np.zeros(5))


def test_validate_expected_dim():
    validate_action(np.zeros(4), expected_dim=4)
    with pytest.raises(ValueError):
        validate_action(np.zeros(3), expected_dim=4)


def test_split_action():
    action = np.array([0.1, 0.2, 0.3, 0.5])
    vpp, agg = split_action(action)
    assert np.allclose(vpp, [0.1, 0.2, 0.3])
    assert agg == pytest.approx(0.5)


def test_split_action_clips_aggressiveness():
    action = np.array([0.0, 0.0, 0.0, 2.0])
    _, agg = split_action(action)
    assert agg == pytest.approx(1.0)
