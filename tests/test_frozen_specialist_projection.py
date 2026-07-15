from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.frozen_specialist_projection import (
    CROSSING_LEGACY_FEATURES,
    FrozenSpecialistProjectionError,
    action_for_frozen_specialist,
    project_crossing_observation,
)


V5_NAMES = CROSSING_LEGACY_FEATURES + ("is_crossing",)


class _Policy:
    def __init__(self, obs_dim: int) -> None:
        self.obs_dim = obs_dim
        self.seen = None

    def get_deterministic_action(self, observation):
        self.seen = np.asarray(observation, dtype=np.float32)
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


def test_crossing_projection_drops_only_the_declared_task_bit():
    source = np.arange(19, dtype=np.float32)
    projected, receipt = project_crossing_observation(source, V5_NAMES, 18)

    assert np.array_equal(projected, source[:18])
    assert receipt["dropped_feature_names"] == ["is_crossing"]
    assert receipt["implicit_truncation_or_padding"] is False


def test_frozen_specialists_receive_exact_widths_without_implicit_adaptation():
    schema = {"feature_names": list(V5_NAMES)}
    source = np.arange(19, dtype=np.float32)
    head, crossing = _Policy(19), _Policy(18)

    head_action, head_receipt = action_for_frozen_specialist(head, source, schema, "frozen_fixed_head_on")
    crossing_action, crossing_receipt = action_for_frozen_specialist(crossing, source, schema, "frozen_fixed_crossing")

    assert head.seen.shape == (19,)
    assert crossing.seen.shape == (18,)
    assert head_action.shape == crossing_action.shape == (3,)
    assert head_receipt["adapter"] == "identity_exact_schema_v1"
    assert crossing_receipt["adapter"] == "explicit_drop_is_crossing_task_bit_v1"


def test_projection_rejects_any_undeclared_schema_or_checkpoint_drift():
    with pytest.raises(FrozenSpecialistProjectionError, match="18-D"):
        project_crossing_observation(np.zeros(19), V5_NAMES, 19)
    with pytest.raises(FrozenSpecialistProjectionError, match="declared 19-D"):
        project_crossing_observation(np.zeros(19), tuple(reversed(V5_NAMES)), 18)
