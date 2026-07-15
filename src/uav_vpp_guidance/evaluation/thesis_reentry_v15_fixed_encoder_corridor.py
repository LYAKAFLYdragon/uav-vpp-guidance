"""V15 fixed-encoder profile helpers for the temporal-corridor probe.

The 66-D observation contract requires one intent target/weight vector.  V15
keeps that vector fixed while recording a separate time-indexed objective and
semantic action profile.  This prevents a dynamic evaluation corridor from
silently changing the observation schema inherited by the frozen collector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile

from uav_vpp_guidance.evaluation.thesis_reentry_v14_temporal_corridor import (
    METHODS,
    OPPONENTS,
    REFERENCE_METHOD,
    is_ego_failure,
    objective_profile,
    scheduled_action,
    summarize_records,
)


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-FIXED-ENCODER-TEMPORAL-CORRIDOR-V15"


class FixedEncoderCorridorError(ValueError):
    """Raised when the fixed encoder-profile contract is malformed."""


def compose_fixed_encoder_observation(
    *,
    contract: FiveStateObservationContract,
    base_geometry: Any,
    explicit_history: Any,
    temporal_embedding: Any,
    encoder_profile: str,
) -> np.ndarray:
    """Compose 66-D input with a declared static profile anchor.

    `encoder_profile` is deliberately independent of the profile that scores
    or actuates any particular temporal-corridor step.
    """

    profile = compile_profile(str(encoder_profile))
    value = contract.compose(
        base_geometry=np.asarray(base_geometry, dtype=np.float32),
        explicit_history=np.asarray(explicit_history, dtype=np.float32),
        intent_targets=profile["target_vector"],
        intent_weights=profile["weight_vector"],
        temporal_embedding=np.asarray(temporal_embedding, dtype=np.float32),
    )
    if value.shape != (66,) or not np.isfinite(value).all():
        raise FixedEncoderCorridorError("V15 fixed-profile 66-D contract failed")
    return value
