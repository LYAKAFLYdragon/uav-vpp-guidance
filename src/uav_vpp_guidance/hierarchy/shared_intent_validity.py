"""Auditable validity mask for shared skill and intent-profile actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .shared_intent_profiles import PROFILE_NAMES


SKILL_NAMES: Tuple[str, ...] = (
    "pursuit_conversion",
    "lead_intercept",
    "defensive_extension",
    "reentry_recovery",
)


@dataclass(frozen=True)
class TacticalContext:
    geometry_state: str
    phase: str
    energy_state: str = "balanced"
    altitude_state: str = "co_altitude"


def _profile_indices(names: Tuple[str, ...]) -> set[int]:
    return {PROFILE_NAMES.index(name) for name in names}


_BASE_ALLOWED = {
    "pursuit_conversion": _profile_indices(("front_intercept", "rear_quarter_alignment", "lateral_displacement", "reentry_preparation")),
    "lead_intercept": _profile_indices(("front_intercept", "rear_quarter_alignment", "lateral_displacement")),
    "defensive_extension": _profile_indices(("lateral_displacement", "defensive_break", "range_extension", "energy_altitude_recovery", "reentry_preparation")),
    "reentry_recovery": _profile_indices(("lateral_displacement", "range_extension", "energy_altitude_recovery", "reentry_preparation")),
}


def build_validity_mask(context: TacticalContext) -> Dict[str, object]:
    """Return an allowed skill/profile matrix and reasons for excluded pairs."""

    if context.geometry_state not in {"advantage", "head_on", "disadvantage", "neutral", "crossing_entry"}:
        raise ValueError(f"Unknown geometry_state: {context.geometry_state}")
    if context.phase not in {"pre_merge", "post_merge", "re_entry"}:
        raise ValueError(f"Unknown phase: {context.phase}")
    if context.energy_state not in {"advantage", "balanced", "disadvantage"}:
        raise ValueError(f"Unknown energy_state: {context.energy_state}")

    mask = np.zeros((len(SKILL_NAMES), len(PROFILE_NAMES)), dtype=bool)
    reasons: Dict[str, str] = {}
    for skill_index, skill in enumerate(SKILL_NAMES):
        for profile_index, profile in enumerate(PROFILE_NAMES):
            allowed = profile_index in _BASE_ALLOWED[skill]
            reason = "base_contract"
            if context.phase == "pre_merge" and skill == "reentry_recovery":
                allowed, reason = False, "reentry_skill_requires_post_merge"
            if context.geometry_state == "disadvantage" and skill in {"pursuit_conversion", "lead_intercept"}:
                allowed, reason = False, "offensive_skill_blocked_in_disadvantage"
            if context.energy_state == "disadvantage" and profile in {"front_intercept", "rear_quarter_alignment"}:
                allowed, reason = False, "offensive_profile_blocked_when_energy_disadvantaged"
            if context.phase == "re_entry" and profile == "front_intercept":
                allowed, reason = False, "front_intercept_deferred_until_reentry_closure"
            if context.geometry_state == "crossing_entry" and skill == "lead_intercept":
                allowed = allowed and True
                reason = "crossing_lead_allowed"
            mask[skill_index, profile_index] = allowed
            if not allowed:
                reasons[f"{skill}:{profile}"] = reason
    if not bool(mask.any()):  # Defensive invariant for any future rule change.
        raise AssertionError("Validity mask removed every action")
    return {"mask": mask, "reasons": reasons, "context": context}
