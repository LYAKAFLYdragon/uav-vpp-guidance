from __future__ import annotations

from uav_vpp_guidance.hierarchy.shared_intent_profiles import PROFILE_NAMES
from uav_vpp_guidance.hierarchy.shared_intent_validity import SKILL_NAMES, TacticalContext, build_validity_mask


def _allowed(result, skill, profile):
    return bool(result["mask"][SKILL_NAMES.index(skill), PROFILE_NAMES.index(profile)])


def test_validity_mask_keeps_at_least_one_action_for_each_core_context():
    for geometry in ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry"):
        result = build_validity_mask(TacticalContext(geometry, "pre_merge"))
        assert result["mask"].shape == (4, 7)
        assert result["mask"].any()


def test_validity_mask_blocks_obviously_offensive_disadvantage_pairs():
    result = build_validity_mask(TacticalContext("disadvantage", "post_merge", "disadvantage"))

    assert not _allowed(result, "lead_intercept", "front_intercept")
    assert _allowed(result, "defensive_extension", "energy_altitude_recovery")
    assert _allowed(result, "reentry_recovery", "reentry_preparation")


def test_validity_mask_keeps_crossing_lead_and_reentry_recovery_available():
    crossing = build_validity_mask(TacticalContext("crossing_entry", "pre_merge"))
    reentry = build_validity_mask(TacticalContext("neutral", "re_entry"))

    assert _allowed(crossing, "lead_intercept", "front_intercept")
    assert _allowed(reentry, "reentry_recovery", "reentry_preparation")
