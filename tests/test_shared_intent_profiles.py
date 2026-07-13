from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.hierarchy.shared_intent_profiles import PROFILE_NAMES, compile_profile


def test_all_global_profiles_compile_to_finite_target_and_weight_slots():
    assert len(PROFILE_NAMES) == 7
    for name in PROFILE_NAMES:
        compiled = compile_profile(name)
        assert compiled["target_vector"].shape == (6,)
        assert compiled["weight_vector"].shape == (6,)
        assert np.all(np.isfinite(compiled["target_vector"]))
        assert np.all((compiled["weight_vector"] >= 0.0) & (compiled["weight_vector"] <= 1.0))


def test_unknown_profile_is_rejected():
    with pytest.raises(KeyError, match="Unknown tactical-intent profile"):
        compile_profile("task_id_disadvantage")
