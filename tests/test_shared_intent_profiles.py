from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.hierarchy.shared_intent_profiles import PROFILE_NAMES, compile_profile, get_profile


REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_profile_yaml_freezes_the_same_targets_and_weights_as_the_compiler():
    path = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_profiles.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert tuple(config["profiles"]) == PROFILE_NAMES
    for name, declared in config["profiles"].items():
        profile = get_profile(name)
        assert tuple(declared["targets"]) == profile.targets
        assert tuple(declared["weights"]) == profile.weights
