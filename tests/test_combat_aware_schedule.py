"""
Tests for the combat-aware intentional update schedule.
"""

import numpy as np
import pytest

from uav_vpp_guidance.ablations.cais_only.combat_aware_schedule import CombatAwareSchedule


class TestCombatAwareSchedule:
    def _make_schedule(self, **overrides):
        config = {
            "eta_actor": 0.01,
            "eta_critic": 0.1,
            "range_thresholds_m": [3000.0, 6000.0],
            "terminal_range_m": 1200.0,
            "aspect_threshold_deg": 30.0,
        }
        config.update(overrides)
        return CombatAwareSchedule(config)

    def test_far_range_is_search_approach(self):
        sched = self._make_schedule()
        phase = sched.classify({"range_m": 7000.0, "aa_rad": 0.0})
        assert phase == "search_approach"
        scales = sched.get_eta_scales({"range_m": 7000.0, "aa_rad": 0.0})
        assert scales["actor"] == pytest.approx(1.5)
        assert scales["critic"] == pytest.approx(1.0)

    def test_terminal_range_overrides_aspect(self):
        sched = self._make_schedule()
        phase = sched.classify({"range_m": 500.0, "aa_rad": 0.0})
        assert phase == "terminal"

    def test_advantage_and_disadvantage(self):
        sched = self._make_schedule()
        adv = sched.classify({"range_m": 2000.0, "aa_rad": 0.0})
        dis = sched.classify({"range_m": 2000.0, "aa_rad": np.pi})
        assert adv == "advantage_position"
        assert dis == "disadvantage_defense"

    def test_batch_scales(self):
        sched = self._make_schedule()
        features = [
            {"range_m": 7000.0, "aa_rad": 0.0},
            {"range_m": 2000.0, "aa_rad": 0.0},
        ]
        scales = sched.get_batch_scales(features)
        # mean of (1.5, 0.5) for actor, (1.0, 0.5) for critic
        assert scales["actor"] == pytest.approx(1.0)
        assert scales["critic"] == pytest.approx(0.75)

    def test_batch_eta(self):
        sched = self._make_schedule()
        features = [
            {"range_m": 7000.0, "aa_rad": 0.0},
            {"range_m": 2000.0, "aa_rad": 0.0},
        ]
        eta = sched.get_batch_eta(features)
        assert eta["actor"] == pytest.approx(0.01 * 1.0)
        assert eta["critic"] == pytest.approx(0.1 * 0.75)

    def test_array_input(self):
        sched = self._make_schedule()
        # Build a (2, F) array using FEATURE_KEYS order
        arr = np.zeros((2, len(sched.FEATURE_KEYS)), dtype=np.float32)
        arr[0, 0] = 7000.0   # range_m
        arr[1, 0] = 500.0    # range_m -> terminal
        scales = sched.get_batch_scales(arr)
        # phases: search_approach (actor 1.5) and terminal (actor 0.3)
        assert scales["actor"] == pytest.approx(0.9)

    def test_custom_phase_scales(self):
        sched = self._make_schedule(
            phase_scales={"search_approach": {"actor": 2.0, "critic": 2.0}}
        )
        scales = sched.get_eta_scales({"range_m": 7000.0, "aa_rad": 0.0})
        assert scales["actor"] == 2.0
        assert scales["critic"] == 2.0
