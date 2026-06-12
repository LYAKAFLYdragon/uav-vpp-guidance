"""
Tests for the curriculum scheduler.
"""

import pytest

from uav_vpp_guidance.training.curriculum import CurriculumScheduler


class TestCurriculumScheduler:
    def test_default_stages_present(self):
        sched = CurriculumScheduler()
        assert len(sched.stages) > 0
        assert sched.current_level == 0
        assert "favorable" in sched.allowed_scenario_names

    def test_uniform_weights_for_allowed_scenarios(self):
        sched = CurriculumScheduler()
        all_scenarios = {
            "favorable": {},
            "neutral": {},
            "disadvantage": {},
            "challenging": {},
        }
        weights = sched.get_current_scenario_weights(all_scenarios)
        assert weights["favorable"] == weights["neutral"]
        assert weights["disadvantage"] == 0.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_custom_weights(self):
        config = {
            "stages": [
                {
                    "name": "stage0",
                    "scenario_names": ["a", "b"],
                    "success_threshold": 1.0,
                    "min_episodes": 0,
                    "weights": {"a": 0.8, "b": 0.2},
                }
            ],
            "gate_mode": "min",
        }
        sched = CurriculumScheduler(config)
        weights = sched.get_current_scenario_weights({"a": {}, "b": {}, "c": {}})
        assert weights["a"] == pytest.approx(0.8)
        assert weights["b"] == pytest.approx(0.2)
        assert weights["c"] == 0.0

    def test_advance_when_gate_passed(self):
        config = {
            "stages": [
                {
                    "name": "stage0",
                    "scenario_names": ["favorable"],
                    "success_threshold": 0.5,
                    "min_episodes": 10,
                },
                {
                    "name": "stage1",
                    "scenario_names": ["favorable", "challenging"],
                    "success_threshold": 0.3,
                    "min_episodes": 5,
                },
            ],
            "gate_mode": "min",
        }
        sched = CurriculumScheduler(config)
        assert sched.current_stage_name == "stage0"

        # Not enough episodes yet
        advanced = sched.update({"n_episodes": 5, "per_scenario_success_rates": {"favorable": 0.9}})
        assert not advanced

        # Enough episodes and gate passed
        advanced = sched.update({"n_episodes": 5, "per_scenario_success_rates": {"favorable": 0.9}})
        assert advanced
        assert sched.current_stage_name == "stage1"
        assert "challenging" in sched.allowed_scenario_names

    def test_do_not_advance_if_gate_failed(self):
        config = {
            "stages": [
                {
                    "name": "stage0",
                    "scenario_names": ["favorable"],
                    "success_threshold": 0.5,
                    "min_episodes": 0,
                },
                {
                    "name": "stage1",
                    "scenario_names": ["challenging"],
                    "success_threshold": 0.3,
                    "min_episodes": 0,
                },
            ],
            "gate_mode": "min",
        }
        sched = CurriculumScheduler(config)
        advanced = sched.update({"n_episodes": 1, "per_scenario_success_rates": {"favorable": 0.2}})
        assert not advanced
        assert sched.current_stage_name == "stage0"

    def test_state_dict_roundtrip(self):
        config = {
            "stages": [
                {
                    "name": "stage0",
                    "scenario_names": ["favorable"],
                    "success_threshold": 0.5,
                    "min_episodes": 0,
                },
                {
                    "name": "stage1",
                    "scenario_names": ["challenging"],
                    "success_threshold": 0.3,
                    "min_episodes": 0,
                },
            ],
            "gate_mode": "min",
        }
        sched = CurriculumScheduler(config)
        sched.update({"n_episodes": 1, "per_scenario_success_rates": {"favorable": 1.0}})
        state = sched.state_dict()

        sched2 = CurriculumScheduler(config)
        sched2.load_state_dict(state)
        assert sched2.current_level == sched.current_level
        assert sched2.episodes_in_stage == sched.episodes_in_stage
        assert sched2.total_episodes == sched.total_episodes

    def test_mean_gate_mode(self):
        config = {
            "stages": [
                {
                    "name": "stage0",
                    "scenario_names": ["a", "b"],
                    "success_threshold": 0.6,
                    "min_episodes": 0,
                },
                {
                    "name": "stage1",
                    "scenario_names": ["c"],
                    "success_threshold": 0.3,
                    "min_episodes": 0,
                },
            ],
            "gate_mode": "mean",
        }
        sched = CurriculumScheduler(config)
        # mean SR = 0.65 -> should advance to stage1
        advanced = sched.update({
            "n_episodes": 1,
            "per_scenario_success_rates": {"a": 0.8, "b": 0.5},
        })
        assert advanced
        assert sched.current_level == len(sched.stages) - 1
