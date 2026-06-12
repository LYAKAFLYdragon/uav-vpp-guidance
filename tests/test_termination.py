"""
Unit tests for termination checker.
"""

from uav_vpp_guidance.envs.termination import TerminationChecker


class TestTerminationChecker:
    def test_init(self):
        config = {"success_hold_time_s": 0.2, "decision_freq": 5}
        checker = TerminationChecker(config)
        assert checker._success_hold_steps == 1  # 0.2 * 5 = 1.0 -> int(1.0) = 1

    def test_reset(self):
        config = {}
        checker = TerminationChecker(config)
        checker._success_counter = 5
        checker.reset()
        assert checker._success_counter == 0

    def test_check_crash_low_altitude(self):
        checker = TerminationChecker(config={"min_altitude_m": 500.0, "max_high_level_steps": 100})
        own_state = {"altitude_m": 400.0}
        metrics = {"range_m": 2000.0, "ata_rad": 0.5}
        done, info = checker.check(own_state, {}, metrics, step=0)
        assert done is True
        assert info["is_crash"] is True
        assert info["reason"] == "crash"

    def test_check_success(self):
        checker = TerminationChecker(config={
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.0,
            "decision_freq": 5,
            "max_high_level_steps": 100,
        })
        own_state = {"altitude_m": 5000.0}
        metrics = {"range_m": 800.0, "ata_rad": 0.0}
        done, info = checker.check(own_state, {}, metrics, step=0)
        assert done is True
        assert info["is_success"] is True
        assert info["reason"] == "success"

    def test_check_timeout(self):
        checker = TerminationChecker(config={"max_high_level_steps": 10})
        own_state = {"altitude_m": 5000.0}
        metrics = {"range_m": 2000.0, "ata_rad": 1.0}
        done, info = checker.check(own_state, {}, metrics, step=10)
        assert done is True
        assert info["is_timeout"] is True
        assert info["reason"] == "timeout"

    def test_check_not_done(self):
        checker = TerminationChecker(config={"max_high_level_steps": 100})
        own_state = {"altitude_m": 5000.0}
        metrics = {"range_m": 2000.0, "ata_rad": 1.0}
        done, info = checker.check(own_state, {}, metrics, step=0)
        assert done is False
        assert info["reason"] is None

    def test_set_success_criteria_updates_hold_steps(self):
        checker = TerminationChecker(config={"success_hold_time_s": 0.2, "decision_freq": 5})
        assert checker._success_hold_steps == 1
        checker.set_success_criteria(success_hold_time_s=1.0)
        assert checker._success_hold_steps == 5
        assert checker.success_range_m == 900.0  # unchanged

    def test_set_success_criteria_resets_counter(self):
        checker = TerminationChecker(config={
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.0,
            "decision_freq": 5,
            "max_high_level_steps": 100,
        })
        own_state = {"altitude_m": 5000.0}
        metrics = {"range_m": 800.0, "ata_rad": 0.0}
        checker.check(own_state, {}, metrics, step=0)
        assert checker._success_counter == 1
        checker.set_success_criteria(success_range_m=700.0)
        assert checker._success_counter == 0
