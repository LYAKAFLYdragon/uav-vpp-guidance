"""Tests for telemetry schema validator."""

import unittest

from uav_vpp_guidance.evaluation.telemetry_schema_validator import (
    validate_episode_telemetry,
    validate_episodes_telemetry,
    render_telemetry_validation_report,
    CORE_EPISODE_FIELDS,
)


class TestValidateEpisodeTelemetry(unittest.TestCase):
    def test_complete_core_passes(self):
        ep = {f: 0 for f in CORE_EPISODE_FIELDS}
        ok, crit, missing = validate_episode_telemetry(ep, require_core=True)
        self.assertTrue(ok)
        self.assertEqual(len(crit), 0)

    def test_missing_core_field_fails(self):
        ep = {f: 0 for f in CORE_EPISODE_FIELDS if f != "is_success"}
        ok, crit, missing = validate_episode_telemetry(ep, require_core=True)
        self.assertFalse(ok)
        self.assertIn("is_success", crit)

    def test_terminal_phase_missing_reported(self):
        ep = {f: 0 for f in CORE_EPISODE_FIELDS}
        ok, crit, missing = validate_episode_telemetry(
            ep, require_core=True, require_terminal_phase=True
        )
        self.assertTrue(ok)  # core still passes
        self.assertIn("terminal_phase", missing)
        self.assertIn("min_range_m", missing["terminal_phase"])

    def test_command_saturation_missing_reported(self):
        ep = {f: 0 for f in CORE_EPISODE_FIELDS}
        ok, crit, missing = validate_episode_telemetry(
            ep, require_core=True, require_command_saturation=True
        )
        self.assertTrue(ok)
        self.assertIn("command_saturation", missing)
        self.assertIn("nz_cmd_max", missing["command_saturation"])


class TestValidateEpisodesTelemetry(unittest.TestCase):
    def test_empty_episodes_fails(self):
        ok, report = validate_episodes_telemetry([])
        self.assertFalse(ok)
        self.assertIn("No episodes provided", report["critical_issues"])

    def test_homogeneous_episodes(self):
        eps = [{f: 0 for f in CORE_EPISODE_FIELDS} for _ in range(100)]
        ok, report = validate_episodes_telemetry(eps, sample_size=5)
        self.assertTrue(ok)
        self.assertEqual(report["sampled"], 5)

    def test_unavailable_categories_detected(self):
        eps = [{f: 0 for f in CORE_EPISODE_FIELDS} for _ in range(10)]
        ok, report = validate_episodes_telemetry(
            eps,
            require_command_saturation=True,
            require_altitude_energy=True,
        )
        self.assertTrue(ok)  # core passes
        self.assertIn("command_saturation", report["unavailable_categories"])
        self.assertIn("altitude_energy", report["unavailable_categories"])


class TestRenderReport(unittest.TestCase):
    def test_report_contains_unavailable_warning(self):
        eps = [{f: 0 for f in CORE_EPISODE_FIELDS} for _ in range(10)]
        _, report = validate_episodes_telemetry(
            eps, require_command_saturation=True
        )
        md = render_telemetry_validation_report(report)
        self.assertIn("Unavailable Categories", md)
        self.assertIn("command_saturation", md)
        self.assertIn("per-step telemetry", md)

    def test_report_shows_critical(self):
        eps = [{"scenario": "s1"}]  # Missing most core fields
        _, report = validate_episodes_telemetry(eps)
        md = render_telemetry_validation_report(report)
        self.assertIn("Critical Issues", md)
        self.assertIn("is_success", md)


if __name__ == "__main__":
    unittest.main()
