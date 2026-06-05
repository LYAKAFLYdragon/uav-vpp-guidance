"""Stage 6G.4R Merge Gate: CI hardening and xfail registry contract tests."""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCICompilesScripts(unittest.TestCase):
    """Ensure CI workflow compiles the scripts directory."""

    def test_ci_yaml_contains_scripts_compile(self):
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"
        self.assertTrue(ci_path.exists(), f"CI workflow not found at {ci_path}")
        content = ci_path.read_text(encoding="utf-8")
        self.assertIn(
            "compileall src/uav_vpp_guidance tests scripts",
            content,
            "CI must compile scripts/ directory to catch syntax errors early",
        )


class TestCriticalScriptsHaveHelp(unittest.TestCase):
    """Critical CLI scripts must expose --help without errors."""

    def _assert_help_ok(self, script_name: str):
        script_path = PROJECT_ROOT / "scripts" / script_name
        self.assertTrue(script_path.exists(), f"{script_name} not found")
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{script_name} --help failed:\n{result.stderr}",
        )
        self.assertIn("usage:", result.stdout.lower())

    def test_stage6g_guidance_limitation_probe_help(self):
        self._assert_help_ok("run_stage6g_guidance_limitation_probe.py")

    def test_stage6g4_smoke_probes_help(self):
        self._assert_help_ok("run_stage6g4_smoke_probes.py")


class TestXfailRegistryDocumented(unittest.TestCase):
    """XFail registry must be present, accurate, and list all categories."""

    def test_registry_file_exists(self):
        registry_path = PROJECT_ROOT / "docs" / "stage6g4r_xfail_registry.md"
        self.assertTrue(registry_path.exists(), "XFail registry markdown missing")

    def test_registry_contains_total_count(self):
        registry_path = PROJECT_ROOT / "docs" / "stage6g4r_xfail_registry.md"
        content = registry_path.read_text(encoding="utf-8")
        self.assertIn("68", content, "Registry must state the total xfailed count")
        normalized = content.lower().replace("**", "")
        self.assertTrue(
            "68 xfailed" in normalized or "total xfailed tests: 68" in normalized,
            "Registry must state the total xfailed count in a recognizable form",
        )

    def test_registry_contains_all_categories(self):
        registry_path = PROJECT_ROOT / "docs" / "stage6g4r_xfail_registry.md"
        content = registry_path.read_text(encoding="utf-8")
        categories = [
            "legacy_stage6f_runner_integration",
            "legacy_stage6f5_runner_analysis",
            "legacy_stage6f6_synthesis_artifacts",
            "legacy_stage6g_runner_evolved",
        ]
        for cat in categories:
            self.assertIn(
                cat,
                content,
                f"Registry must document category '{cat}'",
            )

    def test_registry_declares_triage_not_substitute(self):
        registry_path = PROJECT_ROOT / "docs" / "stage6g4r_xfail_registry.md"
        content = registry_path.read_text(encoding="utf-8")
        # Allow case-insensitive match for the key sentence
        self.assertTrue(
            "triage mechanism" in content.lower()
            and "not a long-term substitute" in content.lower(),
            "Registry must explicitly state that xfail is a triage mechanism, not a substitute for regression health",
        )


if __name__ == "__main__":
    unittest.main()
