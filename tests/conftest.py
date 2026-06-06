"""Pytest configuration for uav-vpp-guidance test suite.

Stage 6H.0-F: Add project root to sys.path so `from scripts.xxx import ...`
works in tests when scripts/ lacks __init__.py.

Stage 6G.5D-R: All legacy xfail markers cleared.
- PREEXISTING_FAILURES is intentionally empty.
- The 68 pre-existing failures from Stage 6G.4R baseline (b246391)
  were incrementally fixed across stages 6G.4–6G.5D.
- Audit report: docs/results/stage6g5d_xpass_audit.md
- New failures must be fixed, not xfailed.
"""

import os
import sys

# Allow `from scripts.xxx import ...` in tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# ------------------------------------------------------------------
# Pre-existing failure registry (CLEARED in Stage 6G.5D-R)
#
# History: Stage 6G.4R introduced auto-xfail for 68 legacy failures
# verified on baseline b246391. Between 6G.4R and 6G.5D, all
# underlying issues were incrementally fixed. The 68 tests now pass
# consistently (xpassed → normal pass).
#
# Audit report: docs/results/stage6g5d_xpass_audit.md
# ------------------------------------------------------------------
PREEXISTING_FAILURES = {}

# Classification for reporting
CLASSIFICATION = {
    "tests/test_comparison_contract.py": "legacy_stage6f_runner_integration",
    "tests/test_stage6f5_reablation.py": "legacy_stage6f5_runner_analysis",
    "tests/test_stage6f6_synthesis.py": "legacy_stage6f6_synthesis_artifacts",
    "tests/test_stage6g_guidance_probe.py": "legacy_stage6g_runner_evolved",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        nodeid = item.nodeid
        if nodeid in PREEXISTING_FAILURES:
            module = nodeid.split("::")[0]
            category = CLASSIFICATION.get(module, "legacy_unknown")
            item.add_marker(
                pytest.mark.xfail(
                    reason=f"pre-existing failure ({category}); verified on baseline b246391",
                    strict=False,
                )
            )
