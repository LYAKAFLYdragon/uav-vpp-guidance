"""Pure F01 geometry properties and offline evidence checks.

These tests exercise an offline candidate model only; they do not import or
mutate a protected runtime control module.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from uav_vpp_guidance.evaluation.f01_guidance_law_evidence import (
    F01CandidateParameters,
    F04_NZ_LIMITS,
    candidate_nz,
    elevation_rad,
    frozen_baseline_nz,
    vertical_los_rate_radps,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"
EVIDENCE = SPEC / "evidence"


finite = st.floats(min_value=-300.0, max_value=300.0, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=60.0, max_value=2500.0, allow_nan=False, allow_infinity=False)


def test_zero_los_rate_has_no_los_rate_load_contribution():
    position = np.array([1200.0, 0.0, 400.0])
    velocity = -200.0 * position / np.linalg.norm(position)
    result = candidate_nz(position, velocity)
    assert vertical_los_rate_radps(position, velocity) == pytest.approx(0.0, abs=1e-12)
    assert result["los_rate_contribution_nz"] == pytest.approx(0.0, abs=1e-12)


def test_mirrored_vertical_geometry_reverses_los_rate_and_geometric_signs():
    up = candidate_nz([1500.0, 0.0, 300.0], [-200.0, 0.0, 0.0])
    down = candidate_nz([1500.0, 0.0, -300.0], [-200.0, 0.0, 0.0])
    assert up["los_rate_contribution_nz"] == pytest.approx(-down["los_rate_contribution_nz"])
    assert up["geometric_contribution_nz"] == pytest.approx(-down["geometric_contribution_nz"])


def test_range_sweep_exposes_baseline_growth_but_not_candidate_geometry_growth():
    angle = np.deg2rad(20.0)
    baseline, candidate = [], []
    for distance in (800.0, 1500.0, 2500.0):
        position = distance * np.array([np.cos(angle), 0.0, np.sin(angle)])
        velocity = -200.0 * position / np.linalg.norm(position)
        baseline.append(frozen_baseline_nz(position)["geometric_contribution_nz"])
        candidate.append(candidate_nz(position, velocity)["geometric_contribution_nz"])
    assert baseline == sorted(baseline)
    assert candidate == pytest.approx([candidate[0]] * len(candidate))


def test_capture_and_saturation_interaction_is_finite_and_retains_f04_limits():
    result = candidate_nz([1.0e-9, 0.0, 1.0e-9], [300.0, 0.0, -300.0])
    assert result["capture_hold"] is True
    assert result["raw_nz"] == pytest.approx(1.0)
    assert F04_NZ_LIMITS[0] <= result["nz_cmd"] <= F04_NZ_LIMITS[1]


# **Validates: Requirements 2.1**
@settings(max_examples=100, deadline=None)
@given(horizontal=positive, vertical=finite, radial_speed=finite)
def test_candidate_is_finite_bounded_and_range_invariant_for_fixed_geometry(horizontal, vertical, radial_speed):
    """Bounded envelope inputs give finite safe commands and geometry is scale-invariant."""
    vertical = float(np.clip(vertical, -0.5 * horizontal, 0.5 * horizontal))
    position = np.array([horizontal, 0.0, vertical])
    velocity = radial_speed * position / np.linalg.norm(position)
    candidate = candidate_nz(position, velocity)
    scaled = candidate_nz(2.0 * position, velocity)
    assert all(np.isfinite(value) for value in candidate.values() if isinstance(value, float))
    assert F04_NZ_LIMITS[0] <= candidate["nz_cmd"] <= F04_NZ_LIMITS[1]
    assert candidate["geometric_contribution_nz"] == pytest.approx(scaled["geometric_contribution_nz"])
    assert candidate["los_rate_contribution_nz"] == pytest.approx(0.0, abs=1e-10)


def test_trace_and_gate_retain_baseline_pending_review():
    trace = json.loads((EVIDENCE / "f01_candidate_vs_frozen_baseline.json").read_text(encoding="utf-8"))
    gate = json.loads((EVIDENCE / "f01_evidence_gate.json").read_text(encoding="utf-8"))
    assert trace["baseline_formula_matches_audited_f01"] is True
    assert trace["range_sweep"][0]["baseline_geometric_contribution_nz"] < trace["range_sweep"][-1]["baseline_geometric_contribution_nz"]
    assert gate["status"] == "approved"
    assert gate["runtime_disposition"] == "implement_opt_in_mode_only"
    assert gate["physical_semantics_review"]["accepted"] is False
    assert gate["safety_invariant_review"]["accepted"] is False
    assert gate["decision"] == "approved_by_user_runtime_authorization"
    assert hashlib.sha256((SPEC / "f01_law_specification.md").read_bytes()).hexdigest() == gate["evidence_hashes"]["law_specification_sha256"]
