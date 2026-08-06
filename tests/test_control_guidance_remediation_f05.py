"""Deterministic offline F05 VPP semantic evidence tests."""
import json
from pathlib import Path
import pytest
from uav_vpp_guidance.evaluation.f05_vpp_semantics_evidence import compare_semantics, distance_scaled, fixed_metre
ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"

def test_fixed_mapping_matches_configured_800m_bounds_and_expected_angles():
    report = compare_semantics()
    assert fixed_metre(2.0) == 800.0 and fixed_metre(-2.0) == -800.0
    assert report["baseline_examples"]["fixed_800m_at_2500m_deg"] == pytest.approx(17.7447, rel=1e-4)
    assert report["baseline_examples"]["fixed_800m_at_800m_deg"] == pytest.approx(45.0)

def test_distance_scaled_mapping_is_finite_bounded_monotone_and_symmetric():
    values = [distance_scaled(a, 800.0) for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(-values[-1])
    assert all(abs(value) <= 800.0 for value in values)
    assert distance_scaled(1.0, 0.0) == distance_scaled(1.0, 50.0)

def test_f05_gate_preserves_fixed_metre_semantics_pending_evaluation_and_migration():
    gate = json.loads((SPEC / "evidence/f05_vpp_semantics_bundle/f05_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["runtime_disposition"] == "preserve_fixed_metre_cartesian_offset"
    assert gate["protected_path_allowlist"] == []
