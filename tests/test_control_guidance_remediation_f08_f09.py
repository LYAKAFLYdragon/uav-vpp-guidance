"""Deterministic F08/F09 evidence tests; no optimizer or training behavior changes."""
import json
from pathlib import Path

from uav_vpp_guidance.evaluation.f08_f09_cem_evidence import compare_fixed_budget, f09_metric_definition

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


def test_f08_comparison_is_seed_controlled_5d_and_fixed_budget():
    report = compare_fixed_budget()
    assert report["budget_objective_evaluations"] == 120
    assert report["seed_list"] == [0, 1, 2]
    assert len(report["plans"]["frozen_12_candidates_3_elites_diagonal"]) == 3
    for summary in report["summary"].values():
        assert summary["all_fixed_budget_evaluations"]


def test_f08_baseline_preserves_active_population_and_elite_count():
    baseline = compare_fixed_budget()["plans"]["frozen_12_candidates_3_elites_diagonal"][0]
    assert baseline["plan"]["candidates"] == 12
    assert baseline["plan"]["elite_ratio"] == 0.25
    assert baseline["plan"]["covariance"] == "diagonal"
    assert baseline["evaluations"] == 120


def test_f09_is_labeled_as_within_batch_empirical_gap_not_paired_regret_or_rollback():
    metric = f09_metric_definition()
    assert metric["truthful_label"] == "within-batch empirical score gap to the best sampled candidate"
    assert not metric["is_paired_regret"]
    assert not metric["is_rollback_signal"]
    assert not metric["training_behavior_change"]


def test_f08_f09_gate_keeps_cem_and_training_behavior_unchanged():
    gate = json.loads((SPEC / "evidence/f08_f09_evidence_bundle/f08_f09_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["protected_path_allowlist"] == []
    assert gate["runtime_disposition"] == "preserve_cem_and_bilevel_training_behavior"
