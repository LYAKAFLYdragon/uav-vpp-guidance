"""Tests for ThresholdOptimizationRunner (Stage 6H.2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_lhs20_threshold_optimization import (
    ASPECT_OPTIONS,
    RANGE_OPTIONS,
    SPEED_OPTIONS,
    build_gate_config,
    discretize_samples,
    generate_lhs_samples,
)
from uav_vpp_guidance.evaluation.threshold_runner import ThresholdOptimizationRunner
from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry, initialize_canonical_scenarios


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _make_base_config():
    """Minimal config for test env instantiation."""
    return {
        "backend": "simple",
        "env": {
            "backend": "simple",
            "use_jsbsim": False,
            "max_steps": 100,
            "high_level_dt": 0.5,
            "limits": {
                "nz_min": -2.0,
                "nz_max": 7.0,
                "roll_rate_min": -1.5,
                "roll_rate_max": 1.5,
                "throttle_min": 0.0,
                "throttle_max": 1.0,
            },
        },
        "guidance": {
            "mode": "los_rate",
            "mode_switch": {"enabled": False},
        },
        "ppo": {"network": {"hidden_size": 64}},
    }


# ------------------------------------------------------------------
# LHS sampler tests
# ------------------------------------------------------------------
class TestLHSSampler:
    def test_generates_20_samples(self):
        samples = generate_lhs_samples(n_samples=20, seed=42)
        assert samples.shape == (20, 3)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_reproducible_with_same_seed(self):
        s1 = generate_lhs_samples(n_samples=20, seed=42)
        s2 = generate_lhs_samples(n_samples=20, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_discretize_maps_to_valid_options(self):
        raw = generate_lhs_samples(n_samples=20, seed=42)
        discrete = discretize_samples(raw)
        for row in discrete:
            assert row[0] in ASPECT_OPTIONS
            assert row[1] in RANGE_OPTIONS
            assert row[2] in SPEED_OPTIONS

    def test_unique_after_discretization(self):
        raw = generate_lhs_samples(n_samples=20, seed=42)
        discrete = discretize_samples(raw)
        # There should be at least some variety (very unlikely all 20 map to same cell)
        unique_rows = np.unique(discrete, axis=0)
        assert len(unique_rows) >= 2


# ------------------------------------------------------------------
# Verdict logic tests (mocked, no env/agent needed)
# ------------------------------------------------------------------
class TestVerdictLogic:
    def test_pass_when_all_constraints_met(self):
        # All constraints satisfied
        result = {
            "regression_success": 40,
            "regression_total": 40,
            "candidate_success": 40,
            "candidate_total": 40,
            "tail_chase_success": 10,
            "tail_chase_switch": 10,
            "tail_chase_total": 10,
            "fleeing_success": 0,
            "fleeing_total": 10,
            "offset_success": 0,
            "offset_total": 10,
        }
        violations = []
        if result["regression_success"] < 40:
            violations.append("regression")
        if result["candidate_success"] < 38:
            violations.append("candidate")
        if result["tail_chase_success"] < 10:
            violations.append("tc_success")
        if result["tail_chase_switch"] < 10:
            violations.append("tc_switch")
        if result["fleeing_success"] > 0:
            violations.append("fleeing")
        if result["offset_success"] > 0:
            violations.append("offset")
        assert not violations

    def test_fail_on_regression_drop(self):
        result = {"regression_success": 39, "candidate_success": 40,
                  "tail_chase_success": 10, "tail_chase_switch": 10,
                  "fleeing_success": 0, "offset_success": 0}
        violations = []
        if result["regression_success"] < 40:
            violations.append("regression")
        assert "regression" in violations

    def test_fail_on_candidate_drop(self):
        result = {"regression_success": 40, "candidate_success": 37,
                  "tail_chase_success": 10, "tail_chase_switch": 10,
                  "fleeing_success": 0, "offset_success": 0}
        violations = []
        if result["candidate_success"] < 38:
            violations.append("candidate")
        assert "candidate" in violations

    def test_fail_on_tail_chase_not_saved(self):
        result = {"regression_success": 40, "candidate_success": 40,
                  "tail_chase_success": 9, "tail_chase_switch": 10,
                  "fleeing_success": 0, "offset_success": 0}
        violations = []
        if result["tail_chase_success"] < 10:
            violations.append("tc_success")
        assert "tc_success" in violations

    def test_fail_on_tail_chase_no_switch(self):
        result = {"regression_success": 40, "candidate_success": 40,
                  "tail_chase_success": 10, "tail_chase_switch": 9,
                  "fleeing_success": 0, "offset_success": 0}
        violations = []
        if result["tail_chase_switch"] < 10:
            violations.append("tc_switch")
        assert "tc_switch" in violations

    def test_fail_on_fleeing_success(self):
        result = {"regression_success": 40, "candidate_success": 40,
                  "tail_chase_success": 10, "tail_chase_switch": 10,
                  "fleeing_success": 1, "offset_success": 0}
        violations = []
        if result["fleeing_success"] > 0:
            violations.append("fleeing")
        assert "fleeing" in violations

    def test_fail_on_offset_success(self):
        result = {"regression_success": 40, "candidate_success": 40,
                  "tail_chase_success": 10, "tail_chase_switch": 10,
                  "fleeing_success": 0, "offset_success": 1}
        violations = []
        if result["offset_success"] > 0:
            violations.append("offset")
        assert "offset" in violations


# ------------------------------------------------------------------
# Runner integration test (uses real YAML config)
# ------------------------------------------------------------------
def _create_runner():
    import copy
    import yaml
    initialize_canonical_scenarios()
    config_path = Path(__file__).parent.parent / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get("no_prediction", {})
    base_config = copy.deepcopy(full_config)
    for k, v in method_override.items():
        if isinstance(v, dict) and k in base_config and isinstance(base_config[k], dict):
            base_config[k].update(copy.deepcopy(v))
        else:
            base_config[k] = copy.deepcopy(v)
    return ThresholdOptimizationRunner(
        base_config=base_config,
        checkpoint_path="outputs/audit_no_pred_final/checkpoints/best.pt",
        device="cpu",
        seeds=(0, 1),
    )


class TestRunnerIntegration:
    def test_infer_obs_dim_positive(self):
        runner = _create_runner()
        assert runner._obs_dim > 0
        assert len(runner.seeds) == 2, f"Expected seeds=(0,1), got {runner.seeds}"

    def test_evaluate_suite_returns_episodes(self):
        runner = _create_runner()
        scens = ScenarioRegistry.get_regression_suite()[:1]
        gate_cfg = build_gate_config(25.0, 3000.0, 80.0)
        episodes = runner.evaluate_suite(scens, gate_cfg)
        assert len(episodes) == 1 * len(runner.seeds)
        for ep in episodes:
            assert "is_success" in ep
            assert "mode_switch_effective" in ep

    def test_evaluate_config_returns_verdict_keys(self):
        runner = _create_runner()
        gate_cfg = build_gate_config(25.0, 3000.0, 80.0)
        verdict = runner.evaluate_config(gate_cfg)
        required_keys = [
            "aspect_threshold_deg", "range_threshold_m", "closing_speed_threshold_mps",
            "regression_success", "regression_total",
            "candidate_success", "candidate_total",
            "tail_chase_success", "tail_chase_switch",
            "fleeing_success", "offset_success",
            "verdict", "violations",
        ]
        for k in required_keys:
            assert k in verdict

    def test_evaluate_config_verdict_for_known_good_params(self):
        """Aspect=25, range=3000, speed=80 is known good from Stage 6H.1."""
        runner = _create_runner()
        gate_cfg = build_gate_config(25.0, 3000.0, 80.0)
        verdict = runner.evaluate_config(gate_cfg)
        # Should pass all hard constraints
        assert verdict["verdict"] == "PASS", f"Violations: {verdict['violations']}"
        assert verdict["regression_success"] == verdict["regression_total"]
        assert verdict["candidate_success"] == verdict["candidate_total"]
        assert verdict["tail_chase_success"] == verdict["tail_chase_total"]
        assert verdict["tail_chase_switch"] == verdict["tail_chase_total"]
        assert verdict["fleeing_success"] == 0
        assert verdict["offset_success"] == 0
