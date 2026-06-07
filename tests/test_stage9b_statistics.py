"""Tests for Stage 9B.1 statistical analysis script."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze_stage9b_statistics import (
    wilson_ci,
    bootstrap_ci,
    cohens_h,
    cohens_d_paired,
    classify_failure,
    compute_method_stats,
    compute_paired_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_raw_episodes(tmp_path):
    """Create a minimal raw_episodes.csv fixture for testing."""
    path = tmp_path / "raw_episodes.csv"
    rows = []
    rng = np.random.default_rng(42)
    for method in ["no_prediction", "cv_prediction"]:
        for scenario in ["regression_neutral", "candidate_head_on_close"]:
            for seed in [0, 1]:
                is_success = rng.random() > 0.3
                rows.append({
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "return": float(rng.normal(100 if is_success else -50, 20)),
                    "length": int(rng.integers(10, 50)),
                    "is_success": is_success,
                    "is_crash": not is_success and rng.random() > 0.5,
                    "is_timeout": False,
                    "is_out_of_bounds": False,
                    "reason": "success" if is_success else "crash",
                    "min_range_m": float(rng.uniform(200, 1000)),
                    "final_range_m": float(rng.uniform(200, 1000)),
                    "final_ata_deg": float(rng.uniform(0, 30)),
                    "nz_cmd_mean": float(rng.uniform(0.5, 1.5)),
                    "roll_rate_cmd_mean": float(rng.uniform(0, 0.01)),
                    "throttle_cmd_mean": float(rng.uniform(0.5, 0.9)),
                    "mode_switch_effective": False,
                    "effective_guidance_mode": "los_rate",
                    "prediction_enabled_rate": 0.0,
                    "prediction_valid_rate": 0.0,
                    "prediction_fallback_rate": 0.0,
                    "mean_prediction_error_m": float("nan"),
                    "mean_virtual_point_shift_m": float(rng.uniform(10, 100)),
                    "energy_proxy": float(rng.uniform(15000, 20000)),
                    "nz_cmd_modification_rate": float(rng.uniform(0.8, 0.99)),
                    "nz_cmd_saturation_rate": float(rng.uniform(0, 0.1)),
                })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def missing_column_csv(tmp_path):
    """CSV missing required columns."""
    path = tmp_path / "bad.csv"
    rows = [
        {"method": "no_prediction", "scenario": "regression_neutral", "seed": 0},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "scenario", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Unit tests for statistical helpers
# ---------------------------------------------------------------------------
class TestWilsonCI:
    def test_perfect_success(self):
        lo, hi = wilson_ci(10, 10)
        assert lo > 0.6
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_zero_success(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0
        assert hi < 0.4

    def test_n_zero(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0


class TestBootstrapCI:
    def test_known_mean(self):
        data = [1, 2, 3, 4, 5]
        lo, hi = bootstrap_ci(data, n_boot=5000, seed=42)
        assert lo < np.mean(data) < hi
        assert hi - lo < 3.0


class TestCohensH:
    def test_identical(self):
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-6)

    def test_direction(self):
        assert cohens_h(0.8, 0.2) > 0
        assert cohens_h(0.2, 0.8) < 0


class TestCohensDPaired:
    def test_zero_diff(self):
        assert cohens_d_paired([0, 0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_nonzero(self):
        d = cohens_d_paired([1, 2, 3, 4, 5])
        assert d > 1.0


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------
class TestFailureTaxonomy:
    def test_timeout_classified(self):
        row = pd.Series({"is_timeout": True, "is_crash": False, "is_out_of_bounds": False,
                         "final_range_m": 100, "scenario": "regression_neutral", "method": "no_prediction",
                         "mean_prediction_error_m": np.nan, "prediction_fallback_rate": 0.0,
                         "nz_cmd_modification_rate": 0.5, "nz_cmd_saturation_rate": 0.0,
                         "energy_proxy": 10000})
        res = classify_failure(row)
        assert "timeout" in res["root_cause"]
        assert res["supporting_telemetry"]

    def test_unclassified_not_silent(self):
        # Episode that somehow misses all heuristics — should still produce "unclassified"
        row = pd.Series({"is_timeout": False, "is_crash": False, "is_out_of_bounds": False,
                         "final_range_m": 50, "scenario": "regression_neutral", "method": "no_prediction",
                         "mean_prediction_error_m": np.nan, "prediction_fallback_rate": 0.0,
                         "nz_cmd_modification_rate": 0.1, "nz_cmd_saturation_rate": 0.0,
                         "energy_proxy": 10000})
        res = classify_failure(row)
        assert res["root_cause"] == "unclassified"
        assert "no heuristic matched" in res["supporting_telemetry"]

    def test_candidate_tail_chase(self):
        row = pd.Series({"is_timeout": False, "is_crash": True, "is_out_of_bounds": False,
                         "final_range_m": 100, "scenario": "candidate_head_on_close", "method": "no_prediction",
                         "mean_prediction_error_m": np.nan, "prediction_fallback_rate": 0.0,
                         "nz_cmd_modification_rate": 0.95, "nz_cmd_saturation_rate": 0.0,
                         "energy_proxy": 10000})
        res = classify_failure(row)
        assert "candidate_tail_chase" in res["root_cause"]
        assert "crash" in res["root_cause"]
        assert "unstable_command" in res["root_cause"]


# ---------------------------------------------------------------------------
# Integration: script run on minimal fixture
# ---------------------------------------------------------------------------
class TestAnalysisScriptIntegration:
    def test_script_runs_on_minimal_fixture(self, tmp_path, minimal_raw_episodes):
        output_dir = tmp_path / "stats"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_stage9b_statistics.py",
                "--input", str(minimal_raw_episodes),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        expected_files = [
            "statistics_summary.md",
            "statistics.json",
            "analysis_manifest.json",
            "failure_root_cause.csv",
            "figures/stage9b_success_rate_ci.png",
            "figures/stage9b_metric_return_ci.png",
            "figures/stage9b_metric_range_ci.png",
            "statistics_tables/method_summary.csv",
            "statistics_tables/pairwise_comparison.csv",
            "statistics_tables/stratified_results.csv",
            "statistics_tables/per_scenario_ranking.csv",
        ]
        for name in expected_files:
            assert (output_dir / name).exists(), f"Missing artifact: {name}"

        # Validate JSON structure
        stats = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))
        assert "methods" in stats
        assert "paired_comparisons" in stats
        assert "stratified" in stats
        assert "failure_taxonomy" in stats

        # Validate manifest
        manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_raw_episodes"]
        assert manifest["git_commit"]
        assert "analysis_script_command" in manifest

    def test_missing_columns_hard_fail(self, tmp_path, missing_column_csv):
        output_dir = tmp_path / "stats"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_stage9b_statistics.py",
                "--input", str(missing_column_csv),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Missing required columns" in (result.stdout + result.stderr)

    def test_empty_input_produces_valid_outputs(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("method,scenario,seed,return,length,is_success,is_crash,is_timeout,is_out_of_bounds,reason,min_range_m,final_range_m,final_ata_deg,nz_cmd_mean,roll_rate_cmd_mean,throttle_cmd_mean\n", encoding="utf-8")
        output_dir = tmp_path / "stats"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_stage9b_statistics.py",
                "--input", str(empty_csv),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (output_dir / "statistics_summary.md").exists()
        assert (output_dir / "analysis_manifest.json").exists()

    def test_failure_taxonomy_no_silent_unknown(self, tmp_path, minimal_raw_episodes):
        output_dir = tmp_path / "stats"
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_stage9b_statistics.py",
                "--input", str(minimal_raw_episodes),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        failure_df = pd.read_csv(output_dir / "failure_root_cause.csv")
        if not failure_df.empty:
            assert failure_df["root_cause"].notna().all(), "Some failures have NaN root_cause"
            assert (failure_df["root_cause"] != "").all(), "Some failures have empty root_cause"
            assert not (failure_df["root_cause"] == "unclassified").any(), "Unclassified failures detected"


# ---------------------------------------------------------------------------
# Computation correctness
# ---------------------------------------------------------------------------
class TestComputationCorrectness:
    def test_method_stats_fields(self, tmp_path, minimal_raw_episodes):
        df = pd.read_csv(minimal_raw_episodes)
        for method in df["method"].unique():
            s = compute_method_stats(df, method)
            assert "n_episodes" in s
            assert "success_rate" in s
            assert "success_rate_ci" in s
            assert "return" in s
            assert "min_range_m" in s
            assert s["success_rate_ci"][0] <= s["success_rate"] <= s["success_rate_ci"][1]

    def test_paired_comparison_structure(self, tmp_path, minimal_raw_episodes):
        df = pd.read_csv(minimal_raw_episodes)
        res = compute_paired_comparison(df, "no_prediction", "cv_prediction", "return")
        assert "n_pairs" in res
        assert "mean_diff" in res
        assert "ci_lo" in res
        assert "ci_hi" in res
        assert "p_value" in res
        assert "effect_size" in res
        assert res["ci_lo"] <= res["mean_diff"] <= res["ci_hi"] or math.isnan(res["ci_lo"])
