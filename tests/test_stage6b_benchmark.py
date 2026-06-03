"""
Tests for Stage 6B simple-backend prediction benchmark.

Covers:
- Benchmark config loading
- Smoke benchmark execution
- Output file presence and format
- Unified metrics fields
- Statistical comparison robustness
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.evaluation.statistical_comparison import (
    mean_std,
    bootstrap_ci,
    paired_delta,
    compare_methods,
    compare_per_scenario,
)


class TestBenchmarkConfig:
    def test_config_loads(self):
        config = load_yaml_config("config/experiment/benchmark_simple_prediction_comparison.yaml")
        assert "methods" in config
        assert "no_prediction" in config["methods"]
        assert "cv_prediction" in config["methods"]
        assert "ca_prediction" in config["methods"]

    def test_config_has_benchmark_defaults(self):
        config = load_yaml_config("config/experiment/benchmark_simple_prediction_comparison.yaml")
        bench = config.get("benchmark", {})
        assert "episodes" in bench
        assert "seeds" in bench
        assert "scenarios" in bench
        assert len(bench["seeds"]) >= 5
        assert len(bench["scenarios"]) >= 4


class TestSmokeBenchmark:
    def test_smoke_runs_and_produces_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark",
                    "--config", "config/experiment/benchmark_simple_prediction_comparison.yaml",
                    "--smoke",
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, f"Smoke benchmark failed: {result.stderr}"

            # Check required output files
            required_files = [
                "prediction_metrics.csv",
                "prediction_metrics.json",
                "scenario_metrics.csv",
                "summary.md",
            ]
            for fname in required_files:
                fpath = os.path.join(tmpdir, fname)
                assert os.path.exists(fpath), f"Missing output file: {fname}"

    def test_csv_contains_unified_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable, "-m",
                    "uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark",
                    "--config", "config/experiment/benchmark_simple_prediction_comparison.yaml",
                    "--smoke",
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            csv_path = os.path.join(tmpdir, "prediction_metrics.csv")
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 3, "CSV should have at least 3 method rows"

            required_fields = [
                "method", "scenario", "seed", "episodes",
                "instant_success_rate", "score_win_rate", "mean_return",
                "mean_final_range_m", "mean_final_ata_deg",
                "prediction_rmse_m", "prediction_fallback_rate",
                "timeout_rate", "crash_rate", "out_of_bounds_rate",
            ]
            for field in required_fields:
                assert field in reader.fieldnames, f"Missing unified field: {field}"

            methods = [r["method"] for r in rows]
            assert "no_prediction" in methods
            assert "cv_prediction" in methods
            assert "ca_prediction" in methods

    def test_json_contains_three_methods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable, "-m",
                    "uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark",
                    "--config", "config/experiment/benchmark_simple_prediction_comparison.yaml",
                    "--smoke",
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            json_path = os.path.join(tmpdir, "prediction_metrics.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == 3
            methods = [m["method"] for m in data]
            assert "no_prediction" in methods
            assert "cv_prediction" in methods
            assert "ca_prediction" in methods

    def test_summary_md_contains_methods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable, "-m",
                    "uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark",
                    "--config", "config/experiment/benchmark_simple_prediction_comparison.yaml",
                    "--smoke",
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            md_path = os.path.join(tmpdir, "summary.md")
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "no_prediction" in content
            assert "cv_prediction" in content
            assert "ca_prediction" in content
            assert "Warning" in content


class TestStatisticalComparison:
    def test_mean_std_ignores_nan(self):
        vals = [1.0, 2.0, np.nan, 3.0]
        m, s = mean_std(vals)
        assert m == pytest.approx(2.0, abs=1e-6)
        assert s > 0

    def test_bootstrap_ci_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean_val, lower, upper = bootstrap_ci(vals, n_bootstrap=500, random_seed=0)
        assert lower <= mean_val <= upper

    def test_bootstrap_ci_handles_empty(self):
        vals = [np.nan, np.nan]
        mean_val, lower, upper = bootstrap_ci(vals)
        assert np.isnan(mean_val)

    def test_paired_delta_basic(self):
        baseline = [1.0, 2.0, 3.0]
        treatment = [2.0, 3.0, 4.0]
        d_mean, d_std, n = paired_delta(baseline, treatment)
        assert d_mean == pytest.approx(1.0, abs=1e-6)
        assert n == 3

    def test_paired_delta_handles_nan(self):
        baseline = [1.0, np.nan, 3.0]
        treatment = [2.0, 3.0, 4.0]
        d_mean, d_std, n = paired_delta(baseline, treatment)
        assert n == 2

    def test_compare_methods_finds_baseline(self):
        metrics = [
            {"method": "no_prediction", "mean_return": 100.0, "success_rate": 0.5},
            {"method": "cv_prediction", "mean_return": 120.0, "success_rate": 0.6},
        ]
        result = compare_methods(metrics, baseline_name="no_prediction")
        assert "no_prediction" in result["per_method"]
        assert "no_prediction_vs_cv_prediction" in result["pairwise"]
        assert result["pairwise"]["no_prediction_vs_cv_prediction"]["delta"] == 20.0

    def test_compare_methods_missing_baseline(self):
        metrics = [
            {"method": "cv_prediction", "mean_return": 120.0},
        ]
        result = compare_methods(metrics, baseline_name="no_prediction")
        assert "error" in result

    def test_compare_per_scenario(self):
        method_metrics = {
            "no_prediction": {
                "per_scenario": {
                    "favorable": {"mean_return": 100.0},
                    "neutral": {"mean_return": 50.0},
                }
            },
            "cv_prediction": {
                "per_scenario": {
                    "favorable": {"mean_return": 110.0},
                    "neutral": {"mean_return": 55.0},
                }
            },
        }
        result = compare_per_scenario(method_metrics, "favorable", "mean_return")
        assert result["scenario"] == "favorable"
        assert result["deltas"]["no_prediction_vs_cv_prediction"]["delta"] == 10.0
