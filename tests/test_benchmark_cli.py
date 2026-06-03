"""Lightweight smoke tests for benchmark CLI argument handling."""

import sys
from unittest.mock import patch

from uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark import main


class TestBenchmarkCLI:
    def test_cli_smoke_flag_parses(self):
        """--smoke should be accepted without error."""
        test_args = [
            "prog",
            "--config",
            "config/experiment/benchmark_simple_prediction_comparison.yaml",
            "--smoke",
            "--force",
        ]
        with patch.object(sys, "argv", test_args):
            try:
                main()
            except SystemExit as exc:
                # We expect the benchmark to run and exit 0
                assert exc.code == 0

    def test_cli_backend_override_parses(self):
        """--backend simple should be accepted."""
        test_args = [
            "prog",
            "--config",
            "config/experiment/benchmark_simple_prediction_comparison.yaml",
            "--smoke",
            "--backend",
            "simple",
            "--force",
        ]
        with patch.object(sys, "argv", test_args):
            try:
                main()
            except SystemExit as exc:
                assert exc.code == 0

    def test_cli_force_flag_parses(self):
        """--force should be accepted."""
        test_args = [
            "prog",
            "--config",
            "config/experiment/benchmark_simple_prediction_comparison.yaml",
            "--smoke",
            "--force",
        ]
        with patch.object(sys, "argv", test_args):
            try:
                main()
            except SystemExit as exc:
                assert exc.code == 0
