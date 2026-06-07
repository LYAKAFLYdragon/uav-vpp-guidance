"""Tests for Stage 6G probe output validator using synthetic fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml


class TestValidateStage6GProbeOutputsSynthetic(unittest.TestCase):
    """Validate validator against synthetic fixtures."""

    def _create_synthetic_probe_output(self, tmpdir, smoke=False, n_episodes_per_cell=10, exclude_cells=None):
        """Create a minimal synthetic probe output directory."""
        root = Path(tmpdir) / "run_test"
        root.mkdir(parents=True, exist_ok=True)

        guidance_modes = ["los_rate", "proportional_navigation", "hybrid"]
        scenarios = ["favorable", "disadvantage", "weaving_pursuit", "weaving_disadvantage"]
        methods = ["no_prediction", "gru_frozen"]
        eval_seeds = [0] if smoke else [0, 1, 2]
        exclude_cells = set(exclude_cells or [])

        total_episodes = 0
        for guidance in guidance_modes:
            for scenario in scenarios:
                cell_name = f"{guidance}_{scenario}"
                if cell_name in exclude_cells:
                    continue
                cell_dir = root / cell_name
                cell_dir.mkdir(parents=True, exist_ok=True)

                # resolved_config.yaml
                cfg = {
                    "guidance": {"mode": guidance},
                    "scenarios": {scenario: {}},
                    "methods": {m: {} for m in methods},
                }
                with open(cell_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)

                # prediction_metrics.json
                methods_data = []
                for method in methods:
                    raw_eps = []
                    for seed in eval_seeds:
                        for ep_idx in range(n_episodes_per_cell):
                            raw_eps.append({
                                "scenario": scenario,
                                "method": method,
                                "method_name": method,
                                "requested_guidance_mode": guidance,
                                "effective_guidance_mode": guidance,
                                "training_seed": 0,
                                "evaluation_seed": seed,
                                "eval_seed": seed,
                                "episode_index": ep_idx,
                                "episode_seed": ep_idx,
                                "is_success": False,
                                "is_crash": scenario in ("favorable", "weaving_pursuit"),
                                "is_out_of_bounds": scenario in ("disadvantage", "weaving_disadvantage"),
                                "is_timeout": False,
                                "reason": "crash" if scenario in ("favorable", "weaving_pursuit") else "out_of_bounds",
                                "return": -100.0,
                                "length": 100,
                                "final_range_m": 11300.0,
                                "final_ata_deg": 90.0,
                                "prediction_fallback_rate": 0.0,
                                "mean_prediction_error_m": 50.0,
                                "mean_virtual_point_shift_m": 200.0,
                                "mean_anchor_shift_m": 100.0,
                                "time_to_first_advantage_s": 10.0,
                                "advantage_hold_time_s": 5.0,
                            })
                            total_episodes += 1
                    methods_data.append({
                        "method": method,
                        "method_name": method,
                        "requested_guidance_mode": guidance,
                        "effective_guidance_mode": guidance,
                        "raw_episodes": raw_eps,
                    })

                with open(cell_dir / "prediction_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(methods_data, f)

        # Global artifacts
        # raw_episodes.csv
        columns = [
            "scenario", "method", "guidance_mode_requested", "effective_guidance_mode",
            "training_seed", "evaluation_seed", "episode_seed", "episode_index",
            "success", "termination_reason", "capture_time", "miss_distance", "min_range",
            "oob", "crash", "fallback_used", "prediction_error",
            "return", "length", "final_range_m", "final_ata_deg",
            "score_win", "mean_virtual_point_shift_m", "mean_anchor_shift_m",
            "time_to_first_advantage_s", "advantage_hold_time_s",
        ]
        csv_rows = []
        for guidance in guidance_modes:
            for scenario in scenarios:
                for method in methods:
                    for seed in eval_seeds:
                        for ep_idx in range(n_episodes_per_cell):
                            csv_rows.append({
                                "scenario": scenario,
                                "method": method,
                                "guidance_mode_requested": guidance,
                                "effective_guidance_mode": guidance,
                                "training_seed": 0,
                                "evaluation_seed": seed,
                                "episode_seed": ep_idx,
                                "episode_index": ep_idx,
                                "success": "False",
                                "termination_reason": "crash" if scenario in ("favorable", "weaving_pursuit") else "out_of_bounds",
                                "capture_time": "20.0",
                                "miss_distance": "11300.0",
                                "min_range": "11300.0",
                                "oob": "False" if scenario in ("favorable", "weaving_pursuit") else "True",
                                "crash": "True" if scenario in ("favorable", "weaving_pursuit") else "False",
                                "fallback_used": "False",
                                "prediction_error": "50.0",
                                "return": "-100.0",
                                "length": "100",
                                "final_range_m": "11300.0",
                                "final_ata_deg": "90.0",
                                "score_win": "False",
                                "mean_virtual_point_shift_m": "200.0",
                                "mean_anchor_shift_m": "100.0",
                                "time_to_first_advantage_s": "10.0",
                                "advantage_hold_time_s": "5.0",
                            })
        with open(root / "raw_episodes.csv", "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(csv_rows)

        # Empty but present global artifacts
        for fname in ["scenario_method_summary.csv", "pairwise_mcnemar.csv"]:
            with open(root / fname, "w", encoding="utf-8") as f:
                f.write("")

        for fname in ["paper_safe_claims.md", "README_result_block.md", "run.log"]:
            with open(root / fname, "w", encoding="utf-8") as f:
                f.write("test\n")

        # run_manifest.json
        manifest = {
            "run_status": "completed",
            "cells_total": 12,
            "cells_passed": 12,
            "cells_failed": 0,
            "total_raw_episodes": total_episodes,
            "artifacts_present": {a: True for a in [
                "raw_episodes.csv", "scenario_method_summary.csv", "pairwise_mcnemar.csv",
                "paper_safe_claims.md", "README_result_block.md", "run.log"
            ]},
        }
        with open(root / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        return root, total_episodes

    def test_full_probe_passes_validation(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmpdir:
            root, total_episodes = self._create_synthetic_probe_output(tmpdir, smoke=False, n_episodes_per_cell=10)
            result = subprocess.run(
                [sys.executable, "scripts/validate_stage6g_probe_outputs.py", "--input", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Validation failed:\n{result.stdout}\n{result.stderr}")
            self.assertIn("All validations passed", result.stdout)

    def test_smoke_probe_passes_validation(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmpdir:
            root, total_episodes = self._create_synthetic_probe_output(tmpdir, smoke=True, n_episodes_per_cell=1)
            result = subprocess.run(
                [sys.executable, "scripts/validate_stage6g_probe_outputs.py", "--input", str(root), "--smoke"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Validation failed:\n{result.stdout}\n{result.stderr}")
            self.assertIn("All validations passed", result.stdout)

    def test_missing_cell_fails_validation(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _ = self._create_synthetic_probe_output(tmpdir, smoke=False, n_episodes_per_cell=10)
            # Remove one cell
            import shutil
            cell_to_remove = root / "los_rate_favorable"
            if cell_to_remove.exists():
                shutil.rmtree(cell_to_remove)
            result = subprocess.run(
                [sys.executable, "scripts/validate_stage6g_probe_outputs.py", "--input", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, f"Expected exit 1, got {result.returncode}\n{result.stdout}")
            self.assertIn("FAIL", result.stdout)

    def test_allow_incomplete_permits_missing_cells(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _ = self._create_synthetic_probe_output(
                tmpdir, smoke=False, n_episodes_per_cell=10, exclude_cells=["los_rate_favorable"]
            )
            # Update manifest to reflect missing cell
            manifest_path = root / "run_manifest.json"
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["cells_passed"] = 11
            manifest["cells_failed"] = 1
            manifest["failed_cells"] = ["los_rate_favorable"]
            manifest["total_raw_episodes"] = 660
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            result = subprocess.run(
                [sys.executable, "scripts/validate_stage6g_probe_outputs.py", "--input", str(root), "--allow-incomplete"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Expected exit 0 with --allow-incomplete\n{result.stdout}")

    def test_guidance_mode_mismatch_fails(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _ = self._create_synthetic_probe_output(tmpdir, smoke=False, n_episodes_per_cell=10)
            # Modify one cell's prediction_metrics to have wrong effective_guidance_mode
            metrics_path = root / "los_rate_favorable" / "prediction_metrics.json"
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for m in data:
                m["effective_guidance_mode"] = "hybrid"  # mismatch
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = subprocess.run(
                [sys.executable, "scripts/validate_stage6g_probe_outputs.py", "--input", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Guidance mode consistency", result.stdout)

    def test_duplicate_episode_keys_fails(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _ = self._create_synthetic_probe_output(tmpdir, smoke=False, n_episodes_per_cell=10)
            metrics_path = root / "los_rate_favorable" / "prediction_metrics.json"
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Duplicate first episode
            for m in data:
                if m["raw_episodes"]:
                    m["raw_episodes"].append(m["raw_episodes"][0])
                    break
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = subprocess.run(
                [sys.executable, "scripts/validate_stage6g_probe_outputs.py", "--input", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            # Duplicate episodes cause either explicit duplicate-key failure or episode-count mismatch
            self.assertTrue(
                "Duplicate" in result.stdout or "episode count" in result.stdout.lower(),
                f"Expected duplicate or episode-count failure in:\n{result.stdout}"
            )


if __name__ == "__main__":
    unittest.main()
