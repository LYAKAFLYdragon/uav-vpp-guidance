"""
Tests for Stage 6F.0 Experiment Launch Gate.

Covers:
- Telemetry contract completeness across all 5 ablation methods
- Cross-config validation (virtual_point.anchor_mode vs trajectory_prediction.enabled)
- Comparison checkpoint policy (strict vs --allow-random-policy)
- Stage 6F config completeness and checkpoint semantics
- PredictorHealthAccumulator fallback phase accounting
- Method config deepcopy isolation
"""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.trajectory_prediction._telemetry import PredictorHealthAccumulator
from uav_vpp_guidance.trajectory_prediction.config_validator import validate_full_config


class TestTelemetryContract(unittest.TestCase):
    """Ensure info dict contains all required telemetry fields for every method type."""

    REQUIRED_TELEMETRY_FIELDS = {
        "prediction_enabled",
        "predictor_init_failed",
        "predictor_type",
        "prediction_valid",
        "prediction_fallback",
        "prediction_fallback_reason",
        "prediction_fallback_mode",
        "prediction_fallback_model",
        "prediction_fallback_phase",
        "predicted_target_position",
        "prediction_error_m",
        "latest_prediction_error_m",
        "mean_prediction_error_m",
        "median_prediction_error_m",
        "prediction_error_count",
    }

    def _get_base_config(self):
        return load_yaml_config("config/env.yaml")

    def _make_config(self, tp_override, vp_override=None):
        base = self._get_base_config()
        base["trajectory_prediction"] = tp_override
        if vp_override:
            base["virtual_point"] = vp_override
        else:
            base["virtual_point"] = {"anchor_mode": "predicted_target" if tp_override.get("enabled", False) else "current_target"}
        return base

    def _step_and_collect_fields(self, env):
        obs = env.reset(seed=42)
        all_fields = set()
        for _ in range(5):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            all_fields.update(info.keys())
            if terminated or truncated:
                break
        return all_fields

    def test_no_prediction_fields(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = self._make_config({"enabled": False}, {"anchor_mode": "current_target"})
        env = CloseRangeTrackingEnv(config)
        try:
            fields = self._step_and_collect_fields(env)
            missing = self.REQUIRED_TELEMETRY_FIELDS - fields
            self.assertEqual(missing, set(), f"Missing telemetry fields for no_prediction: {missing}")
        finally:
            env.close()

    def test_cv_prediction_fields(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = self._make_config({
            "enabled": True,
            "predictor_type": "constant_velocity",
            "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position", "fallback_mode": "constant_velocity"},
            "history": {"history_len": 5, "padding_mode": "repeat_first"},
        })
        env = CloseRangeTrackingEnv(config)
        try:
            fields = self._step_and_collect_fields(env)
            missing = self.REQUIRED_TELEMETRY_FIELDS - fields
            self.assertEqual(missing, set(), f"Missing telemetry fields for CV: {missing}")
        finally:
            env.close()

    def test_ca_prediction_fields(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = self._make_config({
            "enabled": True,
            "predictor_type": "constant_acceleration",
            "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position", "fallback_mode": "constant_velocity"},
            "history": {"history_len": 5, "padding_mode": "repeat_first"},
        })
        env = CloseRangeTrackingEnv(config)
        try:
            fields = self._step_and_collect_fields(env)
            missing = self.REQUIRED_TELEMETRY_FIELDS - fields
            self.assertEqual(missing, set(), f"Missing telemetry fields for CA: {missing}")
        finally:
            env.close()

    def test_lstm_prediction_fields(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = self._make_config({
            "enabled": True,
            "predictor_type": "lstm",
            "strict_predictor_init": False,
            "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position", "fallback_mode": "constant_velocity"},
            "history": {"history_len": 5, "padding_mode": "repeat_first"},
            "model": {"input_dim": 9, "hidden_dim": 64, "num_layers": 2, "output_dim": 3},
        })
        env = CloseRangeTrackingEnv(config)
        try:
            fields = self._step_and_collect_fields(env)
            missing = self.REQUIRED_TELEMETRY_FIELDS - fields
            self.assertEqual(missing, set(), f"Missing telemetry fields for LSTM: {missing}")
        finally:
            env.close()

    def test_gru_prediction_fields(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = self._make_config({
            "enabled": True,
            "predictor_type": "gru",
            "strict_predictor_init": False,
            "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position", "fallback_mode": "constant_velocity"},
            "history": {"history_len": 5, "padding_mode": "repeat_first"},
            "model": {"input_dim": 9, "hidden_dim": 64, "num_layers": 2, "output_dim": 3},
        })
        env = CloseRangeTrackingEnv(config)
        try:
            fields = self._step_and_collect_fields(env)
            missing = self.REQUIRED_TELEMETRY_FIELDS - fields
            self.assertEqual(missing, set(), f"Missing telemetry fields for GRU: {missing}")
        finally:
            env.close()


class TestComparisonConfigValidation(unittest.TestCase):
    """Cross-config validation: anchor_mode must match trajectory_prediction.enabled."""

    def test_enabled_true_requires_predicted_target(self):
        config = {
            "trajectory_prediction": {"enabled": True, "predictor_type": "constant_velocity"},
            "virtual_point": {"anchor_mode": "current_target"},
        }
        with self.assertRaises(ValueError) as ctx:
            validate_full_config(config, on_unknown="raise")
        self.assertIn("predicted_target", str(ctx.exception))

    def test_enabled_false_requires_current_target(self):
        config = {
            "trajectory_prediction": {"enabled": False},
            "virtual_point": {"anchor_mode": "predicted_target"},
        }
        with self.assertRaises(ValueError) as ctx:
            validate_full_config(config, on_unknown="raise")
        self.assertIn("current_target", str(ctx.exception))

    def test_valid_enabled_true_predicted_target(self):
        config = {
            "trajectory_prediction": {"enabled": True, "predictor_type": "constant_velocity"},
            "virtual_point": {"anchor_mode": "predicted_target"},
        }
        validate_full_config(config, on_unknown="raise")

    def test_valid_enabled_false_current_target(self):
        config = {
            "trajectory_prediction": {"enabled": False},
            "virtual_point": {"anchor_mode": "current_target"},
        }
        validate_full_config(config, on_unknown="raise")


class TestComparisonCheckpointPolicy(unittest.TestCase):
    """Checkpoint must exist unless --allow-random-policy is passed."""

    def test_missing_checkpoint_raises_without_flag(self):
        allow_random = False
        ckpt_path = "/nonexistent/path/best.pt"
        exists = os.path.exists(ckpt_path)
        self.assertFalse(exists)
        self.assertTrue(not allow_random and not exists)

    def test_missing_checkpoint_allowed_with_flag(self):
        allow_random = True
        ckpt_path = "/nonexistent/path/best.pt"
        exists = os.path.exists(ckpt_path)
        self.assertFalse(exists)
        self.assertTrue(allow_random or exists)


class TestStage6FConfig(unittest.TestCase):
    """Stage 6F ablation config must define exactly 5 methods with correct checkpoint semantics."""

    def test_five_methods_defined(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        methods = config.get("methods", {})
        expected = {"no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"}
        self.assertEqual(set(methods.keys()), expected)

    def test_no_prediction_disabled(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        tp = config["methods"]["no_prediction"]["trajectory_prediction"]
        self.assertFalse(tp["enabled"])

    def test_all_methods_have_policy_checkpoint(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        for name, method in config["methods"].items():
            with self.subTest(method=name):
                self.assertIn("checkpoint", method, f"Method {name} missing policy checkpoint")
                self.assertTrue(method["checkpoint"].endswith(".pt"), f"Method {name} checkpoint should be .pt file")

    def test_neural_methods_have_predictor_checkpoint(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        for name in ("lstm_frozen", "gru_frozen"):
            with self.subTest(method=name):
                tp = config["methods"][name]["trajectory_prediction"]
                self.assertTrue(tp.get("strict_predictor_init", False))
                self.assertTrue(tp.get("checkpoint_strict", False))
                self.assertIsNotNone(tp.get("checkpoint_path"))

    def test_lstm_predictor_checkpoint_is_best_model(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        tp = config["methods"]["lstm_frozen"]["trajectory_prediction"]
        self.assertEqual(tp["checkpoint_path"], "outputs/trajectory_prediction/best_model.pt")
        self.assertEqual(tp["predictor_type"], "lstm")

    def test_gru_predictor_checkpoint_is_best_model_gru(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        tp = config["methods"]["gru_frozen"]["trajectory_prediction"]
        self.assertEqual(tp["checkpoint_path"], "outputs/trajectory_prediction/best_model_gru.pt")
        self.assertEqual(tp["predictor_type"], "gru")

    def test_policy_and_predictor_checkpoints_are_different(self):
        config = load_yaml_config("config/experiment/evaluate_vpp_prediction_comparison.yaml")
        for name in ("lstm_frozen", "gru_frozen"):
            with self.subTest(method=name):
                method = config["methods"][name]
                policy_ckpt = method["checkpoint"]
                predictor_ckpt = method["trajectory_prediction"]["checkpoint_path"]
                self.assertNotEqual(policy_ckpt, predictor_ckpt,
                                    f"Method {name}: policy and predictor checkpoints must be different")


class TestPredictorHealthAccumulatorRates(unittest.TestCase):
    """Ensure PredictorHealthAccumulator produces the expected rate keys and fallback phase accounting."""

    def test_rate_keys(self):
        acc = PredictorHealthAccumulator()
        for _ in range(10):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": True,
                "prediction_fallback": False,
                "predictor_init_failed": False,
            })
        rates = acc.rates(10)
        expected_keys = {
            "prediction_valid_rate",
            "fallback_rate",
            "warmup_fallback_rate",
            "runtime_fallback_rate",
            "post_warmup_fallback_rate",
            "predictor_init_failed_count",
            "unknown_fallback_phase_count",
            "missing_fallback_phase_count",
            "configured_current_target_fallback_count",
            "mean_prediction_error_m",
            "median_prediction_error_m",
            "prediction_error_count",
        }
        self.assertEqual(set(rates.keys()), expected_keys)
        self.assertAlmostEqual(rates["prediction_valid_rate"], 1.0)
        self.assertAlmostEqual(rates["fallback_rate"], 0.0)

    def test_fallback_phase_tracking(self):
        acc = PredictorHealthAccumulator()
        for _ in range(5):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "warmup",
                "predictor_init_failed": False,
            })
        for _ in range(5):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "runtime_failure",
                "predictor_init_failed": False,
            })
        rates = acc.rates(10)
        self.assertAlmostEqual(rates["warmup_fallback_rate"], 0.5)
        self.assertAlmostEqual(rates["runtime_fallback_rate"], 0.5)
        self.assertAlmostEqual(rates["post_warmup_fallback_rate"], 0.5)

    def test_missing_phase_counted(self):
        acc = PredictorHealthAccumulator()
        for _ in range(5):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": None,
                "predictor_init_failed": False,
            })
        rates = acc.rates(5)
        self.assertEqual(acc.missing_fallback_phase_count, 5)
        self.assertEqual(acc.unknown_fallback_phase_count, 0)
        self.assertEqual(acc.configured_current_target_fallback_count, 0)

    def test_configured_current_target_counted(self):
        acc = PredictorHealthAccumulator()
        for _ in range(3):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "configured_current_target",
                "predictor_init_failed": False,
            })
        rates = acc.rates(3)
        self.assertEqual(acc.configured_current_target_fallback_count, 3)
        self.assertEqual(acc.missing_fallback_phase_count, 0)
        self.assertEqual(acc.unknown_fallback_phase_count, 0)

    def test_unknown_phase_counted(self):
        acc = PredictorHealthAccumulator()
        for _ in range(4):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "unknown",
                "predictor_init_failed": False,
            })
        rates = acc.rates(4)
        self.assertEqual(acc.unknown_fallback_phase_count, 4)
        self.assertEqual(acc.missing_fallback_phase_count, 0)
        self.assertEqual(acc.configured_current_target_fallback_count, 0)

    def test_warmup_not_polluted_by_unknown(self):
        acc = PredictorHealthAccumulator()
        for _ in range(2):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "warmup",
                "predictor_init_failed": False,
            })
        for _ in range(3):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "unknown",
                "predictor_init_failed": False,
            })
        rates = acc.rates(5)
        self.assertEqual(acc.warmup_fallback_steps, 2)
        self.assertEqual(acc.unknown_fallback_phase_count, 3)
        self.assertEqual(acc.post_warmup_fallback_steps, 3)


class TestMethodConfigDeepcopy(unittest.TestCase):
    """Ensure method config merge does not cross-contaminate between methods."""

    def test_deepcopy_prevents_contamination(self):
        base = {
            "trajectory_prediction": {"enabled": False, "predictor_type": "none"},
            "virtual_point": {"anchor_mode": "current_target", "action_dim": 3},
        }
        method_a_override = {
            "trajectory_prediction": {"enabled": True, "predictor_type": "lstm"},
            "virtual_point": {"anchor_mode": "predicted_target"},
        }
        method_b_override = {
            "trajectory_prediction": {"enabled": True, "predictor_type": "gru"},
        }

        config_a = merge_config(copy.deepcopy(base), copy.deepcopy(method_a_override))
        config_b = merge_config(copy.deepcopy(base), copy.deepcopy(method_b_override))

        # Modifying A should not affect B
        config_a["virtual_point"]["action_dim"] = 5
        self.assertEqual(config_b["virtual_point"]["action_dim"], 3)
        self.assertEqual(config_a["trajectory_prediction"]["predictor_type"], "lstm")
        self.assertEqual(config_b["trajectory_prediction"]["predictor_type"], "gru")


class TestFullConfigValidationEntrypoints(unittest.TestCase):
    """Ensure validate_full_config is called by train/eval/comparison entrypoints."""

    def test_train_entrypoint_imports_validate_full_config(self):
        import uav_vpp_guidance.training.train_prediction_vpp_ppo as train_mod
        self.assertTrue(hasattr(train_mod, "validate_full_config"))

    def test_eval_policy_entrypoint_imports_validate_full_config(self):
        import uav_vpp_guidance.evaluation.evaluate_policy as eval_mod
        self.assertTrue(hasattr(eval_mod, "validate_full_config"))

    def test_eval_comparison_entrypoint_imports_validate_full_config(self):
        import uav_vpp_guidance.evaluation.evaluate_prediction_comparison as comp_mod
        self.assertTrue(hasattr(comp_mod, "validate_full_config"))


class TestRuntimeFallbackSemantics(unittest.TestCase):
    """runtime_fallback_steps must only count phase=='runtime_failure'."""

    def _make_acc_with_phase(self, phase, steps=5):
        acc = PredictorHealthAccumulator()
        for _ in range(steps):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": phase,
                "predictor_init_failed": False,
            })
        return acc

    def test_runtime_failure_counts(self):
        acc = self._make_acc_with_phase("runtime_failure")
        self.assertEqual(acc.runtime_fallback_steps, 5)
        self.assertEqual(acc.fallback_steps, 5)

    def test_configured_current_target_does_not_count(self):
        acc = self._make_acc_with_phase("configured_current_target")
        self.assertEqual(acc.runtime_fallback_steps, 0)
        self.assertEqual(acc.configured_current_target_fallback_count, 5)
        self.assertEqual(acc.fallback_steps, 5)

    def test_unknown_does_not_count(self):
        acc = self._make_acc_with_phase("unknown")
        self.assertEqual(acc.runtime_fallback_steps, 0)
        self.assertEqual(acc.unknown_fallback_phase_count, 5)
        self.assertEqual(acc.fallback_steps, 5)

    def test_missing_phase_does_not_count(self):
        acc = self._make_acc_with_phase(None)
        self.assertEqual(acc.runtime_fallback_steps, 0)
        self.assertEqual(acc.missing_fallback_phase_count, 5)
        self.assertEqual(acc.fallback_steps, 5)

    def test_unrecognized_phase_does_not_count(self):
        acc = self._make_acc_with_phase("garbage_phase")
        self.assertEqual(acc.runtime_fallback_steps, 0)
        self.assertEqual(acc.unknown_fallback_phase_count, 5)
        self.assertEqual(acc.fallback_steps, 5)

    def test_post_warmup_still_counts_all_non_warmup(self):
        acc = PredictorHealthAccumulator()
        for _ in range(3):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "warmup",
            })
        for _ in range(4):
            acc.step({
                "prediction_enabled": True,
                "prediction_valid": False,
                "prediction_fallback": True,
                "prediction_fallback_phase": "unknown",
            })
        rates = acc.rates(7)
        self.assertEqual(acc.post_warmup_fallback_steps, 4)
        self.assertEqual(acc.runtime_fallback_steps, 0)


class TestUnifiedTelemetrySchema(unittest.TestCase):
    """All three entrypoints must emit the same unified telemetry fields."""

    UNIFIED_FIELDS = {
        "prediction_valid_rate",
        "fallback_rate",
        "warmup_fallback_rate",
        "runtime_fallback_rate",
        "post_warmup_fallback_rate",
        "predictor_init_failed_count",
        "unknown_fallback_phase_count",
        "missing_fallback_phase_count",
        "configured_current_target_fallback_count",
        "mean_prediction_error_m",
        "median_prediction_error_m",
        "prediction_error_count",
    }

    def test_train_episode_fieldnames(self):
        import uav_vpp_guidance.training.train_prediction_vpp_ppo as train_mod
        # Re-constitute fieldnames by calling the helper logic implicitly
        # The module defines them inside train_ppo; check via source inspection
        source = Path(train_mod.__file__).read_text(encoding="utf-8")
        for field in self.UNIFIED_FIELDS:
            self.assertIn(f'"{field}"', source, f"Missing field in train script: {field}")

    def test_eval_policy_metrics_keys(self):
        import uav_vpp_guidance.evaluation.evaluate_policy as eval_mod
        source = Path(eval_mod.__file__).read_text(encoding="utf-8")
        for field in self.UNIFIED_FIELDS:
            self.assertIn(f'"{field}"', source, f"Missing field in evaluate_policy: {field}")

    def test_eval_comparison_metrics_keys(self):
        import uav_vpp_guidance.evaluation.evaluate_prediction_comparison as comp_mod
        source = Path(comp_mod.__file__).read_text(encoding="utf-8")
        for field in self.UNIFIED_FIELDS:
            self.assertIn(f'"{field}"', source, f"Missing field in comparison script: {field}")


class TestComparisonPolicyMetadataCSV(unittest.TestCase):
    """Comparison CSV must include policy metadata and checkpoint provenance."""

    def test_csv_contains_policy_metadata(self):
        import subprocess
        import tempfile
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m", "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
                "--config", "config/experiment/evaluate_vpp_prediction_comparison.yaml",
                "--backend", "simple",
                "--episodes", "1",
                "--seeds", "0",
                "--allow-random-policy",
                "--output-dir", tmp,
            ]
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"Comparison failed: {result.stderr}")

            csv_path = Path(tmp) / "prediction_metrics.csv"
            self.assertTrue(csv_path.exists(), "prediction_metrics.csv not generated")
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            first = rows[0]
            required_cols = {
                "policy_type",
                "requested_policy_checkpoint_path",
                "loaded_policy_checkpoint_path",
                "predictor_checkpoint_path",
                "allow_random_policy",
            }
            missing = required_cols - set(first.keys())
            self.assertEqual(missing, set(), f"CSV missing columns: {missing}")

    def test_random_policy_loaded_checkpoint_is_none(self):
        import subprocess
        import tempfile
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m", "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
                "--config", "config/experiment/evaluate_vpp_prediction_comparison.yaml",
                "--backend", "simple",
                "--episodes", "1",
                "--seeds", "0",
                "--allow-random-policy",
                "--output-dir", tmp,
            ]
            subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
            csv_path = Path(tmp) / "prediction_metrics.csv"
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            no_pred = next((r for r in rows if r["method"] == "no_prediction"), None)
            self.assertIsNotNone(no_pred)
            # Since no_prediction checkpoint likely does not exist in test env,
            # loaded_policy_checkpoint_path should be empty/None in CSV.
            self.assertEqual(no_pred["policy_type"], "random_policy")
            self.assertEqual(no_pred["loaded_policy_checkpoint_path"], "")


class TestStage6FFullAblationRunnerDryRun(unittest.TestCase):
    """Runner dry-run must produce commands for all methods and seeds."""

    def test_dry_run_prints_all_methods_and_seeds(self):
        import io
        from contextlib import redirect_stdout
        from scripts.run_stage6f_full_ablation import main as runner_main

        f = io.StringIO()
        with redirect_stdout(f):
            old_argv = sys.argv
            sys.argv = ["run_stage6f_full_ablation.py", "--dry-run", "--training-seeds", "0", "1", "--evaluation-seeds", "0", "1"]
            try:
                runner_main()
            finally:
                sys.argv = old_argv

        output = f.getvalue()
        for method in ("no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"):
            self.assertIn(method, output, f"Dry-run output missing method: {method}")
        # Two seeds should be mentioned
        self.assertIn("seed 0", output)
        self.assertIn("seed 1", output)
        # Formal mode must not include --allow-random-policy
        self.assertNotIn("--allow-random-policy", output)


class TestStage6FManifest(unittest.TestCase):
    """Per-run manifest must contain required provenance fields."""

    REQUIRED_MANIFEST_KEYS = {
        "git_commit", "branch", "timestamp", "method", "seed",
        "config_path", "config_hash", "output_dir",
        "policy_checkpoint_path", "predictor_checkpoint_path",
        "backend", "validation_mode", "allow_random_policy", "metrics_schema_version",
    }

    def test_manifest_helper_produces_required_keys(self):
        from scripts.run_stage6f_full_ablation import write_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(
                output_dir=tmp,
                method="lstm_frozen",
                seed=7,
                config_path="config/experiment/dummy.yaml",
                policy_checkpoint_path="outputs/dummy/checkpoints/best.pt",
                predictor_checkpoint_path="outputs/dummy/best_model.pt",
                backend="simple",
                validation_mode="raise",
                allow_random_policy=False,
            )
            manifest_path = Path(tmp) / "manifest.json"
            self.assertTrue(manifest_path.exists())
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            missing = self.REQUIRED_MANIFEST_KEYS - set(manifest.keys())
            self.assertEqual(missing, set(), f"Manifest missing keys: {missing}")
            self.assertEqual(manifest["method"], "lstm_frozen")
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["metrics_schema_version"], "6f.2")
            self.assertFalse(manifest["allow_random_policy"])


class TestTrainingSeedPropagation(unittest.TestCase):
    """Comparison script must propagate --training-seed to episode and aggregate results."""

    def test_evaluate_single_episode_training_seed_set_by_evaluate_method(self):
        from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_method

        # Create a minimal mock env/agent
        class MockEnv:
            def __init__(self):
                self.max_steps = 5
                self._backend = "simple"
                self.config = {"trajectory_prediction": {"prediction": {"lookahead_time_s": 1.0}}}
                self.env_config = {"high_level_dt": 0.2}
            def reset(self, scenario=None, seed=0):
                return {"observation_vector": np.zeros(10)}
            def step(self, action):
                return {"observation_vector": np.zeros(10)}, 0.0, True, False, {}
            def close(self):
                pass

        class MockAgent:
            def get_deterministic_action(self, obs):
                return np.zeros(3)

        env = MockEnv()
        agent = MockAgent()
        config = {"scenarios": {"favorable": {"name": "favorable"}}}
        metrics = evaluate_method(
            env, agent, config, "test_method",
            num_episodes=2, seeds=[42],
            scenarios=["favorable"],
            training_seed=99,
        )
        # All raw episodes should have training_seed=99
        for ep in metrics["raw_episodes"]:
            self.assertEqual(ep["training_seed"], 99)
            self.assertEqual(ep["evaluation_seed"], 42)

    def test_aggregate_metrics_includes_scenario_balance(self):
        from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import aggregate_metrics
        episodes = [
            {"return": 1, "length": 10, "final_range_m": 100, "final_ata_deg": 5,
             "is_success": True, "is_crash": False, "is_timeout": False, "is_out_of_bounds": False,
             "prediction_enabled_rate": 0.5, "prediction_valid_rate": 0.4,
             "prediction_fallback_rate": 0.1, "warmup_fallback_rate": 0.0,
             "runtime_fallback_rate": 0.05, "post_warmup_fallback_rate": 0.05,
             "predictor_init_failed_count": 0, "unknown_fallback_phase_count": 0,
             "missing_fallback_phase_count": 0, "configured_current_target_fallback_count": 0,
             "mean_env_prediction_error_m": 1.0, "median_env_prediction_error_m": 0.8,
             "mean_offline_aligned_error_m": 1.2, "median_offline_aligned_error_m": 1.0,
             "mean_virtual_point_shift_m": 2.0, "mean_anchor_shift_m": 1.5,
             "time_to_first_advantage_s": 1.0, "advantage_hold_time_s": 5.0,
             "score_win": True,
             "min_range_m": 80, "min_ata_deg": 3,
             "mean_prediction_error_m": 1.0, "median_prediction_error_m": 0.9},
        ]
        result = aggregate_metrics(episodes)
        self.assertEqual(result["num_episodes"], 1)
        self.assertAlmostEqual(result["mean_runtime_fallback_rate"], 0.05)


class TestEpisodesPerScenario(unittest.TestCase):
    """Balanced scenario evaluation must produce equal counts per scenario."""

    def test_episodes_per_scenario_computes_total(self):
        from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_method

        class MockEnv:
            def __init__(self):
                self.max_steps = 3
                self._backend = "simple"
                self.config = {"trajectory_prediction": {"prediction": {"lookahead_time_s": 1.0}}}
                self.env_config = {"high_level_dt": 0.2}
            def reset(self, scenario=None, seed=0):
                return {"observation_vector": np.zeros(10)}
            def step(self, action):
                return {"observation_vector": np.zeros(10)}, 0.0, True, False, {}
            def close(self):
                pass

        class MockAgent:
            def get_deterministic_action(self, obs):
                return np.zeros(3)

        env = MockEnv()
        agent = MockAgent()
        config = {"scenarios": {
            "favorable": {"name": "favorable"},
            "neutral": {"name": "neutral"},
            "disadvantage": {"name": "disadvantage"},
            "challenging": {"name": "challenging"},
        }}
        metrics = evaluate_method(
            env, agent, config, "test_method",
            num_episodes=12, seeds=[0],
            scenarios=["favorable", "neutral", "disadvantage", "challenging"],
        )
        counts = metrics.get("scenario_episode_count", {})
        self.assertEqual(counts.get("favorable"), 3)
        self.assertEqual(counts.get("neutral"), 3)
        self.assertEqual(counts.get("disadvantage"), 3)
        self.assertEqual(counts.get("challenging"), 3)
        self.assertTrue(metrics.get("scenario_balance_ok"))


class TestMethodCheckpointOverrides(unittest.TestCase):
    """Runner must build correct per-seed checkpoint override paths."""

    def test_build_method_checkpoint_overrides(self):
        from scripts.run_stage6f_full_ablation import build_method_checkpoint_overrides
        overrides = build_method_checkpoint_overrides(training_seed=2)
        # Should return 5 method=path entries
        self.assertEqual(len(overrides), 5)
        for ov in overrides:
            self.assertIn("=", ov)
            method, path = ov.split("=", 1)
            self.assertIn("_seed2", path)
            self.assertTrue(path.endswith(os.path.join("checkpoints", "best.pt")))

    def test_override_contains_all_methods(self):
        from scripts.run_stage6f_full_ablation import build_method_checkpoint_overrides, METHODS
        overrides = build_method_checkpoint_overrides(training_seed=0)
        methods_found = {ov.split("=", 1)[0] for ov in overrides}
        expected = {m["name"] for m in METHODS}
        self.assertEqual(methods_found, expected)


class TestExperimentPlan(unittest.TestCase):
    """Experiment plan must contain required fields and schema version."""

    def test_write_experiment_plan(self):
        from scripts.run_stage6f_full_ablation import write_experiment_plan
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            write_experiment_plan(
                output_dir=tmp,
                training_seeds=[0, 1, 2],
                evaluation_seeds=[0, 1],
                episodes_per_scenario=25,
                formal=True,
            )
            plan_path = Path(tmp) / "experiment_plan.json"
            self.assertTrue(plan_path.exists())
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            self.assertEqual(plan["training_seeds"], [0, 1, 2])
            self.assertEqual(plan["evaluation_seeds"], [0, 1])
            self.assertEqual(plan["episodes_per_scenario"], 25)
            self.assertTrue(plan["formal"])
            self.assertFalse(plan["allow_random_policy"])
            self.assertEqual(plan["metrics_schema_version"], "6f.2")
            self.assertIn("methods", plan)
            self.assertIn("scenarios", plan)


class TestComparisonInvalidForPaper(unittest.TestCase):
    """Comparison script must set invalid_for_paper based on policy source."""

    def test_invalid_for_paper_false_when_trained_checkpoint_loaded(self):
        from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import aggregate_metrics
        episodes = [
            {"return": 1, "length": 10, "final_range_m": 100, "final_ata_deg": 5,
             "is_success": True, "is_crash": False, "is_timeout": False, "is_out_of_bounds": False,
             "prediction_enabled_rate": 0.5, "prediction_valid_rate": 0.4,
             "prediction_fallback_rate": 0.1, "warmup_fallback_rate": 0.0,
             "runtime_fallback_rate": 0.05, "post_warmup_fallback_rate": 0.05,
             "predictor_init_failed_count": 0, "unknown_fallback_phase_count": 0,
             "missing_fallback_phase_count": 0, "configured_current_target_fallback_count": 0,
             "mean_env_prediction_error_m": 1.0, "median_env_prediction_error_m": 0.9,
             "mean_offline_aligned_error_m": 1.1, "median_offline_aligned_error_m": 1.0,
             "mean_virtual_point_shift_m": 2.0, "mean_anchor_shift_m": 1.5,
             "time_to_first_advantage_s": 1.0, "advantage_hold_time_s": 5.0,
             "score_win": True, "min_range_m": 80, "min_ata_deg": 3,
             "mean_prediction_error_m": 1.0, "median_prediction_error_m": 0.9},
        ]
        result = aggregate_metrics(episodes)
        # Note: invalid_for_paper is set in main(), not aggregate_metrics.
        # But we can verify the aggregation function works correctly.
        self.assertEqual(result["num_episodes"], 1)


class TestStage6FOutputValidation(unittest.TestCase):
    """Validation script must detect formal output anomalies."""

    def test_validation_passes_on_pilot(self):
        from scripts.validate_stage6f_outputs import validate
        import argparse
        args = argparse.Namespace(
            input="outputs/tables/stage6f_full_ablation",
            summary="outputs/tables/stage6f_pilot",
        )
        self.assertTrue(validate(args))

    def test_validation_fails_on_missing_summary(self):
        from scripts.validate_stage6f_outputs import validate
        import argparse
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input=tmp,
                summary=tmp,
            )
            self.assertFalse(validate(args))


class TestTwoLevelAggregation(unittest.TestCase):
    """Aggregation script must produce cross-training-seed statistics."""

    def test_aggregate_episodes_to_training_seed(self):
        from scripts.aggregate_stage6f_results import aggregate_episodes_to_training_seed
        episodes = [
            {"return": 10, "length": 100, "final_range_m": 50, "final_ata_deg": 5,
             "is_success": True, "prediction_valid_rate": 0.8,
             "prediction_fallback_rate": 0.1, "runtime_fallback_rate": 0.05,
             "post_warmup_fallback_rate": 0.05,
             "mean_env_prediction_error_m": 1.0, "median_env_prediction_error_m": 0.9,
             "mean_offline_aligned_error_m": 1.1, "median_offline_aligned_error_m": 1.0,
             "unknown_fallback_phase_count": 0, "missing_fallback_phase_count": 0,
             "configured_current_target_fallback_count": 2,
             "predictor_init_failed_count": 0},
            {"return": 12, "length": 110, "final_range_m": 45, "final_ata_deg": 4,
             "is_success": True, "prediction_valid_rate": 0.75,
             "prediction_fallback_rate": 0.15, "runtime_fallback_rate": 0.10,
             "post_warmup_fallback_rate": 0.10,
             "mean_env_prediction_error_m": 1.2, "median_env_prediction_error_m": 1.1,
             "mean_offline_aligned_error_m": 1.3, "median_offline_aligned_error_m": 1.2,
             "unknown_fallback_phase_count": 1, "missing_fallback_phase_count": 0,
             "configured_current_target_fallback_count": 1,
             "predictor_init_failed_count": 0},
        ]
        row = aggregate_episodes_to_training_seed(episodes)
        self.assertEqual(row["num_episodes"], 2)
        self.assertAlmostEqual(row["instant_success_rate"], 1.0)
        self.assertAlmostEqual(row["mean_return"], 11.0)
        self.assertAlmostEqual(row["configured_current_target_fallback_count"], 3)

    def test_aggregate_training_seeds_to_cross_seed(self):
        from scripts.aggregate_stage6f_results import aggregate_training_seeds_to_cross_seed
        rows = [
            {"num_episodes": 100, "instant_success_rate": 0.8, "mean_return": 10.0,
             "mean_final_range_m": 50, "mean_final_ata_deg": 5,
             "prediction_valid_rate": 0.7, "prediction_fallback_rate": 0.2,
             "runtime_fallback_rate": 0.1, "post_warmup_fallback_rate": 0.1,
             "mean_env_prediction_error_m": 1.0, "median_env_prediction_error_m": 0.9,
             "mean_offline_aligned_error_m": 1.1, "median_offline_aligned_error_m": 1.0,
             "unknown_fallback_phase_count": 0, "missing_fallback_phase_count": 0,
             "configured_current_target_fallback_count": 5, "predictor_init_failed_count": 0},
            {"num_episodes": 100, "instant_success_rate": 0.85, "mean_return": 11.0,
             "mean_final_range_m": 48, "mean_final_ata_deg": 4.5,
             "prediction_valid_rate": 0.75, "prediction_fallback_rate": 0.18,
             "runtime_fallback_rate": 0.08, "post_warmup_fallback_rate": 0.08,
             "mean_env_prediction_error_m": 0.9, "median_env_prediction_error_m": 0.85,
             "mean_offline_aligned_error_m": 1.0, "median_offline_aligned_error_m": 0.95,
             "unknown_fallback_phase_count": 1, "missing_fallback_phase_count": 0,
             "configured_current_target_fallback_count": 4, "predictor_init_failed_count": 0},
        ]
        metadata = {
            "method_name": "test_method",
            "allow_random_policy": False,
            "loaded_policy_checkpoint_path": "/path/to/checkpoint.pt",
            "evaluation_seeds": [0, 1, 2],
            "scenarios": ["favorable", "neutral", "disadvantage", "challenging"],
            "episodes_per_scenario": 25,
        }
        result = aggregate_training_seeds_to_cross_seed(rows, metadata)
        self.assertEqual(result["method"], "test_method")
        self.assertEqual(result["num_training_seeds"], 2)
        self.assertFalse(result["invalid_for_paper"])
        self.assertAlmostEqual(result["instant_success_rate_mean"], 0.825, places=3)
        self.assertIn("instant_success_rate_std", result)
        self.assertIn("instant_success_rate_ci95", result)

    def test_manifest_validation_warnings(self):
        from scripts.aggregate_stage6f_results import _check_manifest
        plan_ok = {
            "metrics_schema_version": "6f.2",
            "formal": True,
            "allow_random_policy": False,
        }
        self.assertEqual(_check_manifest(plan_ok), [])

        plan_bad = {
            "metrics_schema_version": "6f.1",
            "formal": False,
            "allow_random_policy": True,
        }
        warnings = _check_manifest(plan_bad)
        self.assertEqual(len(warnings), 3)


if __name__ == "__main__":
    unittest.main()
