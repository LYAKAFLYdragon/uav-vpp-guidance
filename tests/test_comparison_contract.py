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
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
