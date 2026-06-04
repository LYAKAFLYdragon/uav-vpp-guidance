"""
Shared predictor telemetry utilities for train / eval scripts.

Provides a reusable per-step health collector so that train_prediction_vpp_ppo,
evaluate_policy, and evaluate_prediction_comparison all agree on how
counters are incremented and which info fields are read.
"""

import numpy as np


class PredictorHealthAccumulator:
    """Accumulates per-episode predictor health metrics from env.step info."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_valid_steps = 0
        self.fallback_steps = 0
        self.warmup_fallback_steps = 0
        self.runtime_fallback_steps = 0
        self.post_warmup_fallback_steps = 0
        self.predictor_init_failed_steps = 0
        self.prediction_errors = []
        self.prediction_error_count = 0

    def step(self, info: dict):
        """Process one env.step info dict."""
        if not info.get("prediction_enabled", False):
            return
        if info.get("prediction_valid", False):
            self.pred_valid_steps += 1
        if info.get("fallback", False) or info.get("prediction_fallback_reason") is not None:
            self.fallback_steps += 1
            phase = info.get("prediction_fallback_phase")
            if phase == "warmup":
                self.warmup_fallback_steps += 1
            elif phase == "runtime_failure":
                self.runtime_fallback_steps += 1
            if phase != "warmup":
                self.post_warmup_fallback_steps += 1
        if info.get("predictor_init_failed", False):
            self.predictor_init_failed_steps += 1
        pred_err = info.get("prediction_error_m", np.nan)
        if np.isfinite(pred_err):
            self.prediction_errors.append(float(pred_err))
            self.prediction_error_count += 1

    def rates(self, episode_length: int) -> dict:
        """Compute per-episode rates from accumulated counters."""
        ep_len = max(1, episode_length)
        mean_err = float(np.mean(self.prediction_errors)) if self.prediction_errors else np.nan
        median_err = float(np.median(self.prediction_errors)) if self.prediction_errors else np.nan
        return {
            "prediction_valid_rate": self.pred_valid_steps / ep_len,
            "fallback_rate": self.fallback_steps / ep_len,
            "warmup_fallback_rate": self.warmup_fallback_steps / ep_len,
            "runtime_fallback_rate": self.runtime_fallback_steps / ep_len,
            "post_warmup_fallback_rate": self.post_warmup_fallback_steps / ep_len,
            "predictor_init_failed_count": self.predictor_init_failed_steps,
            "mean_prediction_error_m": mean_err,
            "median_prediction_error_m": median_err,
            "prediction_error_count": self.prediction_error_count,
        }
