"""
Regret computation for gain evaluation.
"""

import os
import warnings

import numpy as np
import yaml


_DEFAULT_WEIGHTS = {
    "return": 1.0,
    "success_rate": 200.0,
    "crash_rate": -300.0,
    "saturation_rate": -50.0,
}

# Fixed bounds for min-max normalizing episode return to [0, 1].
# These should cover the practical range of accumulated rewards.
_RETURN_MIN = -500.0
_RETURN_MAX = 500.0


def _load_weights_from_config(config_path="config/gain_space.yaml"):
    """Load default weights from gain_space.yaml regret block."""
    if not os.path.exists(config_path):
        return _DEFAULT_WEIGHTS.copy()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    regret = data.get("regret", {}) if data else {}
    mapping = {
        "return": regret.get("w_return", 1.0),
        "success_rate": regret.get("w_success", 200.0),
        "crash_rate": -abs(regret.get("w_crash", 300.0)),
        "saturation_rate": -abs(regret.get("w_saturation", 50.0)),
    }
    return mapping


def compute_score(metrics, weights=None):
    """
    Composite score for gain evaluation.

    Computes a weighted sum of evaluation metrics. The ``return`` metric is
    min-max normalized to [0, 1] before weighting so that it is on the same
    scale as ratio metrics (e.g. success_rate, crash_rate). The final score
    is then linearly rescaled to the [0, 1] interval.

    Args:
        metrics (dict): Evaluation metrics. Expected keys include:
            - "return" (float): average episode return
            - "success_rate" (float): success rate in [0, 1]
            - "crash_rate" (float): crash rate in [0, 1]
            - "saturation_rate" (float): actuator saturation rate in [0, 1]
        weights (dict, optional): Score component weights. If None, loads
            defaults from ``config/gain_space.yaml``.

    Returns:
        float: Scalar composite score in [0, 1] (higher is better).
    """
    if weights is None:
        weights = _load_weights_from_config()

    # Min-max normalize return to [0, 1] so it shares the same scale as
    # ratio metrics.
    metrics = dict(metrics)
    if "return" in metrics:
        raw_return = float(metrics["return"])
        norm_return = (raw_return - _RETURN_MIN) / (_RETURN_MAX - _RETURN_MIN)
        metrics["return"] = float(np.clip(norm_return, 0.0, 1.0))

    score = 0.0
    used_keys = set()
    for key, weight in weights.items():
        if key in metrics:
            score += weight * float(metrics[key])
            used_keys.add(key)
        else:
            warnings.warn(
                f"compute_score: metric key '{key}' not found in metrics. "
                f"Skipping this component.",
                stacklevel=2,
            )

    # Map raw weighted sum to [0, 1] based on the theoretical range of the
    # actually-used weights. This makes the score interpretable regardless of
    # the absolute weight magnitudes.
    if used_keys:
        max_possible = sum(max(0.0, weights[k]) for k in used_keys)
        min_possible = sum(min(0.0, weights[k]) for k in used_keys)
        denominator = max_possible - min_possible
        if denominator > 0.0:
            score = (score - min_possible) / denominator

    # Warn if metrics contains keys not used by weights
    unused = set(metrics.keys()) - used_keys
    if unused:
        warnings.warn(
            f"compute_score: metrics contains unused keys: {sorted(unused)}",
            stacklevel=2,
        )

    return float(score)


def compute_empirical_regret(candidate_scores, current_index):
    """
    Compute empirical regret for the current candidate.

    regret = best_candidate_score - current_candidate_score

    Args:
        candidate_scores (np.ndarray): Scores for all candidates.
        current_index (int): Index of the current candidate.

    Returns:
        float: Non-negative regret value.
    """
    best = np.max(candidate_scores)
    current = candidate_scores[current_index]
    return max(0.0, best - current)
