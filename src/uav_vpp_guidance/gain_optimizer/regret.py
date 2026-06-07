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
    "command_smoothness": 10.0,
}


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
        "command_smoothness": regret.get("w_command_smooth", 10.0),
    }
    return mapping


def compute_score(metrics, weights=None):
    """
    Composite score for gain evaluation.

    Computes a weighted sum of evaluation metrics. Positive weights reward
    higher values; negative weights penalize higher values (e.g. crash_rate).

    Args:
        metrics (dict): Evaluation metrics. Expected keys include:
            - "return" (float): average episode return
            - "success_rate" (float): success rate in [0, 1]
            - "crash_rate" (float): crash rate in [0, 1]
            - "saturation_rate" (float): actuator saturation rate in [0, 1]
            - "command_smoothness" (float): command smoothness in [0, 1]
        weights (dict, optional): Score component weights. If None, loads
            defaults from ``config/gain_space.yaml``.

    Returns:
        float: Scalar composite score (higher is better).
    """
    if weights is None:
        weights = _load_weights_from_config()

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
