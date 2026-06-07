"""
Regret computation for gain evaluation.
"""

import os
import warnings

import numpy as np
import yaml


_DEFAULT_WEIGHTS = {
    "return_norm": 0.3,
    "success_rate": 0.4,
    "crash_rate": -0.15,
    "saturation_rate": -0.15,
}


def _load_weights_from_config(config_path="config/gain_space.yaml"):
    """Load default weights from gain_space.yaml regret block."""
    if not os.path.exists(config_path):
        return _DEFAULT_WEIGHTS.copy()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    regret = data.get("regret", {}) if data else {}
    mapping = {
        "return_norm": regret.get("w_return_norm", 0.3),
        "success_rate": regret.get("w_success", 0.4),
        "crash_rate": -abs(regret.get("w_crash", 0.15)),
        "saturation_rate": -abs(regret.get("w_saturation", 0.15)),
    }
    return mapping


def compute_score(metrics, weights=None):
    """
    Composite score for gain evaluation.

    1. Dynamically min-max normalizes ``return`` to ``return_norm`` in [0, 1]
       using ``return_min`` / ``return_max`` from ``metrics`` (defaults to
       ``-1000`` / ``1000``).
    2. Computes a weighted sum of normalized metrics.
    3. Rescales the raw sum to [0, 1] based on the theoretical extrema of the
       actually-used weights.

    Args:
        metrics (dict): Evaluation metrics. Expected keys include:
            - "return" (float): average episode return
            - "return_min" (float, optional): lower bound for return normalization
            - "return_max" (float, optional): upper bound for return normalization
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

    # Build a mutable copy and inject the normalized return.
    evaluated = dict(metrics)
    if "return" in evaluated:
        raw_return = float(evaluated.pop("return"))
        ret_min = float(evaluated.pop("return_min", -1000.0))
        ret_max = float(evaluated.pop("return_max", 1000.0))
        denom = ret_max - ret_min
        if denom <= 0.0:
            ret_norm = 0.5
        else:
            ret_norm = (raw_return - ret_min) / denom
        evaluated["return_norm"] = float(np.clip(ret_norm, 0.0, 1.0))

    score = 0.0
    used_keys = set()
    for key, weight in weights.items():
        if key in evaluated:
            score += weight * float(evaluated[key])
            used_keys.add(key)
        else:
            warnings.warn(
                f"compute_score: metric key '{key}' not found in metrics. "
                f"Skipping this component.",
                stacklevel=2,
            )

    # Map raw weighted sum to [0, 1] based on the theoretical range of the
    # actually-used weights.
    if used_keys:
        max_possible = sum(max(0.0, weights[k]) for k in used_keys)
        min_possible = sum(min(0.0, weights[k]) for k in used_keys)
        denominator = max_possible - min_possible
        if denominator > 0.0:
            score = (score - min_possible) / denominator

    # Warn if metrics contains keys not used by weights
    unused = set(evaluated.keys()) - used_keys
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
