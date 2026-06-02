"""
Regret computation for gain evaluation.

TODO: Formalize the composite score and regret definitions.
"""

import numpy as np


def compute_score(metrics, weights):
    """
    Composite score for gain evaluation.

    Should include return, success rate, crash rate, saturation rate, and command smoothness.

    Args:
        metrics (dict): Evaluation metrics.
        weights (dict): Score component weights.

    Returns:
        float: Scalar composite score (higher is better).
    """
    # TODO: Define composite score based on metrics.
    raise NotImplementedError


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
