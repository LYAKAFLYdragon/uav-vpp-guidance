"""
Evaluation metrics computation.
"""

import numpy as np


def compute_success_rate(outcomes):
    """
    Compute success rate from episode outcomes.

    Args:
        outcomes (list): List of outcome strings.

    Returns:
        float: Success rate in [0, 1].
    """
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o == "success") / len(outcomes)


def compute_crash_rate(outcomes):
    """Compute crash rate."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o == "crash") / len(outcomes)


def compute_timeout_rate(outcomes):
    """Compute timeout rate."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o == "timeout") / len(outcomes)


def compute_mean_return(returns):
    """Compute mean episode return."""
    if not returns:
        return 0.0
    return float(np.mean(returns))
