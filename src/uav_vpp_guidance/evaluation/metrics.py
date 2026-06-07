"""
Evaluation metrics computation.
"""

import warnings

import numpy as np


def _get_reason(ep):
    """Extract termination reason from episode dict (multiple naming conventions)."""
    for key in ("reason", "outcome", "result"):
        val = ep.get(key)
        if val is not None:
            return str(val)
    return ""


def _get_return(ep):
    """Extract episode return from episode dict (multiple naming conventions)."""
    for key in ("return", "episode_return", "ep_return"):
        val = ep.get(key)
        if val is not None:
            return float(val)
    return 0.0


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


def compute_success_rate_with_ci(episodes, ci=0.95):
    """
    Compute success rate with bootstrap confidence interval.

    Args:
        episodes: List of episode dicts, each with an 'outcome' key.
        ci: Confidence level for the bootstrap CI.

    Returns:
        dict: {
            'success_rate': float,
            'ci_lower': float,
            'ci_upper': float,
            'n_total': int,
            'n_success': int,
        }
    """
    from .statistical_comparison import bootstrap_confidence_interval

    outcomes = [_get_reason(ep) for ep in episodes]
    n_total = len(outcomes)
    n_success = sum(1 for o in outcomes if o == "success")

    if n_total == 0:
        return {
            "success_rate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_total": 0,
            "n_success": 0,
        }

    # Binary outcomes for bootstrap: 1 for success, 0 otherwise
    binary = [1.0 if o == "success" else 0.0 for o in outcomes]
    mean, ci_lower, ci_upper = bootstrap_confidence_interval(binary, n_bootstrap=10000, ci=ci)

    return {
        "success_rate": float(mean),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_total": n_total,
        "n_success": n_success,
    }


def aggregate_metrics_with_statistics(episodes_list, method_names):
    """
    Aggregate metrics across multiple method runs with statistical comparison.

    Args:
        episodes_list: List of episode lists, one per method.
        method_names: List of method names matching episodes_list.

    Returns:
        dict: {
            'per_method': list of metric dicts per method,
            'pairwise': dict of pairwise comparisons,
        }
    """
    from .statistical_comparison import paired_t_test, cohens_d

    # Validate seed alignment across methods
    baseline_seeds = {ep.get("seed") for ep in episodes_list[0] if ep.get("seed") is not None}
    for i in range(1, len(episodes_list)):
        other_seeds = {ep.get("seed") for ep in episodes_list[i] if ep.get("seed") is not None}
        common = baseline_seeds & other_seeds
        if len(common) < max(1, len(baseline_seeds) // 2):
            warnings.warn(
                f"Method {method_names[i]} has only {len(common)} common seeds with baseline",
                stacklevel=2,
            )

    per_method = []
    for episodes, name in zip(episodes_list, method_names):
        outcomes = [_get_reason(ep) for ep in episodes]
        returns = [_get_return(ep) for ep in episodes]
        sr_info = compute_success_rate_with_ci(episodes)

        per_method.append({
            "method": name,
            "episodes": len(episodes),
            "success_rate": sr_info["success_rate"],
            "success_rate_ci_lower": sr_info["ci_lower"],
            "success_rate_ci_upper": sr_info["ci_upper"],
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "crash_rate": compute_crash_rate(outcomes),
            "timeout_rate": compute_timeout_rate(outcomes),
        })

    pairwise = {}
    baseline_name = "no_prediction" if "no_prediction" in method_names else (method_names[0] if method_names else None)
    if baseline_name and len(method_names) > 1:
        baseline_idx = method_names.index(baseline_name)
        for i in range(len(method_names)):
            if i == baseline_idx:
                continue
            treatment_name = method_names[i]
            key = f"{baseline_name}_vs_{treatment_name}"

            # Extract paired returns (assumes episodes are aligned by seed/order)
            baseline_returns = [_get_return(ep) for ep in episodes_list[baseline_idx]]
            treatment_returns = [_get_return(ep) for ep in episodes_list[i]]

            pairwise[key] = {
                "baseline": baseline_name,
                "treatment": treatment_name,
                "t_test": paired_t_test(baseline_returns, treatment_returns),
                "cohens_d": cohens_d(baseline_returns, treatment_returns),
            }

    return {
        "per_method": per_method,
        "pairwise": pairwise,
    }
