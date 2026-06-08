"""
Statistical comparison utilities for benchmark results.

Provides:
- mean ± std
- bootstrap confidence interval
- paired delta by scenario/seed
- Pairwise comparisons: no_prediction vs cv_prediction, etc.
- McNemar exact two-sided p-value

No external dependencies beyond numpy and scipy.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import binomtest, ttest_rel, mannwhitneyu


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation, ignoring NaN."""
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def bootstrap_ci(values: List[float],
                 n_bootstrap: int = 1000,
                 confidence: float = 0.95,
                 random_seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    means = np.sort(means)
    alpha = 1.0 - confidence
    lower_idx = int(np.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(np.ceil((1.0 - alpha / 2) * n_bootstrap)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(n_bootstrap - 1, upper_idx)

    return float(np.mean(arr)), float(means[lower_idx]), float(means[upper_idx])


def paired_delta(baseline_values: List[float],
                 treatment_values: List[float]) -> Tuple[float, float, float]:
    """
    Paired difference: treatment - baseline.

    Returns:
        tuple: (mean_delta, std_delta, n_pairs)
    """
    pairs = [(b, t) for b, t in zip(baseline_values, treatment_values)
             if np.isfinite(b) and np.isfinite(t)]
    if not pairs:
        return np.nan, np.nan, 0

    deltas = np.array([t - b for b, t in pairs], dtype=np.float64)
    return float(np.mean(deltas)), float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0, len(deltas)


def compare_methods(metrics_list: List[Dict],
                    baseline_name: str = "no_prediction") -> Dict:
    """
    Compare all methods against a baseline.

    Args:
        metrics_list: List of aggregated metric dicts (one per method).
        baseline_name: Name of the baseline method.

    Returns:
        dict: Comparison results with keys:
            - per_method: dict of method_name -> {mean_return, std_return, ...}
            - pairwise: dict of "baseline_vs_treatment" -> {mean_delta, std_delta, ci_lower, ci_upper}
    """
    per_method = {}
    for m in metrics_list:
        name = m.get("method", "unknown")
        per_method[name] = {
            "method": name,
            "episodes": m.get("episodes", m.get("num_episodes", 0)),
            "mean_return": m.get("mean_return", np.nan),
            "std_return": m.get("std_return", np.nan),
            "success_rate": m.get("instant_success_rate", m.get("success_rate", np.nan)),
            "score_win_rate": m.get("score_win_rate", np.nan),
            "mean_final_range_m": m.get("mean_final_range_m", np.nan),
            "mean_final_ata_deg": m.get("mean_final_ata_deg", np.nan),
            "timeout_rate": m.get("timeout_rate", np.nan),
            "crash_rate": m.get("crash_rate", np.nan),
            "out_of_bounds_rate": m.get("out_of_bounds_rate", np.nan),
            "prediction_rmse_m": m.get("prediction_rmse_m", np.nan),
            "prediction_fallback_rate": m.get("prediction_fallback_rate", np.nan),
        }

    baseline = per_method.get(baseline_name)
    if baseline is None:
        return {"per_method": per_method, "pairwise": {}, "error": f"Baseline '{baseline_name}' not found"}

    pairwise = {}
    for name, stats in per_method.items():
        if name == baseline_name:
            continue
        key = f"{baseline_name}_vs_{name}"

        # For aggregated metrics we only have single values, so delta is just the difference.
        # If raw episodes were available we could do bootstrap on them.
        delta = stats["mean_return"] - baseline["mean_return"]
        pairwise[key] = {
            "baseline": baseline_name,
            "treatment": name,
            "metric": "mean_return",
            "baseline_value": baseline["mean_return"],
            "treatment_value": stats["mean_return"],
            "delta": delta,
            "relative_delta_pct": (delta / abs(baseline["mean_return"]) * 100.0
                                    if baseline["mean_return"] != 0 and np.isfinite(baseline["mean_return"])
                                    else np.nan),
        }

    return {"per_method": per_method, "pairwise": pairwise}


def compare_per_scenario(method_metrics: Dict[str, Dict],
                         scenario_name: str,
                         metric_key: str = "mean_return") -> Dict:
    """
    Compare methods for a single scenario.

    Args:
        method_metrics: dict of method_name -> aggregated_metrics dict
        scenario_name: scenario to compare
        metric_key: metric to compare

    Returns:
        dict: comparison results
    """
    results = {}
    for method_name, metrics in method_metrics.items():
        per_scenario = metrics.get("per_scenario", {})
        sc = per_scenario.get(scenario_name, {})
        results[method_name] = sc.get(metric_key, np.nan)

    # Deltas relative to no_prediction
    baseline_val = results.get("no_prediction", np.nan)
    deltas = {}
    for name, val in results.items():
        if name == "no_prediction":
            continue
        delta = val - baseline_val if np.isfinite(val) and np.isfinite(baseline_val) else np.nan
        deltas[f"no_prediction_vs_{name}"] = {
            "baseline_value": baseline_val,
            "treatment_value": val,
            "delta": delta,
        }

    return {"scenario": scenario_name, "metric": metric_key, "values": results, "deltas": deltas}


def bootstrap_confidence_interval(data: List[float],
                                    n_bootstrap: int = 10000,
                                    ci: float = 0.95,
                                    random_seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean (paper-level API).

    Args:
        data: Sample data points.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95%% CI).
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    return bootstrap_ci(data, n_bootstrap=n_bootstrap, confidence=ci, random_seed=random_seed)


def bootstrap_success_rate_ci(outcomes: List[int],
                               n_bootstrap: int = 1000,
                               confidence: float = 0.95,
                               random_seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for success rate (proportion).

    Args:
        outcomes: Binary list of outcomes (1 = success, 0 = failure).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (success_rate, lower_bound, upper_bound)
    """
    arr = np.array([v for v in outcomes if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_seed)
    rates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        rates.append(float(np.mean(sample)))

    rates = np.sort(rates)
    alpha = 1.0 - confidence
    lower_idx = int(np.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(np.ceil((1.0 - alpha / 2) * n_bootstrap)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(n_bootstrap - 1, upper_idx)

    return float(np.mean(arr)), float(rates[lower_idx]), float(rates[upper_idx])


def paired_t_test(method_a_results: List[float],
                   method_b_results: List[float]) -> Dict:
    """
    Paired t-test between two methods.

    Args:
        method_a_results: Results from method A (one per episode/seed).
        method_b_results: Results from method B (paired with A).

    Returns:
        dict: {
            't_statistic': float,
            'p_value': float,
            'mean_diff': float,
            'std_diff': float,
            'n_pairs': int,
            'significant_at_05': bool,
            'significant_at_01': bool,
        }
    """
    pairs = [(a, b) for a, b in zip(method_a_results, method_b_results)
             if np.isfinite(a) and np.isfinite(b)]
    if not pairs:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'std_diff': np.nan,
            'n_pairs': 0,
            'significant_at_05': False,
            'significant_at_01': False,
        }

    a_vals = np.array([p[0] for p in pairs], dtype=np.float64)
    b_vals = np.array([p[1] for p in pairs], dtype=np.float64)
    diffs = b_vals - a_vals

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # t-test needs df > 0 (at least 2 pairs)
    if len(diffs) < 2:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': mean_diff,
            'std_diff': 0.0,
            'n_pairs': len(diffs),
            'significant_at_05': False,
            'significant_at_01': False,
        }

    # Handle zero-variance differences (all diffs identical) to avoid scipy nan
    if np.isclose(std_diff, 0.0, atol=1e-12):
        if np.isclose(mean_diff, 0.0, atol=1e-12):
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'mean_diff': 0.0,
                'std_diff': 0.0,
                'n_pairs': len(diffs),
                'significant_at_05': False,
                'significant_at_01': False,
            }
        else:
            return {
                't_statistic': float('inf') if mean_diff > 0 else float('-inf'),
                'p_value': 0.0,
                'mean_diff': mean_diff,
                'std_diff': 0.0,
                'n_pairs': len(diffs),
                'significant_at_05': True,
                'significant_at_01': True,
            }

    t_stat, p_val = ttest_rel(a_vals, b_vals)
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'n_pairs': len(diffs),
        'significant_at_05': bool(p_val < 0.05),
        'significant_at_01': bool(p_val < 0.01),
    }


def cohens_d(method_a_results: List[float],
             method_b_results: List[float]) -> Dict:
    """
    Cohen's d effect size for paired samples.

    Args:
        method_a_results: Results from method A.
        method_b_results: Results from method B (paired with A).

    Returns:
        dict: {
            'd': float,            # Cohen's d (mean_diff / std_diff)
            'mean_diff': float,
            'std_diff': float,
            'n_pairs': int,
            'magnitude': str,      # 'negligible', 'small', 'medium', 'large'
        }
    """
    pairs = [(a, b) for a, b in zip(method_a_results, method_b_results)
             if np.isfinite(a) and np.isfinite(b)]
    if not pairs:
        return {
            'd': np.nan,
            'mean_diff': np.nan,
            'std_diff': np.nan,
            'n_pairs': 0,
            'magnitude': 'unknown',
        }

    a_vals = np.array([p[0] for p in pairs], dtype=np.float64)
    b_vals = np.array([p[1] for p in pairs], dtype=np.float64)
    diffs = b_vals - a_vals

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    n_pairs = len(diffs)

    # Handle zero-variance differences to avoid division-by-zero artifacts
    if np.isclose(std_diff, 0.0, atol=1e-12):
        if np.isclose(mean_diff, 0.0, atol=1e-12):
            d = 0.0
        else:
            d = float('inf') if mean_diff > 0 else float('-inf')
    else:
        d = mean_diff / std_diff

    # Standard thresholds for paired Cohen's d (|d|)
    if np.isinf(d):
        abs_d = float('inf')
    elif np.isfinite(d):
        abs_d = abs(d)
    else:
        abs_d = 0.0

    if abs_d < 0.2:
        magnitude = 'negligible'
    elif abs_d < 0.5:
        magnitude = 'small'
    elif abs_d < 0.8:
        magnitude = 'medium'
    else:
        magnitude = 'large'

    return {
        'd': float(d) if np.isfinite(d) else d,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'n_pairs': n_pairs,
        'magnitude': magnitude,
    }


def mann_whitney_u(method_a_results: List[float],
                   method_b_results: List[float]) -> Dict:
    """
    Mann-Whitney U test (non-parametric) for two independent samples.

    Args:
        method_a_results: Results from method A.
        method_b_results: Results from method B.

    Returns:
        dict: {
            'u_statistic': float,
            'p_value': float,
            'n_a': int,
            'n_b': int,
            'significant_at_05': bool,
            'significant_at_01': bool,
        }
    """
    a_vals = np.array([v for v in method_a_results if np.isfinite(v)], dtype=np.float64)
    b_vals = np.array([v for v in method_b_results if np.isfinite(v)], dtype=np.float64)

    if len(a_vals) == 0 or len(b_vals) == 0:
        return {
            'u_statistic': np.nan,
            'p_value': np.nan,
            'n_a': len(a_vals),
            'n_b': len(b_vals),
            'significant_at_05': False,
            'significant_at_01': False,
        }

    try:
        u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
    except ValueError:
        # All numbers are identical
        return {
            'u_statistic': np.nan,
            'p_value': 1.0,
            'n_a': len(a_vals),
            'n_b': len(b_vals),
            'significant_at_05': False,
            'significant_at_01': False,
        }

    return {
        'u_statistic': float(u_stat),
        'p_value': float(p_val),
        'n_a': int(len(a_vals)),
        'n_b': int(len(b_vals)),
        'significant_at_05': bool(p_val < 0.05),
        'significant_at_01': bool(p_val < 0.01),
    }


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant paired outcomes.

    b: A success, B failure
    c: A failure, B success
    """
    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
