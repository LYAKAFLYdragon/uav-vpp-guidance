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
from typing import List, Dict, Tuple, Optional
from scipy.stats import binomtest


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
