"""Pure deterministic F08 CEM credibility and F09 metric-label evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class CEMPlan:
    name: str
    candidates: int
    elite_ratio: float
    iterations: int
    covariance: str = "diagonal"
    early_stop_std: float | None = None


def _objective(points: np.ndarray) -> np.ndarray:
    """Fixed bounded multimodal surrogate; never represents flight performance."""
    target = np.array([0.2, -0.35, 0.1, 0.4, -0.2])
    basin = -np.sum((points - target) ** 2, axis=1)
    ripple = 0.12 * np.cos(8.0 * points[:, 0]) * np.cos(6.0 * points[:, 1])
    return basin + ripple


def run_plan(plan: CEMPlan, seed: int) -> dict:
    if plan.candidates * plan.iterations != 120:
        raise ValueError("each offline plan must use exactly 120 objective evaluations")
    rng = np.random.default_rng(seed)
    mean, std, best, history = np.zeros(5), np.full(5, 0.5), -float("inf"), []
    for iteration in range(plan.iterations):
        points = np.clip(rng.normal(mean, std, size=(plan.candidates, 5)), -1.0, 1.0)
        scores = _objective(points); elite_count = max(1, int(plan.candidates * plan.elite_ratio))
        elite = points[np.argsort(scores)[-elite_count:]]
        mean = elite.mean(axis=0)
        if plan.covariance == "full":
            covariance = np.atleast_2d(np.cov(elite, rowvar=False)) + np.eye(5) * 0.01
            std = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        else:
            std = elite.std(axis=0) + 0.01
        best = max(best, float(scores.max()))
        history.append({"iteration": iteration, "mean_score": float(scores.mean()), "iter_best_score": float(scores.max()), "best_score": best, "coverage_unique_bins": int(np.unique(np.floor((points + 1.0) * 4.0).astype(int), axis=0).shape[0]), "mean_std": float(std.mean())})
        if plan.early_stop_std is not None and np.all(std < plan.early_stop_std):
            break
    return {"plan": asdict(plan), "seed": seed, "evaluations": plan.candidates * len(history), "history": history, "best_score": best, "final_mean_std": float(std.mean()), "early_stopped": len(history) < plan.iterations}


def compare_fixed_budget(seeds=(0, 1, 2)) -> dict:
    plans = (CEMPlan("frozen_12_candidates_3_elites_diagonal", 12, 0.25, 10), CEMPlan("population_24_elite_25pct_diagonal", 24, 0.25, 5), CEMPlan("population_12_elite_50pct_full_covariance", 12, 0.5, 10), CEMPlan("population_12_elite_25pct_early_stop", 12, 0.25, 10, early_stop_std=0.02))
    reports = {plan.name: [run_plan(plan, seed) for seed in seeds] for plan in plans}
    summary = {name: {"best_score_distribution": [item["best_score"] for item in runs], "median_best_score": float(np.median([item["best_score"] for item in runs])), "mean_final_std": float(np.mean([item["final_mean_std"] for item in runs])), "all_fixed_budget_evaluations": all(item["evaluations"] <= 120 for item in runs)} for name, runs in reports.items()}
    return {"objective": "deterministic 5D bounded surrogate; not an environment, policy, gain, safety, or transfer result", "budget_objective_evaluations": 120, "seed_list": list(seeds), "plans": reports, "summary": summary}


def f09_metric_definition() -> dict:
    return {"current_function": "compute_empirical_regret(candidate_scores, current_index)", "truthful_label": "within-batch empirical score gap to the best sampled candidate", "formula": "max(candidate_scores) - candidate_scores[current_index]", "is_paired_regret": False, "is_rollback_signal": False, "training_behavior_change": False, "required_context": "candidate batch, score definition, seed/scenario aggregation, and comparison baseline"}
