"""
Cross-Entropy Method optimizer for guidance gains.

Implements the standard CEM algorithm:
1. Sample candidates from current Gaussian distribution.
2. Evaluate each candidate.
3. Select elite (top-performing) candidates.
4. Fit new Gaussian to elite candidates.
5. Repeat until convergence or max iterations.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np


class CEMGainOptimizer:
    """
    Cross-Entropy Method optimizer for guidance gains.
    """

    def __init__(self, gain_space, config):
        """
        Args:
            gain_space (GainSpace): Gain search space.
            config (dict): CEM hyperparameters.
        """
        self.gain_space = gain_space
        self.config = config
        self.candidates = int(config.get("candidates", 12))
        self.elite_ratio = float(config.get("elite_ratio", 0.25))
        self.mean = (gain_space.low + gain_space.high) / 2.0
        self.std = (gain_space.high - gain_space.low) / 4.0

    def sample_candidates(self) -> np.ndarray:
        """Sample candidate gain vectors from current Gaussian distribution.

        Shape: (candidates, dim)
        Uses truncated normal: sample from N(mean, std), then clip to [low, high].
        """
        rng = np.random.default_rng()
        # Sample from normal then clip to bounds
        candidates = rng.normal(
            loc=self.mean, scale=self.std, size=(self.candidates, len(self.mean))
        )
        return self.gain_space.clip(candidates)

    def update(self, candidates: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update distribution using elite candidates.

        Args:
            candidates: (n_candidates, dim) array of gain vectors
            scores: (n_candidates,) array of scores (higher is better)

        Returns:
            (new_mean, new_std)

        Selects top elite_ratio * n_candidates by score.
        New mean = mean of elite candidates.
        New std = std of elite candidates + minimal noise floor (configurable).
        """
        n_elite = max(1, int(len(scores) * self.elite_ratio))
        elite_idx = np.argsort(scores)[-n_elite:]
        elite = candidates[elite_idx]

        new_mean = np.mean(elite, axis=0)
        new_std = np.std(elite, axis=0) + self.config.get("noise_floor", 0.01)

        # Trust region: clip mean to bounds
        new_mean = self.gain_space.clip(new_mean.reshape(1, -1))[0]

        self.mean = new_mean
        self.std = new_std
        return new_mean, new_std

    def optimize(self, evaluator: Callable[[Dict], float], n_iter: int = 50) -> Tuple[dict, list]:
        """Run full CEM optimization loop.

        Args:
            evaluator: Callable that takes a gain dict and returns a score (float, higher=better).
                Signature: evaluator(gains_dict: dict) -> float
            n_iter: Number of CEM iterations.

        Returns:
            (best_gains_dict, history)
            where history is a list of {
                'iteration': int,
                'mean_score': float,
                'best_score': float,
                'mean': np.ndarray,
                'std': np.ndarray,
            }
        """
        history: List[dict] = []
        best_score = -float("inf")
        best_gains = None

        for i in range(n_iter):
            candidates = self.sample_candidates()
            candidate_dicts = [self.gain_space.vector_to_gains(c) for c in candidates]
            scores = np.array([evaluator(g) for g in candidate_dicts], dtype=np.float64)

            self.update(candidates, scores)

            iter_best_idx = int(np.argmax(scores))
            iter_best_score = float(scores[iter_best_idx])
            if iter_best_score > best_score:
                best_score = iter_best_score
                best_gains = candidate_dicts[iter_best_idx]

            history.append({
                "iteration": i,
                "mean_score": float(np.mean(scores)),
                "best_score": iter_best_score,
                "mean": self.mean.copy(),
                "std": self.std.copy(),
            })

            # Early stop: if std is very small, distribution has converged
            if np.all(self.std < self.config.get("convergence_tol", 0.001)):
                break

        return best_gains, history
