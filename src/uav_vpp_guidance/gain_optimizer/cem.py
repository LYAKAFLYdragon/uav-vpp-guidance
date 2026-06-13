"""
Cross-Entropy Method optimizers for guidance gains.

Provides three variants:
- CEMGainOptimizer: standard CEM (fit Gaussian to elite candidates).
- CEMEMAGainOptimizer: EMA-smoothed CEM (low-pass filter on mean/std).
- CEMGDGainOptimizer: score-weighted gradient ascent on the distribution mean.
"""

import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np


class CEMGainOptimizer:
    """Standard cross-entropy method for guidance gains."""

    def __init__(self, gain_space, config):
        self.gain_space = gain_space
        self.config = config
        self.candidates = int(config.get("candidates", 12))
        self.elite_ratio = float(config.get("elite_ratio", 0.25))
        self.rng = np.random.default_rng(config.get("random_seed", 42))
        self.mean = (gain_space.low + gain_space.high) / 2.0
        self.std = (gain_space.high - gain_space.low) / 4.0

    def sample_candidates(self) -> np.ndarray:
        """Sample candidate gain vectors from current Gaussian distribution."""
        candidates = self.rng.normal(
            loc=self.mean, scale=self.std, size=(self.candidates, len(self.mean))
        )
        return self.gain_space.clip(candidates)

    def update(self, candidates: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update distribution using elite candidates."""
        n_elite = max(1, int(len(scores) * self.elite_ratio))
        elite_idx = np.argsort(scores)[-n_elite:]
        elite = candidates[elite_idx]

        new_mean = np.mean(elite, axis=0)
        new_std = np.std(elite, axis=0) + self.config.get("noise_floor", 0.01)

        new_mean = self.gain_space.clip(new_mean.reshape(1, -1))[0]

        self.mean = new_mean
        self.std = new_std
        return new_mean, new_std

    def optimize(self, evaluator: Callable[[Dict], float], n_iter: int = 50) -> Tuple[dict, list]:
        """Run full CEM optimization loop."""
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
                "best_score": float(best_score),
                "iter_best_score": float(iter_best_score),
                "mean": self.mean.copy(),
                "std": self.std.copy(),
            })

            if np.all(self.std < self.config.get("convergence_tol", 0.001)):
                break

        return best_gains, history


class CEMEMAGainOptimizer(CEMGainOptimizer):
    """CEM with exponential moving average smoothing of mean and std.

    The EMA update reduces high-frequency oscillations caused by noisy elite
    estimates, which is particularly helpful when the gain landscape is flat
    or stochastic.
    """

    def __init__(self, gain_space, config):
        super().__init__(gain_space, config)
        self.beta = float(config.get("beta_ema", 0.7))

    def update(self, candidates: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EMA-smoothed update."""
        n_elite = max(1, int(len(scores) * self.elite_ratio))
        elite_idx = np.argsort(scores)[-n_elite:]
        elite = candidates[elite_idx]

        elite_mean = np.mean(elite, axis=0)
        elite_std = np.std(elite, axis=0) + self.config.get("noise_floor", 0.01)

        new_mean = self.beta * self.mean + (1.0 - self.beta) * elite_mean
        new_std = self.beta * self.std + (1.0 - self.beta) * elite_std

        new_mean = self.gain_space.clip(new_mean.reshape(1, -1))[0]

        self.mean = new_mean
        self.std = new_std
        return new_mean, new_std


class CEMGDGainOptimizer(CEMGainOptimizer):
    """Score-weighted gradient-ascent variant of CEM.

    .. deprecated::

        CEM-GD hybrid is not recommended for flat, noisy landscapes.
        Use ``CEMEMAGainOptimizer`` instead. See Theorem 3' and
        docs/status.md for the canonical optimizer choice.

    Instead of fitting a Gaussian to elites, this optimizer estimates the
    gradient of the expected score with respect to the distribution mean and
    takes a step of size alpha in that direction:

        grad = sum_i (c_i - mean) * score_i / candidates
        mean <- mean + alpha * grad

    This is the simplest CEM-GD hybrid; it can be unstable on flat or noisy
    landscapes, which is why the theoretical analysis recommends EMA instead.
    """

    def __init__(self, gain_space, config):
        super().__init__(gain_space, config)
        warnings.warn(
            "CEMGDGainOptimizer is deprecated. Use CEMEMAGainOptimizer for "
            "flat, noisy gain landscapes. See Theorem 3' and docs/status.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.alpha = float(config.get("alpha_gd", 0.05))

    def update(self, candidates: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradient-ascent update on the distribution mean."""
        centered = candidates - self.mean
        # Normalize scores to zero mean for a stable gradient estimate
        norm_scores = scores - np.mean(scores)
        grad = np.mean(centered * norm_scores[:, None], axis=0)

        new_mean = self.mean + self.alpha * grad
        # Shrink std slightly to concentrate search around the new mean
        shrink = self.config.get("gd_std_shrink", 0.99)
        noise_floor = self.config.get("noise_floor", 0.01)
        new_std = np.maximum(self.std * shrink, noise_floor)

        new_mean = self.gain_space.clip(new_mean.reshape(1, -1))[0]

        self.mean = new_mean
        self.std = new_std
        return new_mean, new_std
