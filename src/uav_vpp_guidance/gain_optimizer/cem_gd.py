"""
CEM-GD hybrid gain optimizer.

Implements the CEM-GD idea (Huang, 2021): use the Cross-Entropy Method for
coarse global search during the first phase, then switch to local gradient
descent refinement of the best elite candidate during the second phase.

Gradient descent is implemented with finite differences so that the optimizer
works with any black-box guidance-cost evaluator.
"""

from typing import Callable, List, Tuple

import numpy as np

from .cem import CEMGainOptimizer


class CEMGDGainOptimizer(CEMGainOptimizer):
    """
    Cross-Entropy Method + Gradient Descent (CEM-GD) optimizer for guidance gains.

    Additional configuration keys:
      - ``gd_ratio`` (float): fraction of total iterations spent in GD phase
        (default 0.5).
      - ``gd_lr`` (float): gradient-ascent step size applied to the best elite
        vector (default 0.05).
      - ``gd_fd_eps`` (float): finite-difference perturbation for gradient
        estimation (default 1e-3).
      - ``gd_max_iter`` (int): hard cap on GD iterations if not derived from
        ``n_iter`` (default None).
    """

    def __init__(self, gain_space, config):
        super().__init__(gain_space, config)
        self.gd_ratio = float(config.get("gd_ratio", 0.5))
        self.gd_lr = float(config.get("gd_lr", 0.05))
        self.gd_fd_eps = float(config.get("gd_fd_eps", 1e-3))
        self.gd_max_iter = config.get("gd_max_iter", None)
        if self.gd_max_iter is not None:
            self.gd_max_iter = int(self.gd_max_iter)

    def _finite_difference_gradient(
        self, x: np.ndarray, objective: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Estimate gradient of ``objective`` at ``x`` using two-point finite differences.

        Args:
            x (np.ndarray): current parameter vector.
            objective (Callable): maps vector to scalar score (higher is better).

        Returns:
            np.ndarray: estimated gradient vector.
        """
        fx = objective(x)
        grad = np.zeros_like(x)
        eps = self.gd_fd_eps
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            # Avoid leaving the gain space by clipping before evaluating
            x_plus = self.gain_space.clip(x_plus.reshape(1, -1))[0]
            f_plus = objective(x_plus)
            # Use actual perturbation size after clipping
            actual_eps = x_plus[i] - x[i]
            if abs(actual_eps) < 1e-12:
                grad[i] = 0.0
            else:
                grad[i] = (f_plus - fx) / actual_eps
        return grad

    def optimize(self, evaluator: Callable[[dict], float], n_iter: int = 50) -> Tuple[dict, list]:
        """
        Run CEM-GD optimization loop.

        Phase 1 (CEM): standard cross-entropy updates.
        Phase 2 (GD): gradient ascent refinement of the best candidate found so far.

        Args:
            evaluator: Callable that takes a gain dict and returns a score
                (higher is better).
            n_iter: total number of optimizer iterations. Half are CEM by default,
                the remainder are GD (or set via ``gd_ratio``).

        Returns:
            (best_gains_dict, history)
        """
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter}")

        gd_iters = self.gd_max_iter
        if gd_iters is None:
            gd_iters = max(0, int(round(n_iter * self.gd_ratio)))
        cem_iters = max(1, n_iter - gd_iters)
        gd_iters = max(0, n_iter - cem_iters)

        history: List[dict] = []
        best_score = -float("inf")
        best_gains = None
        best_vector = None

        # ---- Phase 1: CEM coarse search ----
        for i in range(cem_iters):
            candidates = self.sample_candidates()
            candidate_dicts = [self.gain_space.vector_to_gains(c) for c in candidates]
            scores = np.array([evaluator(g) for g in candidate_dicts], dtype=np.float64)

            self.update(candidates, scores)

            iter_best_idx = int(np.argmax(scores))
            iter_best_score = float(scores[iter_best_idx])
            if iter_best_score > best_score:
                best_score = iter_best_score
                best_gains = candidate_dicts[iter_best_idx]
                best_vector = candidates[iter_best_idx].copy()

            history.append({
                "phase": "cem",
                "iteration": i,
                "mean_score": float(np.mean(scores)),
                "best_score": float(best_score),
                "iter_best_score": float(iter_best_score),
                "mean": self.mean.copy(),
                "std": self.std.copy(),
            })

            if np.all(self.std < self.config.get("convergence_tol", 0.001)):
                # CEM has converged early; switch to GD immediately
                break

        # Ensure we have a starting point for GD
        if best_vector is None:
            best_vector = self.mean.copy()
            best_gains = self.gain_space.vector_to_gains(best_vector)
            best_score = evaluator(best_gains)

        # ---- Phase 2: GD local refinement ----
        def objective_vec(x: np.ndarray) -> float:
            gains = self.gain_space.vector_to_gains(x)
            return float(evaluator(gains))

        for j in range(gd_iters):
            grad = self._finite_difference_gradient(best_vector, objective_vec)
            candidate = self.gain_space.clip((best_vector + self.gd_lr * grad).reshape(1, -1))[0]
            candidate_score = objective_vec(candidate)

            if candidate_score > best_score:
                best_score = candidate_score
                best_vector = candidate.copy()
                best_gains = self.gain_space.vector_to_gains(best_vector)

            history.append({
                "phase": "gd",
                "iteration": cem_iters + j,
                "cem_iter": cem_iters,
                "gd_iter": j,
                "best_score": float(best_score),
                "grad_norm": float(np.linalg.norm(grad)),
                "mean": self.mean.copy(),
                "std": self.std.copy(),
            })

        return best_gains, history
