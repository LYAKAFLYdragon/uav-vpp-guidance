"""
Delayed prediction error tracker.

Tracks predictions made at time t with a lookahead horizon T,
compares them against the actual target position at time t+T,
and computes error statistics.
"""

from typing import List, Tuple, Optional
import numpy as np


class PredictionErrorTracker:
    """Tracks pending predictions and computes delayed prediction errors."""

    def __init__(self, high_level_dt: float = 0.2):
        self.high_level_dt = high_level_dt
        self._pending: List[Tuple[float, float, np.ndarray]] = []
        self._errors: List[float] = []

    def reset(self):
        """Clear all pending predictions and errors."""
        self._pending.clear()
        self._errors.clear()

    def register_prediction(
        self,
        current_time_s: float,
        lookahead_time_s: float,
        predicted_position_neu: np.ndarray,
    ):
        """Register a prediction to be evaluated later.

        Args:
            current_time_s: Simulation time when prediction was made.
            lookahead_time_s: Prediction horizon in seconds.
            predicted_position_neu: Predicted target position [3] in NEU.
        """
        self._pending.append(
            (current_time_s, lookahead_time_s, np.asarray(predicted_position_neu, dtype=np.float64))
        )

    def update(self, current_time_s: float, actual_target_position_neu: np.ndarray):
        """Check for matured predictions and compute errors.

        Args:
            current_time_s: Current simulation time.
            actual_target_position_neu: Actual target position at current time.
        """
        actual = np.asarray(actual_target_position_neu, dtype=np.float64)
        matured = []
        remaining = []
        for t_pred, t_lookahead, pred_pos in self._pending:
            if current_time_s >= t_pred + t_lookahead - 1e-9:
                error = float(np.linalg.norm(pred_pos - actual))
                self._errors.append(error)
                matured.append((t_pred, t_lookahead, pred_pos))
            else:
                remaining.append((t_pred, t_lookahead, pred_pos))
        self._pending = remaining

    @property
    def latest_error(self) -> Optional[float]:
        return self._errors[-1] if self._errors else None

    @property
    def mean_error(self) -> Optional[float]:
        return float(np.mean(self._errors)) if self._errors else None

    @property
    def median_error(self) -> Optional[float]:
        return float(np.median(self._errors)) if self._errors else None

    @property
    def error_count(self) -> int:
        return len(self._errors)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def get_stats(self) -> dict:
        return {
            "latest_prediction_error_m": self.latest_error,
            "mean_prediction_error_m": self.mean_error,
            "median_prediction_error_m": self.median_error,
            "prediction_error_count": self.error_count,
            "pending_prediction_count": self.pending_count,
        }
