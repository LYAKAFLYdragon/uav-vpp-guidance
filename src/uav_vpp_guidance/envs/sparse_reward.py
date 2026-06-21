"""
R2SP-style trajectory-level reward relabelling kernels.

This module implements the linear and Gaussian causal kernels described in
Huang et al., "Trajectory-level reward relabelling and progressive sub-task
flight control for engineering-grade simulation of adversarial aerial
engagements", Engineering Applications of Artificial Intelligence 181 (2026)
115305.

The relabelling operation is *offline*: the original sparse event/outcome
rewards are recorded along a complete trajectory and then redistributed over
time.  No new geometry-derived reward terms are introduced, so the total
reward magnitude of the original sparse MDP is preserved.
"""

from typing import Iterable, Tuple

import numpy as np


def linear_kernel(window_start: int, window_end: int) -> np.ndarray:
    """
    Normalised uniform (linear) kernel over a discrete integer window.

    Args:
        window_start: First time step included in the window.
        window_end: Last time step included in the window.

    Returns:
        Array of weights of length ``window_end - window_start + 1`` that sum
        to one.
    """
    length = window_end - window_start + 1
    if length <= 0:
        return np.ones(1, dtype=float)
    return np.full(length, 1.0 / length, dtype=float)


def gaussian_kernel(
    window_start: int,
    window_end: int,
    sigma: float = None,
    sigma_ratio: float = 0.5,
) -> np.ndarray:
    """
    Normalised truncated Gaussian kernel peaking at ``window_end``.

    The kernel gives the largest credit to the step at which the event
    occurred and decays backwards in time, matching the causal "backward
    smoothing" interpretation in R2SP.

    Args:
        window_start: First time step included in the window.
        window_end: Last time step included in the window (event step).
        sigma: Standard deviation of the Gaussian.  If ``None`` it is set to
            ``max(1.0, sigma_ratio * (window_end - window_start))``.
        sigma_ratio: Fallback relative standard deviation when ``sigma`` is not
            provided.

    Returns:
        Array of weights that sum to one.
    """
    if window_end < window_start:
        return np.ones(1, dtype=float)

    length = window_end - window_start + 1
    t = np.arange(window_start, window_end + 1, dtype=float)
    peak = float(window_end)

    if sigma is None:
        sigma = max(1.0, sigma_ratio * (window_end - window_start))
    sigma = max(1e-6, float(sigma))

    weights = np.exp(-0.5 * ((t - peak) / sigma) ** 2)
    total = weights.sum()
    if total <= 0.0:
        return np.full(length, 1.0 / length, dtype=float)
    return weights / total


def redistribute_rewards(
    trajectory_length: int,
    terminal_reward: float = 0.0,
    events: Iterable[Tuple[int, float]] = None,
    gaussian_window: int = 50,
    gaussian_sigma_ratio: float = 0.5,
    terminal_kernel: str = "linear",
    event_kernel: str = "gaussian",
) -> np.ndarray:
    """
    Redistribute sparse event and terminal rewards over a trajectory.

    Args:
        trajectory_length: Number of time steps in the trajectory.
        terminal_reward: Scalar outcome reward received at the end of the
            trajectory (e.g. success / crash / timeout).
        events: Iterable of ``(step, reward)`` pairs for non-terminal events.
        gaussian_window: Maximum number of steps before an event over which its
            reward is redistributed.
        gaussian_sigma_ratio: Shape parameter for the Gaussian kernel.
        terminal_kernel: Either ``"linear"`` to spread the terminal reward
            uniformly, or ``"none"`` to leave it at the terminal step.
        event_kernel: Either ``"gaussian"`` to redistribute event rewards with
            a backward Gaussian kernel, or ``"none"`` to leave events at their
            occurrence step.

    Returns:
        Per-step redistributed rewards of length ``trajectory_length``.
    """
    relabelled = np.zeros(trajectory_length, dtype=float)

    if trajectory_length <= 0:
        return relabelled

    # Terminal / outcome reward.
    if terminal_reward != 0.0 and terminal_kernel != "none":
        if terminal_kernel == "linear":
            relabelled += terminal_reward / trajectory_length
        else:
            raise ValueError(f"Unknown terminal_kernel: {terminal_kernel}")

    # Event rewards.
    if event_kernel != "none":
        for step, reward in (events or []):
            if reward == 0.0:
                continue
            if not (0 <= step < trajectory_length):
                continue
            window_start = max(0, step - gaussian_window + 1)
            if event_kernel == "gaussian":
                weights = gaussian_kernel(
                    window_start, step, sigma_ratio=gaussian_sigma_ratio
                )
            else:
                raise ValueError(f"Unknown event_kernel: {event_kernel}")
            relabelled[window_start : step + 1] += reward * weights

    return relabelled
