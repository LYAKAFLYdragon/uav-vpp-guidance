"""
Backend-agnostic actuator dynamics model.

Inserts a realistic command-transmission layer between the guidance law
and the aircraft dynamics:
  - first-order lag per channel (time constant tau_s)
  - pure delay in high-level steps (delay_steps)
  - per-channel rate limits (rate_limit_per_s)
  - final saturation using the configured limits

This layer is intentionally placed inside CloseRangeTrackingEnv so that
both the simple point-mass backend and the JSBSim backend can share the
same actuator model without changing either backend.
"""

from collections import deque
from typing import Dict, Any, Optional, Tuple

import numpy as np


# Map command channel names to the corresponding limit keys in the
# standard `limits` / `guidance.limits` config blocks.
_LIMIT_KEYS = {
    "nz_cmd": ("nz_min", "nz_max"),
    "roll_rate_cmd": ("roll_rate_min", "roll_rate_max"),
    "throttle_cmd": ("throttle_min", "throttle_max"),
}


def _resolve_limits(limits: Optional[Dict[str, float]], channel: str) -> Tuple[Optional[float], Optional[float]]:
    """Resolve (min, max) limit pair for a command channel."""
    if limits is None:
        return None, None
    lo_key, hi_key = _LIMIT_KEYS.get(channel, (None, None))
    lo = limits.get(lo_key) if lo_key else None
    hi = limits.get(hi_key) if hi_key else None
    return lo, hi


class ActuatorDynamics:
    """
    Actuator dynamics with lag, delay, rate limiting, and saturation.

    All effects are optional and config-driven. When disabled or when no
    dynamic parameters are specified, ``step`` returns the input command
    unchanged (except for saturation, which is always applied if limits
    are provided and ``apply_saturation`` is True).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, dt: float = 0.2):
        """
        Args:
            config (dict): Configuration dictionary. Expected keys:
                - enabled (bool): master switch, default False.
                - channels (list/tuple): command channel names to process.
                  Default: ["nz_cmd", "roll_rate_cmd", "throttle_cmd"].
                - tau_s (dict): first-order time constant per channel (seconds).
                - delay_steps (int): pure delay in high-level steps.
                - rate_limit_per_s (dict): max rate of change per channel
                  (units per second).
                - limits (dict): saturation limits with keys nz_min/max,
                  roll_rate_min/max, throttle_min/max.
                - apply_saturation (bool): whether to apply final saturation.
                  Default True.
            dt (float): high-level time step in seconds.
        """
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.dt = float(dt)
        self.channels = tuple(config.get("channels", ("nz_cmd", "roll_rate_cmd", "throttle_cmd")))
        self.tau_s = {str(k): float(v) for k, v in config.get("tau_s", {}).items()}
        self.delay_steps = max(0, int(config.get("delay_steps", 0)))
        self.rate_limit_per_s = {
            str(k): float(v) for k, v in config.get("rate_limit_per_s", {}).items()
        }
        self.limits = config.get("limits", {})
        self.apply_saturation = bool(config.get("apply_saturation", True))

        # Internal state
        self._lag_state: Dict[str, float] = {}
        self._delay_buffers: Dict[str, deque] = {}
        self._prev_out: Dict[str, float] = {}

    def reset(self) -> None:
        """Clear all internal dynamic state."""
        self._lag_state = {}
        self._delay_buffers = {ch: deque() for ch in self.channels}
        self._prev_out = {}

    def step(self, command: Dict[str, float]) -> Dict[str, float]:
        """
        Apply actuator dynamics to a command dict.

        Args:
            command (dict): raw command with the configured channels.

        Returns:
            dict: actuated command with the same channels.
        """
        if not self.enabled:
            return dict(command)

        result = {}
        for ch in self.channels:
            u = float(command.get(ch, 0.0))

            # 1. First-order lag: y += alpha * (u - y)
            y = self._lag_state.get(ch, u)
            tau = self.tau_s.get(ch, 0.0)
            if tau is not None and tau > 0.0:
                alpha = self.dt / (tau + self.dt)
                y = y + alpha * (u - y)
            else:
                y = u

            # 2. Pure delay via FIFO buffer
            if self.delay_steps > 0:
                buf = self._delay_buffers.get(ch)
                if buf is None:
                    buf = deque()
                    self._delay_buffers[ch] = buf
                buf.append(y)
                if len(buf) > self.delay_steps:
                    delayed = buf.popleft()
                else:
                    # Buffer not yet full: hold the oldest (first) value so
                    # the output does not jump before the delay is established.
                    delayed = buf[0]
                y = delayed

            # 3. Rate limit
            rate_limit = self.rate_limit_per_s.get(ch)
            if rate_limit is not None and rate_limit >= 0.0 and ch in self._prev_out:
                max_delta = rate_limit * self.dt
                prev = self._prev_out[ch]
                y = float(np.clip(y, prev - max_delta, prev + max_delta))

            # 4. Saturation
            if self.apply_saturation:
                lo, hi = _resolve_limits(self.limits, ch)
                if lo is not None and hi is not None:
                    y = float(np.clip(y, lo, hi))

            result[ch] = y
            self._lag_state[ch] = y
            self._prev_out[ch] = y

        return result

    def get_state(self) -> Dict[str, Any]:
        """Return serializable internal state for checkpointing / tests."""
        return {
            "lag_state": dict(self._lag_state),
            "delay_buffers": {ch: list(buf) for ch, buf in self._delay_buffers.items()},
            "prev_out": dict(self._prev_out),
        }
