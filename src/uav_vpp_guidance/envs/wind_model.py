"""Simplified Dryden turbulence model for the simple point-mass backend.

This is a pragmatic, low-order approximation of the MIL-F-8785C Dryden gust
model.  It generates colored wind components as first-order Gauss-Markov
processes with intensity and scale length chosen to represent light, medium,
and severe turbulence.

The wind vector is returned in NEU coordinates (north, east, up) and is meant
to be added to the aircraft's inertial velocity vector.
"""

from typing import Dict, Optional
import numpy as np


# Intensity table: RMS gust magnitude (sigma) in m/s for each component.
# These are representative low-altitude values rescaled for robustness testing.
_INTENSITY_TABLE: Dict[str, Dict[str, float]] = {
    "light": {"u": 0.5, "v": 0.5, "w": 0.25},
    "medium": {"u": 1.5, "v": 1.5, "w": 0.75},
    "severe": {"u": 3.0, "v": 3.0, "w": 1.5},
}

# Default spatial scale lengths (m).  Dryden uses different values for each
# component; we keep them configurable but provide sensible defaults.
_DEFAULT_SCALE_LENGTH: Dict[str, float] = {
    "u": 530.0,
    "v": 530.0,
    "w": 150.0,
}


class DrydenWindModel:
    """Discrete first-order Gauss-Markov wind gust model.

    State update:
        w_{k+1} = exp(-a dt) w_k + sigma sqrt(1 - exp(-2 a dt)) zeta_k
    where a = speed / scale_length and zeta_k ~ N(0, 1).
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self._intensity = str(cfg.get("intensity", "medium")).lower()
        if self._intensity not in _INTENSITY_TABLE:
            raise ValueError(
                f"Unknown wind intensity '{self._intensity}'. "
                f"Valid: {list(_INTENSITY_TABLE.keys())}"
            )
        sigmas = _INTENSITY_TABLE[self._intensity]
        self._sigma_u = float(cfg.get("sigma_u", sigmas["u"]))
        self._sigma_v = float(cfg.get("sigma_v", sigmas["v"]))
        self._sigma_w = float(cfg.get("sigma_w", sigmas["w"]))
        self._scale_u = float(cfg.get("scale_length_u_m", _DEFAULT_SCALE_LENGTH["u"]))
        self._scale_v = float(cfg.get("scale_length_v_m", _DEFAULT_SCALE_LENGTH["v"]))
        self._scale_w = float(cfg.get("scale_length_w_m", _DEFAULT_SCALE_LENGTH["w"]))

        self._wind = np.zeros(3, dtype=np.float64)  # [north, east, down]
        self._rng = np.random.default_rng(0)

    def reset(self, seed: Optional[int] = None):
        """Reset gust state and optionally seed the internal RNG."""
        self._wind.fill(0.0)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def update(self, speed_mps: float, dt: float) -> np.ndarray:
        """Advance the gust model by one timestep.

        Args:
            speed_mps: Aircraft airspeed (m/s).
            dt: Timestep (s).

        Returns:
            Wind vector in NEU coordinates [north, east, up] (m/s).
        """
        speed = max(float(speed_mps), 1.0)
        dt = float(dt)

        # Discrete first-order Gauss-Markov coefficients.
        def _step(scale: float, sigma: float) -> float:
            a = speed / scale
            decay = np.exp(-a * dt)
            diffusion = sigma * np.sqrt(max(1.0 - decay ** 2, 1e-12))
            return decay, diffusion

        decay_u, diff_u = _step(self._scale_u, self._sigma_u)
        decay_v, diff_v = _step(self._scale_v, self._sigma_v)
        decay_w, diff_w = _step(self._scale_w, self._sigma_w)

        self._wind[0] = decay_u * self._wind[0] + diff_u * self._rng.standard_normal()
        self._wind[1] = decay_v * self._wind[1] + diff_v * self._rng.standard_normal()
        # _wind[2] is stored as down (positive = downward gust).
        self._wind[2] = decay_w * self._wind[2] + diff_w * self._rng.standard_normal()

        # Return NEU: north, east, up.
        return np.array([self._wind[0], self._wind[1], -self._wind[2]], dtype=np.float64)

    def get_wind(self) -> np.ndarray:
        """Return the current wind vector in NEU coordinates without advancing."""
        return np.array([self._wind[0], self._wind[1], -self._wind[2]], dtype=np.float64)
