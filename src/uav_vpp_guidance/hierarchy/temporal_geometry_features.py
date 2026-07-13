"""Explicit past-only geometry statistics for five-state v1."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable

import numpy as np

from .five_state_observation_contract import EXPLICIT_HISTORY_FEATURES


def _signed_angle_delta_deg(current: float, previous: float) -> float:
    delta = (float(current) - float(previous) + 180.0) % 360.0 - 180.0
    return float(delta)


class TemporalGeometryFeatureExtractor:
    """Maintain a reset-safe, fixed-size history of observed geometry only."""

    REQUIRED_FIELDS = (
        "range_rate_mps",
        "aa_deg",
        "ata_deg",
        "specific_energy_height_m",
        "altitude_m",
    )

    def __init__(self, window_steps: int = 10):
        if int(window_steps) < 2:
            raise ValueError("window_steps must be at least 2")
        self.window_steps = int(window_steps)
        self._frames: deque[Dict[str, float]] = deque(maxlen=self.window_steps)

    def reset(self) -> None:
        self._frames.clear()

    @property
    def size(self) -> int:
        return len(self._frames)

    def update(self, frame: Dict[str, float]) -> np.ndarray:
        normalized = {}
        for field in self.REQUIRED_FIELDS:
            value = float(frame[field])
            if not np.isfinite(value):
                raise ValueError(f"Temporal frame contains non-finite {field}")
            normalized[field] = value
        self._frames.append(normalized)
        return self.features()

    def features(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(len(EXPLICIT_HISTORY_FEATURES), dtype=np.float32)
        frames = list(self._frames)
        if len(frames) < 2:
            return np.zeros(len(EXPLICIT_HISTORY_FEATURES), dtype=np.float32)
        rates = np.asarray([item["range_rate_mps"] for item in frames], dtype=np.float64)
        first = frames[0]
        last = frames[-1]
        # Closed-form least-squares slope avoids a LAPACK call in the online loop.
        steps = np.arange(len(rates), dtype=np.float64)
        centered_steps = steps - np.mean(steps)
        slope = float(
            np.sum(centered_steps * (rates - np.mean(rates)))
            / np.sum(centered_steps * centered_steps)
        )
        values = np.asarray(
            [
                float(np.mean(rates)) / 200.0,
                slope / 50.0,
                _signed_angle_delta_deg(last["aa_deg"], first["aa_deg"]) / 180.0,
                _signed_angle_delta_deg(last["ata_deg"], first["ata_deg"]) / 180.0,
                (last["specific_energy_height_m"] - first["specific_energy_height_m"])
                / 1000.0,
                (last["altitude_m"] - first["altitude_m"]) / 1000.0,
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)


def history_feature_names() -> Iterable[str]:
    return EXPLICIT_HISTORY_FEATURES
