from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor


def _frame(range_rate, aa, ata, energy, altitude):
    return {"range_rate_mps": range_rate, "aa_deg": aa, "ata_deg": ata, "specific_energy_height_m": energy, "altitude_m": altitude}


def test_history_features_are_past_only_normalized_and_reset_safe():
    extractor = TemporalGeometryFeatureExtractor(window_steps=10)
    first = extractor.update(_frame(-100.0, 170.0, 10.0, 5000.0, 5000.0))
    second = extractor.update(_frame(-80.0, -170.0, 20.0, 5200.0, 5100.0))

    assert np.allclose(first, np.zeros(6))
    assert second[0] == pytest.approx(-0.45)
    assert second[1] == pytest.approx(0.4)
    assert second[2] == pytest.approx(20.0 / 180.0)
    assert second[3] == pytest.approx(10.0 / 180.0)
    assert second[4] == pytest.approx(0.2)
    assert second[5] == pytest.approx(0.1)
    extractor.reset()
    assert extractor.size == 0
    assert np.allclose(extractor.features(), np.zeros(6))


def test_history_rejects_nonfinite_input():
    extractor = TemporalGeometryFeatureExtractor()
    with pytest.raises(ValueError, match="non-finite"):
        extractor.update(_frame(np.nan, 0.0, 0.0, 0.0, 0.0))
