from __future__ import annotations

import pytest

from uav_vpp_guidance.evaluation.v21_warmup_anchor_protocol import V21AnchorProtocolError, V21WarmupAnchorMixin


class _Env:
    def __init__(self): self.calls = []
    def set_runtime_vpp_anchor_mode(self, **kwargs): self.calls.append(kwargs)


class _Episode(V21WarmupAnchorMixin):
    def __init__(self): self.env = _Env(); self.total_steps = 0; self.history = []; self.last_info = {}; self._v21_reset_anchor_protocol()


def test_v21_releases_once_after_valid_ten_frame_receipt():
    item = _Episode(); item._v21_after_observation(); item.total_steps = 10; item.history = [0] * 10; item.last_info = {"prediction_valid": True, "prediction_fallback": False}; item._v21_after_observation()
    assert item.env.calls == [{"mode": "current_target", "reason": "v21_predictor_warmup"}, {"mode": "predicted_target", "reason": "v21_first_valid_prediction"}]


def test_v21_fails_closed_after_release():
    item = _Episode(); item._v21_released = True; item.total_steps = 11; item.history = [0] * 10; item.last_info = {"prediction_valid": False, "prediction_fallback": True}
    with pytest.raises(V21AnchorProtocolError, match="after V21"):
        item._v21_after_observation()
