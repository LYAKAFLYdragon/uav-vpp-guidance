"""One-way, receipt-gated VPP anchor transition used only by V21 probes."""

from __future__ import annotations

from typing import Any, Mapping


class V21AnchorProtocolError(RuntimeError):
    """Raised when a warm-up anchor transition violates the frozen protocol."""


class V21WarmupAnchorMixin:
    """Mix into a continuous episode that exposes env, history, and last_info."""

    def _v21_reset_anchor_protocol(self) -> None:
        self._v21_released = False
        self._v21_switches: list[dict[str, Any]] = []

    def _v21_after_observation(self) -> None:
        info: Mapping[str, Any] = self.last_info
        if self.total_steps == 0:
            self.env.set_runtime_vpp_anchor_mode(mode="current_target", reason="v21_predictor_warmup")
            self._v21_switches.append({"step": 0, "mode": "current_target", "reason": "v21_predictor_warmup"})
            return
        valid = info.get("prediction_valid") is True
        fallback = bool(info.get("prediction_fallback", False))
        if not self._v21_released and len(self.history) == 10 and valid and not fallback:
            self.env.set_runtime_vpp_anchor_mode(mode="predicted_target", reason="v21_first_valid_prediction")
            self._v21_released = True
            self._v21_switches.append({"step": int(self.total_steps), "mode": "predicted_target", "reason": "v21_first_valid_prediction"})
            return
        if self._v21_released and (not valid or fallback):
            raise V21AnchorProtocolError("prediction receipt invalid after V21 anchor release")
