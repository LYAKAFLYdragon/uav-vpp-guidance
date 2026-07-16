"""Observer-receipt release logic for the V22 non-learning probe."""

from __future__ import annotations

from uav_vpp_guidance.evaluation.v21_warmup_anchor_protocol import (
    V21AnchorProtocolError,
    V21WarmupAnchorMixin,
)


class V22ObserverReleaseMixin(V21WarmupAnchorMixin):
    """Release only from an observer-only, unconsumed valid prediction receipt."""

    def _v22_after_observation(self) -> None:
        info = self.last_info
        if self.total_steps == 0:
            self.env.set_runtime_vpp_anchor_mode(mode="current_target", reason="v22_observer_warmup")
            self._v21_switches.append({"step": 0, "mode": "current_target", "reason": "v22_observer_warmup"})
            return
        observer_valid = info.get("observer_prediction_valid") is True
        observer_fallback = bool(info.get("observer_prediction_fallback", False))
        observer_consumed = bool(info.get("observer_prediction_consumed_by_control", True))
        if not self._v21_released and len(self.history) == 10 and observer_valid and not observer_fallback and not observer_consumed:
            self.env.set_runtime_vpp_anchor_mode(mode="predicted_target", reason="v22_first_valid_observer_receipt")
            self._v21_released = True
            self._v21_switches.append({"step": int(self.total_steps), "mode": "predicted_target", "reason": "v22_first_valid_observer_receipt"})
            return
        if self._v21_released and (info.get("prediction_valid") is not True or bool(info.get("prediction_fallback", False))):
            raise V21AnchorProtocolError("prediction receipt invalid after V22 anchor release")
