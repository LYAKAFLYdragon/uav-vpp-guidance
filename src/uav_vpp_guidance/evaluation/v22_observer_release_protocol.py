"""Observer-receipt release logic for the V22 non-learning probe."""

from __future__ import annotations

from typing import Any

from uav_vpp_guidance.evaluation.v21_warmup_anchor_protocol import (
    V21AnchorProtocolError,
    V21WarmupAnchorMixin,
)


class V22ObserverReleaseMixin(V21WarmupAnchorMixin):
    """Release once from a valid observer receipt that is isolated from control."""

    def _v22_reset_observer_protocol(self) -> None:
        self._v21_reset_anchor_protocol()
        self._v22_observer_trace: list[dict[str, Any]] = []
        self._v22_warmup_control_consumption = False
        self._v22_post_release_valid_steps = 0
        self._v22_post_release_prediction_fallback = False

    def _v22_after_observation(self) -> None:
        info = self.last_info
        if self.total_steps == 0:
            self.env.set_runtime_vpp_anchor_mode(mode="current_target", reason="v22_observer_warmup")
            self._v21_switches.append({"step": 0, "mode": "current_target", "reason": "v22_observer_warmup"})
            self._v22_observer_trace.append(
                {
                    "step": 0,
                    "event": "initial_anchor",
                    "anchor_mode": "current_target",
                    "observer_updated": False,
                    "observer_prediction_valid": False,
                    "observer_prediction_fallback": False,
                    "observer_prediction_consumed_by_control": False,
                }
            )
            return

        receipt = {
            "step": int(self.total_steps),
            "event": "post_step_receipt",
            "anchor_mode": info.get("runtime_vpp_anchor_mode"),
            "observer_updated": bool(info.get("observer_prediction_updated", False)),
            "observer_prediction_valid": info.get("observer_prediction_valid") is True,
            "observer_prediction_fallback": bool(info.get("observer_prediction_fallback", False)),
            "observer_prediction_consumed_by_control": bool(
                info.get("observer_prediction_consumed_by_control", True)
            ),
            "prediction_valid": info.get("prediction_valid") is True,
            "prediction_fallback": bool(info.get("prediction_fallback", False)),
        }
        self._v22_observer_trace.append(receipt)

        if not self._v21_released and receipt["observer_prediction_consumed_by_control"]:
            self._v22_warmup_control_consumption = True
            raise V21AnchorProtocolError("observer prediction entered control during V22 warm-up")
        if (
            not self._v21_released
            and len(self.history) == 10
            and receipt["observer_prediction_valid"]
            and not receipt["observer_prediction_fallback"]
        ):
            self.env.set_runtime_vpp_anchor_mode(
                mode="predicted_target", reason="v22_first_valid_observer_receipt"
            )
            self._v21_released = True
            self._v21_switches.append(
                {"step": int(self.total_steps), "mode": "predicted_target", "reason": "v22_first_valid_observer_receipt"}
            )
            return
        if self._v21_released:
            if not receipt["prediction_valid"] or receipt["prediction_fallback"]:
                self._v22_post_release_prediction_fallback = True
                raise V21AnchorProtocolError("prediction receipt invalid after V22 anchor release")
            self._v22_post_release_valid_steps += 1
