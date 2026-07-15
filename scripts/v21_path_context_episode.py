"""Concrete V17 episode subclass with the V21 warm-up anchor protocol."""
from __future__ import annotations

from typing import Any, Mapping

import run_thesis_five_state_v17_path_context_preflight as v17
from uav_vpp_guidance.evaluation.v21_warmup_anchor_protocol import V21WarmupAnchorMixin


class V21PathContextEpisode(V21WarmupAnchorMixin, v17.PathContextEpisode):
    def run(self, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self._v21_reset_anchor_protocol()
        result = super().run(scenario, seed)
        result["v21_anchor_switches"] = list(self._v21_switches)
        result["v21_anchor_released"] = bool(self._v21_released)
        return result

    def _observe(self, observation: Mapping[str, Any]) -> None:
        super()._observe(observation)
        self._v21_after_observation()
