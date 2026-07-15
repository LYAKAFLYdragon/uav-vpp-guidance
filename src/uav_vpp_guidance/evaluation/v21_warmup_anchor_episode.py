"""V17-compatible episode wrapper for the V21 one-way anchor protocol."""

from __future__ import annotations

from typing import Any, Mapping

from uav_vpp_guidance.evaluation.v21_warmup_anchor_protocol import V21WarmupAnchorMixin


class V21WarmupAnchorEpisode(V21WarmupAnchorMixin):
    """Mixin adapter; combine before a V17-compatible episode class."""

    def _v21_run(self, parent_run, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self._v21_reset_anchor_protocol()
        result = parent_run(scenario, seed)
        result["v21_anchor_switches"] = list(self._v21_switches)
        result["v21_anchor_released"] = bool(self._v21_released)
        return result

    def _v21_observe(self, parent_observe, observation: Mapping[str, Any]) -> None:
        parent_observe(observation)
        self._v21_after_observation()
