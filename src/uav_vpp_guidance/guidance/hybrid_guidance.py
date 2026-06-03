"""
Hybrid guidance: geometric (LOS rate) ↔ proportional navigation switching.

Switches between LOSRateGuidance and ProportionalNavigationGuidance based on
dynamic conditions (range, engagement geometry, energy state). Provides the
best of both worlds: geometric precision in terminal phase and PN efficiency
in midcourse.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

from .los_rate_guidance import LOSRateGuidance, _extract_position, _extract_velocity
from .proportional_navigation import ProportionalNavigationGuidance

logger = logging.getLogger(__name__)


class HybridGuidance:
    """
    Hybrid guidance law with configurable switching logic.

    Switching modes (configurable via guidance.params.hybrid_mode):
        - 'range': Switch based purely on range threshold.
        - 'energy': Switch based on energy state (speed + altitude).
        - 'blended': Continuously blend commands from both laws.

    Config keys (under guidance.params):
        - hybrid_mode: str, one of 'range', 'energy', 'blended' (default 'range')
        - range_threshold_m: Range at which to switch from PN to LOS (default 3000)
        - blend_transition_m: Width of blending zone for 'blended' mode (default 1000)
        - energy_speed_threshold_mps: Speed threshold for 'energy' mode (default 220)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.params = config.get("params", {})
        self.gains = config.get("gains", {})

        self.mode = str(self.params.get("hybrid_mode", "range")).lower()
        self.range_threshold_m = float(self.params.get("range_threshold_m", 3000.0))
        self.blend_transition_m = float(self.params.get("blend_transition_m", 1000.0))
        self.energy_speed_threshold_mps = float(
            self.params.get("energy_speed_threshold_mps", 220.0)
        )
        self.epsilon = float(self.params.get("epsilon", 1.0e-6))

        # Hysteresis / dwell-time anti-chatter
        self.hysteresis_m = float(self.params.get("hysteresis_m", 500.0))
        self.energy_speed_hysteresis_mps = float(
            self.params.get("energy_speed_hysteresis_mps", 20.0)
        )
        self.min_dwell_steps = int(self.params.get("min_dwell_steps", 3))

        # Instantiate sub-guidance laws
        self.los_guidance = LOSRateGuidance(config)
        self.pn_guidance = ProportionalNavigationGuidance(config)

        # Tracking
        self._active_law: str = "pn"
        self._steps_in_law: int = 0
        self._pending_law: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state of both sub-laws and switching logic."""
        self.pn_guidance.reset()
        self.los_guidance.reset()
        self._active_law = "pn"
        self._steps_in_law = 0
        self._pending_law = None

    def compute_command(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Compute hybrid guidance command.

        Args:
            own_state (dict): Own aircraft state.
            target_state (dict, optional): Target aircraft state.
            virtual_point (dict): Virtual pursuit point.
            gains (GuidanceGains, optional): Gain overrides.

        Returns:
            dict: Command with keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
        """
        # Compute range for switching decision using unified position helper
        own_pos = _extract_position(own_state)
        vp_pos = _extract_position(virtual_point)
        rel_pos = vp_pos - own_pos
        range_m = float(np.linalg.norm(rel_pos))

        # Determine active law / blend factor
        if self.mode == "range":
            cmd = self._range_mode(
                own_state, target_state, virtual_point, gains, range_m
            )
        elif self.mode == "energy":
            cmd = self._energy_mode(
                own_state, target_state, virtual_point, gains, range_m
            )
        elif self.mode == "blended":
            cmd = self._blended_mode(
                own_state, target_state, virtual_point, gains, range_m
            )
        else:
            logger.warning(
                f"Unknown hybrid_mode '{self.mode}', falling back to range mode"
            )
            cmd = self._range_mode(
                own_state, target_state, virtual_point, gains, range_m
            )

        return cmd

    # ------------------------------------------------------------------
    # Switching modes
    # ------------------------------------------------------------------

    def _range_mode(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any],
        range_m: float,
    ) -> Dict[str, float]:
        """Hysteresis switch at range_threshold_m with dwell time."""
        lower = self.range_threshold_m - 0.5 * self.hysteresis_m
        upper = self.range_threshold_m + 0.5 * self.hysteresis_m

        # Determine desired law based on hysteresis bands
        if self._active_law == "pn":
            desired = "los" if range_m < lower else "pn"
        else:
            desired = "pn" if range_m > upper else "los"

        return self._apply_switch(
            desired, own_state, target_state, virtual_point, gains
        )

    def _energy_mode(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any],
        range_m: float,
    ) -> Dict[str, float]:
        """Hysteresis switch based on energy state with dwell time."""
        vel = _extract_velocity(own_state)
        speed = float(np.linalg.norm(vel))

        lower_range = self.range_threshold_m - 0.5 * self.hysteresis_m
        upper_range = self.range_threshold_m + 0.5 * self.hysteresis_m
        lower_speed = (
            self.energy_speed_threshold_mps - 0.5 * self.energy_speed_hysteresis_mps
        )
        upper_speed = (
            self.energy_speed_threshold_mps + 0.5 * self.energy_speed_hysteresis_mps
        )

        if self._active_law == "pn":
            desired = "los" if (range_m < lower_range or speed < lower_speed) else "pn"
        else:
            desired = "pn" if (range_m > upper_range and speed > upper_speed) else "los"

        return self._apply_switch(
            desired, own_state, target_state, virtual_point, gains
        )

    def _apply_switch(
        self,
        desired: str,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any],
    ) -> Dict[str, float]:
        """Apply switching with dwell-time anti-chatter."""
        if desired == self._active_law:
            self._steps_in_law += 1
            self._pending_law = None
        else:
            if self._pending_law == desired:
                self._steps_in_law += 1
            else:
                self._pending_law = desired
                self._steps_in_law = 1

            if self._steps_in_law >= self.min_dwell_steps:
                self._active_law = desired
                self._pending_law = None
                self._steps_in_law = 0
                logger.debug(f"Hybrid switched to {desired}")

        if self._active_law == "los":
            return self.los_guidance.compute_command(
                own_state, target_state, virtual_point, gains
            )
        return self.pn_guidance.compute_command(
            own_state, target_state, virtual_point, gains
        )

    def _blended_mode(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any],
        range_m: float,
    ) -> Dict[str, float]:
        """
        Continuously blend commands from PN and LOS based on range.

        Blend factor w=0 => pure PN (long range), w=1 => pure LOS (short range).
        """
        upper = self.range_threshold_m + 0.5 * self.blend_transition_m
        lower = self.range_threshold_m - 0.5 * self.blend_transition_m

        if range_m >= upper:
            w = 0.0
            self._active_law = "pn"
        elif range_m <= lower:
            w = 1.0
            self._active_law = "los"
        else:
            w = (upper - range_m) / max(self.blend_transition_m, self.epsilon)
            self._active_law = "blend"

        pn_cmd = self.pn_guidance.compute_command(
            own_state, target_state, virtual_point, gains
        )
        los_cmd = self.los_guidance.compute_command(
            own_state, target_state, virtual_point, gains
        )

        def _blend(key: str) -> float:
            return (1.0 - w) * pn_cmd[key] + w * los_cmd[key]

        return {
            "nz_cmd": float(_blend("nz_cmd")),
            "roll_rate_cmd": float(_blend("roll_rate_cmd")),
            "throttle_cmd": float(_blend("throttle_cmd")),
        }
