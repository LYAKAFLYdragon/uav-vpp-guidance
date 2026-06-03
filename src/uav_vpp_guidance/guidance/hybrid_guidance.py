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

from .los_rate_guidance import LOSRateGuidance
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

        # Instantiate sub-guidance laws
        self.los_guidance = LOSRateGuidance(config)
        self.pn_guidance = ProportionalNavigationGuidance(config)

        # Tracking
        self._active_law: str = "pn"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state of both sub-laws."""
        self.pn_guidance.reset()
        self._active_law = "pn"

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
        # Compute range for switching decision
        own_pos_raw = own_state.get("position_ned")
        if own_pos_raw is None:
            own_pos_raw = own_state.get("position_m")
        if own_pos_raw is None:
            own_pos_raw = [0.0, 0.0, 0.0]
        own_pos = np.asarray(own_pos_raw, dtype=np.float64)

        vp_pos_raw = virtual_point.get("position_ned")
        if vp_pos_raw is None:
            vp_pos_raw = virtual_point.get("position_m")
        if vp_pos_raw is None:
            vp_pos_raw = [0.0, 0.0, 0.0]
        vp_pos = np.asarray(vp_pos_raw, dtype=np.float64)
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
        """Pure switch at range_threshold_m."""
        if range_m < self.range_threshold_m:
            self._active_law = "los"
            return self.los_guidance.compute_command(
                own_state, target_state, virtual_point, gains
            )
        self._active_law = "pn"
        return self.pn_guidance.compute_command(
            own_state, target_state, virtual_point, gains
        )

    def _energy_mode(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any],
        range_m: float,
    ) -> Dict[str, float]:
        """Switch to LOS when low on energy (speed or terminal range)."""
        vel = own_state.get("velocity_ned")
        if vel is None:
            vel = own_state.get("velocity_vector_mps")
        if vel is None:
            vel = [0.0, 0.0, 0.0]
        speed = float(np.linalg.norm(np.asarray(vel, dtype=np.float64)))

        if range_m < self.range_threshold_m or speed < self.energy_speed_threshold_mps:
            self._active_law = "los"
            return self.los_guidance.compute_command(
                own_state, target_state, virtual_point, gains
            )
        self._active_law = "pn"
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
