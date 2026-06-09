"""
No-VPP baseline guidance.

Bypasses the virtual pursuit point offset computation by always returning
the target's current position as the virtual point (zero offset).

This maintains the same interface as :class:`VirtualPointGenerator` so it
can be used as a drop-in replacement in :class:`CloseRangeTrackingEnv`.
"""

import numpy as np


class NoVPPGuidance:
    """
    No-VPP baseline: directly track the target's current position.

    The policy action is ignored (or conceptually forced to zero), so the
    virtual point is always identical to the target position.  This serves
    as an ablation baseline to quantify the tactical value added by the
    VPP offset layer.

    Interface compatibility:
    - :meth:`action_to_virtual_point` matches the signature of
      :meth:`VirtualPointGenerator.action_to_virtual_point`.
    - Returns a dict with ``position`` and ``offset`` keys, plus an
      optional info dict when ``return_info=True``.
    """

    def action_to_virtual_point(
        self,
        action,
        own_state,
        target_state,
        anchor_mode: str = "current_target",
        lookahead_time_s: float = 1.0,
        trajectory_predictor_adapter=None,
        predicted_target_position=None,
        return_info: bool = False,
    ):
        """
        Convert policy action to a virtual pursuit point.

        In the No-VPP baseline the action is ignored and the offset is
        always zero, so the virtual point equals the target's current
        position.

        Args:
            action (np.ndarray): Policy action (ignored).
            own_state (dict): Own aircraft state (ignored).
            target_state (dict): Target aircraft state. Must contain a
                position field.
            anchor_mode (str): Ignored — always treated as "current_target".
            lookahead_time_s (float): Ignored.
            trajectory_predictor_adapter: Ignored.
            predicted_target_position: Ignored.
            return_info (bool): If True, also return an info dict.

        Returns:
            dict or tuple: ``virtual_point`` dict, or
            ``(virtual_point, info)`` when ``return_info=True``.
        """
        target_pos = self._get_target_position(target_state)
        offset = np.zeros(3, dtype=np.float64)

        virtual_point = {
            "position": target_pos,
            "offset": offset,
        }

        if not return_info:
            return virtual_point

        info = {
            "anchor_mode": "current_target",
            "anchor_pos": target_pos,
            "offset": offset,
            "pred_var": None,
            "prediction_info": {
                "anchor_mode": "current_target",
                "vpp_enabled": False,
                "note": "No-VPP baseline: offset forced to zero",
            },
        }
        return virtual_point, info

    @staticmethod
    def _get_target_position(target_state):
        """Extract target position from state dict.

        Tries keys in order: ``position_neu``, ``position_m``, ``position``.
        """
        pos = target_state.get("position_neu")
        if pos is None:
            pos = target_state.get("position_m")
        if pos is None:
            pos = target_state.get("position")
        if pos is None:
            raise ValueError(
                "target_state must contain 'position_neu', 'position_m', or 'position'"
            )
        arr = np.asarray(pos, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"Target position must be a 3-element vector, got shape {arr.shape}"
            )
        return arr
