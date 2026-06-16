"""Fixed-offset VPP ablation.

Returns a constant offset relative to the target's current position. The policy
action is ignored, so this isolates the effect of a static lead/lag offset from
the learned adaptive offset.
"""
import numpy as np


class FixedOffsetGuidance:
    """Virtual-point generator that always adds a fixed offset to the target.

    This is useful for ablations such as "does a constant +500m lead offset
    rescue the No-VPP timeout behavior?" or "can a fixed 2987m offset replicate
    the learned VPP disadvantage behavior?".
    """

    def __init__(self, offset_m=None):
        """Args:
            offset_m (array-like): 3-D offset [longitudinal, lateral, vertical] in meters.
                Defaults to zero offset (No-VPP behavior).
        """
        if offset_m is None:
            offset_m = [0.0, 0.0, 0.0]
        self.offset_m = np.asarray(offset_m, dtype=np.float64)

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
        initial_range_m: float = None,
    ):
        target_pos = self._get_target_position(target_state)
        virtual_point = {
            "position": target_pos + self.offset_m,
            "offset": self.offset_m.copy(),
        }
        if not return_info:
            return virtual_point
        info = {
            "anchor_mode": "current_target",
            "anchor_pos": target_pos,
            "offset": self.offset_m.copy(),
            "pred_var": None,
            "prediction_info": {
                "anchor_mode": "current_target",
                "fixed_offset": True,
                "offset_m": self.offset_m.tolist(),
            },
        }
        return virtual_point, info

    @staticmethod
    def _get_target_position(target_state):
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
            raise ValueError(f"Target position must be a 3-element vector, got shape {arr.shape}")
        return arr
