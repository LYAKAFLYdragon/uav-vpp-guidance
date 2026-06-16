"""Fixed-offset virtual-point generator for ablation studies."""
import numpy as np


class FixedOffsetGuidance:
    """
    Place a fixed NEU offset relative to the target's current position.

    This is useful for interpretability ablations (Wave 3/4): it replaces the
    learned VPP offset with a manually chosen displacement, while keeping the
    rest of the guidance chain unchanged.
    """

    def __init__(self, fixed_offset):
        self.fixed_offset = np.asarray(fixed_offset, dtype=np.float64)
        if self.fixed_offset.shape != (3,):
            raise ValueError("fixed_offset must be a 3-element vector")

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
        virtual_point_pos = target_pos + self.fixed_offset
        virtual_point = {
            "position": virtual_point_pos,
            "offset": self.fixed_offset.copy(),
        }
        if not return_info:
            return virtual_point
        info = {
            "anchor_mode": "current_target",
            "anchor_pos": target_pos,
            "offset": self.fixed_offset.copy(),
            "pred_var": None,
            "prediction_info": {
                "anchor_mode": "current_target",
                "vpp_enabled": True,
                "note": f"Fixed offset ablation: {self.fixed_offset.tolist()}",
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
            raise ValueError(
                f"Target position must be a 3-element vector, got shape {arr.shape}"
            )
        return arr
