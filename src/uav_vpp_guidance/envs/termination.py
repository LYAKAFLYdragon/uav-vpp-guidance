"""
Termination condition checker.

Migrated from legacy project:
  <JSBSIM_ROOT>/envs/JSBSim/termination_conditions/*.py

检查成功、失败、坠毁、超时、越界等终止条件。
"""

import numpy as np
from typing import Tuple


class TerminationChecker:
    """
    Check success, failure, crash, out-of-bounds, and max-step termination.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Termination configuration dictionary.
        """
        self.config = config
        self.success_range_m = config.get("success_range_m", 900.0)
        self.success_ata_deg = config.get("success_ata_deg", 25.0)
        self.success_hold_time_s = config.get("success_hold_time_s", 0.2)
        self.hysteresis_range_m = config.get("hysteresis_range_m", 950.0)
        self.hysteresis_ata_deg = config.get("hysteresis_ata_deg", 30.0)
        self.min_altitude_m = config.get("min_altitude_m", 500.0)
        self.max_altitude_m = config.get("max_altitude_m", 15000.0)
        self.max_range_m = config.get("max_range_m", 8000.0)
        self.max_steps = config.get("max_high_level_steps", 512)
        self.decision_freq = config.get("decision_freq", 5)

        # 成功条件需要连续保持的步数
        self._success_hold_steps = int(self.success_hold_time_s * self.decision_freq)
        self._success_counter = 0
        self._in_success_zone = False

    def reset(self):
        """Reset internal termination state."""
        self._success_counter = 0
        self._in_success_zone = False

    def check(self, own_state, target_state, metrics, step) -> Tuple[bool, dict]:
        """
        Check termination conditions.

        Args:
            own_state (dict): Own aircraft state.
            target_state (dict): Target aircraft state.
            metrics (dict): Current tracking metrics (range_m, ata_deg, etc.).
            step (int): Current high-level step count.

        Returns:
            tuple: (done, termination_info)
                done (bool): True if episode should terminate.
                termination_info (dict): Contains reason, is_success, is_crash,
                    is_timeout, is_out_of_bounds, success_hold_steps.
        """
        range_m = metrics.get("range_m", float("inf"))
        ata_deg = np.rad2deg(metrics.get("ata_rad", np.pi))
        altitude_m = own_state.get("altitude_m", 5000.0)

        info = {
            "reason": None,
            "is_success": False,
            "is_crash": False,
            "is_timeout": False,
            "is_out_of_bounds": False,
            "success_hold_steps": self._success_counter,
        }

        # 1. 坠毁检查（低空或极端高度）
        if altitude_m < self.min_altitude_m or altitude_m > self.max_altitude_m:
            info["reason"] = "crash"
            info["is_crash"] = True
            return True, info

        # 2. 越界检查（距离过大）
        if range_m > self.max_range_m:
            info["reason"] = "out_of_bounds"
            info["is_out_of_bounds"] = True
            return True, info

        # 3. 成功检查（带滞回）
        in_zone = (range_m <= self.success_range_m) and (ata_deg <= self.success_ata_deg)

        if in_zone:
            self._success_counter += 1
            info["success_hold_steps"] = self._success_counter
            if self._success_counter >= self._success_hold_steps:
                info["reason"] = "success"
                info["is_success"] = True
                return True, info
        else:
            # 滞回：若超出较宽阈值，重置成功计数器
            out_zone = (range_m > self.hysteresis_range_m) or (ata_deg > self.hysteresis_ata_deg)
            if out_zone:
                self._success_counter = 0
                info["success_hold_steps"] = 0

        # 4. 超时检查
        if step >= self.max_steps:
            info["reason"] = "timeout"
            info["is_timeout"] = True
            return True, info

        return False, info
