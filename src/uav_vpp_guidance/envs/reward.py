"""
Modular reward calculator.

Migrated from legacy project:
  <JSBSIM_ROOT>/envs/JSBSim/reward_functions/*.py

第一版奖励项：
- range_reward: 鼓励进入合理距离区间
- angle_reward: 鼓励 ATA 变小
- safety_penalty: 低空惩罚
- saturation_penalty: 指令饱和惩罚
- smooth_penalty: 指令变化率惩罚
- terminal_reward: 终端成功/失败/坠毁奖励
"""

import math
import numpy as np


def _stable_angle_diff(a, b):
    """Signed smallest angle difference in radians."""
    delta = a - b
    if not np.isfinite(delta):
        return float(delta)
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


class RewardCalculator:
    """
    Modular reward calculator.

    Reward terms:
    - range_reward: encourage reasonable range interval
    - angle_reward: encourage small ATA
    - safety_penalty: low altitude
    - saturation_penalty: command limits
    - command smoothness penalty
    - terminal reward
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Reward configuration dictionary.
        """
        self.config = config.get("reward", {})
        self.w_range = self.config.get("w_range", 0.5)
        self.w_angle = self.config.get("w_angle", 0.8)
        self.w_energy = self.config.get("w_energy", 0.2)
        self.w_safety = self.config.get("w_safety", 2.0)
        self.w_saturation = self.config.get("w_saturation", 1.0)
        self.w_smooth = self.config.get("w_smooth", 0.1)
        self.w_turn_rate = self.config.get("w_turn_rate", 0.5)
        # F-16 typical max heading rate ≈ 0.3 rad/s (≈17°/s) at cruise speed
        self.max_heading_rate = self.config.get("max_heading_rate", 0.3)
        self.terminal_success = self.config.get("terminal_success", 200.0)
        self.terminal_failure = self.config.get("terminal_failure", -200.0)
        self.terminal_crash = self.config.get("terminal_crash", -300.0)

        # 归一化参考值
        self._ref_range_m = 2000.0
        self._ref_altitude_m = 5000.0
        self._prev_command = None

    def compute(self, info):
        """
        Compute the scalar reward for the current step.

        Args:
            info (dict): Auxiliary information containing:
                - own_state, target_state
                - relative_state (from compute_relative_geometry)
                - command (current command dict)
                - terminal_reward (optional): injected by env when episode ends

        Returns:
            tuple: (reward, reward_terms)
                reward (float): Total scalar reward.
                reward_terms (dict): Per-term breakdown.
        """
        rel = info.get("relative_state", {})
        own_state = info.get("own_state", {})
        command = info.get("command", {})

        range_m = rel.get("range_m", 2000.0)
        ata_rad = rel.get("ata_rad", np.pi)
        aa_rad = rel.get("aa_rad", np.pi)
        altitude_m = own_state.get("altitude_m", 5000.0)

        # 1. 距离奖励：鼓励进入合理距离区间（不是越近越好）
        # 理想区间 [800, 1200]m，区间外按距离惩罚
        reward_range = self._compute_range_reward(range_m)

        # 2. 角度奖励（ATA 和 AA 越小越好）
        # ATA: 目标机速度与本机-目标视线的夹角，越小表示目标正对
        # AA: 本机速度与本机-目标视线的夹角，越小表示本机正对目标
        ata_deg = np.rad2deg(ata_rad)
        aa_deg = np.rad2deg(aa_rad)
        angle_error = (ata_deg + aa_deg) / 180.0
        reward_angle = -self.w_angle * angle_error

        # 3. 安全惩罚（低空）
        min_alt = self.config.get("min_altitude_m", 500.0)
        altitude_margin = (altitude_m - min_alt) / self._ref_altitude_m
        # 当高度接近下限时，惩罚增大
        safety_penalty = 0.0
        if altitude_m < min_alt + 1000.0:
            safety_penalty = self.w_safety * max(0.0, 1.0 - altitude_margin)
        reward_safety = -safety_penalty

        # 4. 指令饱和惩罚
        saturation_penalty = 0.0
        nz_cmd = abs(command.get("nz_cmd", 0.0))
        roll_rate_cmd = abs(command.get("roll_rate_cmd", 0.0))
        if nz_cmd > 6.5:
            saturation_penalty += (nz_cmd - 6.5) / 7.0
        if roll_rate_cmd > 1.4:
            saturation_penalty += (roll_rate_cmd - 1.4) / 1.5
        reward_saturation = -self.w_saturation * saturation_penalty

        # 5. 平滑性惩罚（指令变化率）
        smooth_penalty = 0.0
        if self._prev_command is not None:
            for key in ["nz_cmd", "roll_rate_cmd", "throttle_cmd"]:
                delta = abs(command.get(key, 0.0) - self._prev_command.get(key, 0.0))
                smooth_penalty += delta
        reward_smooth = -self.w_smooth * smooth_penalty

        # 6. 转弯速率惩罚（动力学可行性）
        # 惩罚需要 F-16 无法完成的航向变化率的场景
        turn_rate_penalty = self._compute_turn_rate_penalty(rel, own_state, command)
        reward_turn = -self.w_turn_rate * turn_rate_penalty

        # 7. 终端奖励（由调用方根据 done/reason 注入，这里预留接口）
        terminal_reward = info.get("terminal_reward", 0.0)

        # 汇总
        reward = (
            reward_range
            + reward_angle
            + reward_safety
            + reward_saturation
            + reward_smooth
            + reward_turn
            + terminal_reward
        )

        reward_terms = {
            "reward_range": reward_range,
            "reward_angle": reward_angle,
            "reward_safety": reward_safety,
            "reward_saturation": reward_saturation,
            "reward_smooth": reward_smooth,
            "reward_turn": reward_turn,
            "terminal_reward": terminal_reward,
            "reward_total": reward,
        }

        # 记录当前指令用于下一步平滑性计算
        self._prev_command = dict(command)

        return reward, reward_terms

    def _compute_range_reward(self, range_m: float) -> float:
        """
        距离奖励：鼓励进入合理距离区间。

        理想区间 [ideal_min, ideal_max]，区间内奖励最高。
        区间外按偏离程度线性惩罚。
        """
        ideal_min = 800.0
        ideal_max = 1200.0
        max_penalty_range = 4000.0

        if ideal_min <= range_m <= ideal_max:
            # 在理想区间内，给予正奖励
            return self.w_range * 0.5
        elif range_m < ideal_min:
            # 过近：惩罚
            error = (ideal_min - range_m) / ideal_min
            return -self.w_range * error
        else:
            # 过远：惩罚
            error = min(1.0, (range_m - ideal_max) / (max_penalty_range - ideal_max))
            return -self.w_range * error

    def _compute_turn_rate_penalty(self, rel, own_state, command):
        """
        Compute turn-rate penalty based on required heading rate vs F-16 capability.

        Penalizes two conditions:
        1. Large heading errors at close range (crossing scenario signature).
        2. Required heading rate exceeding aircraft max capability.

        Args:
            rel (dict): Relative geometry dict.
            own_state (dict): Own aircraft state.
            command (dict): Current command dict.

        Returns:
            float: Turn-rate penalty (0.0 if feasible, positive if infeasible).
        """
        range_m = rel.get("range_m", 2000.0)
        speed_mps = own_state.get("speed_mps", 200.0)
        if range_m <= 0 or speed_mps <= 0:
            return 0.0

        # Current heading error from LOS
        los_az = rel.get("los_azimuth_rad", 0.0)
        own_yaw = own_state.get("yaw_rad", 0.0)
        heading_error = abs(_stable_angle_diff(los_az, own_yaw))

        penalty = 0.0

        # Condition 1: large heading error at close range (crossing signature)
        # F-16 needs time to establish roll; >60° error within 3000m is risky
        if heading_error > math.pi / 3 and range_m < 3000.0:
            base_penalty = (heading_error - math.pi / 3) / (math.pi / 2)
            range_factor = (3000.0 - range_m) / 3000.0
            penalty += base_penalty * range_factor

        # Condition 2: required heading rate exceeds max capability
        tgo = range_m / speed_mps
        required_heading_rate = heading_error / max(tgo, 0.1)
        excess = required_heading_rate - self.max_heading_rate
        if excess > 0:
            penalty += excess / self.max_heading_rate

        return penalty

    def reset(self):
        """Reset internal state (e.g., previous command buffer)."""
        self._prev_command = None
