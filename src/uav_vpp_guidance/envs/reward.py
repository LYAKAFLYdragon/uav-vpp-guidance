"""
Modular reward calculator.

Self-contained JSBSim integration.

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

from .sparse_reward import redistribute_rewards


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
        self.w_closing = self.config.get("w_closing", 0.0)
        self.w_alive = self.config.get("w_alive", 0.0)
        self.w_overshoot = self.config.get("w_overshoot", 0.0)
        self.w_boundary = self.config.get("w_boundary", 0.0)
        self.w_adversarial_bonus = self.config.get("w_adversarial_bonus", 0.0)
        self.w_position_advantage = self.config.get("w_position_advantage", 0.0)
        self.position_advantage_tail_deg = float(
            self.config.get("position_advantage_tail_deg", 30.0)
        )
        self.boundary_range_m = float(self.config.get("boundary_range_m", 6000.0))
        self.boundary_alt_min_m = float(self.config.get("boundary_alt_min_m", 1000.0))
        self.boundary_alt_max_m = float(self.config.get("boundary_alt_max_m", 9000.0))
        # F-16 typical max heading rate ≈ 0.3 rad/s (≈17°/s) at cruise speed
        self.max_heading_rate = self.config.get("max_heading_rate", 0.3)
        self.terminal_success = self.config.get("terminal_success", 200.0)
        self.terminal_failure = self.config.get("terminal_failure", -200.0)
        self.terminal_crash = self.config.get("terminal_crash", -300.0)

        # Angle reward formula ("quadratic_sum" recommended)
        self.angle_reward_formula = self.config.get("angle_reward_formula", "quadratic_sum")

        # Turn-rate penalty config (crossing-scenario aware)
        trp = self.config.get("turn_rate_penalty", {})
        self.turn_rate_penalty_enabled = bool(trp.get("enabled", True))
        self.turn_rate_penalty_range_min_m = float(trp.get("range_min_m", 1000.0))
        self.turn_rate_penalty_range_max_m = float(trp.get("range_max_m", 3000.0))
        self.turn_rate_penalty_heading_error_threshold_deg = float(
            trp.get("heading_error_threshold_deg", 60.0)
        )
        self.turn_rate_penalty_require_closing = bool(trp.get("require_closing", True))

        # 归一化参考值
        self._ref_range_m = 2000.0
        self._ref_altitude_m = 5000.0
        self._prev_command = None

        # 理想距离区间（可配置，用于与成功条件对齐）
        self.ideal_range_min = float(self.config.get("ideal_range_min", 800.0))
        self.ideal_range_max = float(self.config.get("ideal_range_max", 1200.0))

        # 势能奖励塑形（Potential-Based Reward Shaping）
        pbs = self.config.get("potential_based_shaping", {})
        self.pbs_enabled = bool(pbs.get("enabled", False))
        self.pbs_C = float(pbs.get("C", 0.001))
        self.pbs_gamma = float(pbs.get("gamma", 0.99))
        self._prev_rel_state = None

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
        if self.angle_reward_formula == "quadratic_sum":
            # Audit-recommended formula: separately penalize ATA and AA using
            # a quadratic mapping that saturates beyond 90°. This has a clear
            # physical interpretation and avoids the meaningless (ATA+AA) sum.
            ata_norm = min(1.0, ata_deg / 90.0)
            aa_norm = min(1.0, aa_deg / 90.0)
            angle_error = (ata_norm ** 2 + aa_norm ** 2) / 2.0
        elif self.angle_reward_formula == "max_ata_aa":
            # Use max(ATA, AA) to guarantee monotonic [-w_angle, 0] mapping.
            angle_error = max(ata_deg, aa_deg) / 180.0
        else:
            # Legacy (sum) formula — kept for backward compatibility only.
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

        # 7. 接近奖励：鼓励减小距离（range_rate 为负表示接近）
        range_rate_mps = rel.get("range_rate_mps", 0.0)
        reward_closing = self.w_closing * (-range_rate_mps / 200.0)

        # 8. 存活奖励：每步小额正奖励，帮助端到端基线避免早期出界
        reward_alive = self.w_alive

        # 9. 过冲惩罚：当距离已经小于理想下界的一半且还在继续接近时惩罚
        # 这防止飞机飞过目标导致坠毁/越界，同时避免在成功边界（理想区间）
        # 附近犹豫。触发阈值默认使用 ideal_range_min * 0.5，可通过配置覆盖。
        reward_overshoot = 0.0
        if self.w_overshoot > 0.0:
            overshoot_threshold_m = float(
                self.config.get("overshoot_range_threshold_m", self.ideal_range_min * 0.5)
            )
            if range_m < overshoot_threshold_m and range_rate_mps < 0.0:
                # 越接近目标、速度越快，惩罚越大
                proximity_factor = (overshoot_threshold_m - range_m) / overshoot_threshold_m
                reward_overshoot = -self.w_overshoot * proximity_factor * (-range_rate_mps / 200.0)

        # 10. 边界接近惩罚：防止飞出 max_range 或极端高度
        reward_boundary = 0.0
        if self.w_boundary > 0.0:
            boundary_penalty = 0.0
            if range_m > self.boundary_range_m:
                boundary_penalty += (range_m - self.boundary_range_m) / self.boundary_range_m
            if altitude_m < self.boundary_alt_min_m:
                boundary_penalty += (self.boundary_alt_min_m - altitude_m) / 1000.0
            if altitude_m > self.boundary_alt_max_m:
                boundary_penalty += (altitude_m - self.boundary_alt_max_m) / 1000.0
            reward_boundary = -self.w_boundary * boundary_penalty

        # 11. 对抗奖励：当目标正在机动逃逸且本机正在接近时给予额外奖励
        reward_adversarial = 0.0
        if self.w_adversarial_bonus > 0.0 and info.get("adversarial_maneuvering", False):
            range_rate_mps = rel.get("range_rate_mps", 0.0)
            closing_signal = max(0.0, -range_rate_mps / 200.0)
            reward_adversarial = self.w_adversarial_bonus * closing_signal

        # 12. 位置优势奖励：鼓励从“被目标追尾”转换为“追尾目标”。
        # 在 disadvantage 等场景中，目标初始位于本机后方；单纯的角度/距离奖励
        # 不足以引导智能体完成 lead-turn / 高悠悠等位置转换机动。
        # v2: 使用更柔和、非对称的信号，避免 agent 为了“绕后”而过度远离目标。
        reward_position_advantage = 0.0
        if self.w_position_advantage > 0.0:
            tail_thresh = self.position_advantage_tail_deg
            own_on_tail = abs(aa_deg) <= tail_thresh
            bandit_on_tail = abs(ata_deg) <= tail_thresh
            if own_on_tail and not bandit_on_tail:
                # 本机已成功处于目标后方：强正奖励
                advantage_signal = 1.0
            elif bandit_on_tail and not own_on_tail:
                # 被目标追尾：轻微负奖励，鼓励逐步脱离，但不要过度反应
                advantage_signal = -0.3
            else:
                # 过渡状态：AA 越小（本机越在目标后方）越好，
                # ATA 越小（本机越对准目标）越好。
                # AA 权重更高，因为绕到目标后方是 disadvantage 的核心目标。
                aa_component = 1.0 - min(1.0, abs(aa_deg) / 90.0)
                ata_component = 1.0 - min(1.0, abs(ata_deg) / 90.0)
                advantage_signal = 0.6 * aa_component + 0.4 * ata_component - 0.5
            reward_position_advantage = self.w_position_advantage * advantage_signal

        # 13. 终端奖励（由调用方根据 done/reason 注入，这里预留接口）
        terminal_reward = info.get("terminal_reward", 0.0)

        # 14. 势能奖励塑形：提供密集的距离梯度信号
        reward_potential = 0.0
        if self.pbs_enabled:
            reward_potential = self._compute_potential_shaping(rel)

        # 汇总
        reward = (
            reward_range
            + reward_angle
            + reward_safety
            + reward_saturation
            + reward_smooth
            + reward_turn
            + reward_closing
            + reward_alive
            + reward_overshoot
            + reward_boundary
            + reward_adversarial
            + reward_position_advantage
            + terminal_reward
            + reward_potential
        )

        reward_terms = {
            "reward_range": reward_range,
            "reward_angle": reward_angle,
            "reward_safety": reward_safety,
            "reward_saturation": reward_saturation,
            "reward_smooth": reward_smooth,
            "reward_turn": reward_turn,
            "reward_closing": reward_closing,
            "reward_alive": reward_alive,
            "reward_overshoot": reward_overshoot,
            "reward_boundary": reward_boundary,
            "reward_adversarial": reward_adversarial,
            "reward_position_advantage": reward_position_advantage,
            "terminal_reward": terminal_reward,
            "reward_potential": reward_potential,
            "reward_total": reward,
        }

        # 记录当前指令用于下一步平滑性计算
        self._prev_command = dict(command)

        return reward, reward_terms

    def _compute_potential_shaping(self, rel_next: dict) -> float:
        """
        Potential-based reward shaping using relative distance.

        phi(s) = -C * range(s)
        r_shape = gamma * phi(s') - phi(s)
              = -C * (gamma * range' - range)
        """
        range_next = float(rel_next.get("range_m", 0.0))
        phi_next = -self.pbs_C * range_next

        if self._prev_rel_state is None:
            # First call: no previous state, initialize and return zero shaping.
            self._prev_rel_state = dict(rel_next)
            return 0.0

        range_prev = float(self._prev_rel_state.get("range_m", 0.0))
        phi_prev = -self.pbs_C * range_prev
        self._prev_rel_state = dict(rel_next)
        return self.pbs_gamma * phi_next - phi_prev

    def _compute_range_reward(self, range_m: float) -> float:
        """
        距离奖励：鼓励进入合理距离区间。

        理想区间 [ideal_min, ideal_max]，区间内奖励最高。
        区间外按偏离程度线性惩罚。
        """
        ideal_min = self.ideal_range_min
        ideal_max = self.ideal_range_max
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

        if not self.turn_rate_penalty_enabled:
            return penalty

        range_min = self.turn_rate_penalty_range_min_m
        range_max = self.turn_rate_penalty_range_max_m
        # Only apply turn-rate penalty within the configured range window.
        # This exempts very close range (crossing passes) and very far range
        # where the aircraft has ample time to maneuver.
        if not (range_min < range_m < range_max):
            return penalty

        # Condition 1: large heading error at close range (crossing signature)
        # F-16 needs time to establish roll; >60° error within 3000m is risky.
        # Require closing to avoid punishing legitimate crossing geometries.
        threshold_rad = math.radians(self.turn_rate_penalty_heading_error_threshold_deg)
        if heading_error > threshold_rad:
            if (not self.turn_rate_penalty_require_closing) or (rel.get("range_rate_mps", 0.0) < 0.0):
                base_penalty = (heading_error - threshold_rad) / (math.pi / 2)
                range_factor = (range_max - range_m) / (range_max - range_min)
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
        self._prev_rel_state = None


def create_reward_calculator(config: dict):
    """根据配置选择奖励实现（非破坏式工厂，需求 4.11 / 7.2 / 7.7）。

    - ``air_combat.reward.use_sparse=True`` → 返回 ``AirCombatSparseReward``
      （R2SP 风格事件稀疏奖励，使用 ``air_combat.reward`` 子配置）。
    - 否则 → 返回现有 ``RewardCalculator``（密集奖励，行为完全不变）。

    采用惰性导入 ``AirCombatSparseReward`` 以避免模块加载期的循环依赖。
    本工厂不修改 ``RewardCalculator`` 现有逻辑（需求 7.7）。

    Args:
        config: 完整配置字典。沿用 ``RewardCalculator(config)`` 既有约定，
            回退分支直接将整个 ``config`` 传入 ``RewardCalculator``（其内部
            自行读取 ``config.get("reward", {})``）。

    Returns:
        奖励实现实例：``AirCombatSparseReward`` 或 ``RewardCalculator``。
    """
    ac = config.get("air_combat", {})
    if ac.get("reward", {}).get("use_sparse", False):
        from .sparse_reward_air_combat import AirCombatSparseReward
        return AirCombatSparseReward(ac.get("reward", {}))
    return RewardCalculator(config)


class SparseRewardCalculator:
    """
    Sparse event reward calculator with R2SP trajectory-level relabelling.

    This calculator replaces dense geometry-based shaping with a small set of
    sparse, semantically meaningful rewards:

    - Outcome rewards at episode end: success (+10), crash (-10),
      timeout / out-of-bounds (-5).
    - Event reward for entering the proximity zone (e.g. ideal engagement
      range with acceptable aspect angle).
    - Optional per-step lock reward when the aircraft is inside the success
      zone (can be supplied by the environment via ``info["in_lock"]`` or
      inferred from relative geometry).

    The per-step ``compute()`` output follows the original sparse MDP and can
    be consumed directly by an environment.  After an episode finishes,
    ``finalize()`` returns the full trajectory with outcome rewards redistributed
    via a linear kernel and event rewards redistributed via a truncated Gaussian
    kernel looking backward up to 50 steps.  Total reward is preserved exactly.

    Because this class only changes the reward signal, it is fully compatible
    with the existing CBF safety filter, which operates on states and actions
    before the reward is ever computed.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary.  Sparse-reward parameters
                are read from ``config["sparse_reward"]``; dense ``reward``
                keys are used only as fallbacks for terminal magnitudes and
                ideal range defaults.
        """
        self._full_config = config
        self._sparse_cfg = config.get("sparse_reward", {})
        self._dense_cfg = config.get("reward", {})

        # Outcome magnitudes (also exposed for env injection)
        self.terminal_success = float(
            self._sparse_cfg.get(
                "terminal_success", self._dense_cfg.get("terminal_success", 10.0)
            )
        )
        self.terminal_crash = float(
            self._sparse_cfg.get(
                "terminal_crash", self._dense_cfg.get("terminal_crash", -10.0)
            )
        )
        self.terminal_failure = float(
            self._sparse_cfg.get(
                "terminal_failure", self._dense_cfg.get("terminal_failure", -5.0)
            )
        )

        # Proximity event definition: entering this zone triggers a one-shot
        # event reward.  Defaults align with the existing ideal range.
        self.event_range_m = float(
            self._sparse_cfg.get(
                "event_range_m", self._dense_cfg.get("ideal_range_max", 1200.0)
            )
        )
        self.event_ata_deg = float(
            self._sparse_cfg.get("event_ata_deg", 45.0)
        )
        self.event_reward = float(
            self._sparse_cfg.get("event_reward", 1.0)
        )

        # Continuous lock reward (optional, disabled by default).
        self.lock_reward = float(
            self._sparse_cfg.get("lock_reward", 0.0)
        )
        self.success_range_m = float(
            self._sparse_cfg.get(
                "success_range_m",
                self._full_config.get("env", {}).get("success_range_m", 900.0),
            )
        )
        self.success_ata_deg = float(
            self._sparse_cfg.get(
                "success_ata_deg",
                self._full_config.get("env", {}).get("success_ata_deg", 25.0),
            )
        )

        # Relabelling kernel parameters.
        self.gaussian_window = int(
            self._sparse_cfg.get("gaussian_window", 50)
        )
        self.gaussian_sigma_ratio = float(
            self._sparse_cfg.get("gaussian_sigma_ratio", 0.5)
        )
        rel_cfg = self._sparse_cfg.get("relabelling", {})
        self.relabelling_enabled = bool(rel_cfg.get("enabled", True))
        self.terminal_kernel = str(rel_cfg.get("terminal_kernel", "linear"))
        self.event_kernel = str(rel_cfg.get("event_kernel", "gaussian"))

        self.reset()

    def reset(self):
        """Reset all per-episode buffers."""
        self._step = 0
        self._continuous_rewards = []  # per-step lock rewards
        self._events = []              # (step, event_reward)
        self._base_step_rewards = []   # original sparse reward per step
        self._terminal_reward = 0.0
        self._terminal_step = None
        self._proximity_triggered = False
        self._redistributed = None

    def compute(self, info):
        """
        Compute the sparse reward for the current step.

        Args:
            info (dict): Same interface as :class:`RewardCalculator`:
                ``relative_state``, ``own_state``, ``command`` and optional
                ``terminal_reward`` / ``termination_info``.

        Returns:
            tuple: ``(reward, reward_terms)``.  ``reward`` is the original
            sparse reward for this step (including terminal outcome reward on
            the last step).  ``reward_terms`` contains a per-component
            breakdown.
        """
        rel = info.get("relative_state", {})

        range_m = float(rel.get("range_m", 2000.0))
        ata_deg = float(np.rad2deg(rel.get("ata_rad", np.pi)))

        # ------------------------------------------------------------------
        # 1. Proximity event: one-shot reward on first entry to the event zone.
        # ------------------------------------------------------------------
        event_reward = 0.0
        if not self._proximity_triggered:
            if range_m <= self.event_range_m and abs(ata_deg) <= self.event_ata_deg:
                event_reward = self.event_reward
                self._proximity_triggered = True
                self._events.append((self._step, event_reward))

        # ------------------------------------------------------------------
        # 2. Continuous lock reward (optional).
        # ------------------------------------------------------------------
        lock_reward = 0.0
        if self.lock_reward != 0.0:
            in_lock = info.get("in_lock", False)
            if not in_lock:
                in_lock = (
                    range_m <= self.success_range_m
                    and abs(ata_deg) <= self.success_ata_deg
                )
            if in_lock:
                lock_reward = self.lock_reward
        self._continuous_rewards.append(lock_reward)

        # ------------------------------------------------------------------
        # 3. Terminal outcome reward (recorded separately for relabelling).
        # ------------------------------------------------------------------
        terminal_reward = info.get("terminal_reward", 0.0)
        if terminal_reward == 0.0:
            term_info = info.get("termination_info", {})
            if term_info.get("is_success"):
                terminal_reward = self.terminal_success
            elif term_info.get("is_crash"):
                terminal_reward = self.terminal_crash
            elif term_info.get("is_timeout") or term_info.get("is_out_of_bounds"):
                terminal_reward = self.terminal_failure

        if terminal_reward != 0.0 and self._terminal_step is None:
            self._terminal_reward = terminal_reward
            self._terminal_step = self._step

        # Step reward that the environment sees (original sparse MDP).
        step_reward = event_reward + lock_reward + terminal_reward
        self._base_step_rewards.append(float(step_reward))

        reward_terms = {
            "reward_event_proximity": event_reward,
            "reward_lock": lock_reward,
            "terminal_reward": terminal_reward,
            "reward_sparse_step": event_reward + lock_reward,
            "reward_total": step_reward,
        }

        self._step += 1
        return float(step_reward), reward_terms

    def finalize(self) -> np.ndarray:
        """
        Redistribute the recorded trajectory and return the per-step rewards.

        Outcome rewards are spread uniformly across the whole trajectory.
        Event rewards are spread backwards from the event step using a
        truncated Gaussian kernel over at most ``gaussian_window`` steps.

        Returns:
            np.ndarray: Relabelled per-step rewards.  ``sum(finalize())``
            equals the sum of all sparse rewards returned by ``compute()``.
        """
        trajectory_length = len(self._continuous_rewards)
        if trajectory_length == 0:
            self._redistributed = np.array([], dtype=float)
            return self._redistributed

        if not self.relabelling_enabled:
            self._redistributed = np.asarray(self._base_step_rewards, dtype=float)
            return self._redistributed

        # Only redistribute the terminal reward if it was observed on the last
        # step; otherwise treat it as already delivered in the step reward.
        terminal = self._terminal_reward if self._terminal_step == trajectory_length - 1 else 0.0

        event_relabel = redistribute_rewards(
            trajectory_length,
            terminal_reward=0.0,
            events=self._events,
            gaussian_window=self.gaussian_window,
            gaussian_sigma_ratio=self.gaussian_sigma_ratio,
            terminal_kernel="none",
            event_kernel=self.event_kernel,
        )
        terminal_relabel = redistribute_rewards(
            trajectory_length,
            terminal_reward=terminal,
            events=[],
            terminal_kernel=self.terminal_kernel,
            event_kernel="none",
        )

        continuous = np.asarray(self._continuous_rewards, dtype=float)
        self._redistributed = continuous + event_relabel + terminal_relabel
        return self._redistributed

    def get_redistributed_trajectory(self) -> np.ndarray:
        """Return the last ``finalize()`` result, or ``finalize()`` if needed."""
        if self._redistributed is None:
            return self.finalize()
        return self._redistributed


def build_reward_calculator(config):
    """
    Factory that selects the dense or sparse reward calculator.

    The sparse calculator is chosen when ``config["reward"]["use_sparse"]`` is
    ``True`` or when ``config["sparse_reward"]["enabled"]`` is ``True``.
    Otherwise the original dense :class:`RewardCalculator` is returned.
    """
    use_sparse = (
        config.get("reward", {}).get("use_sparse", False)
        or config.get("sparse_reward", {}).get("enabled", False)
    )
    if use_sparse:
        return SparseRewardCalculator(config)
    return RewardCalculator(config)
