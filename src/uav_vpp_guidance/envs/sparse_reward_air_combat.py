"""
AirCombatSparseReward：R2SP 风格的空战事件稀疏奖励。

本模块实现一个**独立**的事件稀疏奖励类 ``AirCombatSparseReward``，用于以稀疏的
空战事件信号（雷达锁定/丢失、导弹发射/命中/未命中、被锁定/被击中、脱离发射包线）
训练策略，并通过轨迹级重标（relabelling）缓解奖励稀疏问题：

    - 结局奖励（terminal/outcome reward）：用**线性核**在整条轨迹上均匀分配，
      总量守恒等于 ``terminal_reward``（需求 4.9，对应 Property 15）。
    - 事件奖励（event reward）：对发生于步 ``t_e`` 的事件，用**高斯核**在窗口
      ``[max(0, t_e - window), t_e]`` 内反向衰减分配（``t > t_e`` 的步获得零信用，
      体现因果性），归一化后总量守恒等于该事件奖励值（需求 4.10，对应 Property 16）。

设计要点：
    - 本类**不修改**现有 ``RewardCalculator`` 的密集奖励逻辑（需求 4.11）。
    - 模块仅依赖 ``numpy`` 与标准库 ``enum`` / ``math``，**不依赖 JSBSim**，
      可在纯 Python 环境中导入（供 ``reward.py`` 工厂与 ``AirCombatMVPEnv`` 引用）。

事件名称约定：``AirCombatEvent`` 为 ``(str, Enum)``，其成员值与 ``EVENT_REWARDS``
的键一一对应。由于 str 枚举成员与其字符串值相等且哈希一致，
``event_rewards.get(member)`` 与 ``event_rewards.get("name")`` 行为一致，
因此 ``compute_step`` / ``relabel_trajectory`` 同时兼容字符串与枚举成员输入。
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

# 事件奖励表（需求 4 验收标准 1–8）
EVENT_REWARDS = {
    "radar_lock": 0.1,        # 雷达完成锁定（需求 4.1）
    "radar_lock_lost": -0.1,  # 雷达丢失锁定（需求 4.2）
    "missile_launch": 1.0,    # 本机发射导弹（需求 4.3）
    "missile_hit": 10.0,      # 本机导弹命中（需求 4.4）
    "missile_miss": -1.0,     # 本机导弹未命中（需求 4.5）
    "being_locked": -0.1,     # 本机被目标雷达锁定（需求 4.6）
    "being_hit": -10.0,       # 本机被击中（需求 4.7）
    "out_of_envelope": -0.5,  # 目标脱离发射包线（需求 4.8）
}


class AirCombatEvent(str, Enum):
    """空战事件枚举（需求 4 / 设计 §4.3）。

    继承 ``str`` 使各成员可直接当作字符串键使用：``AirCombatEvent.RADAR_LOCK``
    与字符串 ``"radar_lock"`` 相等且哈希一致，故可直接用于 ``EVENT_REWARDS`` 查表。
    成员值与 ``EVENT_REWARDS`` 的键一一对应。
    """

    RADAR_LOCK = "radar_lock"
    RADAR_LOCK_LOST = "radar_lock_lost"
    MISSILE_LAUNCH = "missile_launch"
    MISSILE_HIT = "missile_hit"
    MISSILE_MISS = "missile_miss"
    BEING_LOCKED = "being_locked"
    BEING_HIT = "being_hit"
    OUT_OF_ENVELOPE = "out_of_envelope"


class AirCombatSparseReward:
    """R2SP 风格的空战事件稀疏奖励（需求 4）。

    逐步奖励 ``compute_step`` 对当前步触发的所有事件求和；轨迹级重标
    ``relabel_trajectory`` 将结局奖励（线性核）与事件奖励（高斯核）重新分配到
    整条轨迹，两部分均保持总量守恒。
    """

    def __init__(self, config: dict | None = None):
        """构造稀疏奖励。

        Args:
            config: 配置字典。可含：
                - ``event_rewards``：覆盖默认事件奖励值的字典（与 ``EVENT_REWARDS``
                  合并，后者为基础默认值）。
                - ``relabelling``：重标配置子字典，可含 ``enabled``（默认 True）、
                  ``window``（默认 50）、``gaussian_sigma``（默认 ``window/3.0``）。
        """
        config = config or {}
        # 合并事件奖励：默认表为基础，config 覆盖（需求 4.11 不破坏默认值）
        self.event_rewards = {**EVENT_REWARDS, **config.get("event_rewards", {})}
        # 重标配置
        self.relabelling = config.get("relabelling", {})
        self.relabel_enabled = self.relabelling.get("enabled", True)
        self.window = self.relabelling.get("window", 50)
        self.gaussian_sigma = self.relabelling.get(
            "gaussian_sigma", self.window / 3.0
        )

    def reset(self) -> None:
        """回合开始时复位每回合状态。

        当前实现为无状态（逐步奖励与轨迹重标均为纯函数式计算），此处为接口占位/
        缓存清理，保留以与 ``RewardCalculator`` 的生命周期接口一致。
        """
        return None

    def compute_step(self, events) -> float:
        """逐步即时奖励：对本步触发的所有事件求和（需求 4.1–4.8）。

        Args:
            events: 本步触发的事件名称序列。元素可为字符串或 ``AirCombatEvent``
                成员（str 枚举与其字符串值等价，查表行为一致）。

        Returns:
            各事件奖励值之和；未知事件按 0.0 计。
        """
        return float(sum(self.event_rewards.get(e, 0.0) for e in events))

    def relabel_trajectory(
        self,
        step_events: list,
        terminal_reward: float,
        episode_length: int,
    ) -> np.ndarray:
        """回合结束后的轨迹级奖励重分配（需求 4.9/4.10）。

        Args:
            step_events: 长度为 ``episode_length`` 的列表，第 ``t`` 个元素为该步触发的
                事件名称列表（可为字符串或 ``AirCombatEvent``）。
            terminal_reward: 结局奖励（成功为正、失败为负）。
            episode_length: 回合步数。

        Returns:
            长度为 ``episode_length`` 的 ``np.ndarray``，为重标后的逐步奖励。

        分配规则：
            A) 结局奖励——线性核在整条轨迹上**均匀分配**：
               ``rewards[t] += terminal_reward / episode_length``。
               总量恰为 ``terminal_reward``（守恒，Property 15）。
            B) 事件奖励——高斯核在窗口 ``[max(0, t_e - window), t_e]`` 内反向衰减
               分配，归一化后乘以事件奖励值。``t > t_e`` 获得零信用（因果性）。
               每个事件的分配总量恰为其奖励值（守恒，Property 16）。

        边界：
            - ``episode_length <= 0`` 返回空数组。
            - ``window == 0`` 或 ``sigma <= 0`` 时窗口退化为单点 ``{t_e}``，
              全部信用落在事件步 ``t_e``（仍满足守恒与因果性）。
        """
        if episode_length <= 0:
            return np.zeros(0, dtype=np.float64)

        rewards = np.zeros(episode_length, dtype=np.float64)

        # (A) 结局奖励：线性核均匀分配，总量守恒 == terminal_reward
        rewards += terminal_reward / episode_length

        # (B) 事件奖励：高斯核反向衰减分配，每个事件总量守恒 == event_reward
        sigma = self.gaussian_sigma
        for t_e, events in enumerate(step_events):
            if t_e >= episode_length:
                break
            if not events:
                continue
            start = max(0, t_e - self.window)
            indices = list(range(start, t_e + 1))  # 仅 t <= t_e（因果性）

            # 计算归一化高斯权重；sigma<=0 或单点窗口退化为全权重落在 t_e
            if sigma is not None and sigma > 0 and len(indices) > 1:
                raw = [
                    math.exp(-((t_e - t) ** 2) / (2.0 * sigma * sigma))
                    for t in indices
                ]
                total = sum(raw)
                if total <= 0:
                    weights = [0.0] * len(indices)
                    weights[-1] = 1.0  # 兜底：全部落在 t_e
                else:
                    weights = [w / total for w in raw]
            else:
                weights = [0.0] * len(indices)
                weights[-1] = 1.0  # 退化：全部信用落在事件步 t_e

            for e in events:
                event_reward = self.event_rewards.get(e, 0.0)
                if event_reward == 0.0:
                    continue
                for idx, t in enumerate(indices):
                    rewards[t] += event_reward * weights[idx]

        return rewards
