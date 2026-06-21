"""Property-based tests for AirCombatSparseReward (air-combat-mvp).

本文件实现 AirCombatSparseReward 的属性测试：
    - Property 14：事件奖励求和（compute_step == sum of event rewards）。
    - Property 15：结局奖励线性核守恒（无事件轨迹，总量守恒等于 terminal_reward）。
    - Property 16：事件奖励高斯核守恒与因果性（terminal_reward=0 隔离事件信用，
      总量守恒等于事件奖励值，且分配仅落在 t<=t_e）。

运行（Python 3.11）：
    PYTHONPATH=src python -m pytest tests/test_sparse_reward_air_combat.py -q
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from uav_vpp_guidance.envs.sparse_reward_air_combat import (
    EVENT_REWARDS,
    AirCombatEvent,
    AirCombatSparseReward,
)

# 全部事件成员列表，供 hypothesis 抽样使用。
_ALL_EVENTS = list(AirCombatEvent)


# Feature: air-combat-mvp, Property 14: 事件奖励求和
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
#
# 从 8 个事件枚举成员随机抽取子集（允许重复、允许空列表），断言 compute_step
# 返回各事件奖励值之和。覆盖全部 8 个事件奖励值（4.1–4.8）。
@settings(max_examples=200)
@given(
    events=st.lists(
        st.sampled_from(_ALL_EVENTS),
        min_size=0,
        max_size=20,
    )
)
def test_event_reward_summation(events):
    reward = AirCombatSparseReward()

    expected = sum(EVENT_REWARDS[e] for e in events)
    actual = reward.compute_step(events)

    assert actual == pytest.approx(expected)


# Feature: air-combat-mvp, Property 15: 结局奖励线性核守恒
# Validates: Requirements 4.9
#
# 对一条没有任何事件的轨迹（step_events 全为空列表），线性核将 terminal_reward
# 在整条轨迹上均匀分配。断言：
#   (1) 守恒：分配后总量等于 terminal_reward；
#   (2) 均匀：每一步取值均等于 terminal_reward / episode_length。
@settings(max_examples=200)
@given(
    terminal_reward=st.floats(
        min_value=-1000.0,
        max_value=1000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    episode_length=st.integers(min_value=1, max_value=200),
)
def test_terminal_reward_linear_kernel_conservation(terminal_reward, episode_length):
    reward = AirCombatSparseReward()
    step_events = [[] for _ in range(episode_length)]

    rewards = reward.relabel_trajectory(
        step_events=step_events,
        terminal_reward=terminal_reward,
        episode_length=episode_length,
    )

    assert rewards.shape == (episode_length,)
    # (1) 守恒：总量等于 terminal_reward。
    assert rewards.sum() == pytest.approx(terminal_reward)
    # (2) 均匀：每一步均等于 terminal_reward / episode_length。
    expected_per_step = terminal_reward / episode_length
    np.testing.assert_allclose(
        rewards,
        np.full(episode_length, expected_per_step),
        rtol=1e-9,
        atol=1e-9,
    )


# Feature: air-combat-mvp, Property 16: 事件奖励高斯核守恒与因果性
# Validates: Requirements 4.10
#
# 设 terminal_reward=0.0 以隔离事件信用（线性核贡献为零，数组中只剩事件信用）。
# 在步 t_e 放置恰好一个事件，调用 relabel_trajectory 后断言：
#   (1) 守恒：分配总量等于该事件的奖励值；
#   (2) 因果性：t > t_e 的所有步信用为零（高斯核仅在 [max(0,t_e-window), t_e] 分配）。
# 同时通过 config 生成 window（含退化路径 window=0）与 gaussian_sigma，
# 以同时覆盖高斯核与单点退化路径，确保两种路径下守恒与因果性均成立。
@settings(max_examples=200)
@given(
    data=st.data(),
    episode_length=st.integers(min_value=1, max_value=200),
    event=st.sampled_from(_ALL_EVENTS),
    window=st.integers(min_value=0, max_value=200),
)
def test_event_reward_gaussian_kernel_conservation_and_causality(
    data, episode_length, event, window
):
    # t_e 落在 [0, episode_length - 1] 内。
    t_e = data.draw(st.integers(min_value=0, max_value=episode_length - 1))

    reward = AirCombatSparseReward(config={"relabelling": {"window": window}})

    # 在步 t_e 放置恰好一个事件；其余步无事件。
    step_events = [[] for _ in range(episode_length)]
    step_events[t_e] = [event]

    rewards = reward.relabel_trajectory(
        step_events=step_events,
        terminal_reward=0.0,  # 隔离事件信用
        episode_length=episode_length,
    )

    event_reward = EVENT_REWARDS[event]

    assert rewards.shape == (episode_length,)
    # (1) 守恒：高斯核分配总量等于事件奖励值。
    assert rewards.sum() == pytest.approx(event_reward)
    # (2) 因果性：t > t_e 的所有步信用为零。
    if t_e + 1 < episode_length:
        assert rewards[t_e + 1 :].sum() == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            rewards[t_e + 1 :],
            np.zeros(episode_length - t_e - 1),
            atol=1e-12,
        )
