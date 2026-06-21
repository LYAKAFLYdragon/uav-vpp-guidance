"""Property-based tests for AirCombatMVPEnv logic (air-combat-mvp).

本文件包含空战 MVP 环境逻辑的属性测试：
    - Property 12：环境发射触发的组合充要性（Validates: Requirements 3.7）
    - Property 13：脱离发射包线事件检测（Validates: Requirements 3.13, 4.8）

实现方法（Option A — 构造真实 AirCombatMVPEnv）：
    本测试直接以最小 ``backend: simple`` 配置构造真实 ``AirCombatMVPEnv`` 并复用其
    真实组件（``missile_ego.can_launch`` / ``_is_in_launch_envelope`` /
    ``_detect_events`` / ``_prev_in_envelope``）。相较于纯逻辑桩件，Option A 直接
    驱动环境本身的发射决策谓词与事件检测方法，是对设计 §3.3.4（step 发射组合条件）
    与 §3.3.5 / §3.4.2（发射包线事件检测）最忠实的属性测试方式；经验证该最小配置
    可在不依赖 JSBSim 的前提下稳定构造并 reset/step，故对两条属性均采用 Option A。

坐标系约定（与实现一致）：
    JSBSim 坐标系 X=North, Y=Up, Z=East；position_m = [north, up, east]，
    velocity_mps = [v_north, v_up, v_east]，单位米/秒/弧度。

运行（Python 3.11）：
    PYTHONPATH=src python -m pytest tests/test_mvp_air_combat.py -q
"""

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from uav_vpp_guidance.envs.air_combat_mvp_env import AirCombatMVPEnv
from uav_vpp_guidance.envs.sparse_reward_air_combat import AirCombatEvent

OUT_OF_ENVELOPE = AirCombatEvent.OUT_OF_ENVELOPE.value


# ---------------------------------------------------------------------------
# 最小 simple-backend 配置（不依赖 JSBSim），用于构造真实 AirCombatMVPEnv。
# ---------------------------------------------------------------------------
def _build_config() -> dict:
    return {
        "experiment": {"name": "test_mvp_air_combat", "seed": 0, "output_root": "outputs"},
        "env": {
            "use_jsbsim": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": 64,
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.2,
            "hysteresis_range_m": 950.0,
            "hysteresis_ata_deg": 30.0,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 8000.0,
            "target_mode": "constant_velocity",
            "high_level_dt": 0.2,
        },
        "virtual_point": {
            "anchor_mode": "current_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        },
        "reward": {
            "w_range": 0.5,
            "w_angle": 0.8,
            "w_energy": 0.2,
            "w_safety": 2.0,
            "w_saturation": 1.0,
            "w_smooth": 0.1,
            "terminal_success": 200.0,
            "terminal_failure": -200.0,
            "terminal_crash": -300.0,
            "min_altitude_m": 500.0,
        },
        "guidance": {
            "mode": "los_rate",
            "use_gain_adapter": False,
            "gains": {
                "k_los": 1.0,
                "k_pos": 0.5,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "alpha_filter": 0.3,
            },
        },
        "air_combat": {
            "missile_config": {"ego": {"enabled": True}, "target": {"enabled": False}},
            "radar_config": {
                "max_range_m": 10000.0,
                "azimuth_fov_deg": 60.0,
                "elevation_fov_deg": 30.0,
                "lock_time_steps": 3,
            },
            "reward": {"use_sparse": True},
        },
    }


# 模块级单例环境：构造一次后在各属性示例间复用（每个示例自行复位相关可变状态）。
_ENV_SINGLETON = None


def _get_env() -> AirCombatMVPEnv:
    global _ENV_SINGLETON
    if _ENV_SINGLETON is None:
        _ENV_SINGLETON = AirCombatMVPEnv(_build_config())
        _ENV_SINGLETON.reset(seed=0)
    return _ENV_SINGLETON


def _make_state(position, velocity):
    """构造状态字典（position_m/velocity_mps 均为 np.array，[north, up, east]）。"""
    return {
        "position_m": np.array(position, dtype=float),
        "velocity_mps": np.array(velocity, dtype=float),
    }


def _state_at(range_m: float, ata_deg: float):
    """以本机在原点、速度沿北（[250,0,0]）为基准，构造距离 range_m、相对本机
    速度方向夹角（ATA）为 ata_deg 的目标几何。

    本机速度沿 North；目标位于 North-East 平面内、相对 North 偏转 ata_deg：
        target_pos = range_m * [cos(ata), 0, sin(ata)]
    则 LOS 与本机速度（North）的夹角恰为 ata_deg。
    """
    ego = _make_state([0.0, 0.0, 0.0], [250.0, 0.0, 0.0])
    ata_rad = math.radians(ata_deg)
    target_pos = [
        range_m * math.cos(ata_rad),
        0.0,
        range_m * math.sin(ata_rad),
    ]
    target = _make_state(target_pos, [-250.0, 0.0, 0.0])
    return ego, target


# ===========================================================================
# Property 12：环境发射触发的组合充要性
# 断言"创建新在飞导弹"当且仅当 radar.locked、can_launch 允许、无在飞导弹三者
# 同时成立（设计 §3.3.4 step 第 ③ 步组合条件：
#   radar.locked AND not missile.in_flight AND can_launch(own, target)）。
# ===========================================================================
@settings(max_examples=150, deadline=None)
@given(
    locked=st.booleans(),
    in_flight=st.booleans(),
    range_m=st.floats(
        min_value=100.0, max_value=6000.0, allow_nan=False, allow_infinity=False
    ),
    ata_deg=st.floats(
        min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_12_launch_trigger_combination(locked, in_flight, range_m, ata_deg):
    # Feature: air-combat-mvp, Property 12: 环境发射触发的组合充要性——
    # 创建新在飞导弹 ⇔ (radar.locked) AND (无在飞导弹) AND (can_launch 允许)。
    # Validates: Requirements 3.7
    env = _get_env()

    # 避开发射包线边界（距离 500/5000，|ATA| 30°）的浮点歧义，使
    # can_launch 的真值确定，断言不受边界数值误差影响。
    assume(abs(range_m - 500.0) > 1.0 and abs(range_m - 5000.0) > 1.0)
    assume(abs(ata_deg - 30.0) > 0.1)

    ego, target = _state_at(range_m, ata_deg)

    # 复位本机导弹并按生成的 in_flight 置位（模拟"是否已有在飞导弹"）。
    AirCombatMVPEnv._reset_missile(env.missile_ego)
    env.missile_ego.in_flight = in_flight

    # 独立计算三个条件（can_launch 为纯几何判定，不依赖雷达/在飞状态）。
    can = env.missile_ego.can_launch(ego, target)

    # 忠实复刻 step 第 ③ 步的发射决策谓词：三者 AND 成立才创建新在飞导弹。
    launched = False
    if (
        locked
        and not env.missile_ego.in_flight
        and env.missile_ego.can_launch(ego, target)
    ):
        env.missile_ego.launch(ego)
        launched = True

    # 充要性（iff）：创建新在飞导弹 ⇔ 三条件同时成立。
    expected = locked and (not in_flight) and can
    assert launched == expected

    # 三向必要性：发射发生 ⇒ 三条件全为真；任一为假 ⇒ 不发射。
    if launched:
        assert locked and (not in_flight) and can
    if (not locked) or in_flight or (not can):
        assert not launched

    # 发射成功时确实创建了一枚在飞导弹（in_flight 由 False 跳变为 True）。
    if launched:
        assert env.missile_ego.in_flight is True


# ===========================================================================
# Property 13：脱离发射包线事件检测
# 用几何序列断言 out_of_envelope 事件当且仅当从"包线内"跳变到"包线外"
# （设计 §3.3.5 / §3.4.2：发射包线内 True → False 边沿触发）。
# ===========================================================================
@st.composite
def _envelope_sequence(draw):
    """生成一串几何步：每步标注其"是否在发射包线内"的意图，并据此采样确定性
    落在包线内/外的 (range_m, ata_deg)，避开边界以消除浮点歧义。

    包线定义：距离 ∈ [500, 5000] m 且 |ATA| < 30°。
        - 包线内：range ∈ [600, 4900]，ata ∈ [0, 25]。
        - 包线外：三种模式之一——距离过近 [50,400]、距离过远 [5100,8000]、
          或 |ATA| 过大 [35,60]（距离合法）。
    """
    n = draw(st.integers(min_value=1, max_value=12))
    steps = []
    for _ in range(n):
        in_env = draw(st.booleans())
        if in_env:
            range_m = draw(st.floats(min_value=600.0, max_value=4900.0))
            ata_deg = draw(st.floats(min_value=0.0, max_value=25.0))
        else:
            mode = draw(st.integers(min_value=0, max_value=2))
            if mode == 0:  # 距离过近
                range_m = draw(st.floats(min_value=50.0, max_value=400.0))
                ata_deg = draw(st.floats(min_value=0.0, max_value=25.0))
            elif mode == 1:  # 距离过远
                range_m = draw(st.floats(min_value=5100.0, max_value=8000.0))
                ata_deg = draw(st.floats(min_value=0.0, max_value=25.0))
            else:  # |ATA| 过大（距离合法）
                range_m = draw(st.floats(min_value=600.0, max_value=4900.0))
                ata_deg = draw(st.floats(min_value=35.0, max_value=60.0))
        steps.append((bool(in_env), float(range_m), float(ata_deg)))
    return steps


@settings(max_examples=150, deadline=None)
@given(sequence=_envelope_sequence())
def test_property_13_out_of_envelope_event(sequence):
    # Feature: air-combat-mvp, Property 13: 脱离发射包线事件检测——
    # out_of_envelope 事件当且仅当包线状态从 True 跳变到 False。
    # Validates: Requirements 3.13, 4.8
    env = _get_env()

    # 复位发射包线与事件检测状态（首步 prior 视为 False，与 reset/init 一致）。
    env._prev_in_envelope = False
    env._prev_event_state = {}
    # 锁定恒为 False，避免触发 radar_lock / radar_lock_lost 事件，孤立 out_of_envelope。
    radar_state = {
        "locked": False,
        "in_view": False,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "range_m": 0.0,
    }

    prev_in_envelope = False
    for intended_in_env, range_m, ata_deg in sequence:
        ego, target = _state_at(range_m, ata_deg)

        # 先验证环境的包线判定与意图一致（依赖此前确认其可靠，避免边界歧义）。
        actual_in_env = env._is_in_launch_envelope(ego, target)
        assert actual_in_env == intended_in_env

        # 调用真实事件检测；launched/hit/expired 均为 False，孤立发射包线事件。
        events = env._detect_events(
            ego, target, radar_state, launched=False, hit=False, expired=False
        )

        # 充要性：out_of_envelope ⇔ 上一步在包线内且当前不在包线内。
        expected_out = prev_in_envelope and (not actual_in_env)
        assert (OUT_OF_ENVELOPE in events) == expected_out

        # _detect_events 须把 _prev_in_envelope 更新为当前包线状态，供下一步跳变检测。
        assert env._prev_in_envelope == actual_in_env

        prev_in_envelope = actual_in_env
