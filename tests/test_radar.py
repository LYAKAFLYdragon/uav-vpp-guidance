"""Property-based tests for FireControlRadar (air-combat-mvp).

本文件包含火控雷达的属性测试：
    - Property 10：雷达视场判定的充要性（Validates: Requirements 2.2, 2.3）
    - Property 11：雷达锁定状态机的充要性（Validates: Requirements 2.4, 2.5, 2.6）

坐标系约定（与实现一致）：position_m = [north, up, east]，velocity_mps = [v_north, v_up, v_east]。

运行（Python 3.11）：
    PYTHONPATH=src python -m pytest tests/test_radar.py -q
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from uav_vpp_guidance.sensors.radar import FireControlRadar, wrap_to_pi

# 与实现保持一致的常量与防奇异小量。
MAX_RANGE_M = FireControlRadar.MAX_RANGE_M
AZIMUTH_FOV_DEG = FireControlRadar.AZIMUTH_FOV_DEG
ELEVATION_FOV_DEG = FireControlRadar.ELEVATION_FOV_DEG
LOCK_TIME_STEPS = FireControlRadar.LOCK_TIME_STEPS
_EPS = 1e-8


def _make_state(position, velocity):
    """构造状态字典（position_m/velocity_mps 均为 np.array）。"""
    return {
        "position_m": np.array(position, dtype=float),
        "velocity_mps": np.array(velocity, dtype=float),
    }


def _oracle_geometry(ego_state, target_state):
    """独立复算 range/azimuth/elevation，使用与实现完全相同的公式。

    返回 (range_m, azimuth_deg, elevation_deg)。
    """
    ego_pos = ego_state["position_m"]
    ego_vel = ego_state["velocity_mps"]
    target_pos = target_state["position_m"]

    rel_r = target_pos - ego_pos
    r = float(np.linalg.norm(rel_r))

    if r < _EPS:
        return r, 0.0, 0.0

    rel_north = float(rel_r[0])
    rel_up = float(rel_r[1])
    rel_east = float(rel_r[2])

    psi_ego = np.arctan2(float(ego_vel[2]), float(ego_vel[0]))
    psi_los = np.arctan2(rel_east, rel_north)
    azimuth_rad = wrap_to_pi(psi_los - psi_ego)
    azimuth_deg = float(np.degrees(azimuth_rad))

    elevation_rad = np.arcsin(np.clip(rel_up / r, -1.0, 1.0))
    elevation_deg = float(np.degrees(elevation_rad))

    return r, azimuth_deg, elevation_deg


# 生成策略：位置在 [-15000, 15000]^3 跨越视场内外与最大距离边界。
_coord = st.floats(
    min_value=-15000.0,
    max_value=15000.0,
    allow_nan=False,
    allow_infinity=False,
)
# 本机速度量值与航向：覆盖任意航向（含近零速度由 arctan2 兜底为 0 航向）。
_vel_component = st.floats(
    min_value=-400.0,
    max_value=400.0,
    allow_nan=False,
    allow_infinity=False,
)


# Feature: air-combat-mvp, Property 10: 雷达视场判定的充要性
# Validates: Requirements 2.2, 2.3
#
# 断言 update 返回的 in_view 为真，当且仅当
#   (range_m <= MAX_RANGE_M) and (|azimuth_deg| <= AZIMUTH_FOV_DEG)
#   and (|elevation_deg| <= ELEVATION_FOV_DEG)。
# 同时用独立 oracle（同公式）核对返回的几何量，保证边界处一致。
@settings(max_examples=200)
@given(
    ego_n=_coord, ego_u=_coord, ego_e=_coord,
    tgt_n=_coord, tgt_u=_coord, tgt_e=_coord,
    v_n=_vel_component, v_u=_vel_component, v_e=_vel_component,
)
def test_radar_in_view_iff(ego_n, ego_u, ego_e, tgt_n, tgt_u, tgt_e, v_n, v_u, v_e):
    ego_state = _make_state([ego_n, ego_u, ego_e], [v_n, v_u, v_e])
    target_state = _make_state([tgt_n, tgt_u, tgt_e], [0.0, 0.0, 0.0])

    radar = FireControlRadar()
    result = radar.update(ego_state, target_state)

    range_m = result["range_m"]
    azimuth_deg = result["azimuth_deg"]
    elevation_deg = result["elevation_deg"]
    in_view = result["in_view"]

    # 自洽 iff：直接用返回的几何量在相同数值上验证视场判定的充要性。
    expected_in_view = (
        (range_m <= MAX_RANGE_M)
        and (abs(azimuth_deg) <= AZIMUTH_FOV_DEG)
        and (abs(elevation_deg) <= ELEVATION_FOV_DEG)
    )
    assert in_view == expected_in_view

    # 独立 oracle（同公式）核对返回的几何量，确保边界处的判定基于一致的数值。
    o_range, o_az, o_el = _oracle_geometry(ego_state, target_state)
    assert range_m == o_range
    assert azimuth_deg == o_az
    assert elevation_deg == o_el


# 锁定状态机参考模型：给定布尔序列，逐步推进。
def _reference_lock(in_view_seq):
    """返回每步参考的 (consecutive_count, locked) 列表。"""
    out = []
    count = 0
    for iv in in_view_seq:
        if iv:
            count += 1
            locked = count >= LOCK_TIME_STEPS
        else:
            count = 0
            locked = False
        out.append((count, locked))
    return out


# Feature: air-combat-mvp, Property 11: 雷达锁定状态机的充要性
# Validates: Requirements 2.4, 2.5, 2.6
#
# 用 hypothesis 生成任意 in_view 布尔序列驱动状态机：每步根据布尔值构造
# 一个"正前方在距离内"（in_view True）或"正后方/视场外"（in_view False）的
# 目标几何，调用 update，并与参考模型对比：
#   - locked == (连续 in_view 步数 >= LOCK_TIME_STEPS)
#   - in_view 为 False 的步立即清零计数且 locked 必为 False。
@settings(max_examples=200)
@given(in_view_seq=st.lists(st.booleans(), min_size=1, max_size=30))
def test_radar_lock_state_machine_iff(in_view_seq):
    radar = FireControlRadar()

    # 本机位于原点，速度朝正北 [250, 0, 0]（航向 0）。
    ego_state = _make_state([0.0, 0.0, 0.0], [250.0, 0.0, 0.0])
    # 正前方在距离内：方位 0、俯仰 0、距离 1000 -> in_view True。
    target_in = _make_state([1000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    # 正后方：方位 180° -> in_view False。
    target_out = _make_state([-1000.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    reference = _reference_lock(in_view_seq)

    for step_idx, want_in_view in enumerate(in_view_seq):
        target_state = target_in if want_in_view else target_out
        result = radar.update(ego_state, target_state)

        # 先验证构造的几何确实产生了预期的 in_view（避免依赖未验证的假设）。
        assert result["in_view"] == want_in_view

        ref_count, ref_locked = reference[step_idx]

        # 锁定充要性：locked == (连续 in_view 步数 >= LOCK_TIME_STEPS)。
        assert radar._consecutive_in_view_steps == ref_count
        assert result["locked"] == ref_locked
        assert radar.locked == ref_locked

        # 脱离即清零：in_view 为 False 的步必须立即清零计数且解除锁定。
        if not want_in_view:
            assert radar._consecutive_in_view_steps == 0
            assert result["locked"] is False
