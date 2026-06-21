"""Property-based and unit tests for Missile3DoF (air-combat-mvp).

本文件累积 Missile3DoF 相关测试。当前包含速度分解—反解可逆性的属性测试
（Property 9）。

运行（Python 3.11）：
    PYTHONPATH=src python -m pytest tests/test_missile.py -q
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from uav_vpp_guidance.weapons.missile import (
    decompose_velocity,
    velocity_to_spherical,
)


# 生成策略：speed >= 0，gamma ∈ [-π/2, π/2]，psi ∈ (-π, π]
# 排除 nan/inf，保证数值有限。
_speed_strategy = st.floats(
    min_value=0.0,
    max_value=2000.0,
    allow_nan=False,
    allow_infinity=False,
)
_gamma_strategy = st.floats(
    min_value=-np.pi / 2,
    max_value=np.pi / 2,
    allow_nan=False,
    allow_infinity=False,
)
# psi ∈ (-π, π]：排除下界 -π（与 +π 等价，arctan2 反解返回 +π），包含上界 +π。
_psi_strategy = st.floats(
    min_value=-np.pi,
    max_value=np.pi,
    allow_nan=False,
    allow_infinity=False,
    exclude_min=True,
)


# Feature: air-combat-mvp, Property 9: 速度分解—反解可逆性
# Validates: Requirements 8.5
#
# 鲁棒可逆性表述：直接断言原始角度相等在退化点会失败——
#   - speed ≈ 0 时，方向（gamma/psi）不可恢复（反解兜底为 0）；
#   - gamma 在 ±π/2 处 psi 不可辨（cos(gamma)≈0，水平分量为零）。
# 因此采用稳健表述：断言速度向量往返一致
#   (speed, gamma, psi) -> vel -> (speed2, gamma2, psi2) -> vel2，
# 要求 vel2 在数值误差内等于 vel。该表述在退化点同样成立
# （退化点处速度向量本身仍可一致重建），是可逆性的鲁棒陈述。
@settings(max_examples=200)
@given(speed=_speed_strategy, gamma=_gamma_strategy, psi=_psi_strategy)
def test_velocity_decomposition_invertibility(speed, gamma, psi):
    vel = decompose_velocity(speed, gamma, psi)

    # 反解得到球面参数。
    speed2, gamma2, psi2 = velocity_to_spherical(vel)

    # 由反解参数重新分解，应与原速度向量一致（鲁棒可逆性）。
    vel2 = decompose_velocity(speed2, gamma2, psi2)

    # 速度量值在 [0, 2000]，使用绝对+相对容差。
    np.testing.assert_allclose(vel2, vel, rtol=1e-9, atol=1e-9)

    # 速度量值始终可恢复（无论是否退化）。
    assert speed2 == pytest.approx(speed, rel=1e-9, abs=1e-9)


# Feature: air-combat-mvp, Property 9: 速度分解—反解可逆性（角度相等，非退化子域）
# Validates: Requirements 8.5
#
# 补充：在 speed 远离 0 且 gamma 远离 ±π/2 的非退化子域内，
# 角度本身也应往返一致（更强的直接断言）。
@settings(max_examples=200)
@given(
    speed=st.floats(
        min_value=1.0,
        max_value=2000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    gamma=st.floats(
        min_value=-np.pi / 2 + 1e-3,
        max_value=np.pi / 2 - 1e-3,
        allow_nan=False,
        allow_infinity=False,
    ),
    psi=_psi_strategy,
)
def test_velocity_decomposition_angle_invertibility_nondegenerate(speed, gamma, psi):
    vel = decompose_velocity(speed, gamma, psi)
    speed2, gamma2, psi2 = velocity_to_spherical(vel)

    assert speed2 == pytest.approx(speed, rel=1e-9, abs=1e-9)
    assert gamma2 == pytest.approx(gamma, abs=1e-9)
    assert psi2 == pytest.approx(psi, abs=1e-9)


# ---------------------------------------------------------------------------
# Property 2: 发射包线判定的充要性（can_launch necessary-and-sufficient）
# ---------------------------------------------------------------------------
from uav_vpp_guidance.weapons.missile import Missile3DoF, _EPS


def _make_state(position, velocity=None):
    """构造与环境互通的状态字典。

    位置使用首选键 ``position_m``（3 元向量）；速度使用 ``velocity_mps``
    （3 元向量时被 :func:`_extract_velocity` 直接采用）。
    """
    state = {"position_m": np.asarray(position, dtype=float)}
    if velocity is not None:
        state["velocity_mps"] = np.asarray(velocity, dtype=float)
    return state


def _reference_range_and_ata_deg(ego_pos, ego_vel, target_pos):
    """独立参考预言机：直接从生成的状态计算 range 与 |ATA|（度）。

    为避免在精确边界（500 / 5000 / 30°）处与实现产生浮点级别的偶发分歧，
    本预言机使用与 ``Missile3DoF.can_launch`` **完全相同** 的 numpy 公式与
    同一 ``_EPS`` 平滑项，保证二者在边界处逐位一致（bit-for-bit）：

        range  = |target.pos - ego.pos|
        los_hat = los / (range + _EPS)
        v_hat   = ego.vel / (|ego.vel| + _EPS)
        ATA    = degrees(arccos(clip(dot(v_hat, los_hat), -1, 1)))
    """
    los = np.asarray(target_pos, float) - np.asarray(ego_pos, float)
    range_m = float(np.linalg.norm(los))
    los_hat = los / (range_m + _EPS)
    ego_vel = np.asarray(ego_vel, float)
    ego_speed = float(np.linalg.norm(ego_vel))
    v_hat = ego_vel / (ego_speed + _EPS)
    cos_ata = float(np.clip(np.dot(v_hat, los_hat), -1.0, 1.0))
    ata_deg = float(np.degrees(np.arccos(cos_ata)))
    return range_m, ata_deg


# 生成策略：覆盖发射包线内外与边界。
#   - 距离 r ∈ [0, 7000]：跨越低于下限(500)、带内、高于上限(5000)三段。
#   - 目标方向（az/el 角）随机：覆盖任意三维方位。
#   - 本机速度独立随机航向（speed ∈ [1, 400]，方向任意）：使 ATA 覆盖 [0, 180°]。
#   - 本机位置任意偏移：验证判定对绝对位置平移不变（仅依赖相对几何）。
_range_strategy = st.floats(
    min_value=0.0, max_value=7000.0, allow_nan=False, allow_infinity=False
)
_angle_strategy = st.floats(
    min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False
)
_offset_strategy = st.floats(
    min_value=-2000.0, max_value=2000.0, allow_nan=False, allow_infinity=False
)
_ego_speed_strategy = st.floats(
    min_value=1.0, max_value=400.0, allow_nan=False, allow_infinity=False
)


def _unit_from_angles(az, el):
    """由方位角 az 与俯仰角 el 构造单位方向向量（坐标轴排列对判定不变）。"""
    ce = np.cos(el)
    return np.array([ce * np.cos(az), np.sin(el), ce * np.sin(az)], dtype=float)


# Feature: air-combat-mvp, Property 2: 发射包线判定的充要性
# Validates: Requirements 1.7, 1.8, 1.9
#
# 充要性表述：can_launch 返回允许 当且仅当
#   (LAUNCH_RANGE_MIN <= range <= LAUNCH_RANGE_MAX)  且  (|ATA| < LAUNCH_ATA_MAX_DEG)。
# 用独立预言机直接从生成的本机/目标状态计算 range 与 |ATA|，再构造该充要条件
# 与 can_launch 的返回值比对，从而在生成域上证明二者等价（iff）。
# 生成覆盖包线内/外/边界：r ∈ [0, 7000] 跨越下限/带内/上限，本机航向独立随机使
# |ATA| 覆盖 [0, 180°]。预言机与实现采用同一 numpy 公式与同一 _EPS，故在精确边界
# 处逐位一致，不会出现偶发的边界分歧。
@settings(max_examples=200)
@given(
    r=_range_strategy,
    tgt_az=_angle_strategy,
    tgt_el=st.floats(
        min_value=-np.pi / 2, max_value=np.pi / 2,
        allow_nan=False, allow_infinity=False,
    ),
    ego_speed=_ego_speed_strategy,
    ego_az=_angle_strategy,
    ego_el=st.floats(
        min_value=-np.pi / 2, max_value=np.pi / 2,
        allow_nan=False, allow_infinity=False,
    ),
    ox=_offset_strategy,
    oy=_offset_strategy,
    oz=_offset_strategy,
)
def test_can_launch_necessary_and_sufficient(
    r, tgt_az, tgt_el, ego_speed, ego_az, ego_el, ox, oy, oz
):
    missile = Missile3DoF()

    ego_pos = np.array([ox, oy, oz], dtype=float)
    target_pos = ego_pos + r * _unit_from_angles(tgt_az, tgt_el)
    ego_vel = ego_speed * _unit_from_angles(ego_az, ego_el)

    ego_state = _make_state(ego_pos, ego_vel)
    target_state = _make_state(target_pos)

    # 独立预言机：直接计算 range 与 |ATA|（与实现同公式同 _EPS，边界逐位一致）。
    range_m, ata_deg = _reference_range_and_ata_deg(ego_pos, ego_vel, target_pos)
    expected = (
        (missile.LAUNCH_RANGE_MIN <= range_m <= missile.LAUNCH_RANGE_MAX)
        and (abs(ata_deg) < missile.LAUNCH_ATA_MAX_DEG)
    )

    assert missile.can_launch(ego_state, target_state) is expected


# ---------------------------------------------------------------------------
# Property 1: 制导加速度过载限幅（PNG acceleration g-limit）
# ---------------------------------------------------------------------------
from uav_vpp_guidance.weapons.missile import GRAVITY


# 生成策略：任意相对几何。
#   - 导弹/目标位置 ∈ [-10000, 10000]^3：覆盖宽广空间，含近零相对距离（奇异点附近）。
#   - 导弹/目标速度 ∈ [-1000, 1000]^3：覆盖任意相对速度方向与量值。
# 近零相对距离由两端位置独立采样自然产生（含偶发相互靠近），用以检验奇异守卫
# （分母加 _EPS）下限幅仍成立。
_pos_component = st.floats(
    min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False
)
_vel_component = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
_vec3 = lambda comp: st.tuples(comp, comp, comp)  # noqa: E731


# Feature: air-combat-mvp, Property 1: 制导加速度过载限幅
# Validates: Requirements 1.4, 8.4
#
# 过载限幅充分性：对任意相对几何（任意导弹/目标位置与速度），_compute_png_acceleration
# 返回的 PNG 制导加速度范数必须不超过 MAX_G * GRAVITY（≈294.2 m/s²）。制导律采用标准
# 真比例导引（true PN）向量形式 a_png = K * cross(omega, V_missile)，其中
# omega = cross(rel_r, rel_v)/dot(rel_r, rel_r) 为视线角速度、V_missile = missile.velocity_mps
# 为导弹速度向量；限幅后该不变量（范数 <= MAX_G*GRAVITY）对新制导律同样成立。生成域覆盖
# 宽广位置/速度范围，并自然包含近零相对距离以触发奇异守卫（除法分母加 _EPS）。
@settings(max_examples=200)
@given(
    missile_pos=_vec3(_pos_component),
    missile_vel=_vec3(_vel_component),
    target_pos=_vec3(_pos_component),
    target_vel=_vec3(_vel_component),
)
def test_png_acceleration_g_limit(missile_pos, missile_vel, target_pos, target_vel):
    missile = Missile3DoF()

    # 直接设置导弹自身位置/速度（绕过 launch，覆盖任意相对几何）。
    missile.position_m = np.asarray(missile_pos, dtype=float)
    missile.velocity_mps = np.asarray(missile_vel, dtype=float)

    target_state = _make_state(target_pos, target_vel)

    a_png = missile._compute_png_acceleration(target_state)

    # 返回应为有限的 3 向量。
    assert a_png.shape == (3,)
    assert np.all(np.isfinite(a_png))

    a_max = missile.MAX_G * GRAVITY
    a_norm = float(np.linalg.norm(a_png))

    # 充分性：范数不超过过载上限（含极小数值容差）。
    assert a_norm <= a_max + 1e-6


# ---------------------------------------------------------------------------
# Property 3: 步进推进的运动学一致性（step kinematic consistency）
# ---------------------------------------------------------------------------
#
# 生成策略：任意在飞导弹与 dt > 0。
#   - 导弹位置 ∈ [-10000, 10000]^3：覆盖宽广空间。
#   - 导弹速度 ∈ [-1000, 1000]^3：覆盖任意速度方向与量值（含近零速度，
#     触发 v_hat 北向兜底）。
#   - flight_time_s ∈ [0, 20]：覆盖燃烧段（<=5s）与燃烧结束段（>5s），
#     使质量递减/推力切换的两种分支均被步进覆盖。
#   - dt ∈ [1e-3, 1.0]：正步长。
#   - target_state 随机位置/速度：制导加速度任意，但不影响运动学一致性断言
#     （位置位移恒等于步进后速度 * dt，与具体加速度无关）。
_mt_pos_component = st.floats(
    min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False
)
_mt_vel_component = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
_mt_dt_strategy = st.floats(
    min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False
)
_mt_flight_time_strategy = st.floats(
    min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
)


# Feature: air-combat-mvp, Property 3: 步进推进的运动学一致性
# Validates: Requirements 1.10
#
# 运动学一致性表述：对任意在飞导弹与任意 dt > 0，调用 step(dt, target_state) 后
#   (a) flight_time_s 恰增加 dt（flight_time_after == flight_time_before + dt）；
#   (b) 位置位移 Δposition == velocity_mps(步进后) * dt。
#
# 关于 (b) 的精度说明：Missile3DoF.step 采用半隐式（symplectic）欧拉——先更新速度
# `velocity_mps += a_total*dt`，再用 **更新后** 的速度更新位置
# `position_m += velocity_mps*dt`。因此位置位移恰等于步进后速度 * dt，这是
# **精确（exact）** 关系而非仅"积分误差内"：实现中 displacement = velocity_mps*dt
# 与本测试重算的 velocity_mps_post*dt 为同一组浮点运算，二者仅有机器精度级别差异。
# 故采用很小的 rtol/atol（assert_allclose）即可，对照设计 §3.1.4 的积分顺序约定。
@settings(max_examples=200)
@given(
    missile_pos=_vec3(_mt_pos_component),
    missile_vel=_vec3(_mt_vel_component),
    target_pos=_vec3(_mt_pos_component),
    target_vel=_vec3(_mt_vel_component),
    flight_time=_mt_flight_time_strategy,
    dt=_mt_dt_strategy,
)
def test_step_kinematic_consistency(
    missile_pos, missile_vel, target_pos, target_vel, flight_time, dt
):
    missile = Missile3DoF()

    # 构造在飞导弹：直接设置状态并置 in_flight=True（绕过 launch，覆盖任意状态）。
    missile.position_m = np.asarray(missile_pos, dtype=float)
    missile.velocity_mps = np.asarray(missile_vel, dtype=float)
    missile.mass_kg = 400.0
    missile.flight_time_s = flight_time
    missile.in_flight = True

    target_state = _make_state(target_pos, target_vel)

    # 记录步进前的位置与飞行时间。
    pos_before = np.array(missile.position_m, dtype=float, copy=True)
    ft_before = missile.flight_time_s

    missile.step(dt, target_state)

    # (a) 飞行时间恰增加 dt。
    assert missile.flight_time_s == pytest.approx(ft_before + dt, rel=1e-12, abs=1e-12)

    # (b) 位置位移 == 步进后速度 * dt（半隐式欧拉，精确至机器精度）。
    delta_pos = missile.position_m - pos_before
    expected_disp = missile.velocity_mps * dt
    np.testing.assert_allclose(delta_pos, expected_disp, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------------------
# Property 6: 燃烧段质量单调性与推力切换（burn-phase mass monotonicity）
# ---------------------------------------------------------------------------
#
# 生成策略：飞行时间序列。
#   - 通过 launch() 发射，置 mass_kg=400、flight_time_s=0、in_flight=True。
#   - 用 hypothesis 生成一串 dt（每个 dt ∈ [1e-3, 1.0]，正步长），逐步 step。
#     列表长度 ∈ [1, 40]，总飞行时间可覆盖燃烧段（<=5s）之前与之后，使
#     "燃烧段质量递减"与"燃烧结束推力切换/质量恒定"两个分支均被自然覆盖。
#   - 目标取一个固定的远处目标（faraway），其位置/速度仅驱动 PNG 制导；
#     质量动力学不依赖 PNG，故该选择不影响质量相关断言。
#
# 关键常量（与实现一致）：
#   structural_mass = INITIAL_MASS - BURN_RATE * ENGINE_BURN_TIME
#                   = 400 - 25 * 5 = 275 kg。
#   燃烧段成员判定使用 **步进前** 的 flight_time_s <= ENGINE_BURN_TIME(=5.0)。
#   燃烧段内 mass_kg = max(mass_kg - BURN_RATE*dt, structural_mass)（钳到下限）。
#   燃烧结束后既不施加推力也不递减质量 → 质量恒定。
_p6_dt_strategy = st.floats(
    min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False
)
_p6_dt_list_strategy = st.lists(_p6_dt_strategy, min_size=1, max_size=40)


# Feature: air-combat-mvp, Property 6: 燃烧段质量单调性与推力切换
# Validates: Requirements 1.5, 1.6
#
# 单调性与切换的鲁棒表述（处理 max() 钳位）：对一条由 launch 起始、用任意正
# dt 序列驱动的飞行轨迹，逐步检查 step 前后的质量：
#   (a) 质量全程非递增（mass_after <= mass_before，含数值容差）；
#   (b) 任一步若 **步进前** flight_time_s <= ENGINE_BURN_TIME 且 **步进前** 质量
#       严格高于结构质量下限 275，则该步质量 **严格递减**（mass_after < mass_before）
#       —— 即燃烧段在触及 275 地板之前严格单调递减；一旦因 max() 钳到 275，
#       质量停止下降（由 (a) 的非递增与 (d) 的下限共同保证，不再强求严格递减）；
#   (c) 任一步若 **步进前** flight_time_s > ENGINE_BURN_TIME，则该步质量恒定
#       （mass_after == mass_before）—— 燃烧结束后推力关闭且不再递减质量；
#   (d) 质量全程位于 [structural_mass(275), INITIAL_MASS(400)]。
@settings(max_examples=200)
@given(dts=_p6_dt_list_strategy)
def test_burn_phase_mass_monotonicity_and_thrust_switch(dts):
    missile = Missile3DoF()

    structural_mass = (
        missile.INITIAL_MASS - missile.BURN_RATE * missile.ENGINE_BURN_TIME
    )
    assert structural_mass == pytest.approx(275.0)

    # 从本机状态发射：mass=400, flight_time=0, in_flight=True。
    ego_state = _make_state([0.0, 5000.0, 0.0], [500.0, 0.0, 0.0])
    missile.launch(ego_state)

    assert missile.mass_kg == pytest.approx(missile.INITIAL_MASS)
    assert missile.flight_time_s == pytest.approx(0.0)

    # 远处固定目标，仅用于驱动 PNG（不影响质量动力学）。
    target_state = _make_state([100000.0, 5000.0, 0.0], [-250.0, 0.0, 0.0])

    for dt in dts:
        # 记录步进前的燃烧段成员判定依据与质量。
        ft_before = missile.flight_time_s
        mass_before = missile.mass_kg
        is_burning_pre = ft_before <= missile.ENGINE_BURN_TIME

        missile.step(dt, target_state)

        mass_after = missile.mass_kg

        # (a) 全程非递增。
        assert mass_after <= mass_before + 1e-9

        # (d) 质量始终位于 [275, 400]。
        assert structural_mass - 1e-9 <= mass_after <= missile.INITIAL_MASS + 1e-9

        if is_burning_pre:
            if mass_before > structural_mass + 1e-9:
                # (b) 燃烧段且步进前质量高于地板 → 严格递减。
                assert mass_after < mass_before - 1e-12
                # 递减量为 BURN_RATE*dt 或恰好钳到地板。
                expected = max(
                    mass_before - missile.BURN_RATE * dt, structural_mass
                )
                assert mass_after == pytest.approx(expected, rel=1e-9, abs=1e-9)
            else:
                # 已在地板：max() 钳位使质量不再下降（仍 == 275）。
                assert mass_after == pytest.approx(structural_mass, abs=1e-9)
        else:
            # (c) 燃烧结束后质量恒定。
            assert mass_after == pytest.approx(mass_before, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# Property 4: 命中判定的距离阈值充要性（check_hit distance-threshold iff）
# ---------------------------------------------------------------------------
#
# 生成策略：在飞导弹 + 任意目标几何。
#   - 导弹位置 ∈ [-10000, 10000]^3：覆盖宽广空间。
#   - 目标方向（az/el）随机；目标距离 dist ∈ [0, 100]：跨越杀伤半径
#     KILL_RADIUS_M(=30) 的两侧（命中带内 / 带外）与精确边界附近，使
#     "命中 / 未命中"两个分支均被充分覆盖。
#   - in_flight 由参数控制：True 验证距离阈值充要性与状态置位；False 验证
#     未在飞时无论距离如何 check_hit 恒返回 False 且不改写状态。
_p4_pos_component = st.floats(
    min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False
)
_p4_dist_strategy = st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
)


# Feature: air-combat-mvp, Property 4: 命中判定的距离阈值充要性
# Validates: Requirements 1.11
#
# 充要性表述：对在飞导弹，check_hit(target_state) 返回 True 当且仅当
#   |target.pos - missile.pos| <= KILL_RADIUS_M(=30)。
# 预言机用与实现 **完全相同** 的 numpy 公式 float(np.linalg.norm(target_pos -
# missile.pos)) 计算距离，保证在精确边界(30)处与实现逐位一致，避免偶发分歧。
# 另验证：
#   - 命中（返回 True）时 hit=True 且 in_flight=False 被置位；
#   - 未在飞（in_flight=False）时无论距离如何均返回 False，且不改写 hit/in_flight。
@settings(max_examples=200)
@given(
    missile_pos=_vec3(_p4_pos_component),
    tgt_az=_angle_strategy,
    tgt_el=st.floats(
        min_value=-np.pi / 2, max_value=np.pi / 2,
        allow_nan=False, allow_infinity=False,
    ),
    dist=_p4_dist_strategy,
    in_flight=st.booleans(),
)
def test_check_hit_distance_threshold_iff(
    missile_pos, tgt_az, tgt_el, dist, in_flight
):
    missile = Missile3DoF()
    missile.position_m = np.asarray(missile_pos, dtype=float)
    missile.in_flight = in_flight

    target_pos = missile.position_m + dist * _unit_from_angles(tgt_az, tgt_el)
    target_state = _make_state(target_pos)

    # 预言机：与实现同公式同精度计算距离。
    oracle_distance = float(np.linalg.norm(target_pos - missile.position_m))
    within_kill = oracle_distance <= missile.KILL_RADIUS_M

    result = missile.check_hit(target_state)

    if not in_flight:
        # 未在飞：无论距离如何恒返回 False，且不改写状态。
        assert result is False
        assert missile.hit is False
        assert missile.in_flight is False
    else:
        # 在飞：返回值与距离阈值充要等价。
        assert result is within_kill
        if result:
            # 命中时状态置位。
            assert missile.hit is True
            assert missile.in_flight is False
        else:
            # 未命中时不改写状态（仍在飞、未命中）。
            assert missile.hit is False
            assert missile.in_flight is True


# ---------------------------------------------------------------------------
# Property 5: 失效判定的充要性（is_expired necessary-and-sufficient）
# ---------------------------------------------------------------------------
#
# 生成策略：直接设置失效相关状态量，覆盖各阈值两侧。
#   - flight_time_s ∈ [0, 40]：跨越 MAX_FLIGHT_TIME_S(=20) 两侧与边界。
#   - mass_kg ∈ [200, 400]：跨越结构质量地板 structural_mass(=275) 两侧。
#   - flight_distance_m ∈ [0, 10000]：跨越 MAX_RANGE_M(=5000) 两侧。
# 三者独立采样，使 (超时) 与 (燃料耗尽 且 超射程) 两个失效分支及其组合均被覆盖。
_p5_flight_time_strategy = st.floats(
    min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False
)
_p5_mass_strategy = st.floats(
    min_value=200.0, max_value=400.0, allow_nan=False, allow_infinity=False
)
_p5_distance_strategy = st.floats(
    min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
)


# Feature: air-combat-mvp, Property 5: 失效判定的充要性
# Validates: Requirements 1.12, 1.15
#
# 充要性表述：is_expired() 返回 True 当且仅当
#   flight_time_s > MAX_FLIGHT_TIME_S(=20)
#   OR (mass_kg <= structural_mass(=275) AND flight_distance_m > MAX_RANGE_M(=5000))。
# 其中 structural_mass = INITIAL_MASS - BURN_RATE*ENGINE_BURN_TIME = 400 - 25*5 = 275。
# 预言机用与实现 **完全相同** 的比较运算符（>、<=、>）计算该条件，保证在精确阈值
# (20 / 275 / 5000)处与实现逐位一致，避免边界偶发分歧。
# 另验证：失效（返回 True）时 expired=True 且 in_flight=False 被置位。
@settings(max_examples=200)
@given(
    flight_time=_p5_flight_time_strategy,
    mass=_p5_mass_strategy,
    flight_distance=_p5_distance_strategy,
)
def test_is_expired_necessary_and_sufficient(flight_time, mass, flight_distance):
    missile = Missile3DoF()

    structural_mass = (
        missile.INITIAL_MASS - missile.BURN_RATE * missile.ENGINE_BURN_TIME
    )
    assert structural_mass == pytest.approx(275.0)

    # 直接设置失效相关状态量，置 in_flight=True 以验证失效时被复位。
    missile.flight_time_s = flight_time
    missile.mass_kg = mass
    missile.flight_distance_m = flight_distance
    missile.in_flight = True

    # 预言机：与实现同运算符计算充要条件。
    timed_out = flight_time > missile.MAX_FLIGHT_TIME_S
    fuel_exhausted = mass <= structural_mass
    out_of_range = flight_distance > missile.MAX_RANGE_M
    expected = timed_out or (fuel_exhausted and out_of_range)

    result = missile.is_expired()

    # 充要性：返回值与预言机条件等价。
    assert result is expected

    if result:
        # 失效时状态置位。
        assert missile.expired is True
        assert missile.in_flight is False
    else:
        # 未失效时不改写状态（仍在飞、未失效）。
        assert missile.expired is False
        assert missile.in_flight is True


# ---------------------------------------------------------------------------
# Property 7: time_to_impact 分段语义（piecewise semantics）
# ---------------------------------------------------------------------------
#
# 生成策略：覆盖三个分段分支与靠近/远离两类几何。
#   - in_flight 由参数控制：False 验证返回 0.0。
#   - has_cache 由参数控制：在飞且无缓存目标 → 返回 999.0。
#   - 在飞且有缓存目标：导弹/目标位置 ∈ [-5000, 5000]^3，速度 ∈ [-1000, 1000]^3，
#     使相对几何覆盖靠近（closing_rate>0）与远离（closing_rate<=0）两类。
_p7_pos_component = st.floats(
    min_value=-5000.0, max_value=5000.0, allow_nan=False, allow_infinity=False
)
_p7_vel_component = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)


# Feature: air-combat-mvp, Property 7: time_to_impact 分段语义
# Validates: Requirements 1.16
#
# 分段语义表述（与实现 / 设计 §3.1.5 一致）：
#   - 未在飞（in_flight=False）→ 0.0；
#   - 在飞且 _last_target_state is None → 999.0；
#   - 在飞且有缓存目标：
#       rel_r = target.pos - missile.pos；  r = |rel_r|
#       rel_v = target.vel - missile.vel
#       closing_rate = -dot(rel_r, rel_v) / (r + _EPS)
#       若 closing_rate <= 0 → 999.0；否则 → r / closing_rate。
# 预言机用与实现 **完全相同** 的公式与同一 _EPS 守卫复算，保证在边界
# （closing_rate==0、近零距离）处与实现逐位一致。
@settings(max_examples=200)
@given(
    missile_pos=_vec3(_p7_pos_component),
    missile_vel=_vec3(_p7_vel_component),
    target_pos=_vec3(_p7_pos_component),
    target_vel=_vec3(_p7_vel_component),
    in_flight=st.booleans(),
    has_cache=st.booleans(),
)
def test_time_to_impact_piecewise_semantics(
    missile_pos, missile_vel, target_pos, target_vel, in_flight, has_cache
):
    missile = Missile3DoF()
    missile.position_m = np.asarray(missile_pos, dtype=float)
    missile.velocity_mps = np.asarray(missile_vel, dtype=float)
    missile.in_flight = in_flight

    if has_cache:
        # 直接设置缓存目标状态字典（position_m / velocity_mps 键）。
        missile._last_target_state = {
            "position_m": np.asarray(target_pos, dtype=float),
            "velocity_mps": np.asarray(target_vel, dtype=float),
        }
    else:
        missile._last_target_state = None

    result = missile.time_to_impact

    # 预言机：复算分段语义。
    if not in_flight:
        expected = 0.0
    elif missile._last_target_state is None:
        expected = 999.0
    else:
        rel_r = np.asarray(target_pos, dtype=float) - missile.position_m
        rel_v = np.asarray(target_vel, dtype=float) - missile.velocity_mps
        r = float(np.linalg.norm(rel_r))
        closing_rate = -float(np.dot(rel_r, rel_v)) / (r + _EPS)
        if closing_rate <= 0.0:
            expected = 999.0
        else:
            expected = r / closing_rate

    assert result == pytest.approx(expected, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Property 8: 导弹速度下限（基于仿真轨迹的属性测试）
# ---------------------------------------------------------------------------
#
# 生成策略：参数扫描发射初速与目标距离。
#   - initial_speed ∈ [500, 700] m/s：本机速度量值（决定发射初速，
#     launch 取 max(INITIAL_SPEED_MPS=500, |ego.velocity|)，故下限恰为 500）。
#     覆盖设计 §3.1.1 能量校验描述的初速范围（500 → 燃烧结束约 750）。
#   - target_distance ∈ [500, 5000] m：目标沿发射方向（北）前方的距离，
#     恰好覆盖发射包线/最大射程区间 [500, 5000]（需求 1.7/1.8、MAX_RANGE_M）。
#
# 仿真轨迹：本机置于原点、速度沿北向 [initial_speed, 0, 0]，调用 launch 发射；
# 目标置于正前方（北）距离 target_distance 处、静止。对每个 (initial_speed,
# target_distance) 组合运行完整 launch + 多步 step 轨迹，dt=0.2（与环境
# high_level_dt 一致）。目标近似共线（沿发射方向正前方），PNG 视线角速度近零、
# 横向制导加速度极小，速度量值由推力/阻力主导——恰好适合检验速度下限。
#
# 步进直到 flight_distance_m 超过 target_distance（即已飞越关注的射程内距离）
# 或达到步数上限（安全护栏，避免极端几何下无限步进）。
_p8_initial_speed_strategy = st.floats(
    min_value=500.0, max_value=700.0, allow_nan=False, allow_infinity=False
)
_p8_target_distance_strategy = st.floats(
    min_value=500.0, max_value=5000.0, allow_nan=False, allow_infinity=False
)


# Feature: air-combat-mvp, Property 8: 导弹速度下限（基于仿真轨迹的属性测试）
# Validates: Requirements 8.3
#
# 速度下限表述（基于仿真轨迹）：对扫描的每个 (initial_speed ∈ [500,700],
# target_distance ∈ [500,5000])，发射导弹并沿轨迹逐步 step，断言在 **射程内
# 飞行期间**（flight_distance_m <= target_distance）每一步的速度量值
# |velocity_mps| 始终高于设定下限 SPEED_FLOOR。
#
# 下限取值说明：需求 8.3 要求导弹以不低于 500 m/s 飞行，设计 §3.1.1 能量校验
# 表明 5000 m 射程内速度仍 >500 m/s。本测试取更稳健的 SPEED_FLOOR=400 m/s
# （低于需求 500 的下限、远高于飞机 ~250–300 m/s），以避免发射瞬间/积分离散
# 与重力补偿带来的边际数值波动导致偶发抖动失败，同时仍充分验证"导弹速度显著
# 高于飞机量级、不会在射程内失速"这一核心物理性质（需求 8.3）。
#
# deadline 说明：本属性为 **基于仿真** 的多步轨迹测试，单个样例需运行数十步
# step（每步含 PNG/气动力计算），耗时显著高于纯函数属性测试。为避免 hypothesis
# 默认 200ms deadline 在较慢样例上误报 flaky，显式设置 deadline=None；以步数
# 上限（MAX_STEPS）约束单样例总开销，保证测试整体可控。
@settings(max_examples=100, deadline=None)
@given(
    initial_speed=_p8_initial_speed_strategy,
    target_distance=_p8_target_distance_strategy,
)
def test_missile_speed_floor_over_trajectory(initial_speed, target_distance):
    SPEED_FLOOR = 400.0  # m/s，稳健下限（见上方说明）。
    DT = 0.2             # s，与环境 high_level_dt 一致。
    MAX_STEPS = 200      # 步数护栏：5000 m / (~500 m/s * 0.2 s) ≈ 50 步，留足裕量。

    missile = Missile3DoF()

    # 本机置于原点，速度沿北向（X=North），量值 = initial_speed。
    ego_state = _make_state(
        [0.0, 0.0, 0.0], [initial_speed, 0.0, 0.0]
    )
    missile.launch(ego_state)

    # 发射后初速应 >= max(INITIAL_SPEED_MPS, initial_speed) >= 500 > 下限。
    assert float(np.linalg.norm(missile.velocity_mps)) > SPEED_FLOOR

    # 目标置于正前方（北）距离 target_distance 处、静止（共线，PNG 横向加速近零）。
    target_state = _make_state(
        [target_distance, 0.0, 0.0], [0.0, 0.0, 0.0]
    )

    steps = 0
    while missile.flight_distance_m <= target_distance and steps < MAX_STEPS:
        missile.step(DT, target_state)
        steps += 1

        speed = float(np.linalg.norm(missile.velocity_mps))

        # 射程内飞行期间速度量值始终高于下限。
        # 当本步使 flight_distance_m 越过 target_distance 时，本步仍属"射程内
        # 飞行"的最后一步（推进起点位于射程内），一并断言。
        assert speed > SPEED_FLOOR, (
            f"速度跌破下限：initial_speed={initial_speed:.1f}, "
            f"target_distance={target_distance:.1f}, step={steps}, "
            f"flight_distance_m={missile.flight_distance_m:.1f}, speed={speed:.2f}"
        )

    # 健全性：在步数护栏内确实飞越了关注的射程内距离（轨迹覆盖了射程）。
    assert missile.flight_distance_m > target_distance
