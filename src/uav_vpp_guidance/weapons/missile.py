"""
Missile3DoF：3 自由度（3-DoF）空空导弹模型。

坐标系约定（与 JSBSim 一致）：
    X = North（北，正向北）
    Y = Up（上，正向上）
    Z = East（东，正向东）

单位约定：
    - 长度 / 位置：米（m）
    - 速度：米/秒（m/s）
    - 角度：弧度（rad）

说明：
    本模块实现简化 3-DoF 空空导弹 ``Missile3DoF`` 的常量与构造函数。
    发射判定（can_launch/launch）、3-DoF 动力学步进（step）、命中/失效判定
    （check_hit/is_expired）与 ``time_to_impact`` 将在后续任务中实现。
    该模块不依赖 JSBSim，可在纯 Python 环境中导入。
"""

from __future__ import annotations

import numpy as np

# 模块级重力加速度常量（m/s^2）
GRAVITY = 9.80665

# 速度分解/反解的零速度兜底阈值（m/s）
_SPEED_EPSILON = 1e-8

# 防奇异小量：所有除法分母加此值以避免零距离/零速度奇异。
_EPS = 1e-8


def _extract_position(state: dict) -> np.ndarray:
    """从状态字典中稳健提取位置向量（3 元）。

    键优先级（与 ``observation.py`` 一致，保证与环境互通）：
        1. ``position_m``   （本特性/设计 §4.1 首选键）
        2. ``position_neu``
        3. ``position``

    所有候选键均假定为同一坐标系下的 3 元向量；``can_launch`` 仅依赖
    ``target.pos - ego.pos`` 的模长与其相对速度的夹角，二者对坐标轴的具体
    排列（NEU vs [north, up, east]）不变，只要 ego 与 target 使用一致排列。

    Args:
        state: 状态字典。

    Returns:
        np.ndarray: 形状 (3,) 的位置向量。

    Raises:
        ValueError: 当未找到任何受支持的位置键时。
    """
    for key in ("position_m", "position_neu", "position"):
        pos = state.get(key)
        if pos is not None:
            return np.asarray(pos, dtype=float)
    raise ValueError(
        "状态字典缺少位置字段（position_m / position_neu / position）。"
    )


def _extract_velocity(state: dict) -> np.ndarray:
    """从状态字典中稳健提取速度向量（3 元）。

    键优先级：
        1. ``velocity_mps``        （设计 §4.1 首选键；仅当其为 3 元向量时采用，
                                     场景初始化字典中 ``velocity_mps`` 为标量速率，
                                     此时跳过）
        2. ``velocity_vector_mps`` （环境 NEU 速度向量）
        3. ``velocity_ned``        （NED，转换为 NEU：``[vn, ve, -vd]``）
        4. ``velocity``

    Args:
        state: 状态字典。

    Returns:
        np.ndarray: 形状 (3,) 的速度向量。

    Raises:
        ValueError: 当未找到任何受支持的速度键时。
    """
    vel = state.get("velocity_mps")
    if vel is not None:
        arr = np.asarray(vel, dtype=float)
        if arr.shape == (3,):
            return arr

    vel = state.get("velocity_vector_mps")
    if vel is not None:
        return np.asarray(vel, dtype=float)

    vel_ned = state.get("velocity_ned")
    if vel_ned is not None:
        v = np.asarray(vel_ned, dtype=float)
        return np.array([v[0], v[1], -v[2]], dtype=float)

    vel = state.get("velocity")
    if vel is not None:
        return np.asarray(vel, dtype=float)

    raise ValueError(
        "状态字典缺少速度字段（velocity_mps / velocity_vector_mps / "
        "velocity_ned / velocity）。"
    )


def decompose_velocity(speed: float, gamma: float, psi: float) -> np.ndarray:
    """按 JSBSim 坐标系将 (speed, gamma, psi) 分解为速度向量。

    坐标系约定（X=North, Y=Up, Z=East），分解公式（设计 §3.1.4）::

        v_north = speed * cos(gamma) * cos(psi)
        v_up    = speed * sin(gamma)
        v_east  = speed * cos(gamma) * sin(psi)
        vel     = [v_north, v_up, v_east]

    其中第 2 分量为 Up（正向上），**不沿用参考项目 vz_ms 的负号写法**
    （需求 8.5）。该分解与 :func:`velocity_to_spherical` 互为可逆。

    Args:
        speed: 速度量值（m/s），非负。
        gamma: 航迹倾角（rad），相对水平面，向上为正，约定域 [-π/2, π/2]。
        psi: 航向角（rad），绕 Up 轴，从 North 起、向 East 为正。

    Returns:
        np.ndarray: 形状 (3,) 的速度向量 ``[v_north, v_up, v_east]``（m/s）。
    """
    cos_gamma = np.cos(gamma)
    v_north = speed * cos_gamma * np.cos(psi)
    v_up = speed * np.sin(gamma)
    v_east = speed * cos_gamma * np.sin(psi)
    return np.array([v_north, v_up, v_east], dtype=float)


def velocity_to_spherical(vel) -> tuple[float, float, float]:
    """将速度向量反解为 (speed, gamma, psi)（设计 §3.1.4，需求 8.5/8.6）。

    反解公式::

        speed = |vel|
        gamma = arcsin(v_up / speed)
        psi   = arctan2(v_east, v_north)

    与 :func:`decompose_velocity` 互为可逆。当 ``speed`` 近似为 0（小于
    ``_SPEED_EPSILON``）时，方向不可定义，兜底返回 ``gamma=0, psi=0``，
    以避免除零与 ``arcsin`` 定义域错误。

    Args:
        vel: 速度向量 ``[v_north, v_up, v_east]``（m/s），可为序列或 ndarray。

    Returns:
        tuple[float, float, float]: ``(speed, gamma, psi)``，单位 (m/s, rad, rad)。
        ``gamma ∈ [-π/2, π/2]``，``psi ∈ (-π, π]``。
    """
    vel = np.asarray(vel, dtype=float)
    v_north, v_up, v_east = vel[0], vel[1], vel[2]
    speed = float(np.linalg.norm(vel))

    # 零速度兜底：方向不可定义，返回中性默认值。
    if speed < _SPEED_EPSILON:
        return 0.0, 0.0, 0.0

    # clip 防止浮点误差导致 arcsin 超出 [-1, 1] 定义域。
    gamma = float(np.arcsin(np.clip(v_up / speed, -1.0, 1.0)))
    psi = float(np.arctan2(v_east, v_north))
    return speed, gamma, psi


class Missile3DoF:
    """3 自由度空空导弹模型。

    构造后导弹处于"未发射/未在飞"状态，需调用 ``launch()`` 进入在飞。
    所有常量可通过 ``config`` 字典覆盖；显式传入的 ``nav_constant`` 与
    ``kill_radius_m`` 优先级高于 ``config``。
    """

    # 气动与性能常量（来自参考项目物理参数，需求 1.2 / 8.6）
    DRAG_COEFF = 0.6          # 阻力系数
    LIFT_COEFF = 0.15         # 升力系数
    REF_AREA = 0.4            # 参考面积 m^2
    MAX_G = 30.0              # 过载上限（需求 1.2 / 8.4）
    ENGINE_BURN_TIME = 5.0    # 发动机燃烧时间 s（需求 1.2）

    # 推进与质量（需求 1.14）
    THRUST = 20000.0          # 恒定推力 N
    INITIAL_MASS = 400.0      # 初始质量 kg
    BURN_RATE = 25.0          # 燃烧速度 kg/s

    # 包线与失效（需求 1.7/1.8/1.12/1.15）
    LAUNCH_RANGE_MIN = 500.0  # 发射最小距离 m
    LAUNCH_RANGE_MAX = 5000.0  # 发射最大距离 m
    LAUNCH_ATA_MAX_DEG = 30.0  # 发射 |ATA| 上限（不含）
    MAX_FLIGHT_TIME_S = 20.0  # 最大飞行时间 s
    MAX_RANGE_M = 5000.0      # 最大射程 m

    # 制导与命中
    NAV_CONSTANT = 3.0        # PNG 比例系数 K（默认）
    KILL_RADIUS_M = 30.0      # 杀伤半径 m
    AIR_DENSITY = 1.225       # 空气密度 kg/m^3（用于气动力）
    INITIAL_SPEED_MPS = 500.0  # 发射初速（继承本机速度方向，量值≥500）

    # 可由 config 字典覆盖的常量名集合
    _OVERRIDABLE_CONSTANTS = (
        "DRAG_COEFF",
        "LIFT_COEFF",
        "REF_AREA",
        "MAX_G",
        "ENGINE_BURN_TIME",
        "THRUST",
        "INITIAL_MASS",
        "BURN_RATE",
        "LAUNCH_RANGE_MIN",
        "LAUNCH_RANGE_MAX",
        "LAUNCH_ATA_MAX_DEG",
        "MAX_FLIGHT_TIME_S",
        "MAX_RANGE_M",
        "NAV_CONSTANT",
        "KILL_RADIUS_M",
        "AIR_DENSITY",
        "INITIAL_SPEED_MPS",
    )

    def __init__(
        self,
        config: dict | None = None,
        nav_constant: float | None = None,
        kill_radius_m: float | None = None,
    ):
        """初始化导弹。

        Args:
            config: 可选覆盖常量的配置字典（如 ``missile_config.ego``）。
                支持以小写键名覆盖任意类常量（如 ``nav_constant``、
                ``kill_radius_m``、``drag_coeff`` 等）。
            nav_constant: PNG 比例系数 K（默认 ``NAV_CONSTANT``）。显式传入时
                优先级高于 ``config``。
            kill_radius_m: 杀伤半径覆盖（默认 ``KILL_RADIUS_M``）。显式传入时
                优先级高于 ``config``。

        构造后导弹处于"未发射/未在飞"状态，需调用 ``launch()`` 进入在飞。
        """
        config = config or {}

        # 1) 以类常量为默认值，初始化全部实例常量；
        #    若 config 提供对应的小写键则覆盖。
        for name in self._OVERRIDABLE_CONSTANTS:
            value = config.get(name.lower(), getattr(type(self), name))
            setattr(self, name, value)

        # 2) 显式参数优先级最高，覆盖 config / 类常量。
        if nav_constant is not None:
            self.NAV_CONSTANT = nav_constant
        if kill_radius_m is not None:
            self.KILL_RADIUS_M = kill_radius_m

        # 3) 初始化全部状态量为"未发射/未在飞"。
        self.position_m = np.zeros(3, dtype=float)   # [north, up, east]
        self.velocity_mps = np.zeros(3, dtype=float)  # [vn, vu, ve]
        self.mass_kg = float(self.INITIAL_MASS)
        self.flight_time_s = 0.0
        self.flight_distance_m = 0.0
        self.in_flight = False
        self.hit = False
        self.expired = False
        self._last_target_state = None

    # ------------------------------------------------------------------
    # 速度分解 / 反解（静态方法别名，便于通过类直接调用；需求 8.5/8.6）
    # 实际逻辑见模块级 decompose_velocity / velocity_to_spherical。
    # ------------------------------------------------------------------
    decompose_velocity = staticmethod(decompose_velocity)
    velocity_to_spherical = staticmethod(velocity_to_spherical)

    # ------------------------------------------------------------------
    # 公开方法：发射判定（需求 1.7 / 1.8 / 1.9）
    # ------------------------------------------------------------------
    def can_launch(self, ego_state: dict, target_state: dict) -> bool:
        """仅检查几何发射条件（需求 1.7 / 1.8 / 1.9）。

        返回 ``True`` 当且仅当同时满足：

            ``LAUNCH_RANGE_MIN <= range <= LAUNCH_RANGE_MAX``  且
            ``|ATA| < LAUNCH_ATA_MAX_DEG``（度）

        其中::

            range = |target.pos - ego.pos|
            LOS   = (target.pos - ego.pos) / range          # 视线单位向量
            ATA   = angle(ego.velocity, LOS)
                  = arccos( clip( dot(v_hat, los_hat), -1, 1 ) )

        本方法 **仅检查几何条件**（距离与 ``|ATA|``），不检查雷达视场或雷达
        锁定——雷达锁定由环境层在调用 ``can_launch`` 之前独立检查并以 AND
        组合（需求 1.9）。本方法为纯函数，不修改任何实例状态。

        状态字典键约定（与环境 ``_get_current_states`` / ``observation.py``
        互通）：位置取 ``position_m``（首选）/ ``position_neu`` / ``position``；
        速度取 ``velocity_mps``（3 元向量时）/ ``velocity_vector_mps`` /
        ``velocity_ned`` / ``velocity``。range 与 ATA 对坐标轴排列不变，
        只要 ego 与 target 使用一致的坐标约定即可。

        所有除法分母均加 ``1e-8`` 以避免零距离 / 零速度奇异。

        Args:
            ego_state: 本机状态字典（含位置与速度）。
            target_state: 目标状态字典（含位置）。

        Returns:
            bool: 是否允许发射（仅几何判定）。
        """
        ego_pos = _extract_position(ego_state)
        target_pos = _extract_position(target_state)
        ego_vel = _extract_velocity(ego_state)

        # 视线向量与距离。
        los = target_pos - ego_pos
        range_m = float(np.linalg.norm(los))

        # 距离包线判定（需求 1.7 / 1.8）。
        if not (self.LAUNCH_RANGE_MIN <= range_m <= self.LAUNCH_RANGE_MAX):
            return False

        # ATA = 本机速度方向与视线方向的夹角。
        # 所有除法分母加 _EPS 防止零距离 / 零速度奇异。
        los_hat = los / (range_m + _EPS)
        ego_speed = float(np.linalg.norm(ego_vel))
        v_hat = ego_vel / (ego_speed + _EPS)

        cos_ata = float(np.clip(np.dot(v_hat, los_hat), -1.0, 1.0))
        ata_deg = float(np.degrees(np.arccos(cos_ata)))

        # |ATA| 包线判定（需求 1.8：>= 上限不允许，故此处为严格小于）。
        return ata_deg < self.LAUNCH_ATA_MAX_DEG

    # ------------------------------------------------------------------
    # 公开方法：发射初始化（需求 8.3；设计 §3.1.3）
    # ------------------------------------------------------------------
    def launch(self, ego_state: dict) -> None:
        """从本机当前状态发射导弹（需求 8.3；设计 §3.1.3）。

        发射后导弹状态初始化为：

            - ``position_m``：取本机位置的 **副本**（避免后续步进修改导弹位置
              时反向污染本机状态）。
            - ``velocity_mps``：方向取本机速度单位向量，量值取
              ``max(INITIAL_SPEED_MPS, |ego.velocity|)``（保证发射初速
              ≥ ``INITIAL_SPEED_MPS``，需求 8.3）。当本机速度量值近似为 0
              （< ``1e-8``）时，方向退化为北向单位向量 ``[1, 0, 0]``。
            - ``mass_kg = INITIAL_MASS``、``flight_time_s = 0``、
              ``in_flight = True``。

        同时复位 ``flight_distance_m``、``hit``、``expired`` 与
        ``_last_target_state``，确保每次发射均为干净的初始状态。

        状态字典键约定与 ``can_launch`` 一致（见 :func:`_extract_position` /
        :func:`_extract_velocity`）。

        Args:
            ego_state: 本机状态字典（含位置与速度）。

        Returns:
            None
        """
        ego_pos = _extract_position(ego_state)
        ego_vel = _extract_velocity(ego_state)

        # 位置取本机位置副本，避免与本机状态共享底层缓冲区。
        self.position_m = np.array(ego_pos, dtype=float, copy=True)

        # 速度方向取本机速度单位向量；速度≈0 时退化为北向单位向量。
        ego_speed = float(np.linalg.norm(ego_vel))
        if ego_speed < _SPEED_EPSILON:
            v_hat = np.array([1.0, 0.0, 0.0], dtype=float)  # 北向
        else:
            v_hat = ego_vel / ego_speed

        # 量值取 max(INITIAL_SPEED_MPS, |ego.velocity|)，保证初速下限。
        speed = max(self.INITIAL_SPEED_MPS, ego_speed)
        self.velocity_mps = v_hat * speed

        # 复位质量 / 计时 / 计程与状态标志，进入在飞。
        self.mass_kg = float(self.INITIAL_MASS)
        self.flight_time_s = 0.0
        self.flight_distance_m = 0.0
        self.in_flight = True
        self.hit = False
        self.expired = False
        self._last_target_state = None

    # ------------------------------------------------------------------
    # PNG 制导加速度计算（需求 1.3 / 1.4；设计 §3.1.4）
    # ------------------------------------------------------------------
    def _compute_png_acceleration(self, target_state: dict) -> np.ndarray:
        """计算 PNG（标准真比例导引）制导加速度并施加过载限幅（需求 1.3 / 1.4）。

        制导律（设计 §3.1.4，标准真比例导引 / true PN 向量形式）::

            rel_r = target.pos - missile.pos          # 视线向量
            rel_v = target.vel - missile.vel          # 相对速度
            omega = cross(rel_r, rel_v) / dot(rel_r, rel_r)   # 视线角速度（rad/s）
            a_png = K * cross(omega, V_missile)       # V_missile = self.velocity_mps

        即标准真比例导引：制导加速度正比于（导航系数 K × 视线角速度 omega ×
        导弹速度向量 V_missile）的叉积，等价于 ``a = N · Vc · Ω`` 施加于垂直
        速度方向。**此式替换了先前 ``a_png = K*(rel_r·rel_v)/|rel_r|^4 *
        cross(rel_r, omega)`` 的 |rel_r|^4 形式**（经授权的 spec 变更）：旧式在
        量纲上于公里级交战距离被严重抑制（量级 ~K·|rel_v|²/|rel_r|³，3 km 处仅
        ~2.8e-5 m/s²），导弹近乎弹道飞行、命中率为 0；标准真 PN（闭合速度 ×
        视线角速度）在公里级距离上提供有效制导。

        过载限幅（需求 1.4 / 8.4）：将 ``a_png`` 的范数限制在
        ``MAX_G * GRAVITY ≈ 30 * 9.80665 ≈ 294.2 m/s²`` 以内；若超过则按比例
        缩放至上限::

            a_max = MAX_G * GRAVITY
            if |a_png| > a_max:
                a_png = a_png / |a_png| * a_max

        **数值防奇异方案（一致性约定）**：所有除法分母均加 ``_EPS=1e-8``。
        即 ``omega`` 的分母用 ``dot(rel_r, rel_r) + _EPS``，限幅时的分母用
        ``|a_png| + _EPS``。该约定避免零距离（``rel_r ≈ 0``）时的除零奇异。

        本方法为 **纯函数**：不修改任何实例状态（``position_m`` /
        ``velocity_mps`` 等），便于独立的单元 / 属性测试调用（Property 1）。
        导弹位置 / 速度取自 ``self.position_m`` / ``self.velocity_mps``；目标
        位置 / 速度通过 :func:`_extract_position` / :func:`_extract_velocity`
        从 ``target_state`` 提取。

        Args:
            target_state: 目标状态字典（含位置与速度）。

        Returns:
            np.ndarray: 形状 (3,) 的 PNG 制导加速度（m/s²），范数不超过
            ``MAX_G * GRAVITY``（允许极小数值容差）。
        """
        target_pos = _extract_position(target_state)
        target_vel = _extract_velocity(target_state)

        # 相对几何：视线向量与相对速度。
        rel_r = target_pos - self.position_m
        rel_v = target_vel - self.velocity_mps

        rr = float(np.dot(rel_r, rel_r))      # = r^2

        # 视线角速度 omega = cross(rel_r, rel_v) / dot(rel_r, rel_r)。
        # 分母加 _EPS 防零距离奇异。
        omega = np.cross(rel_r, rel_v) / (rr + _EPS)

        # 标准真比例导引（true PN）：a_png = K * cross(omega, V_missile)。
        # 等价于 a = N·Vc·Ω 施加于垂直速度方向；以闭合速度 × 视线角速度构成
        # 有效制导加速度。替换先前 |rel_r|^4 形式（其在公里级交战距离量纲被严重
        # 抑制，导致导弹近乎弹道飞行、命中率为 0），见 docstring 与设计 §3.1.4。
        a_png = self.NAV_CONSTANT * np.cross(omega, self.velocity_mps)

        # 过载限幅：范数超过 MAX_G * GRAVITY 时按比例缩放至上限。
        a_max = self.MAX_G * GRAVITY
        a_norm = float(np.linalg.norm(a_png))
        if a_norm > a_max:
            a_png = a_png / (a_norm + _EPS) * a_max

        return np.asarray(a_png, dtype=float)

    # ------------------------------------------------------------------
    # 3-DoF 动力学步进（需求 1.5 / 1.6 / 1.10；设计 §3.1.4）
    # ------------------------------------------------------------------
    def step(self, dt: float, target_state: dict) -> None:
        """推进在飞导弹一步（半隐式欧拉积分，需求 1.5 / 1.6 / 1.10）。

        **无返回值（返回 None）**：所有状态变更通过实例属性
        （``position_m`` / ``velocity_mps`` / ``mass_kg`` / ``flight_time_s`` /
        ``flight_distance_m``）读取，而非返回值。未在飞（``in_flight=False``）
        时在流程最前直接 ``return None``，不积分、不缓存、不返回任何值。

        合成加速度（设计 §3.1.4）::

            a_total = a_png + a_thrust + a_drag + a_lift + a_gravity

        其中各分量：

        - **PNG 制导**：``a_png`` 由 :meth:`_compute_png_acceleration` 计算并已
          限幅到 ``MAX_G * GRAVITY``。
        - **推力**（需求 1.5 / 1.6）：燃烧段（``flight_time_s <= ENGINE_BURN_TIME``）
          施加 ``a_thrust = (THRUST / mass_kg) * v_hat``，沿速度方向；燃烧结束后
          推力为 0。
        - **阻力**（经验修正公式，设计 §3.1.4）：动压
          ``q = 0.5 * AIR_DENSITY * |v|^2``，
          ``F_drag = 0.05 * DRAG_COEFF / 3 * q * REF_AREA``，
          ``a_drag = -(F_drag / mass_kg) * v_hat``（沿速度反方向）。
        - **升力**：``F_lift = q * LIFT_COEFF * REF_AREA``，方向取"竖直—速度"
          平面内、与速度正交、含正向上分量的单位向量
          （对 Up 向量相对 ``v_hat`` 作 Gram-Schmidt 正交化后归一化）；速度近似
          竖直（正交分量退化）时升力取 0。
        - **重力**：``a_gravity = [0, -GRAVITY, 0]``（仅作用于 Up 分量）。

        **积分方案（半隐式欧拉 / Symplectic Euler，与 Property 3 / 6 对齐，必须严格遵守此顺序）**::

            v_hat 由步进前的 velocity_mps 计算（速度≈0 时退化为北向 [1,0,0]）
            velocity_mps += a_total * dt          # 先更新速度
            position_m   += velocity_mps * dt     # 用 **更新后** 的速度更新位置
            flight_distance_m += |velocity_mps * dt|
            flight_time_s += dt
            若燃烧段: mass_kg = max(mass_kg - BURN_RATE * dt, burnout_mass)

        即位置位移基于 **更新后** 的速度（半隐式 / symplectic Euler 风格；
        注意这并非显式欧拉——显式欧拉用步进 **前** 的速度更新位置），
        因此 Property 3 的运动学一致性断言应针对步进 **后** 的 ``velocity_mps``：
        ``Δposition == velocity_mps_post * dt``（积分误差内），且 ``flight_time_s``
        恰增加 ``dt``。

        **燃烧段质量单调性（Property 6 对齐）**：是否处于燃烧段由步进 **前** 的
        ``flight_time_s <= ENGINE_BURN_TIME`` 一致判定，推力施加与质量递减使用
        同一判定。质量按 ``BURN_RATE * dt`` 递减并钳到燃尽质量下限
        ``burnout_mass = INITIAL_MASS - BURN_RATE * ENGINE_BURN_TIME``
        （= 400 - 25*5 = 275 kg；即燃料耗尽后的总质量，非壳体结构干重），
        保证燃烧段严格单调递减、燃烧结束后恒定。

        所有除法分母均加 ``_EPS`` 或显式判零，避免零速度奇异。

        Args:
            dt: 步长（s），应为正。
            target_state: 目标状态字典（含位置与速度），缓存供 ``time_to_impact`` 用。

        Returns:
            None
        """
        # 未在飞：流程最前直接返回，不积分、不缓存。
        if not self.in_flight:
            return None

        # --- 速度方向 v_hat（基于步进前速度；零速度退化为北向）。 ---
        speed = float(np.linalg.norm(self.velocity_mps))
        if speed < _SPEED_EPSILON:
            v_hat = np.array([1.0, 0.0, 0.0], dtype=float)  # 北向兜底
        else:
            v_hat = self.velocity_mps / speed

        # --- 1) PNG 制导加速度（已在内部限幅到 MAX_G * GRAVITY）。 ---
        a_png = self._compute_png_acceleration(target_state)

        # 是否处于燃烧段：由步进前 flight_time_s 一致判定（推力与质量递减共用）。
        is_burning = self.flight_time_s <= self.ENGINE_BURN_TIME

        # --- 2) 推力加速度（仅燃烧段，沿速度方向）。 ---
        if is_burning:
            a_thrust = (self.THRUST / (self.mass_kg + _EPS)) * v_hat
        else:
            a_thrust = np.zeros(3, dtype=float)

        # --- 动压 q（阻力与升力共用）。 ---
        q = 0.5 * self.AIR_DENSITY * speed * speed

        # --- 3) 阻力加速度（经验修正公式，沿速度反方向）。 ---
        f_drag = 0.05 * self.DRAG_COEFF / 3.0 * q * self.REF_AREA
        a_drag = -(f_drag / (self.mass_kg + _EPS)) * v_hat

        # --- 4) 升力加速度（竖直—速度平面内与速度正交的单位向量）。 ---
        f_lift = q * self.LIFT_COEFF * self.REF_AREA
        # 对 Up=[0,1,0] 相对 v_hat 作 Gram-Schmidt 正交化，取正向上分量方向。
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        lift_dir = up - float(np.dot(up, v_hat)) * v_hat
        lift_norm = float(np.linalg.norm(lift_dir))
        if lift_norm < _SPEED_EPSILON:
            # 速度近似竖直，正交方向退化，升力取 0。
            a_lift = np.zeros(3, dtype=float)
        else:
            lift_hat = lift_dir / lift_norm
            a_lift = (f_lift / (self.mass_kg + _EPS)) * lift_hat

        # --- 5) 重力加速度（仅 Up 分量）。 ---
        a_gravity = np.array([0.0, -GRAVITY, 0.0], dtype=float)

        # --- 合成总加速度。 ---
        a_total = a_png + a_thrust + a_drag + a_lift + a_gravity

        # --- 半隐式欧拉积分（Symplectic Euler：先更新速度，再用更新后的速度更新位置）。 ---
        self.velocity_mps = self.velocity_mps + a_total * dt
        displacement = self.velocity_mps * dt
        self.position_m = self.position_m + displacement
        self.flight_distance_m += float(np.linalg.norm(displacement))
        self.flight_time_s += dt

        # --- 燃烧段质量递减（钳到燃尽质量下限 burnout_mass）。 ---
        if is_burning:
            burnout_mass = (
                self.INITIAL_MASS - self.BURN_RATE * self.ENGINE_BURN_TIME
            )
            self.mass_kg = max(self.mass_kg - self.BURN_RATE * dt, burnout_mass)

        # --- 缓存目标状态供 time_to_impact 使用。 ---
        self._last_target_state = target_state

        return None

    # ------------------------------------------------------------------
    # 命中判定（需求 1.11；设计 §3.1.3）
    # ------------------------------------------------------------------
    def check_hit(self, target_state: dict) -> bool:
        """命中判定（需求 1.11）。

        在飞（``in_flight=True``）且导弹与目标的距离
        ``|target.pos - missile.pos| <= KILL_RADIUS_M`` 时，置 ``hit=True``、
        ``in_flight=False`` 并返回 ``True``；否则返回 ``False`` 且不修改任何状态。

        判定为闭区间（``<=``，与 Property 4 的距离阈值充要性一致）。未在飞时
        直接返回 ``False``，不读取 ``target_state``、不改写状态（保持幂等）。

        目标位置通过 :func:`_extract_position` 从 ``target_state`` 提取，坐标约定
        与 ``can_launch`` / ``step`` 一致。

        Args:
            target_state: 目标状态字典（含位置）。

        Returns:
            bool: 是否命中。
        """
        # 未在飞：直接返回 False，不改写状态。
        if not self.in_flight:
            return False

        target_pos = _extract_position(target_state)
        distance = float(np.linalg.norm(target_pos - self.position_m))

        if distance <= self.KILL_RADIUS_M:
            self.hit = True
            self.in_flight = False
            return True
        return False

    # ------------------------------------------------------------------
    # 失效判定（需求 1.12 / 1.15；设计 §3.1.3）
    # ------------------------------------------------------------------
    def is_expired(self) -> bool:
        """失效判定（需求 1.12 / 1.15）。

        满足以下任一条件即判定失效：

        - 飞行时间超时：``flight_time_s > MAX_FLIGHT_TIME_S``（20.0 s），或
        - 燃料耗尽且超出射程：``mass_kg <= burnout_mass`` 且
          ``flight_distance_m > MAX_RANGE_M``，其中
          ``burnout_mass = INITIAL_MASS - BURN_RATE * ENGINE_BURN_TIME``
          （= 400 - 25*5 = 275 kg；燃料耗尽后的总质量，非壳体结构干重）。

        失效时置 ``expired=True``、``in_flight=False`` 并返回 ``True``；否则返回
        ``False`` 且不修改任何状态（保持幂等，与 Property 5 的充要性一致）。

        本方法不依赖 ``in_flight`` 作为前置过滤——即便已不在飞也按状态量重新评估
        失效条件；仅在条件成立时才置位，故重复调用幂等。

        Returns:
            bool: 是否失效。
        """
        burnout_mass = (
            self.INITIAL_MASS - self.BURN_RATE * self.ENGINE_BURN_TIME
        )

        timed_out = self.flight_time_s > self.MAX_FLIGHT_TIME_S
        fuel_exhausted = self.mass_kg <= burnout_mass
        out_of_range = self.flight_distance_m > self.MAX_RANGE_M

        if timed_out or (fuel_exhausted and out_of_range):
            self.expired = True
            self.in_flight = False
            return True
        return False

    # ------------------------------------------------------------------
    # 剩余飞行时间（需求 1.16；设计 §3.1.3 / §3.1.5）
    # ------------------------------------------------------------------
    @property
    def time_to_impact(self) -> float:
        """剩余飞行时间估计（需求 1.16）。

        分段语义（与 Property 7 一致）：

        - 未在飞（``in_flight=False``）：返回 ``0.0``。
        - 在飞但缺少缓存的目标状态（``_last_target_state is None``）：相对速度
          不可知，视为相对速度为 0 → ``closing_rate <= 0`` → 返回 ``999.0``。
        - 在飞且有缓存目标状态：

          .. code-block:: text

              rel_r        = target.pos - missile.pos
              r            = |rel_r|
              rel_v        = target.vel - missile.vel
              closing_rate = -dot(rel_r, rel_v) / r
              若 closing_rate <= 0 返回 999.0；否则返回 r / closing_rate

          其中 ``r`` 的除法分母加 ``_EPS`` 防零距离奇异。目标位置 / 速度取自最近
          一次 ``step`` 缓存的 ``_last_target_state``。

        Returns:
            float: 剩余飞行时间（s）；``0.0`` 未在飞，``999.0`` 无法命中
            （远离 / 缺速度），否则为 ``r / closing_rate``。
        """
        # 未在飞：返回 0.0。
        if not self.in_flight:
            return 0.0

        # 缺少缓存目标状态：相对速度视为 0 → closing_rate <= 0 → 999.0。
        if self._last_target_state is None:
            return 999.0

        target_pos = _extract_position(self._last_target_state)
        target_vel = _extract_velocity(self._last_target_state)

        rel_r = target_pos - self.position_m
        rel_v = target_vel - self.velocity_mps
        r = float(np.linalg.norm(rel_r))

        # closing_rate = -dot(rel_r, rel_v) / r；分母加 _EPS 防零距离奇异。
        closing_rate = -float(np.dot(rel_r, rel_v)) / (r + _EPS)

        if closing_rate <= 0.0:
            return 999.0
        return r / closing_rate
