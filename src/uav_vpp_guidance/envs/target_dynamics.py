"""
机动目标运动学模型。

提供可配置的目标机动动力学，支持：
  - constant_velocity: 匀速直线
  - sinusoidal_weaving: 正弦规避（法向正弦过载）
  - bang_bang: 开关法向加速度
  - barrel_roll: 简化滚桶（水平面匀速圆周 + 垂直振荡）

坐标系：NEU (North-East-Up)。

用法：
    dynamics = create_target_dynamics(config)
    acc = dynamics.get_acceleration(t, target_state)  # [a_n, a_e, a_u]
    dynamics.update_state(target_state, dt, t)
"""

import numpy as np
from typing import Dict, Protocol


# ---------------------------------------------------------------------------
# Protocol / base
# ---------------------------------------------------------------------------


class TargetDynamics(Protocol):
    """目标机动动力学协议。"""

    def get_acceleration(self, t: float, state: dict) -> np.ndarray:
        """
        计算目标在当前时刻的加速度向量。

        Args:
            t (float): 当前仿真时间 [s]。
            state (dict): 目标状态，至少包含 velocity_vector_mps。

        Returns:
            np.ndarray: 加速度 [a_north, a_east, a_up]，shape (3,)。
        """
        ...

    def update_state(self, state: dict, dt: float, t: float) -> None:
        """
        根据当前动力学更新目标状态（位置、速度、姿态）。

        Args:
            state (dict): 目标状态字典（就地修改）。
            dt (float): 时间步长 [s]。
            t (float): 当前仿真时间 [s]。
        """
        ...


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_heading(vel: np.ndarray) -> float:
    """从 NEU 速度向量提取水平航向角 [rad]。"""
    return float(np.arctan2(vel[1], vel[0]))


def _get_speed(vel: np.ndarray) -> float:
    """提取速度大小 [m/s]。"""
    return float(np.linalg.norm(vel))


def _normalize_horizontal(v: np.ndarray) -> np.ndarray:
    """归一化水平分量，若水平速度接近零则返回 [1, 0, 0]。"""
    vh = v.copy()
    vh[2] = 0.0
    norm = np.linalg.norm(vh)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vh / norm


# ---------------------------------------------------------------------------
# Constant velocity
# ---------------------------------------------------------------------------


class ConstantVelocityDynamics:
    """匀速直线运动：加速度恒为零。"""

    def get_acceleration(self, t: float, state: dict) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    def update_state(self, state: dict, dt: float, t: float) -> None:
        pos = state.get("position_m", np.zeros(3))
        vel = state.get("velocity_vector_mps", np.zeros(3))
        pos += vel * dt
        state["position_m"] = pos
        state["altitude_m"] = float(pos[2])


# ---------------------------------------------------------------------------
# Sinusoidal weaving
# ---------------------------------------------------------------------------


class SinusoidalWeavingDynamics:
    """
    正弦规避机动。

    目标保持恒定速度大小，法向（水平横向）加速度按正弦变化：
        a_lat(t) = amplitude * sin(omega * t)

    这导致目标在水平面内做蛇形轨迹。

    Args:
        amplitude_g (float): 横向过载峰值 [g]。默认 3.0。
        frequency_rad_s (float): 正弦角频率 [rad/s]。默认 1.0。
        gravity (float): 重力加速度 [m/s^2]。默认 9.81。
    """

    def __init__(self, amplitude_g: float = 3.0, frequency_rad_s: float = 1.0, gravity: float = 9.81):
        self.amplitude = float(amplitude_g) * gravity
        self.omega = float(frequency_rad_s)

    def get_acceleration(self, t: float, state: dict) -> np.ndarray:
        vel = state.get("velocity_vector_mps", np.zeros(3))
        speed = _get_speed(vel)
        if speed < 1e-6:
            return np.zeros(3, dtype=np.float64)

        # 横向法向加速度（垂直于速度方向，在水平面内）
        a_lat = self.amplitude * np.sin(self.omega * t)

        heading = _get_heading(vel)
        # 横向单位向量（垂直于 heading，左转为正）
        lat_n = -np.sin(heading)
        lat_e = np.cos(heading)

        return np.array([lat_n * a_lat, lat_e * a_lat, 0.0], dtype=np.float64)

    def update_state(self, state: dict, dt: float, t: float) -> None:
        vel = state.get("velocity_vector_mps", np.zeros(3))
        pos = state.get("position_m", np.zeros(3))
        speed = _get_speed(vel)
        if speed < 1e-6:
            # 若速度为零，按匀速处理
            pos += vel * dt
            state["position_m"] = pos
            state["altitude_m"] = float(pos[2])
            return

        heading = _get_heading(vel)

        # 法向加速度改变航向，不改变速度大小
        a_lat = self.amplitude * np.sin(self.omega * t)
        heading_rate = a_lat / speed  # d(psi)/dt = a_lat / V
        new_heading = heading + heading_rate * dt

        # 重建速度向量（速度大小不变）
        new_vel = np.array([
            speed * np.cos(new_heading),
            speed * np.sin(new_heading),
            0.0,
        ], dtype=np.float64)

        # 更新位置（使用平均速度，二阶精度）
        avg_vel = (vel + new_vel) * 0.5
        pos += avg_vel * dt

        state["velocity_vector_mps"] = new_vel
        state["position_m"] = pos
        state["altitude_m"] = float(pos[2])
        state["heading_rad"] = float(new_heading)
        state["speed_mps"] = speed


# ---------------------------------------------------------------------------
# Bang-bang acceleration
# ---------------------------------------------------------------------------


class BangBangDynamics:
    """
    开关法向加速度机动。

    目标在 +a_max 和 -a_max 之间周期性切换：
        a_lat(t) = amplitude * sign(sin(2*pi*t / T))

    轨迹为锯齿形（sawtooth-like）规避路径。

    Args:
        max_acceleration_g (float): 最大横向过载 [g]。默认 3.0。
        switch_interval_s (float): 加速度切换周期 [s]。默认 2.0。
        gravity (float): 重力加速度 [m/s^2]。默认 9.81。
    """

    def __init__(self, max_acceleration_g: float = 3.0, switch_interval_s: float = 2.0, gravity: float = 9.81):
        self.amplitude = float(max_acceleration_g) * gravity
        self.switch_interval = float(switch_interval_s)
        if self.switch_interval <= 0:
            raise ValueError("switch_interval_s must be positive")

    def get_acceleration(self, t: float, state: dict) -> np.ndarray:
        vel = state.get("velocity_vector_mps", np.zeros(3))
        speed = _get_speed(vel)
        if speed < 1e-6:
            return np.zeros(3, dtype=np.float64)

        # bang-bang: 周期性符号切换
        a_lat = self.amplitude * np.sign(np.sin(2.0 * np.pi * t / self.switch_interval))

        heading = _get_heading(vel)
        lat_n = -np.sin(heading)
        lat_e = np.cos(heading)

        return np.array([lat_n * a_lat, lat_e * a_lat, 0.0], dtype=np.float64)

    def update_state(self, state: dict, dt: float, t: float) -> None:
        vel = state.get("velocity_vector_mps", np.zeros(3))
        pos = state.get("position_m", np.zeros(3))
        speed = _get_speed(vel)
        if speed < 1e-6:
            pos += vel * dt
            state["position_m"] = pos
            state["altitude_m"] = float(pos[2])
            return

        heading = _get_heading(vel)

        a_lat = self.amplitude * np.sign(np.sin(2.0 * np.pi * t / self.switch_interval))
        heading_rate = a_lat / speed
        new_heading = heading + heading_rate * dt

        new_vel = np.array([
            speed * np.cos(new_heading),
            speed * np.sin(new_heading),
            0.0,
        ], dtype=np.float64)

        avg_vel = (vel + new_vel) * 0.5
        pos += avg_vel * dt

        state["velocity_vector_mps"] = new_vel
        state["position_m"] = pos
        state["altitude_m"] = float(pos[2])
        state["heading_rad"] = float(new_heading)
        state["speed_mps"] = speed


# ---------------------------------------------------------------------------
# Barrel roll
# ---------------------------------------------------------------------------


class BarrelRollDynamics:
    """
    简化滚桶机动。

    目标在水平面内做匀速圆周运动（向心加速度），
    同时叠加一个小的垂直正弦振荡，模拟滚转过程中的高度变化。

    参数 roll_rate_rad_s 同时控制水平圆周角速度和垂直振荡频率。

    Args:
        roll_rate_rad_s (float): 滚转角速度 [rad/s]。默认 0.5。
        vertical_amplitude_m (float): 垂直振荡振幅 [m]。默认 50.0。
    """

    def __init__(self, roll_rate_rad_s: float = 0.5, vertical_amplitude_m: float = 50.0):
        self.roll_rate = float(roll_rate_rad_s)
        self.vert_amp = float(vertical_amplitude_m)
        if self.roll_rate <= 0:
            raise ValueError("roll_rate_rad_s must be positive")

    def get_acceleration(self, t: float, state: dict) -> np.ndarray:
        vel = state.get("velocity_vector_mps", np.zeros(3))
        speed = _get_speed(vel)
        if speed < 1e-6:
            return np.zeros(3, dtype=np.float64)

        # 水平向心加速度（指向圆心）
        a_c = speed * self.roll_rate
        heading = _get_heading(vel)
        # 向心方向 = heading + pi/2（左转，即向心方向在速度左侧）
        centripetal_heading = heading + np.pi / 2.0
        a_n = a_c * np.cos(centripetal_heading)
        a_e = a_c * np.sin(centripetal_heading)

        # 垂直振荡加速度
        a_u = -self.vert_amp * (self.roll_rate ** 2) * np.sin(self.roll_rate * t)

        return np.array([a_n, a_e, a_u], dtype=np.float64)

    def update_state(self, state: dict, dt: float, t: float) -> None:
        vel = state.get("velocity_vector_mps", np.zeros(3))
        pos = state.get("position_m", np.zeros(3))
        speed = _get_speed(vel)
        if speed < 1e-6:
            pos += vel * dt
            state["position_m"] = pos
            state["altitude_m"] = float(pos[2])
            return

        heading = _get_heading(vel)

        # 水平面：匀速圆周运动，航向持续变化
        new_heading = heading + self.roll_rate * dt
        new_vel = np.array([
            speed * np.cos(new_heading),
            speed * np.sin(new_heading),
            0.0,
        ], dtype=np.float64)

        # 垂直方向：正弦运动
        # z(t) = z0 + A*sin(roll_rate*t)
        # 用速度更新更准确
        z_offset = self.vert_amp * np.sin(self.roll_rate * t)
        vz = self.vert_amp * self.roll_rate * np.cos(self.roll_rate * t)
        new_vel[2] = vz

        # 位置更新（平均速度 + 垂直）
        avg_vel = (vel + new_vel) * 0.5
        pos += avg_vel * dt

        state["velocity_vector_mps"] = new_vel
        state["position_m"] = pos
        state["altitude_m"] = float(pos[2])
        state["heading_rad"] = float(new_heading)
        state["speed_mps"] = speed


# ---------------------------------------------------------------------------
# Legacy sinusoidal (backward-compatible)
# ---------------------------------------------------------------------------


class LegacySinusoidalDynamics:
    """
    保留原有 'sinusoidal' 模式的行为，用于向后兼容。

    该模式使用简单的横向速度波动，而非法向加速度积分，
    轨迹与原版 simple_point_mass_env._update_target 完全一致。
    """

    def __init__(self, lateral_rate_scale: float = 0.05, lateral_freq: float = 0.5):
        self._lateral_rate_scale = lateral_rate_scale
        self._lateral_freq = lateral_freq

    def get_acceleration(self, t: float, state: dict) -> np.ndarray:
        # 原逻辑没有显式加速度概念，返回零以保持兼容
        return np.zeros(3, dtype=np.float64)

    def update_state(self, state: dict, dt: float, t: float) -> None:
        pos = state.get("position_m", np.zeros(3))
        vel = state.get("velocity_vector_mps", np.zeros(3))
        base_speed = float(np.linalg.norm(vel))
        heading = float(np.arctan2(vel[1], vel[0]))

        lateral_rate = self._lateral_rate_scale * np.sin(self._lateral_freq * t)
        heading += lateral_rate * dt

        vel = np.array([
            base_speed * np.cos(heading),
            base_speed * np.sin(heading),
            0.0,
        ], dtype=np.float64)
        pos += vel * dt

        state["velocity_vector_mps"] = vel
        state["position_m"] = pos
        state["altitude_m"] = float(pos[2])
        state["heading_rad"] = heading
        state["speed_mps"] = base_speed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_target_dynamics(config: dict) -> TargetDynamics:
    """
    根据配置创建目标动力学实例。

    Args:
        config (dict): 环境配置字典，需包含 target_mode 键。
            - target_mode == "constant_velocity" -> ConstantVelocityDynamics
            - target_mode == "sinusoidal"        -> LegacySinusoidalDynamics
            - target_mode == "sinusoidal_weaving" -> SinusoidalWeavingDynamics
            - target_mode == "bang_bang"         -> BangBangDynamics
            - target_mode == "barrel_roll"       -> BarrelRollDynamics
            其他参数（如 weaving_amplitude_g 等）从 config 中读取。

    Returns:
        TargetDynamics: 对应的目标动力学实例。

    Raises:
        ValueError: 遇到未知的 target_mode。
    """
    mode = config.get("target_mode", "constant_velocity")
    gravity = config.get("gravity_mps2", 9.81)

    if mode == "constant_velocity":
        return ConstantVelocityDynamics()

    if mode == "sinusoidal":
        return LegacySinusoidalDynamics()

    if mode == "sinusoidal_weaving":
        return SinusoidalWeavingDynamics(
            amplitude_g=config.get("weaving_amplitude_g", 3.0),
            frequency_rad_s=config.get("weaving_frequency_rad_s", 1.0),
            gravity=gravity,
        )

    if mode == "bang_bang":
        return BangBangDynamics(
            max_acceleration_g=config.get("bang_bang_max_g", 3.0),
            switch_interval_s=config.get("bang_bang_switch_interval_s", 2.0),
            gravity=gravity,
        )

    if mode == "barrel_roll":
        return BarrelRollDynamics(
            roll_rate_rad_s=config.get("barrel_roll_rate_rad_s", 0.5),
            vertical_amplitude_m=config.get("barrel_roll_vertical_amp_m", 50.0),
        )

    raise ValueError(
        f"Unknown target_mode: {mode}. "
        f"Supported: constant_velocity, sinusoidal, sinusoidal_weaving, bang_bang, barrel_roll"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import math

    dt = 0.2
    gravity = 9.81

    def _make_state(vx: float = 200.0, vy: float = 0.0, vz: float = 0.0) -> dict:
        return {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([vx, vy, vz], dtype=np.float64),
            "speed_mps": math.sqrt(vx**2 + vy**2 + vz**2),
            "heading_rad": math.atan2(vy, vx),
            "altitude_m": 5000.0,
        }

    print("=" * 60)
    print("Target Dynamics Smoke Test")
    print("=" * 60)

    # 1. constant_velocity -------------------------------------------------
    print("\n[1] constant_velocity")
    dyn = create_target_dynamics({"target_mode": "constant_velocity"})
    state = _make_state()
    for step in range(5):
        t = step * dt
        acc = dyn.get_acceleration(t, state)
        dyn.update_state(state, dt, t)
        print(f"  t={t:.1f}s | pos=({state['position_m'][0]:.1f}, {state['position_m'][1]:.1f}) | "
              f"acc=[{acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}]")
    assert np.allclose(acc, 0.0), "constant_velocity acceleration should be zero"
    assert state["position_m"][0] > 100.0, "target should have moved forward"
    print("  -> PASS")

    # 2. sinusoidal_weaving ------------------------------------------------
    print("\n[2] sinusoidal_weaving (A=3g, w=1.0)")
    dyn = create_target_dynamics({
        "target_mode": "sinusoidal_weaving",
        "weaving_amplitude_g": 3.0,
        "weaving_frequency_rad_s": 1.0,
    })
    state = _make_state()
    headings = []
    for step in range(50):
        t = step * dt
        acc = dyn.get_acceleration(t, state)
        dyn.update_state(state, dt, t)
        headings.append(state["heading_rad"])
    heading_range = max(headings) - min(headings)
    print(f"  final pos=({state['position_m'][0]:.1f}, {state['position_m'][1]:.1f}) | "
          f"heading_range={np.rad2deg(heading_range):.1f} deg")
    assert heading_range > 0.1, "weaving should change heading significantly"
    assert abs(state["speed_mps"] - 200.0) < 1.0, "speed should remain ~200 m/s"
    print("  -> PASS")

    # 3. bang_bang ---------------------------------------------------------
    print("\n[3] bang_bang (max_g=3, T=2.0s)")
    dyn = create_target_dynamics({
        "target_mode": "bang_bang",
        "bang_bang_max_g": 3.0,
        "bang_bang_switch_interval_s": 2.0,
    })
    state = _make_state()
    acc_signs = []
    for step in range(50):
        t = step * dt
        acc = dyn.get_acceleration(t, state)
        dyn.update_state(state, dt, t)
        # 记录横向加速度符号
        vel = state["velocity_vector_mps"]
        heading = math.atan2(vel[1], vel[0])
        lat_n, lat_e = -math.sin(heading), math.cos(heading)
        a_lat = acc[0] * lat_n + acc[1] * lat_e
        acc_signs.append(np.sign(a_lat))
    # 应该观察到符号切换
    unique_signs = set(acc_signs)
    print(f"  final pos=({state['position_m'][0]:.1f}, {state['position_m'][1]:.1f}) | "
          f"acc signs={unique_signs}")
    assert len(unique_signs) > 1, "bang-bang should show sign changes"
    print("  -> PASS")

    # 4. barrel_roll -------------------------------------------------------
    print("\n[4] barrel_roll (roll_rate=0.5 rad/s)")
    dyn = create_target_dynamics({
        "target_mode": "barrel_roll",
        "barrel_roll_rate_rad_s": 0.5,
    })
    state = _make_state()
    altitudes = []
    headings = []
    for step in range(100):
        t = step * dt
        acc = dyn.get_acceleration(t, state)
        dyn.update_state(state, dt, t)
        altitudes.append(state["altitude_m"])
        headings.append(state["heading_rad"])
    heading_range = max(headings) - min(headings)
    alt_range = max(altitudes) - min(altitudes)
    print(f"  final pos=({state['position_m'][0]:.1f}, {state['position_m'][1]:.1f}) | "
          f"heading_range={np.rad2deg(heading_range):.1f} deg | alt_range={alt_range:.1f} m")
    assert heading_range > 0.5, "barrel roll should change heading significantly"
    assert alt_range > 10.0, "barrel roll should show altitude variation"
    print("  -> PASS")

    # 5. backward-compatible sinusoidal ------------------------------------
    print("\n[5] legacy sinusoidal (backward-compatible)")
    dyn = create_target_dynamics({"target_mode": "sinusoidal"})
    state = _make_state()
    for step in range(10):
        t = step * dt
        dyn.update_state(state, dt, t)
    print(f"  final pos=({state['position_m'][0]:.1f}, {state['position_m'][1]:.1f})")
    assert state["position_m"][0] > 0.0, "legacy sinusoidal should move forward"
    print("  -> PASS")

    # 6. unknown mode ------------------------------------------------------
    print("\n[6] unknown mode -> ValueError")
    try:
        create_target_dynamics({"target_mode": "hyperspace_jump"})
        raise AssertionError("should have raised ValueError")
    except ValueError as exc:
        print(f"  -> PASS ({exc})")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
