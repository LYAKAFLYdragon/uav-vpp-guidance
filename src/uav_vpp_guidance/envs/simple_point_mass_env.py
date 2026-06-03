"""
简化质点运动环境，用于在 JSBSim 完全迁移前验证闭环逻辑。

本机采用简化的 3D 质点运动学模型：
  - roll_rate_cmd 控制滚转变化率
  - nz_cmd 粗略影响俯仰或垂向速度
  - throttle_cmd 控制速度增减

目标机采用匀速直线或简单正弦机动。

状态统一使用 NEU 坐标系（North-East-Up）。

状态字段要求：
  own_state:
    - position_m: np.ndarray shape [3]
    - velocity_vector_mps: np.ndarray shape [3]
    - speed_mps: float
    - heading_rad: float
    - pitch_rad: float
    - roll_rad: float
    - altitude_m: float
    - nz: float
  target_state:
    - position_m: np.ndarray shape [3]
    - velocity_vector_mps: np.ndarray shape [3]
    - speed_mps: float
    - heading_rad: float
    - altitude_m: float
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class SimplePointMassEnv:
    """
    简化 3D 质点运动环境，支持本机和目标机的基本运动学。
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): 环境配置，需包含 decision_freq 或 dt。
        """
        self.config = config
        self.dt = 1.0 / config.get("decision_freq", 5)
        self.gravity = 9.81

        # 运动学约束
        self.min_speed = config.get("min_speed_mps", 150.0)
        self.max_speed = config.get("max_speed_mps", 400.0)
        self.max_roll = np.deg2rad(config.get("max_roll_deg", 60.0))
        self.max_pitch = np.deg2rad(config.get("max_pitch_deg", 45.0))

        # 动力学增益（简化）
        self._heading_rate_per_roll = 1.5  # roll=1rad 时的航向变化率增益
        self._pitch_rate_per_nz = 0.3      # nz=1g 时的俯仰变化率增益
        self._accel_per_throttle = 20.0    # throttle=1 时的加速度 m/s^2

        # 目标机运动模式
        self._target_mode = config.get("target_mode", "constant_velocity")

        # 内部状态
        self.own_state: Optional[dict] = None
        self.target_state: Optional[dict] = None
        self.time = 0.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, own_init: Optional[dict] = None, target_init: Optional[dict] = None, scenario=None, seed=None):
        """
        重置本机和目标机状态。

        Args:
            own_init (dict): 本机初始状态。
            target_init (dict): 目标初始状态。
            scenario: 场景对象（可选）。
            seed: 随机种子（可选）。

        Returns:
            tuple: (own_state, target_state)
        """
        self.time = 0.0

        # 默认本机初始状态
        if own_init is None:
            own_init = {
                "position_m": np.array([0.0, 0.0, 5000.0]),
                "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
                "heading_rad": 0.0,
                "altitude_m": 5000.0,
                "roll_rad": 0.0,
                "pitch_rad": 0.0,
                "yaw_rad": 0.0,
                "nz": 1.0,
            }
        self.own_state = self._build_state(own_init)
        self._ensure_derived_fields(self.own_state)

        # 默认目标初始状态
        if target_init is None:
            target_init = {
                "position_m": np.array([2000.0, 0.0, 5000.0]),
                "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
                "heading_rad": np.pi,
                "altitude_m": 5000.0,
            }
        self.target_state = self._build_state(target_init)
        self._ensure_derived_fields(self.target_state)

        return self.own_state.copy(), self.target_state.copy()

    @staticmethod
    def _build_state(init: dict) -> dict:
        """统一状态字段格式。"""
        state = {}
        for k, v in init.items():
            state[k] = np.asarray(v, dtype=np.float64).copy() if isinstance(v, (list, np.ndarray)) else v
        return state

    @staticmethod
    def _ensure_derived_fields(state: dict):
        """确保派生字段（speed_mps 等）存在。缺失时填充合理默认值。"""
        vel = state.get("velocity_vector_mps")
        if vel is not None and "speed_mps" not in state:
            state["speed_mps"] = float(np.linalg.norm(vel))
        if "altitude_m" not in state:
            pos = state.get("position_m")
            if pos is not None:
                state["altitude_m"] = float(pos[2])
        # 简化模型中姿态和过载可能未提供，填充默认值以避免下游 KeyError
        if "pitch_rad" not in state:
            state["pitch_rad"] = 0.0
        if "roll_rad" not in state:
            state["roll_rad"] = 0.0
        if "yaw_rad" not in state:
            vel = state.get("velocity_vector_mps")
            if vel is not None:
                state["yaw_rad"] = float(np.arctan2(vel[1], vel[0]))
            else:
                state["yaw_rad"] = 0.0
        if "nz" not in state:
            state["nz"] = 1.0  # 默认平飞

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, own_command: dict, target_command: Optional[dict] = None) -> Tuple[dict, dict]:
        """
        执行一步环境推进。

        Args:
            own_command (dict): 本机控制指令，含 "nz_cmd", "roll_rate_cmd", "throttle_cmd"。
            target_command (dict, optional): 目标控制指令，若不提供则按目标模式自动更新。

        Returns:
            tuple: (own_state, target_state)
        """
        self._update_own(own_command)
        self._update_target(target_command)
        self.time += self.dt
        return self.own_state.copy(), self.target_state.copy()

    def _update_own(self, command: dict):
        """更新本机状态（简化动力学）。"""
        s = self.own_state
        dt = self.dt

        # 读取控制指令
        nz_cmd = float(command.get("nz_cmd", 1.0))
        roll_rate_cmd = float(command.get("roll_rate_cmd", 0.0))
        throttle_cmd = float(command.get("throttle_cmd", 0.5))

        # 1. 更新滚转
        roll = s.get("roll_rad", 0.0) + roll_rate_cmd * dt
        roll = np.clip(roll, -self.max_roll, self.max_roll)
        s["roll_rad"] = roll

        # 2. 更新俯仰（nz 近似引起俯仰变化）
        pitch = s.get("pitch_rad", 0.0) + self._pitch_rate_per_nz * nz_cmd * dt
        pitch = np.clip(pitch, -self.max_pitch, self.max_pitch)
        s["pitch_rad"] = pitch

        # 3. 更新航向（协调转弯简化：滚转引起航向变化）
        yaw = s.get("yaw_rad", 0.0) + self._heading_rate_per_roll * roll * dt
        s["yaw_rad"] = yaw
        s["heading_rad"] = yaw

        # 4. 更新速度大小
        vel = s.get("velocity_vector_mps", np.zeros(3))
        speed = float(np.linalg.norm(vel))
        speed += self._accel_per_throttle * (throttle_cmd - 0.5) * dt
        speed = np.clip(speed, self.min_speed, self.max_speed)

        # 5. 重建速度向量
        vx = speed * np.cos(pitch) * np.cos(yaw)
        vy = speed * np.cos(pitch) * np.sin(yaw)
        vz = speed * np.sin(pitch)
        s["velocity_vector_mps"] = np.array([vx, vy, vz], dtype=np.float64)
        s["speed_mps"] = speed

        # 6. 更新位置
        pos = s.get("position_m", np.zeros(3))
        pos += s["velocity_vector_mps"] * dt
        s["position_m"] = pos
        s["altitude_m"] = float(pos[2])

        # 7. 简化法向过载记录（用于 reward）
        s["nz"] = nz_cmd

    def _update_target(self, command: Optional[dict]):
        """更新目标机状态。"""
        s = self.target_state
        dt = self.dt

        if command is not None:
            # 若提供显式控制指令，按本机类似方式更新
            self._apply_point_mass_dynamics(s, command, dt)
            return

        if self._target_mode == "constant_velocity":
            # 匀速直线
            pos = s.get("position_m", np.zeros(3))
            vel = s.get("velocity_vector_mps", np.zeros(3))
            pos += vel * dt
            s["position_m"] = pos
            s["altitude_m"] = float(pos[2])

        elif self._target_mode == "sinusoidal":
            # 简单正弦机动：横向速度小幅波动
            pos = s.get("position_m", np.zeros(3))
            vel = s.get("velocity_vector_mps", np.zeros(3))
            base_speed = float(np.linalg.norm(vel))
            heading = np.arctan2(vel[1], vel[0])
            # 横向机动角速度
            lateral_rate = 0.05 * np.sin(0.5 * self.time)
            heading += lateral_rate * dt
            vel = np.array([
                base_speed * np.cos(heading),
                base_speed * np.sin(heading),
                0.0
            ], dtype=np.float64)
            pos += vel * dt
            s["velocity_vector_mps"] = vel
            s["position_m"] = pos
            s["altitude_m"] = float(pos[2])
            s["heading_rad"] = heading
            s["speed_mps"] = base_speed

        else:
            raise ValueError(f"Unknown target_mode: {self._target_mode}")

    @staticmethod
    def _apply_point_mass_dynamics(state: dict, command: dict, dt: float):
        """对 state 应用简化质点动力学更新。"""
        nz_cmd = float(command.get("nz_cmd", 1.0))
        roll_rate_cmd = float(command.get("roll_rate_cmd", 0.0))
        throttle_cmd = float(command.get("throttle_cmd", 0.5))

        roll = state.get("roll_rad", 0.0) + roll_rate_cmd * dt
        pitch = state.get("pitch_rad", 0.0) + 0.3 * nz_cmd * dt
        yaw = state.get("yaw_rad", 0.0) + 1.5 * roll * dt

        vel = state.get("velocity_vector_mps", np.zeros(3))
        speed = float(np.linalg.norm(vel))
        speed += 20.0 * (throttle_cmd - 0.5) * dt
        speed = np.clip(speed, 150.0, 400.0)

        vx = speed * np.cos(pitch) * np.cos(yaw)
        vy = speed * np.cos(pitch) * np.sin(yaw)
        vz = speed * np.sin(pitch)
        vel = np.array([vx, vy, vz], dtype=np.float64)

        pos = state.get("position_m", np.zeros(3))
        pos += vel * dt

        state["roll_rad"] = roll
        state["pitch_rad"] = pitch
        state["yaw_rad"] = yaw
        state["heading_rad"] = yaw
        state["velocity_vector_mps"] = vel
        state["position_m"] = pos
        state["altitude_m"] = float(pos[2])
        state["speed_mps"] = speed

    def get_state(self) -> Tuple[dict, dict]:
        """返回当前本机和目标状态。"""
        return self.own_state.copy(), self.target_state.copy()
