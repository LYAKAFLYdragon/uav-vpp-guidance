"""
FireControlRadar：火控雷达几何视场与锁定状态机模型。

坐标系约定（与 JSBSim 一致）：
    X = North（北，正向北）
    Y = Up（上，正向上）
    Z = East（东，正向东）

单位约定：
    - 长度 / 位置：米（m）
    - 速度：米/秒（m/s）
    - 角度：弧度（rad）；对外暴露的 azimuth_deg / elevation_deg 为度（deg）

说明：
    本模块实现仅基于几何约束的火控雷达模型 ``FireControlRadar``：
    根据本机与目标的相对几何（距离、方位角、俯仰角）判定目标是否在视场内
    （in_view），并通过"连续视场内步数"状态机产生稳定的锁定信号（locked）。
    雷达**仅做几何判定**，不计算雷达方程、多普勒效应或搜索模式（需求 2.8）。
    该模块不依赖 JSBSim，可在纯 Python 环境中导入。

    状态字典键与轴索引约定（与 ``weapons/missile.py`` 一致）：
        - 位置键 ``position_m`` / ``position_neu`` / ``position``，3 元向量
          ``[north, up, east]``：index 0=north, 1=up, 2=east。
        - 速度键 ``velocity_mps``（3 元）/ ``velocity_vector_mps`` / ``velocity_ned`` /
          ``velocity``：``[v_north, v_up, v_east]``。
    位置 / 速度提取复用 ``weapons.missile`` 的 ``_extract_position`` /
    ``_extract_velocity``，避免重复实现并保证与导弹模块行为一致。
"""

import numpy as np

# 复用导弹模块的状态提取辅助函数，保证键约定与轴索引完全一致。
# sensors 依赖 weapons 不会造成循环导入（weapons 不依赖 sensors）。
from ..weapons.missile import _extract_position, _extract_velocity

# 防奇异小量：零距离时避免除零奇异。
_EPS = 1e-8


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 (-pi, pi] 区间。

    Args:
        angle: 任意弧度角。

    Returns:
        float: 归一化到 (-pi, pi] 的等价角。
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class FireControlRadar:
    """火控雷达几何模型（视场判定 + 连续锁定状态机）。

    仅依据几何约束（距离 / 方位角 / 俯仰角 / 连续视场内步数）判定目标是否
    在视场内与是否锁定，不计算雷达方程、多普勒或搜索模式（需求 2.8）。
    """

    # 设计常量（需求 2.1 / 2.4）。
    MAX_RANGE_M = 10000.0       # 最大探测距离（m）
    AZIMUTH_FOV_DEG = 60.0      # 方位视场半角 ±60°
    ELEVATION_FOV_DEG = 30.0    # 俯仰视场半角 ±30°
    LOCK_TIME_STEPS = 3         # 连续视场内步数达到该阈值即锁定

    def __init__(self, config: dict | None = None):
        """构造火控雷达。

        Args:
            config: 可选配置字典（如 ``radar_config``），支持以下小写键覆盖常量：
                ``max_range_m``、``azimuth_fov_deg``、``elevation_fov_deg``、
                ``lock_time_steps``。未提供的键沿用类常量默认值。

        构造后雷达处于复位状态：``in_view=False``、``locked=False``、
        ``_consecutive_in_view_steps=0``。
        """
        cfg = config or {}

        # 实例级常量（允许 radar_config 覆盖类常量）。
        self.max_range_m = float(cfg.get("max_range_m", self.MAX_RANGE_M))
        self.azimuth_fov_deg = float(cfg.get("azimuth_fov_deg", self.AZIMUTH_FOV_DEG))
        self.elevation_fov_deg = float(
            cfg.get("elevation_fov_deg", self.ELEVATION_FOV_DEG)
        )
        self.lock_time_steps = int(cfg.get("lock_time_steps", self.LOCK_TIME_STEPS))

        # 运行状态。
        self.in_view: bool = False
        self.locked: bool = False
        self._consecutive_in_view_steps: int = 0

    def reset(self) -> None:
        """回合开始时复位状态：``in_view=False``、``locked=False``、连续计数清零。"""
        self.in_view = False
        self.locked = False
        self._consecutive_in_view_steps = 0

    def update(self, ego_state: dict, target_state: dict) -> dict:
        """几何更新一步：计算视场判定并推进锁定状态机（需求 2.2–2.8）。

        计算流程（JSBSim 坐标系，轴索引 0=north, 1=up, 2=east）：
            1. 视线向量 ``rel_r = target.pos - ego.pos``，距离 ``r = |rel_r|``。
            2. 方位角：本机航向 ``psi_ego = arctan2(v_east, v_north)``；
               视线方位 ``psi_los = arctan2(rel_east, rel_north)``；
               ``azimuth = wrap_to_pi(psi_los - psi_ego)``，转换为度后取绝对值比较。
            3. 俯仰角：``elevation = arcsin(clip(rel_up / r, -1, 1))``，转换为度。
            4. ``in_view = (r <= max_range_m) and (|az| <= azimuth_fov_deg)
               and (|el| <= elevation_fov_deg)``。
            5. 若 in_view：连续计数 +1；否则连续计数清零且 ``locked=False``（需求 2.6）。
            6. ``locked = (连续计数 >= lock_time_steps)``（需求 2.4 / 2.5）。

        边界处理：当 ``r < _EPS``（目标与本机几乎重合）时，方位角与俯仰角均取 0，
        视为在视场内（距离条件天然满足），避免除零奇异。

        Args:
            ego_state: 本机状态字典（含位置与速度）。
            target_state: 目标状态字典（含位置）。

        Returns:
            dict: ``{'locked': bool, 'in_view': bool, 'azimuth_deg': float,
                     'elevation_deg': float, 'range_m': float}``。
        """
        ego_pos = _extract_position(ego_state)
        ego_vel = _extract_velocity(ego_state)
        target_pos = _extract_position(target_state)

        # 视线向量与距离（轴索引 0=north, 1=up, 2=east）。
        rel_r = target_pos - ego_pos
        r = float(np.linalg.norm(rel_r))

        if r < _EPS:
            # 目标与本机几乎重合：几何角退化为 0，视为在视场内。
            azimuth_deg = 0.0
            elevation_deg = 0.0
        else:
            rel_north = float(rel_r[0])
            rel_up = float(rel_r[1])
            rel_east = float(rel_r[2])

            # 方位角：视线方位相对本机航向，归一化到 (-pi, pi]。
            psi_ego = np.arctan2(float(ego_vel[2]), float(ego_vel[0]))
            psi_los = np.arctan2(rel_east, rel_north)
            azimuth_rad = wrap_to_pi(psi_los - psi_ego)
            azimuth_deg = float(np.degrees(azimuth_rad))

            # 俯仰角：相对本机水平面（MVP 近似本机俯仰为水平基准）。
            elevation_rad = np.arcsin(np.clip(rel_up / r, -1.0, 1.0))
            elevation_deg = float(np.degrees(elevation_rad))

        # 几何视场判定（需求 2.2 / 2.3）。
        in_view = bool(
            (r <= self.max_range_m)
            and (abs(azimuth_deg) <= self.azimuth_fov_deg)
            and (abs(elevation_deg) <= self.elevation_fov_deg)
        )

        # 连续视场内步数状态机（需求 2.4 / 2.5 / 2.6）。
        if in_view:
            self._consecutive_in_view_steps += 1
        else:
            self._consecutive_in_view_steps = 0
            self.locked = False

        self.in_view = in_view
        self.locked = bool(self._consecutive_in_view_steps >= self.lock_time_steps)

        return {
            "locked": self.locked,
            "in_view": self.in_view,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "range_m": r,
        }
