"""
weapons 子包：武器模型（导弹等）。

当前导出 3 自由度空空导弹模型 ``Missile3DoF``。
统一坐标系约定（与 JSBSim 一致）：X=North, Y=Up, Z=East；单位米/秒/弧度。
"""

from .missile import Missile3DoF

__all__ = ["Missile3DoF"]
