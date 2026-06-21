"""
sensors 子包：传感器模型（雷达等）。

当前导出火控雷达模型 ``FireControlRadar``。
统一坐标系约定（与 JSBSim 一致）：X=North, Y=Up, Z=East；单位米/秒/弧度。
"""

from .radar import FireControlRadar, wrap_to_pi

__all__ = ["FireControlRadar", "wrap_to_pi"]
