"""Registry of available maneuver primitives."""
from __future__ import annotations

from typing import Dict, Type

from .base import Maneuver
from .maneuvers.straight_level import StraightLevel
from .maneuvers.coordinated_turn import CoordinatedTurn
from .maneuvers.dive import Dive
from .maneuvers.loop import Loop
from .maneuvers.barrel_roll import BarrelRoll
from .maneuvers.high_yoyo import HighYoYo
from .maneuvers.immelmann import Immelmann
from .maneuvers.low_yoyo import LowYoYo
from .maneuvers.scissors import Scissors
from .maneuvers.split_s import SplitS


class ManeuverLibrary:
    """Factory/registry for maneuver primitives."""

    _registry: Dict[str, Type[Maneuver]] = {
        StraightLevel.name: StraightLevel,
        CoordinatedTurn.name: CoordinatedTurn,
        Dive.name: Dive,
        Loop.name: Loop,
        BarrelRoll.name: BarrelRoll,
        HighYoYo.name: HighYoYo,
        LowYoYo.name: LowYoYo,
        Scissors.name: Scissors,
        SplitS.name: SplitS,
        Immelmann.name: Immelmann,
    }

    @classmethod
    def list_maneuvers(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def register(cls, name: str, maneuver_cls: Type[Maneuver]):
        cls._registry[name] = maneuver_cls

    @classmethod
    def create(cls, name: str, params: dict | None = None) -> Maneuver:
        if name not in cls._registry:
            raise KeyError(f"Unknown maneuver '{name}'. Available: {cls.list_maneuvers()}")
        return cls._registry[name](params)
