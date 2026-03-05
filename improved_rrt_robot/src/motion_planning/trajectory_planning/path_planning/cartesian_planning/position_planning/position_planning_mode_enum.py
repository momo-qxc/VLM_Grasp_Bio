from enum import unique
from src.interface import ModeEnum


@unique
class PositionPlanningModeEnum(ModeEnum):
    LINE = 'line'
    ARC_CENTER = 'arc_center'
