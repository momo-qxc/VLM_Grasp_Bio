from enum import unique
from src.interface import ModeEnum


@unique
class PathPlanningModeEnum(ModeEnum):
    JOINT = 'joint'
    CARTESIAN = 'cartesian'
