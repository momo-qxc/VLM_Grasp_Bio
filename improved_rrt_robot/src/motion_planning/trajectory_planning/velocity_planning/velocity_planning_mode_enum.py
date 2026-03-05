from enum import unique
from src.interface import ModeEnum


@unique
class VelocityPlanningModeEnum(ModeEnum):
    CUBIC = 'cubic'
    QUINTIC = 'quintic'
    T_CURVE = 't_curve'
