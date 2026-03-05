from enum import unique
from src.interface import ModeEnum


@unique
class AttitudePlannerModeEnum(ModeEnum):
    ONE = 'one'
    TWO = 'two'
    THREE = 'three'
