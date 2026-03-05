from abc import ABC

from src.interface import Strategy, Parameter


class VelocityPlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
