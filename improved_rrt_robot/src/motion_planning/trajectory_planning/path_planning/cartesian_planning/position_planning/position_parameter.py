from abc import ABC

from src.interface import Parameter


class PositionParameter(Parameter, ABC):
    def get_length(self):
        pass
