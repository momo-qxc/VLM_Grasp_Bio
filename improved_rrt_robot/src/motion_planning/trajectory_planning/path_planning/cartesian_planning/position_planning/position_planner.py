import numpy as np

from src.interface import StrategyWrapper


class PositionPlanner(StrategyWrapper):

    def interpolate(self, s) -> np.ndarray:
        return self.strategy.interpolate(s)
