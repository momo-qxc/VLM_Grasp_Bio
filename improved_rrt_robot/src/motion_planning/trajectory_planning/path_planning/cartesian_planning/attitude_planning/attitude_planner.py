from spatialmath import SO3

from src.interface.strategy_wrapper import StrategyWrapper


class AttitudePlanner(StrategyWrapper):

    def interpolate(self, s) -> SO3:
        return self.strategy.interpolate(s)
