from .simplex_parameter import SimplexParameter
from src.geometry.simplex.simplex import Simplex


class SimplexFactoryInterface:
    def create_product(self, simplex_parameter: SimplexParameter) -> Simplex:
        pass
