import abc
import multiprocessing

from src.geometry import LineSegment


class ICheckCollision(abc.ABC):

    @abc.abstractmethod
    def check_collision(self, line_segment: LineSegment, pool: multiprocessing.Pool = None):
        pass
