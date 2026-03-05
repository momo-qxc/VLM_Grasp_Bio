import sys
from pathlib import Path
from typing import Any, Dict

_CACHED: Dict[str, Any] = {}


def get_improved_modules() -> Dict[str, Any]:
    """Load improved_rrt_robot modules once and return required classes."""
    if _CACHED:
        return _CACHED

    project_root = Path(__file__).resolve().parents[1]
    improved_root = project_root / "improved_rrt_robot"
    if str(improved_root) not in sys.path:
        sys.path.insert(0, str(improved_root))

    from src.robot import Robot as ImprovedRobot
    from src.geometry import Sphere as ImprovedSphere, Brick as ImprovedBrick, LineSegment
    from src.motion_planning import RRTMap, RobotRRTParameter, RRTPlanner
    from src.motion_planning.trajectory_planning.path_planning.rrt_planning.check_collision_robot import (
        CheckCollisionRobot,
    )

    _CACHED.update(
        {
            "ImprovedRobot": ImprovedRobot,
            "ImprovedSphere": ImprovedSphere,
            "ImprovedBrick": ImprovedBrick,
            "LineSegment": LineSegment,
            "RRTMap": RRTMap,
            "RobotRRTParameter": RobotRRTParameter,
            "RRTPlanner": RRTPlanner,
            "CheckCollisionRobot": CheckCollisionRobot,
        }
    )
    return _CACHED
