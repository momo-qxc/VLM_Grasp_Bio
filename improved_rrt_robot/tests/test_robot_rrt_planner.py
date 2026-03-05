import os
import sys

import numpy as np
from spatialmath import SE3
import mujoco
import mujoco.viewer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.robot import Robot
from src.geometry import Brick
from src.motion_planning import RobotRRTParameter, RRTMap, BlendPlanner, RRTPlanner, RRTStarPlanner, \
    InformedRRTStarPlanner

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory("C://Users//Cybaster//.mujoco//mjpro150//bin")
import multiprocessing

robot = Robot()

obstacles = [
    Brick(SE3.Trans(0.5, 0.0, 0.8), np.array([0.4, 0.4, 0.01])),
]

rrt_map = RRTMap(
    area=[
        (-np.pi / 2, np.pi / 2),
        (-np.pi / 2, np.pi / 2),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi / 2, np.pi / 2)
    ],
    obstacles=obstacles)
rrt_parameter = RobotRRTParameter(start=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  goal=[0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0],
                                  robot=robot, expand_dis=np.pi / 12, max_iter=500, radius=5.0)


def visualize(rrt_planner):
    path_parameters = rrt_planner.get_path_parameters()
    radii = [0.0 for _ in range(len(path_parameters) - 1)]
    blend_planner = BlendPlanner(path_parameters, radii)

    if not rrt_planner.success:
        return

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "universal_robots_ur5e", "scene.xml")
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    dof = 6

    num = 1001
    ss = np.linspace(0.0, 1.0, num)
    joints = np.zeros((num, dof))

    s_step = 0
    forward = True
    j = 0

    for i, si in enumerate(ss):
        qi = blend_planner.interpolate(si)
        robot.set_joint(qi)
        joints[i, :] = robot.get_joint()

    robot.set_joint(joints[0, :])
    T0 = robot.get_cartesian()
    robot.set_joint(joints[-1, :])
    T1 = robot.get_cartesian()
    coms = [T0.t, T1.t]

    # Keep path replay behavior; marker rendering from mujoco_py is omitted in this migration.
    _ = coms
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i in range(dof):
                data.qpos[i] = joints[s_step, i]
                data.qvel[i] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            j += 1
            if j == 10:
                j = 0
                if forward:
                    s_step += 1
                    if s_step == num - 1:
                        forward = False
                else:
                    s_step -= 1
                    if s_step == 0:
                        forward = True


def test_robot_rrt():
    rrt_planner = RRTPlanner(rrt_map, rrt_parameter)
    visualize(rrt_planner)


def test_robot_rrt_star():
    rrt_planner = RRTStarPlanner(rrt_map, rrt_parameter, pool)
    visualize(rrt_planner)


def test_robot_informed_rrt_star():
    rrt_planner = InformedRRTStarPlanner(rrt_map, rrt_parameter, pool)
    visualize(rrt_planner)


if __name__ == "__main__":
    pool = multiprocessing.Pool()

    for i in range(10):
        test_robot_rrt()
        # test_robot_rrt_star()
        # test_robot_informed_rrt_star()

    pool.close()
