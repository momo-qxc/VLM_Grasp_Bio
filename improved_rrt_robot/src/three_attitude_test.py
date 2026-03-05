import os
import sys

import numpy as np
from spatialmath import SO3
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory("C://Users//Cybaster//.mujoco//mjpro150//bin")

from motion_planning import *

from robot import Robot

if __name__ == '__main__':
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "universal_robots_ur5e", "scene.xml")
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    N = 6
    control_angle = list(np.zeros(N))

    ur_robot = Robot()
    q0 = [0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0]
    ur_robot.set_joint(q0)
    T0 = ur_robot.get_cartesian()

    t0 = T0.t + np.array([0.1 * np.sqrt(2), -0.1 * np.sqrt(2), 0.0])
    R0 = SO3.Rz(-np.pi / 4) * SO3.Ry(-np.pi / 6) * SO3(T0.R)
    t1 = T0.t + np.array([0.1 * np.sqrt(2), 0.1 * np.sqrt(2), 0.0])
    R1 = SO3.Rz(np.pi / 4) * SO3.Ry(-np.pi / 3) * SO3(T0.R)

    tc = T0.t
    vec_n = np.array([0, 1, 1])

    arc_center_position_parameter = ArcCenterPositionParameter(t0, t1, tc)
    three_attitude_parameter = ThreeAttitudeParameter(R0, R1, vec_n)
    cartesian_parameter = CartesianParameter(arc_center_position_parameter, three_attitude_parameter)
    cubic_velocity_parameter = CubicVelocityParameter(10.0)
    trajectory_parameter = TrajectoryParameter(cartesian_parameter, cubic_velocity_parameter)

    trajectory_planner = TrajectoryPlanner(trajectory_parameter)

    dt = 0.02
    t_end = 20.0
    t_array = np.arange(0.0, t_end, dt)
    t_len = len(t_array)

    joints = np.zeros((t_len, 6))

    t_start = 5.0
    for i, ti in enumerate(t_array):
        Ti = trajectory_planner.interpolate(ti - t_start)
        ur_robot.move_cartesian(Ti)
        joints[i, :] = ur_robot.get_joint()

    t_step = 0
    forward = True
    j = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i in range(6):
                data.qpos[i] = joints[t_step, i]
                data.qvel[i] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            j += 1
            if j == 10:
                j = 0
                if forward:
                    t_step += 1
                    if t_step == t_len - 1:
                        forward = False
                else:
                    t_step -= 1
                    if t_step == 0:
                        forward = True
