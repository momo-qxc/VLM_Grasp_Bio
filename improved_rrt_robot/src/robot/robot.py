import copy
from typing import List

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from src.geometry import Geometry, Capsule


class Robot:
    def __init__(self):
        d1 = 0.163
        d4 = 0.134
        d5 = 0.1
        d6 = 0.1

        a3 = 0.425
        a4 = 0.392

        self.dof = 6
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]
        a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]
        d_array = [d1, 0.0, 0.0, d4, d5, d6]
        theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]
        tool = SE3.Trans(0.0, 0.0, 0.2)

        links = []
        for i in range(6):
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True))
        self.robot = rtb.DHRobot(links)

        self.alpha_array = alpha_array
        self.a_array = a_array
        self.d_array = d_array
        self.theta_array = theta_array

    def fkine(self, q) -> SE3:
        return self.robot.fkine(q)

    def ikine(self, Tep):
        result = self.robot.ikine_NR(Tep, q0=self.q0)
        if result.success:
            return result.q
        return []

    def move_cartesian(self, T: SE3):
        q = self.ikine(T)

        assert len(q)  # inverse kinematics failure
        self.set_joint(q)

    def set_joint(self, q):
        self.q0 = q[:]

    def get_joint(self):
        return copy.deepcopy(self.q0)

    def get_cartesian(self):
        return self.fkine(self.q0)

    def get_geometries(self) -> List[Geometry]:
        Ts = []
        T = SE3()
        for i in range(self.dof):
            T = T * Robot.transform_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.q0[i])
            Ts.append(T)

        T1 = Ts[0] * SE3.Trans(0, 0, -0.04)
        geometry1 = Capsule(T1, 0.06, 0.12)

        T2 = Ts[1] * SE3.Trans(0.2, 0, 0.138) * SE3.Ry(-np.pi / 2)
        geometry2 = Capsule(T2, 0.05, 0.4)

        T3 = Ts[2] * SE3.Trans(0.2, 0, 0.007) * SE3.Ry(-np.pi / 2)
        geometry3 = Capsule(T3, 0.038, 0.38)

        T4 = Ts[3] * SE3.Trans(0.0, 0.0, -0.07)
        geometry4 = Capsule(T4, 0.04, 0.14)

        T5 = Ts[4] * SE3.Trans(0.0, 0.0, -0.06)
        geometry5 = Capsule(T5, 0.04, 0.12)

        T6 = Ts[5] * SE3.Trans(0.0, 0.0, -0.06)
        geometry6 = Capsule(T6, 0.04, 0.12)

        return [geometry1, geometry2, geometry3, geometry4, geometry5, geometry6]

    def __getstate__(self):
        state = {"dof": self.dof,
                 "q0": self.q0,
                 "alpha_array": self.alpha_array,
                 "a_array": self.a_array,
                 "d_array": self.d_array,
                 "theta_array": self.theta_array,
                 # "robot": self.robot
                 }
        return state

    def __setstate__(self, state):
        self.dof = state["dof"]
        self.q0 = state["q0"]
        self.alpha_array = state["alpha_array"]
        self.a_array = state["a_array"]
        self.d_array = state["d_array"]
        self.theta_array = state["theta_array"]
        links = []
        for i in range(6):
            links.append(
                rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                           mdh=True))
        self.robot = rtb.DHRobot(links)

    @staticmethod
    def transform_mdh(alpha, a, d, theta, q) -> SE3:
        return SE3.Rx(alpha) * SE3.Trans(a, 0, d) * SE3.Rz(theta + q)


if __name__ == '__main__':
    ur_robot = Robot()
    q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T1 = ur_robot.fkine(q0)
    print(T1)
    ur_robot.move_cartesian(T1)
    q_new = ur_robot.get_joint()
    print(q_new)
