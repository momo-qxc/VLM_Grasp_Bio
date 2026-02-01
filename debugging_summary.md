# 机械臂放置策略调试技术报告 (Technical Report on Robotic Placement Strategy)

**日期**：2025-12-28
**主题**：UR5e 机械臂在复杂工作空间内的运动规划策略优化
**摘要**：本文档记录了针对不同空间象限（侧方与后方）目标点的抓取放置策略的调试过程。重点分析了笛卡尔空间路径规划在奇异点附近的失效机理，以及轴角表示法（Axis-Angle）在零旋转条件下的数值不稳定性问题，并详述了最终的自适应混合规划策略。

---

## 1. 策略演变历程 (Strategy Evolution)

### 阶段一：笛卡尔空间直线规划 (Cartesian Linear Planning)
*   **工况**：目标点位于机械臂侧前方 `(1.4, 0.3)`。
*   **方法**：使用 `LinePositionParameter` 对末端执行器 (End-Effector) 进行直线插补。
*   **结果**：成功。由于起点与终点间无障碍且远离奇点，运动轨迹平滑。

### 阶段二：后方目标点的路径失效
*   **工况**：目标点调整至机械臂背部区域 `(0.2, 0.2)`。
*   **现象**：程序抛出 `AssertionError`，运动规划失败。
*   **原因分析**：从前方抓取点至后方放置点的直线路径穿过了机械臂基座 (Base Link) 的圆柱体包围盒。这一区域属于运动学奇异区 (Kinematic Singularity)，逆运动学 (IK) 求解器无法在关节速度限制内计算出满足直线轨迹的解。

### 阶段三：关节空间规划 (Joint Space Planning)
*   **方法**：计算目标位姿的 IK 解，使用 `JointParameter` 在关节空间 (Configuration Space) 进行线性插值。
*   **结果**：解决了可达性问题，机械臂通过旋转基座关节 (Joint 0) 成功到达后方。
*   **衍生问题**：
    1.  **除零异常**：在回退测试侧方放置时，因姿态未发生变化触发 `ValueError: zero norm vector`。
    2.  **IK 无解**：在特定高难度位姿下，解析解 IK 仍可能因为构型限制无法求出逆解。

### 阶段四：自适应混合策略 (Adaptive Hybrid Strategy)
*   **最终方案**：基于目标坐标的空间分布特征，动态切换运动规划算法。
    *   **侧方区域**：保持笛卡尔规划，确保轨迹精确性。
    *   **后方区域**：优先使用关节规划优化构型，并引入中间航点 (Waypoint) 机制作为 IK 求解失败的鲁棒性兜底。

---

## 2. 故障机理深度分析 (Failure Analysis)

### 2.1 零向量归一化异常 (Zero Norm Vector Exception)

**故障现象**：`ValueError: zero norm vector`
**代码定位**：`two_attitude_planner.py` 第 29 行
**数学原理**：
该规划器使用 **轴角 (Axis-Angle)** 方法计算姿态插值。给定两个旋转矩阵 $R_{start}$ 和 $R_{end}$，需计算等效旋转轴 $\vec{k}$ 和角度 $\theta$。
旋转轴 $\vec{k}$ 的计算依赖于旋转向量的归一化：
$$ \vec{k} = \frac{\vec{v}}{||\vec{v}||} $$
当 $R_{start} = R_{end}$ 时（即姿态保持不变），旋转向量 $\vec{v} \to \vec{0}$，模长 $||\vec{v}|| = 0$。
计算机执行浮点除法 `[0,0,0] / 0` 导致异常。
**修正代码**：
在 `grasp_process.py` 中引入类型检查：对于姿态不变的轨迹，强制使用 `OneAttitudeParameter` 类，该类仅保持初始四元数，不涉及轴角计算。

### 2.2 逆运动学求解失败 (Analytical IK Failure)

**故障现象**：`AssertionError` (触发于 `robot.move_cartesian`)
**代码定位**：`ur5e.py` 第 168-171 行
**几何原理**：
UR5e 的解析 IK 算法通过几何法求解。其中一步利用余弦定理计算肘部关节 (`theta3`)：
```python
# ur5e.py
theta3_condition = (x**2 + y**2 - a2**2 - a3**2) / (2 * a2 * a3)
if np.abs(theta3_condition) > 1.0:
    return [] # 无解
```
若目标路径通过基座中心轴线（Z轴），所需的构型可能导致三角形边长关系无法满足（即 `cos(theta) > 1`），或需要关节角发生瞬时跃变。此时数学上无实数解。

**工程修正**：
引入由于两个线段组成的折线路径：
1.  **Segment 1**: 当前点 $\to$ 安全中间点 `(0.8, 0.1)` (避开基座奇异区)。
2.  **Segment 2**: 中间点 $\to$ 后方目标点 `(0.2, 0.2)`。
这种拓扑路径修正确保了每段轨迹都在机器人的可达工作空间 (Reachability Workspace) 内。

---

## 3. 最终代码逻辑规范 (Implementation Specification)

目前的控制逻辑位于 `grasp_process.py`，采用条件分支处理：

1.  **目标评估 (Heuristic Evaluation)**:
    根据 `target_pos` 坐标判定目标区域（Front/Side vs Back）。

2.  **侧方规划 (Side/Front Strategy)**:
    *   **姿态约束**: `OneAttitudeParameter` (保持抓取姿态，规避零向量计算)。
    *   **路径约束**: `CartesianParameter` (直线轨迹)。

3.  **后方规划 (Back Strategy)**:
    *   **姿态约束**: `TwoAttitudeParameter` (允许手腕翻转 `Rz(pi)*Rx(pi)` 以适应对侧放置)。
    *   **主路径**: 关节空间插值 (`move_joint`)。
    *   **冗余路径 (Fallback)**: 若主路径 IK 为空，自动切换至 `Lift -> Waypoint -> Target` 的两段式笛卡尔规划。
