"""
UR5e 工作空间采样脚本
在桌面高度（z=0.76m）上撒点，逐个用 IK 检验是否可达，输出可达范围并保存图像。
"""
import sys, os
import numpy as np
import spatialmath as sm
import matplotlib
matplotlib.use('Agg')  # 纯文件输出，避免 Qt 版本冲突
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp'))

from config import Config
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

env = UR5GraspEnv()
env.reset()

# 末端朝下的姿态（放置时的典型姿态）
# x < 0.7: Rz(π)*Rx(π) — 面向 -X 的垂直向下（背后区域）
# x >= 0.7: Rx(π)       — 面向 +X 的垂直向下（前方区域）
R_down_front = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
])  # Rx(π)

R_down_back = np.array([
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0,  0, -1],
])  # Rz(π)*Rx(π)

TURN_AROUND_X = 0.7  # 与 grasp_process_optimized.py 一致

Z_TABLE = 0.76   # 桌面放置高度
STEP    = 0.04   # 采样间隔（米），越小越精细但越慢

# 在姿态切换区域（x=0.6~0.8）尝试两种姿态，选择可达的那个
TRANSITION_ZONE = (0.6, 0.8)

xs = np.arange(-0.2, 2.2, STEP)
ys = np.arange(-0.8, 1.6, STEP)

reachable_pts = []   # list of (x, y, q)
unreachable   = []

total = len(xs) * len(ys)
count = 0

print(f"采样 {total} 个点，间隔 {STEP}m ...")

for x in xs:
    for y in ys:
        count += 1
        if count % 200 == 0:
            print(f"  进度: {count}/{total}")

        # 在过渡区域尝试两种姿态
        if TRANSITION_ZONE[0] <= x <= TRANSITION_ZONE[1]:
            orientations = [R_down_back, R_down_front]
        else:
            R_down = R_down_back if x < TURN_AROUND_X else R_down_front
            orientations = [R_down]

        reached = False
        for R in orientations:
            T_target = sm.SE3.Rt(R, [x, y, Z_TABLE])
            try:
                q = env.robot.ikine(T_target)
                if len(q) > 0:
                    reachable_pts.append((x, y, q))
                    reached = True
                    break
            except Exception:
                continue

        if not reached:
            unreachable.append([x, y])

unreachable = np.array(unreachable)
reachable   = np.array([[x, y] for x, y, _ in reachable_pts])

print(f"可达点: {len(reachable)}, 不可达点: {len(unreachable)}")
if len(reachable) > 0:
    print(f"X 范围: [{reachable[:,0].min():.3f}, {reachable[:,0].max():.3f}] m")
    print(f"Y 范围: [{reachable[:,1].min():.3f}, {reachable[:,1].max():.3f}] m")

# ── 绘图 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 12))

if len(unreachable) > 0:
    ax.scatter(unreachable[:,0], unreachable[:,1], c='#8b0000', s=12, label='不可达', alpha=0.8)
if len(reachable) > 0:
    ax.scatter(reachable[:,0], reachable[:,1], c='#22aa44', s=12, label='可达', alpha=0.8)

# 机械臂底座
base_x, base_y = Config.ROBOT_BASE_X, Config.ROBOT_BASE_Y
ax.plot(base_x, base_y, 'k*', markersize=13, label=f'底座 ({base_x}, {base_y})', zorder=10)
circle = plt.Circle((base_x, base_y), 0.85, color='blue', fill=False,
                     linestyle='--', linewidth=2.5, label='理论臂展 0.85m')
ax.add_patch(circle)

# 奇异点区域（overhead singularity，UR5e d4=0.134m）
sing_circle = plt.Circle((base_x, base_y), 0.134, color='purple', fill=True,
                          alpha=0.25, linestyle='-', linewidth=2.5,
                          label='奇异区域 (r≈0.13m)')
ax.add_patch(sing_circle)
sing_border = plt.Circle((base_x, base_y), 0.134, color='purple', fill=False,
                          linestyle='-', linewidth=2.5)
ax.add_patch(sing_border)

# 当前代码有效工作空间（环形，来自 Config）
ws_inner = plt.Circle((base_x, base_y), Config.WORKSPACE_R_MIN, color='orange', fill=False,
                       linestyle=':', linewidth=2.5)
ax.add_patch(ws_inner)
ws_outer = plt.Circle((base_x, base_y), Config.WORKSPACE_R_MAX, color='orange', fill=False,
                       linestyle=':', linewidth=2.5,
                       label=f'代码工作空间 r=[{Config.WORKSPACE_R_MIN},{Config.WORKSPACE_R_MAX}]m')
ax.add_patch(ws_outer)
# 桌面边界框
table_rect = patches.Rectangle(
    (Config.TABLE_X_MIN, Config.TABLE_Y_MIN),
    Config.TABLE_X_MAX - Config.TABLE_X_MIN,
    Config.TABLE_Y_MAX - Config.TABLE_Y_MIN,
    linewidth=2, edgecolor='orange', facecolor='none',
    linestyle='--', label='桌面边界'
)
ax.add_patch(table_rect)

# 货架区域（shelf_collisions: center=1.79,0.6, half-size=0.18,0.6）
shelf_rect = patches.Rectangle(
    (1.79 - 0.18, 0.6 - 0.6),
    0.36, 1.2,
    linewidth=2.5, edgecolor='gray', facecolor='#80808033', linestyle='-.', label='货架区域'
)
ax.add_patch(shelf_rect)

# 绿色区域（zone_pickup: center=1.4,0.6, half-size=0.2,0.6）
green = patches.Rectangle((1.2, 0.0), 0.4, 1.2,
                            linewidth=2.5, edgecolor='green', facecolor='#00ff0033', label='绿色区域')
ax.add_patch(green)

# 红色区域（zone_drop: center=0.2,0.2, half-size=0.2,0.2）
red = patches.Rectangle((0.0, 0.0), 0.4, 0.4,
                          linewidth=2.5, edgecolor='red', facecolor='#ff000033', label='红色区域')
ax.add_patch(red)

ax.set_xlabel('X (m)', fontsize=14)
ax.set_ylabel('Y (m)', fontsize=14)
ax.set_title(f'UR5e 工作空间（z={Z_TABLE}m，末端朝下）',
             fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=11, markerscale=1.5, framealpha=0.95,
          facecolor='white', edgecolor='0.5', shadow=True)
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)

# 添加说明文本
info_text = (
    f"采样间隔: {STEP}m\n"
    f"姿态切换: x={TURN_AROUND_X}m\n"
    f"过渡区域: x∈[{TRANSITION_ZONE[0]}, {TRANSITION_ZONE[1]}]m\n"
    f"（在过渡区域尝试两种姿态）"
)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))
plt.tight_layout()
plt.savefig('workspace_map.png', dpi=150)
print("\n已保存: workspace_map.png")

env.close()
