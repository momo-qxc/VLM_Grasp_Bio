# VLM 抓取算法优化与排错记录 (VLM Grasping Optimization)

本文件记录了 VLM 抓取逻辑及其核心算法的优化历程和排错经验，旨在帮助开发者快速通过对比“旧逻辑”与“新逻辑”来理解系统演进。

## 一、系统架构与运行模式 (Architecture)

### 1.1 双版本算法说明
为了兼顾不同场景下的兼容性，系统保留了两套 `grasp_process` 逻辑：

| 文件名 | 算法特性 | 适用场景 |
| :--- | :--- | :--- |
| `grasp_process.py` | **Legacy**: 相机坐标系过滤，无调平，无补偿 | 基础测试，验证物理一致性 |
| `grasp_process_optimized.py` | **Advanced**: 世界坐标系过滤，自动调平，1.5cm 接近补偿 | 正式任务，应对倾斜角度 |

### 1.2 切换方法
在 `main_vlm.py` 中修改导入语句即可：
```python
# 切换到优化版
from grasp_process_optimized import run_grasp_inference, execute_grasp
# 或者切换回基础版
from grasp_process import run_grasp_inference, execute_grasp
```

---

## 二、排错历史记录 (Debugging History)

### 2.1 OpenAI 客户端代理报错 (2026-01-17)

**问题现象：** 运行 `main_vlm.py` 时报 `ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7890/')`。
**原因：** 系统中设置了 `socks://` 全局代理，而 `httpx` 库不支持此协议头。

```python
# 【旧代码】(直接初始化，受环境代理干扰)
client = OpenAI(api_key='...', base_url="...")

# 【新代码】(强制清除环境代理，设置 trust_env=False)
import os, httpx
for key in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
    os.environ.pop(key, None)
client = OpenAI(..., http_client=httpx.Client(trust_env=False))
```

### 2.2 抓取角度倾斜导致碰撞 (2026-01-17)

**问题现象：** 夹爪倾斜抓取，导致下侧手指直接撞到桌面，无法抓起贴地的薄片。
**原因：** 原始代码在摄像头坐标系过滤垂直度，受摄像头倾角影响，选出的“垂直”其实是斜的。

```python
# 【旧逻辑】(参考向量不对)
vertical = np.array([0, 0, 1]) # 这是摄像头方向

# 【新逻辑】(利用相机外参 T_wc 计算世界坐标系的正下方)
R_cw = T_wc.R.T
world_down_c = R_cw @ np.array([0, 0, -1]) # 真正的世界正下方
# 增加 Auto-leveling 逻辑，强制旋转夹爪使手指水平。
```

### 2.3 轻拿轻放：放置逻辑优化 (2026-01-17)

**问题现象：** 机器人移动到目标点上方后，会从 30cm 的高处直接松开夹爪，“垂直空投”。
**原因：** 放置点 `T6` 的 Z 坐标错误引用了抬升后的点 `T4`。

```python
# 【旧代码】(垂直空投)
T6 = sm.SE3.Trans(target_pos[0], target_pos[1], T4.t[2]) * sm.SE3(sm.SO3(T_target_high.R))

# 【新代码】(弯下腰放：降至抓取时的原始 Z 高度)
T6 = sm.SE3.Trans(target_pos[0], target_pos[1], T3.t[2]) * sm.SE3(sm.SO3(T_target_high.R))
```

---
## 三、多相机融合与 VLM 抓取综合排错日记 (2026-01-20)

### 3.1 摄像头架构调整
为了解决桌面相机无法覆盖货架盲区的问题，我们引入了侧向相机 `cam_shelf`。
*   **配置**：在 `scene.xml` 中添加 `cam_shelf`，初始位置 `X=2.75`。
*   **调整**：后续为了增加物体像素占比，一度调整至 `X=2.25`，但最终确认 **Prompt 增强** 才是关键，故恢复至 `X=2.75` 以保持视野广度。
*   **代码**：`get_image(camera_name)` 封装了底层 `render` 接口，支持双路图像采集。

### 3.2 点云融合逻辑
单相机点云存在大量盲区（如物体背面），导致 GraspNet 抓取姿态受限。
*   **坐标变换**：桌面相机 (`cam`) 和货架相机 (`cam_shelf`) 的点云分别通过其外参矩阵 `T_wc` 统一转换到 **世界坐标系**。
*   **融合算法**：使用 open3d 将两路点云合并，并进行体素降采样（Voxel Downsample）去重。
*   **裁剪**：利用 VLM 返回的 2D Bounding Box，分别裁剪两路相机的点云，只保留目标物体部分的点云进行融合，大幅降低了背景噪声干扰。

### 3.3 VLM 提示词敏感性与光影干扰（关键问题）
**现象**：
当机械臂静止在原位 (X=0.8) 时，抓取任务成功；当机械臂移动到抓取位 (X=1.0) 后，VLM 突然识别不到“培养皿”，尽管货架相机画面中物体清晰且无遮挡。

**深度分析**：
1.  **隐形依赖**：在原位能够成功，实际上是**桌面相机**识别成功了，而**货架相机**一直识别失败（bbox: [-1,-1,-1,-1]），但系统会自动使用成功的那一路数据，掩盖了货架相机的问题。
2.  **机械臂遮挡**：移动机械臂后，机械臂躯干挡住了**桌面相机**的视线，导致桌面相机“致盲”。
3.  **光影蝴蝶效应**：此时系统全靠货架相机。虽然画面看起来没变，但机械臂的移动改变了场景的全局光照渲染。对于“培养皿”这种**半透明、小体积**的边缘物体，VLM (Qwen) 的识别信心本来就在临界值（Threshold）附近，光影的微小扰动导致识别率从“勉强能认”跌落到“认不出”。

**解决方案（Prompt Engineering）**：
不修改模型，而是通过**提示词增强**来辅助模型。
在 `vlm_process.py` 中，当检测到命令为“培养皿”时，自动追加视觉描述：
`"培养皿 (green cylinder, small container, cup)"`
这相当于给模型提供了“先验知识”，极大地提高了其对模糊目标的召回率。

### 3.4 泛化性验证 (Duck Test)
为了验证这不是针对培养皿的过拟合，我们添加了 **黄色鸭子 (Duck)** 模型到货架上。
*   **结果**：Qwen 在没有任何 Prompt 增强的情况下，直接准确识别出了鸭子。
*   **结论**：证明了货架相机的物理位置是合理的，之前的失败纯粹是因为培养皿的视觉特征太弱，且 Prompt 增强方案是针对困难样本的有效补丁。

### 3.5 逆运动学 (IK) 与机械臂基座调整
*   **问题**：当货架位于世界坐标 X=1.79 时，机械臂原位 (X=0.8) 无法规划出到达深层货架的路径。
*   **调整**：将 `ur5e_base` 和 `mocap` 的位置调整至 **X=1.2, Y=0.6**。此位置处于工作空间中心，使得机械臂能以更自然的姿态触达货架各层，彻底解决了 `Inverse kinematics failed` 报错。

---
*记录人：GG-bond*
*最后更新：2026-01-20*
