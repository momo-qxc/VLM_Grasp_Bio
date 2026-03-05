# VLM_Grasp_Bio-UI

[English](README-EN.md)

基于 MuJoCo + UR5e + VLM + GraspNet 的实验室智能抓取与放置系统。项目面向“自然语言驱动机器人执行”场景，支持从一句口语化指令出发，自动完成目标理解、视觉定位、抓取推理、放置推理与运动执行；在障碍场景下可切换在线 RRT 避障规划，实现复杂环境中的安全搬运与放置。

## 1. 项目概览

本项目构建了一个完整的“语言-视觉-动作”闭环：

1. 指令理解：解析抓取对象、放置描述、是否为放回任务。
2. 视觉感知：多相机搜索目标，VLM 识别 + SAM 分割。
3. 抓取推理：基于深度/点云与 GraspNet 生成抓取姿态。
4. 放置推理：将“显微镜右边”等语义映射为像素点，再转换为世界坐标。
5. 运动执行：根据场景与规划模式自动选择执行链路（普通场景默认执行 / 障碍场景在线 RRT 避障），完成抓取、搬运、放置、回撤与回位。

系统提供两类入口：

- `mujoco_vlm.py`：CLI 交互入口。
- `ui_main_qt.py`：图形界面入口（推荐，支持场景与规划模式切换）。

## 2. 系统执行模式

项目目前支持两条执行链路：

### 2.1 默认执行链路（普通场景）

- 主执行器：`grasp_process_optimized.py`
- 典型场景：普通实验台、无复杂障碍绕行需求
- 特点：流程成熟、执行速度快、适合日常抓放与放回任务

### 2.2 障碍场景在线 RRT 链路

- 主执行器：`obstacle_rrt/online_rrt_executor.py`
- 触发条件（`task_executor.py`）：
  - `scene_name == "障碍场景"`
  - `planner_mode == "RRT避障算法"`
  - 放置点识别成功（`target_pos is not None`）
- 典型场景：显微镜、障碍球、货架等结构造成的窄通道或避障需求
- 核心机制：在线 RRT（规划一段，执行一段，失败即重规划），并结合中转点、救援重规划、分层下降放置策略

### 2.3 场景示意

普通场景（默认执行链路）：

![一般场景](Visual%20results/一般场景.png)

障碍场景（在线 RRT 避障链路）：

![障碍场景](Visual%20results/障碍场景.png)

## 3. 核心能力

- 自然语言任务执行
  - 支持中英文等自然语言输入。
  - 支持“抓取 + 放置”与“放回”语义。
- 模糊语义澄清
  - 对不明确位置描述（如“右边一点”）进行追问与补全。
- 多相机目标搜索
  - 在 `cam`、`cam_2` 等视角依次搜索目标，提升检出鲁棒性。
- 自适应抓取策略
  - 区分桌面/货架目标，自动选择抓取推理与过滤策略。
  - 支持单相机与融合点云推理。
- 语义放置推理
  - 支持参考物体关系（如“显微镜右边”）和颜色区域描述。
  - 自动保存预测可视化结果到 `PredictionResults/`。
- 障碍场景避障执行
  - 在线 RRT 周期性重规划与分段执行。
  - 障碍物自动抽取：障碍球、显微镜碰撞体、货架碰撞体。
  - 近目标下降采用分层候选搜索与避障约束。
- 历史记忆与放回
  - 记录抓取原位与放置结果，支持后续放回指令。

## 4. 项目结构

```text
VLM_Grasp_Bio-UI/
├── mujoco_vlm.py                    # CLI 入口
├── ui_main_qt.py                    # UI 入口（场景/规划切换、日志、画面）
├── task_executor.py                 # 统一任务编排（UI/CLI 共用）
├── vlm_process.py                   # 指令解析、VLM识别、SAM分割、像素->世界坐标
├── grasp_process_optimized.py       # 默认抓取执行链路
├── obstacle_rrt/                    # 障碍场景在线 RRT 模块
│   ├── online_rrt_executor.py       # 在线RRT主执行器
│   ├── obstacle_extractor.py        # MuJoCo几何体到碰撞体构建
│   ├── improved_adapter.py          # improved_rrt_robot 适配
│   └── __init__.py
├── manipulator_grasp/               # 仿真环境与机器人运动学/轨迹规划
│   └── assets/scenes/
│       ├── scene.xml                # 普通场景
│       └── scene_obstacle.xml       # 障碍场景
├── graspnet-baseline/               # GraspNet baseline 与算子
├── model/                           # 模型与测试脚本
├── config.py                        # 模型/API/界面配置
├── workspace_sampler.py             # 工作空间采样与可视化
├── logs/log_rs/checkpoint-rs.tar    # GraspNet 权重（需存在）
├── sam_b.pt                         # SAM 权重（需存在）
├── PredictionResults/               # 运行时放置预测可视化
├── Visual results/                  # README 展示资源
└── video/                           # 演示视频
```

## 5. 安装与环境配置

项目环境参考仓库根目录 `实验环境.txt`。以下是与当前代码匹配的推荐安装流程。

### 5.1 关键版本参考

| 组件 | 版本 |
|------|------|
| Python | 3.11.14 |
| PyTorch | 2.9.1 |
| TorchVision | 0.24.1 |
| MuJoCo | 3.3.0 |
| Open3D | 0.19.0 |
| Ultralytics | 8.3.98 |
| OpenAI SDK | 2.14.0 |
| NumPy | 1.26.4 |
| OpenCV | 4.7.0.72（另有 headless 4.5.5.64） |
| PyQt5 | 5.15.11 |
| spatialmath-python | 1.1.14 |
| roboticstoolbox-python | 1.1.1 |
| modern-robotics | 1.1.1 |
| graspnetapi | 1.2.11 |

### 5.2 安装步骤

1. 创建并激活环境

```bash
conda create -n vlm_graspnet python=3.11 -y
conda activate vlm_graspnet
pip install --upgrade pip
```

2. 安装 GraspNet 基础依赖

```bash
cd graspnet-baseline
pip install -r requirements.txt
```

3. 安装 PyTorch（按本机 CUDA 版本选择）

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio
```

4. 安装机器人与仿真依赖

```bash
pip install spatialmath-python==1.1.14
pip install roboticstoolbox-python==1.1.1
pip install modern-robotics==1.1.1
pip install mujoco==3.3.0
```

5. 编译 GraspNet 自定义算子

```bash
cd pointnet2
python setup.py install

cd ../knn
python setup.py install

cd ../..
```

6. 安装 GraspNet API

```bash
cd graspnet-baseline/graspnetAPI
pip install .
cd ../..
```

7. 安装 UI / VLM / 视觉相关依赖

```bash
pip install pyqt5==5.15.11
pip install open3d==0.19.0
pip install ultralytics==8.3.98
pip install openai==2.14.0 httpx==0.28.1
pip install numpy==1.26.4 pillow
pip install opencv-python==4.7.0.72
```

8. 可选：语音交互依赖

```bash
pip install openai-whisper soundfile sounddevice pydub
```

9. 检查关键权重文件

```text
logs/log_rs/checkpoint-rs.tar
sam_b.pt
```

## 6. 配置说明

### 6.1 VLM 与 API 配置

项目通过 `config.py` 的 `Config` 类管理模型配置（如 `MODELS`、`ACTIVE_MODEL`、`QWEN_*`）。

在 UI 中可以通过“**大模型设置**”界面增删与切换模型，并写回 `config.py`。

### 6.2 RRT 参数配置（障碍场景）

障碍场景在线 RRT 参数集中在 `obstacle_rrt/online_rrt_executor.py` 的 `DEFAULT_RRT_CFG`，包括：

- 基础规划参数：`expand_dis`、`goal_sample_rate`、`max_iter`、`max_cycles`
- 碰撞参数：`obstacle_inflation`、`collision_check_expand_dis`
- 失败恢复参数：`failure_*`、`enable_rescue_replan`、`rescue_*`
- 下降放置参数：`descent_*`
- 姿态稳定参数：`enable_wrist3_lock`、`wrist3_lock_deg` 等

## 7. 快速开始

### 7.1 CLI 模式

```bash
python mujoco_vlm.py
```

输入示例：

- `把培养皿放到显微镜右边`
- `把培养皿放回原处`
- `把培养皿放回货架`

输入 `q` 退出。

说明：CLI 默认调用 `execute_smart_task(user_input)`，默认参数为“普通场景 + 默认算法”。

### 7.2 UI 模式（推荐）

```bash
python ui_main_qt.py
```

UI 提供：

- 场景切换（普通场景 / 障碍场景）
- 规划模式切换（默认算法 / RRT避障算法）
- 聊天式指令输入
- 双视角画面与实时日志
- 模型设置与界面设置

### 7.3 障碍场景 RRT 运行示例

1. 启动 `ui_main_qt.py`。
2. 切换到 `障碍场景`。
3. 规划模式选择 `RRT避障算法`。
4. 输入：`把培养皿放到显微镜右边`。
5. 日志中可观察到 RRT 阶段执行（阶段5~8）。

## 8. 复现建议流程

1. 完成环境安装并确认权重存在。
2. 配置可用 VLM API（`config.py` 或 UI 模型设置）。
3. 运行场景冒烟测试：

```bash
python model/test_lab_equipment.py
```

4. 运行端到端任务：

```bash
python ui_main_qt.py
# 或 python mujoco_vlm.py
```

5. 使用示例指令验证：

```text
把培养皿放到显微镜右边
```

6. 检查输出：

- UI 实时日志与画面是否完整
- `PredictionResults/` 是否生成放置预测标注图

## 9. 实验效果展示

### 9.1 系统界面与配置

交互主界面：

![交互界面](Visual%20results/交互界面.png)

大模型设置界面：

![大模型设置界面](Visual%20results/大模型设置界面.png)

界面设置：

![界面设置](Visual%20results/界面设置.png)

调试日志界面：

![调试日志](Visual%20results/调试日志.png)

### 9.2 感知与空间能力

点云融合效果：

![点云融合效果](Visual%20results/点云融合效果.png)

机械臂工作空间地图：

![workspace_map](Visual%20results/workspace_map.png)

### 9.3 示例任务：“把培养皿放到显微镜右边” / “Put the petri dish to the right of the microscope”

本组结果对应一次完整任务链路：目标识别 -> 放置点预测 -> 抓取执行 -> 放置完成。

正面识别效果（系统预测放置点）：

![正面识别效果](Visual%20results/正面识别效果.jpg)

俯视预测图（系统预测放置点）：

![俯视预测图](Visual%20results/俯视预测图.jpg)

抓取货架物品（机械臂开始抓取）：

![抓取货架物品](Visual%20results/抓取货架物品.png)

MuJoCo 原生界面放置完成：

![放置完成](Visual%20results/放置完成.png)

UI 界面中的放置结果：

![放置结果](Visual%20results/放置结果.png)

### 9.4 演示视频

- [演示效果.webm](video/演示效果.webm)

## 10. 常见问题

1. `checkpoint-rs.tar` 找不到

请确认路径为 `logs/log_rs/checkpoint-rs.tar`。

2. SAM 分割初始化失败

请确认根目录存在 `sam_b.pt`，并已安装 `ultralytics`。

3. UI 启动时报 Qt 库冲突

`ui_main_qt.py` 已包含 Qt 库路径修正与自重启逻辑；若仍报错，请检查 conda/base 环境中的 Qt 动态库冲突。

4. 没有触发 RRT 执行

请确认同时满足以下条件：

- 场景为 `障碍场景`
- 规划模式为 `RRT避障算法`
- 放置点识别成功（存在 `target_pos`）

5. 放置描述导致结果不稳定

尽量给出明确方向和距离（例如“显微镜右边 5 厘米”）。系统会在部分模糊描述下触发澄清提问。

## 11. 致谢

- [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)
- [MuJoCo](https://mujoco.org/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [VLM_Grasp_Interactive](https://github.com/hangtingLiu/VLM_Grasp_Interactive)
