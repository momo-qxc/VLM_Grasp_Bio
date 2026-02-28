# VLM_Grasp_Bio-UI

[English](README-EN.md)

基于 MuJoCo + UR5e + VLM + GraspNet 的智能抓取与放置系统。当前版本支持**纯自然语言交互**：用户可以直接输入中文或英文等自然语言指令（例如“把培养皿放到显微镜右边”），系统即可自动完成指令解析、目标分割、抓取姿态推理、放置点预测与机械臂执行。

## 1. 项目介绍

本项目围绕实验室场景下的抓取与放置任务，构建了一个“感知-推理-执行”一体化流程：

1. 自然语言理解：解析抓取目标与放置描述。
2. 视觉定位与分割：VLM 找目标，SAM 生成掩码。
3. 抓取推理：结合深度图与点云，使用 GraspNet 预测抓取姿态。
4. 多相机融合：桌面/侧视/俯视/全局视角协同，提高遮挡场景鲁棒性。
5. 放置预测：将“显微镜右边”“绿色区域偏左”等描述落到像素点，再映射到世界坐标。
6. 运动执行：UR5e 在 MuJoCo 中完成抓取、搬运、放置与放回任务。

系统提供两种使用方式：

- `mujoco_vlm.py`：CLI（命令行）模式。
- `ui_main_qt.py`：PyQt5 图形界面模式（包含模型设置、界面设置、实时日志与相机画面）。

## 2. 核心能力

- 纯自然语言交互：
  - 支持“抓取 + 放置”组合指令。
  - 支持“放回原处 / 放回货架 / 放回上次位置”等放回语义。
- 指令澄清机制：
  - 对“右边一点”“旁边”等模糊描述自动追问，减少错误执行。
- 多相机目标搜索：
  - 先后在 `cam`、`cam_2` 等视角中搜索目标，提高检出率。
- 抓取策略自适应：
  - 根据目标在“桌面/货架”位置切换抓取过滤逻辑。
  - 支持单相机推理与多相机点云融合推理。
- 放置点预测与可视化：
  - 支持基于参考物体（如显微镜）和颜色区域（红/绿区域）定位放置点。
  - 自动在 `PredictionResults/` 保存放置预测标注图。
- 历史位置记忆：
  - 记录物体抓取/放置历史，支持后续“放回”指令。

## 3. 项目结构

```text
VLM_Grasp_Bio-UI/
├── mujoco_vlm.py                 # CLI 入口
├── ui_main_qt.py                 # PyQt5 图形界面入口
├── task_executor.py              # 共享任务执行主流程（UI/CLI 共用）
├── vlm_process.py                # 指令解析、VLM识别、SAM分割、像素转世界坐标
├── grasp_process_optimized.py    # 抓取推理与执行（融合、调平、放回记忆）
├── config.py                     # 模型/API/工作空间配置
├── workspace_sampler.py          # 工作空间采样与可视化
├── manipulator_grasp/            # 机械臂仿真、运动学、规划控制
├── graspnet-baseline/            # GraspNet baseline 与自定义算子
├── model/                        # MuJoCo 场景与物体模型
├── logs/log_rs/checkpoint-rs.tar # GraspNet 权重（需存在）
├── Visual results/               # README 展示图片与视频
└── PredictionResults/            # 运行时保存的放置预测结果
```

## 4. 安装与环境配置

本项目当前可用环境已记录在仓库根目录 `实验环境.txt`。下面给出一套与当前代码匹配的安装流程，并附关键版本参考。

### 4.1 实验环境关键版本（摘自 `实验环境.txt`）

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

### 4.2 安装步骤（推荐）

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

3. 安装 PyTorch（请按本机 CUDA 版本选择）

```bash
# 示例：与实验环境一致的版本号
pip install torch==2.9.1 torchvision==0.24.1 torchaudio
```

4. 安装机器人与仿真核心依赖

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

8. 安装语音与音频依赖（如需语音交互）

```bash
pip install openai-whisper soundfile sounddevice pydub
```

9. 检查模型与权重文件

```text
logs/log_rs/checkpoint-rs.tar
sam_b.pt
```

### 4.3 与 `实验环境.txt` 对齐校验（可选）

```bash
python -V
pip show torch torchvision mujoco open3d ultralytics openai pyqt5
conda list > current_env.txt
```

然后将 `current_env.txt` 与仓库中的 `实验环境.txt` 对比，排查版本差异。

## 5. 模型与配置

### 5.1 GraspNet 权重

请确认以下文件存在：

```text
logs/log_rs/checkpoint-rs.tar
```

### 5.2 VLM API 配置

当前项目通过 `config.py` 的 `Config` 类读取模型配置（`MODELS`、`ACTIVE_MODEL`、`QWEN_*` 等）。

图形界面中可直接在“**大模型设置界面**”增删/切换模型配置，保存后会写回 `config.py`。


## 6. 快速开始

### 6.1 CLI 运行

```bash
python mujoco_vlm.py
```

运行后直接输入自然语言指令（支持中英等多语言），例如：

- `把培养皿放到显微镜右边` / `Put the petri dish to the right of the microscope`
- `把培养皿放回原处` / `Put the petri dish back to its original position`
- `把培养皿放回货架` / `Put the petri dish back on the shelf`

输入 `q` 退出。

### 6.2 UI 运行

```bash
python ui_main_qt.py
```

UI 包含：

- 指令输入与聊天式交互区。
- 实时相机画面区。
- 状态与执行反馈。
- 底层调试日志窗口。
- 模型设置与界面设置窗口。

## 7. 复现指南

建议按以下顺序复现：

1. 环境与依赖安装完成。
2. 确认 `logs/log_rs/checkpoint-rs.tar` 和 `sam_b.pt` 存在。
3. 配置可用的 VLM API（`config.py` 或 UI 设置）。
4. 执行场景冒烟测试：

```bash
python model/test_lab_equipment.py
```

5. 执行端到端任务：

```bash
python mujoco_vlm.py
# 或
python ui_main_qt.py
```

6. 使用示例指令验证：

```text
把培养皿放到显微镜右边
```

7. 检查输出：

- UI 中查看相机画面与日志。
- `PredictionResults/` 中查看放置预测标注图。

## 8. 实验效果展示

### 8.1 系统界面与配置

交互主界面：

![交互界面](Visual%20results/交互界面.png)

大模型设置界面：

![大模型设置界面](Visual%20results/大模型设置界面.png)

界面设置：

![界面设置](Visual%20results/界面设置.png)

调试日志界面：

![调试日志](Visual%20results/调试日志.png)

### 8.2 感知与空间能力

点云融合效果：

![点云融合效果](Visual%20results/点云融合效果.png)

机械臂工作空间地图：

![workspace_map](Visual%20results/workspace_map.png)

### 8.3 示例任务：“把培养皿放到显微镜右边” / “Put the petri dish to the right of the microscope”

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

### 8.4 演示视频

- [演示效果.webm](video/演示效果.webm)

## 9. 常见问题

1. `checkpoint-rs.tar` 找不到

请确认路径为 `logs/log_rs/checkpoint-rs.tar`。

2. SAM 分割初始化失败

请确认根目录存在 `sam_b.pt`，并已安装 `ultralytics`。

3. UI 启动时 Qt 库冲突

`ui_main_qt.py` 已包含 Qt 库路径修正与自重启逻辑；若仍报错，请检查 conda/base 环境中的 Qt 动态库冲突。

4. 放置描述过于模糊导致结果不稳定

尽量给出明确方向和距离，例如“显微镜右边 5 厘米”。系统也会在部分模糊描述下自动追问澄清。

## 10. 致谢

- [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)
- [MuJoCo](https://mujoco.org/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [VLM_Grasp_Interactive](https://github.com/hangtingLiu/VLM_Grasp_Interactive)
