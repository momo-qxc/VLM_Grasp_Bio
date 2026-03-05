# VLM_Grasp_Bio-UI

[中文](README.md)

An intelligent lab robotic grasp-and-place system built on MuJoCo + UR5e + VLM + GraspNet. The project targets natural-language-driven robot execution: from one user instruction, it completes target understanding, visual localization, grasp inference, place inference, and motion execution. In obstacle scenes, it can switch to online RRT obstacle-avoidance planning for safer transfer and placement in cluttered environments.

## 1. Project Overview

This project provides a complete Language-Vision-Action loop:

1. Instruction understanding: parse grasp target, place description, and return intent.
2. Visual perception: multi-camera target search, VLM recognition, and SAM segmentation.
3. Grasp inference: generate grasp poses with depth/point cloud and GraspNet.
4. Place inference: map relational language such as "to the right of the microscope" to image pixels, then to world coordinates.
5. Motion execution: select execution pipeline by scene and planner mode, then perform grasp, transfer, placement, retreat, and return.

Two interfaces are provided:

- `mujoco_vlm.py`: CLI entry.
- `ui_main_qt.py`: GUI entry (recommended, supports scene and planner-mode switching).

## 2. System Execution Modes

The project currently supports two execution pipelines:

### 2.1 Default Pipeline (Normal Scene)

- Main executor: `grasp_process_optimized.py`
- Typical use: standard bench tasks without complex obstacle detours
- Characteristics: mature flow, faster execution, suitable for daily grasp/place and return tasks

### 2.2 Online RRT Pipeline (Obstacle Scene)

- Main executor: `obstacle_rrt/online_rrt_executor.py`
- Activation conditions (in `task_executor.py`):
  - `scene_name == "障碍场景"`
  - `planner_mode == "RRT避障算法"`
  - place target is successfully recognized (`target_pos is not None`)
- Typical use: narrow passages or constrained motion around microscope, obstacle spheres, shelf structures, etc.
- Core mechanism: online RRT (plan one segment -> execute one segment -> replan on failure), with transit waypoints, rescue replanning, and layered descent placement strategy

### 2.3 Scene Preview

Normal scene (default pipeline):

![General Scene](Visual%20results/一般场景.png)

Obstacle scene (online RRT pipeline):

![Obstacle Scene](Visual%20results/障碍场景.png)

## 3. Core Capabilities

- Natural-language task execution
  - Supports Chinese/English and other natural-language inputs.
  - Supports both "grasp + place" and "return" task semantics.
- Clarification for ambiguous language
  - Prompts follow-up questions for vague placement descriptions.
- Multi-camera target search
  - Sequential target search across `cam`, `cam_2`, and related views for better robustness.
- Adaptive grasp strategy
  - Automatically switches strategy for table vs. shelf objects.
  - Supports both single-camera and fused point-cloud inference.
- Semantic place inference
  - Supports reference-object relations (for example, "right of microscope") and color-region constraints.
  - Saves placement prediction visualizations to `PredictionResults/`.
- Obstacle-scene avoidance execution
  - Online RRT with cycle-based replanning and segmented execution.
  - Automatic obstacle extraction: obstacle spheres, microscope collision body, shelf collision body.
  - Layered near-target descent with collision-aware candidate selection.
- Memory-assisted return tasks
  - Records grasp origin and placement history for subsequent return instructions.

## 4. Project Structure

```text
VLM_Grasp_Bio-UI/
├── mujoco_vlm.py                    # CLI entry
├── ui_main_qt.py                    # GUI entry (scene/planner switching, logs, camera views)
├── task_executor.py                 # Unified task orchestration (shared by UI/CLI)
├── vlm_process.py                   # Instruction parsing, VLM recognition, SAM segmentation, pixel->world
├── grasp_process_optimized.py       # Default execution pipeline
├── obstacle_rrt/                    # Obstacle-scene online RRT module
│   ├── online_rrt_executor.py       # Online RRT main executor
│   ├── obstacle_extractor.py        # MuJoCo geometry -> collision primitives
│   ├── improved_adapter.py          # improved_rrt_robot adapter
│   └── __init__.py
├── manipulator_grasp/               # Simulation environment, kinematics, trajectory planning
│   └── assets/scenes/
│       ├── scene.xml                # Normal scene
│       └── scene_obstacle.xml       # Obstacle scene
├── graspnet-baseline/               # GraspNet baseline and custom ops
├── model/                           # Models and test scripts
├── config.py                        # Model/API/UI configuration
├── workspace_sampler.py             # Workspace sampling and visualization
├── logs/log_rs/checkpoint-rs.tar    # GraspNet checkpoint (required)
├── sam_b.pt                         # SAM checkpoint (required)
├── PredictionResults/               # Runtime place-prediction visualizations
├── Visual results/                  # README assets
└── video/                           # Demo videos
```

## 5. Installation and Environment Setup

Environment details are documented in `实验环境.txt` at the repository root. The following setup is aligned with the current codebase.

### 5.1 Key Version Reference

| Component | Version |
|------|------|
| Python | 3.11.14 |
| PyTorch | 2.9.1 |
| TorchVision | 0.24.1 |
| MuJoCo | 3.3.0 |
| Open3D | 0.19.0 |
| Ultralytics | 8.3.98 |
| OpenAI SDK | 2.14.0 |
| NumPy | 1.26.4 |
| OpenCV | 4.7.0.72 (plus headless 4.5.5.64) |
| PyQt5 | 5.15.11 |
| spatialmath-python | 1.1.14 |
| roboticstoolbox-python | 1.1.1 |
| modern-robotics | 1.1.1 |
| graspnetapi | 1.2.11 |

### 5.2 Installation Steps

1. Create and activate environment

```bash
conda create -n vlm_graspnet python=3.11 -y
conda activate vlm_graspnet
pip install --upgrade pip
```

2. Install GraspNet base dependencies

```bash
cd graspnet-baseline
pip install -r requirements.txt
```

3. Install PyTorch (choose based on your CUDA setup)

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio
```

4. Install robotics and simulation dependencies

```bash
pip install spatialmath-python==1.1.14
pip install roboticstoolbox-python==1.1.1
pip install modern-robotics==1.1.1
pip install mujoco==3.3.0
```

5. Build GraspNet custom operators

```bash
cd pointnet2
python setup.py install

cd ../knn
python setup.py install

cd ../..
```

6. Install GraspNet API

```bash
cd graspnet-baseline/graspnetAPI
pip install .
cd ../..
```

7. Install UI / VLM / vision dependencies

```bash
pip install pyqt5==5.15.11
pip install open3d==0.19.0
pip install ultralytics==8.3.98
pip install openai==2.14.0 httpx==0.28.1
pip install numpy==1.26.4 pillow
pip install opencv-python==4.7.0.72
```

8. Optional: audio/voice dependencies

```bash
pip install openai-whisper soundfile sounddevice pydub
```

9. Verify required model files

```text
logs/log_rs/checkpoint-rs.tar
sam_b.pt
```

## 6. Configuration

### 6.1 VLM and API Configuration

The project reads model settings from `Config` in `config.py` (`MODELS`, `ACTIVE_MODEL`, `QWEN_*`, etc.).

In GUI mode, you can add/remove/switch models in **Large Model Settings**, then save back to `config.py`.

### 6.2 RRT Configuration (Obstacle Scene)

Online RRT parameters are centralized in `DEFAULT_RRT_CFG` inside `obstacle_rrt/online_rrt_executor.py`, including:

- Basic planning params: `expand_dis`, `goal_sample_rate`, `max_iter`, `max_cycles`
- Collision params: `obstacle_inflation`, `collision_check_expand_dis`
- Failure recovery params: `failure_*`, `enable_rescue_replan`, `rescue_*`
- Descent placement params: `descent_*`
- Pose-stability params: `enable_wrist3_lock`, `wrist3_lock_deg`, etc.

## 7. Quick Start

### 7.1 CLI Mode

```bash
python mujoco_vlm.py
```

Example instructions:

- `Put the petri dish to the right of the microscope`
- `Put the petri dish back to its original position`
- `Put the petri dish back on the shelf`

Type `q` to quit.

Note: CLI calls `execute_smart_task(user_input)` with default arguments (`normal scene + default planner`).

### 7.2 GUI Mode (Recommended)

```bash
python ui_main_qt.py
```

GUI features:

- Scene switching (`普通场景` / `障碍场景`)
- Planner-mode switching (`默认算法` / `RRT避障算法`)
- Chat-style command input
- Dual camera panels and realtime logs
- Model settings and UI settings

### 7.3 Obstacle-Scene RRT Example

1. Launch `ui_main_qt.py`.
2. Switch to `障碍场景`.
3. Select planner mode `RRT避障算法`.
4. Enter: `把培养皿放到显微镜右边`.
5. Observe RRT stage execution in logs (Stage 5-8).

## 8. Reproduction Workflow

1. Finish installation and verify required checkpoints.
2. Configure a valid VLM API (`config.py` or GUI settings).
3. Run scene smoke test:

```bash
python model/test_lab_equipment.py
```

4. Run end-to-end tasks:

```bash
python ui_main_qt.py
# or python mujoco_vlm.py
```

5. Validate with example instruction:

```text
Put the petri dish to the right of the microscope
```

6. Check outputs:

- Realtime logs and camera views in GUI
- Placement visualization files in `PredictionResults/`

## 9. Experimental Results

### 9.1 System UI and Settings

Main interaction UI:

![UI](Visual%20results/交互界面.png)

Large model settings page:

![Large model settings](Visual%20results/大模型设置界面.png)

UI settings page:

![UI settings](Visual%20results/界面设置.png)

Debug log page:

![Debug logs](Visual%20results/调试日志.png)

### 9.2 Perception and Workspace Capability

Point-cloud fusion result:

![Point cloud fusion](Visual%20results/点云融合效果.png)

Robot workspace map:

![Workspace map](Visual%20results/workspace_map.png)

### 9.3 Example Task: "把培养皿放到显微镜右边" / "Put the petri dish to the right of the microscope"

This set corresponds to one complete pipeline: target recognition -> place-point prediction -> grasp execution -> placement completion.

Front-view prediction:

![Front prediction](Visual%20results/正面识别效果.jpg)

Top-view prediction:

![Top prediction](Visual%20results/俯视预测图.jpg)

Shelf object grasp start:

![Shelf grasp](Visual%20results/抓取货架物品.png)

Placement completed in native MuJoCo viewer:

![Placed in MuJoCo](Visual%20results/放置完成.png)

Final placement result in the GUI:

![Placed in GUI](Visual%20results/放置结果.png)

### 9.4 Demo Video

- [Demo](video/演示效果.webm)

## 10. FAQ

1. `checkpoint-rs.tar` not found

Verify path: `logs/log_rs/checkpoint-rs.tar`.

2. SAM initialization failed

Make sure `sam_b.pt` exists in the project root and `ultralytics` is installed.

3. Qt conflict when launching GUI

`ui_main_qt.py` includes Qt library path adjustment and self-restart logic. If the issue remains, check Qt dynamic-library conflicts in your conda/base environment.

4. RRT pipeline is not triggered

Make sure all conditions are met:

- scene is `障碍场景`
- planner mode is `RRT避障算法`
- place target is detected successfully (`target_pos` exists)

5. Placement result is unstable due to vague language

Use explicit direction and distance (for example, "5 cm to the right of the microscope"). The system may ask clarification questions for ambiguous descriptions.

## 11. Acknowledgements

- [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)
- [MuJoCo](https://mujoco.org/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [VLM_Grasp_Interactive](https://github.com/hangtingLiu/VLM_Grasp_Interactive)
