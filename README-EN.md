# VLM_Grasp_Bio-UI

[中文](README.md)

An intelligent robotic grasp-and-place system built on MuJoCo + UR5e + VLM + GraspNet. The current version supports **pure natural-language interaction**: the user can issue instructions in Chinese, English, and other natural languages (for example, "Put the petri dish to the right of the microscope"), and the system will complete instruction parsing, target segmentation, grasp pose inference, place-point prediction, and robot execution end to end.

## 1. Project Overview

This project targets lab-scene grasping and placing tasks with an integrated perception-to-action pipeline:

1. Natural-language understanding: parse grasp target and place description.
2. Visual grounding and segmentation: VLM locates the object, SAM generates the mask.
3. Grasp inference: use depth + point cloud with GraspNet to predict grasp pose.
4. Multi-camera fusion: table/side/top/global views improve robustness under occlusion.
5. Place prediction: map descriptions such as "to the right of the microscope" or "left side of the green zone" to image pixels, then to world coordinates.
6. Motion execution: UR5e executes grasp, transfer, placement, and return operations in MuJoCo.

Two interfaces are provided:

- `mujoco_vlm.py`: CLI mode.
- `ui_main_qt.py`: PyQt5 GUI mode (model settings, UI settings, real-time logs, camera feeds).

## 2. Core Capabilities

- Pure natural-language interaction:
  - Supports combined "grasp + place" instructions.
  - Supports return intents such as "return to original position", "return to shelf", and "return to last grasp position".
- Clarification for ambiguous instructions:
  - The system asks follow-up questions for vague descriptions (for example, "a bit to the right" or "nearby").
- Multi-camera object search:
  - Searches targets across `cam`, `cam_2`, and other views to improve detection success.
- Adaptive grasp strategy:
  - Switches filtering logic based on whether the object is on the table or shelf.
  - Supports both single-camera inference and fused multi-camera point-cloud inference.
- Place prediction and visualization:
  - Supports reference-object-based placement (for example, microscope-based) and color-zone-based placement (red/green area).
  - Automatically saves placement visualization images to `PredictionResults/`.
- Position memory for return tasks:
  - Records grasp/place history and uses it for later return instructions.

## 3. Project Structure

```text
VLM_Grasp_Bio-UI/
├── mujoco_vlm.py                 # CLI entry
├── ui_main_qt.py                 # PyQt5 GUI entry
├── task_executor.py              # Shared task pipeline (used by both CLI and UI)
├── vlm_process.py                # Instruction parsing, VLM reasoning, SAM segmentation, pixel-to-world
├── grasp_process_optimized.py    # Grasp inference and execution (fusion, leveling, memory)
├── config.py                     # Model/API/workspace configuration
├── workspace_sampler.py          # Workspace sampling and visualization
├── manipulator_grasp/            # Robot simulation, kinematics, planning, control
├── graspnet-baseline/            # GraspNet baseline and custom ops
├── model/                        # MuJoCo scenes and assets
├── logs/log_rs/checkpoint-rs.tar # GraspNet checkpoint (required)
├── Visual results/               # Images/videos used in README
└── PredictionResults/            # Runtime placement prediction outputs
```

## 4. Installation and Environment Setup

The full runnable environment for this project is recorded in `实验环境.txt` at the repository root. The steps below provide a practical setup flow aligned with the current codebase.

### 4.1 Key package versions (from `实验环境.txt`)

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

### 4.2 Installation steps (recommended)

1. Create and activate the environment

```bash
conda create -n vlm_graspnet python=3.11.14 -y
conda activate vlm_graspnet
pip install --upgrade pip
```

2. Install GraspNet base dependencies

```bash
cd graspnet-baseline
pip install -r requirements.txt
```

3. Install PyTorch (choose by your CUDA setup)

```bash
# Example version aligned with the recorded environment
pip install torch==2.9.1 torchvision==0.24.1 torchaudio
```

4. Install robotics and simulation core dependencies

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

8. Install audio/voice dependencies (optional)

```bash
pip install openai-whisper soundfile sounddevice pydub
```

9. Verify required model files

```text
logs/log_rs/checkpoint-rs.tar
sam_b.pt
```

### 4.3 Environment consistency check (optional)

```bash
python -V
pip show torch torchvision mujoco open3d ultralytics openai pyqt5
conda list > current_env.txt
```

Then compare `current_env.txt` with `实验环境.txt` to identify version mismatches.

## 5. Models and Configuration

### 5.1 GraspNet checkpoint

Make sure this file exists:

```text
logs/log_rs/checkpoint-rs.tar
```

### 5.2 VLM API configuration

The project reads model configuration from `Config` in `config.py` (`MODELS`, `ACTIVE_MODEL`, `QWEN_*`, etc.).

In GUI mode, you can directly add/delete/switch models in the **Large Model Settings** page. Saving writes updates back to `config.py`.

Security recommendation:

- Do not commit real API keys to public repositories.
- Prefer local untracked configs or environment variables for secret management.

## 6. Quick Start

### 6.1 Run in CLI mode

```bash
python mujoco_vlm.py
```

Then enter natural-language instructions (Chinese/English and other languages are supported), for example:

- `Put the petri dish to the right of the microscope`
- `Put the petri dish back to its original position`
- `Put the petri dish back on the shelf`

Type `q` to quit.

### 6.2 Run in GUI mode

```bash
python ui_main_qt.py
```

The GUI includes:

- Chat-style instruction input.
- Real-time camera view panels.
- Status and execution feedback.
- Low-level debug log window.
- Model settings and UI settings pages.

## 7. Reproduction Guide

Recommended validation order:

1. Finish environment and dependency setup.
2. Confirm `logs/log_rs/checkpoint-rs.tar` and `sam_b.pt` exist.
3. Configure a valid VLM API (`config.py` or GUI settings).
4. Run scene smoke test:

```bash
python model/test_lab_equipment.py
```

5. Run end-to-end tasks:

```bash
python mujoco_vlm.py
# or
python ui_main_qt.py
```

6. Validate with this example command:

```text
Put the petri dish to the right of the microscope
```

7. Check outputs:

- Inspect camera and logs in the GUI.
- Check placement prediction visualizations under `PredictionResults/`.

## 8. Experimental Results

### 8.1 System UI and settings

Main interaction UI:

![UI](Visual%20results/交互界面.png)

Large model settings page:

![Large model settings](Visual%20results/大模型设置界面.png)

UI settings page:

![UI settings](Visual%20results/界面设置.png)

Debug log page:

![Debug logs](Visual%20results/调试日志.png)

### 8.2 Perception and workspace capability

Point-cloud fusion result:

![Point cloud fusion](Visual%20results/点云融合效果.png)

Robot workspace map:

![Workspace map](Visual%20results/workspace_map.png)

### 8.3 Example task: "把培养皿放到显微镜右边" / "Put the petri dish to the right of the microscope"

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

### 8.4 Demo videos

- [演示效果.webm](Visual%20results/演示效果.webm)
- [mujoco运行效果.webm](Visual%20results/mujoco运行效果.webm)

## 9. FAQ

1. `checkpoint-rs.tar` not found

Verify the path: `logs/log_rs/checkpoint-rs.tar`.

2. SAM initialization failed

Make sure `sam_b.pt` exists in the project root and `ultralytics` is installed.

3. Qt conflict when launching GUI

`ui_main_qt.py` already includes an LD-library-path fix and self-restart logic. If issues remain, check Qt dynamic library conflicts in your conda/base environment.

4. Unstable placement due to ambiguous language

Use clearer direction/distance expressions, for example: "显微镜右边 5 厘米" or "5 cm to the right of the microscope". The system can ask clarification questions for some ambiguous inputs.

## 10. Acknowledgements

- [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)
- [MuJoCo](https://mujoco.org/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [VLM_Grasp_Interactive](https://github.com/hangtingLiu/VLM_Grasp_Interactive)
