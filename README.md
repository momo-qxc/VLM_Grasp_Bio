# VLM_Grasp_Bio

[中文](README-cn.md)

A vision‑language‑model (VLM) driven robotic grasping and placing demo. The project simulates a UR5e arm in MuJoCo and uses **fully natural language instructions + VLM segmentation + point‑cloud grasp inference** to pick and place target objects. It supports **single‑camera**, **dual‑camera fusion**, and **intelligent placement** modes with end‑to‑end natural language control.

## Features
- **Full natural language control**: complete instructions like "Place the petri dish to the right of the microscope" for end‑to‑end grasping and placement.
- **VLM target grounding**: natural language command → 2D bounding box of the target object.
- **Intelligent placement detection**: VLM identifies placement positions from multi‑camera views based on spatial descriptions.
- **Grasp pose inference**: depth + mask → GraspNet inference.
- **Dual‑camera fusion**: table + shelf cameras merged to improve occlusion robustness.
- **Execution & placing**: auto‑leveling and approach compensation to reduce collisions.

## Structure
- `main_vlm.py`: main entry; interactive mode selection and grasp execution.
- `vlm_process.py`: VLM prompting and segmentation (Qwen/Gemini).
- `grasp_process_optimized.py`: grasp inference and execution (with leveling/compensation).
- `manipulator_grasp/`: UR5e simulation environment and control.
- `graspnet-baseline/`: GraspNet inference code and dependencies.

## Installation
> The versions below are known‑good references; adjust to your system/driver if needed.

1) Create env
```bash
conda create -n vlm_graspnet python=3.11
conda activate vlm_graspnet
```

2) Install GraspNet deps
```bash
cd graspnet-baseline
pip install -r requirements.txt
```

3) Install PyTorch (example: CUDA 11.3)
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

4) Robotics & simulation deps
```bash
pip install spatialmath-python==1.1.14
pip install roboticstoolbox-python==1.1.1
pip install modern-robotics==1.1.1
pip install mujoco==3.3.1
```

5) Build PointNet++ and k‑NN ops
```bash
cd graspnet-baseline/pointnet2
python setup.py install
cd ../knn
python setup.py install
cd ../..
```

6) Install GraspNet API
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
# Replace sklearn with scikit-learn in setup.py
pip install .
cd ..
```

7) Extra dependencies
```bash
pip install opencv-python
pip install --force-reinstall "numpy==1.26.4"
pip install ultralytics==8.3.98
pip install "opencv-python==4.5.5.64" --force-reinstall
pip install openai-whisper
pip install soundfile
pip install sounddevice
pip install pydub
pip install openai
pip install --upgrade transforms3d
```

8) GraspNet weights
Place `checkpoint-rs.tar` at:
```
logs/log_rs/checkpoint-rs.tar
```

9) VLM keys
- Qwen/OpenAI compatible endpoint: set `api_key` and `base_url` in `vlm_process.py`.
- Gemini: set `genai.Client(api_key=...)` in `vlm_process.py`.

## Run
```bash
python main_vlm.py
```
Select a mode:
- `1`: single camera (table `cam`)
- `2`: single camera (shelf `cam_shelf`)
- `3`: dual‑camera fusion (recommended for occluded scenes)
- `4`: **intelligent placement mode** 🔥 - full natural language control for grasping and placing

### Mode 4: Intelligent Placement
In this mode, you can provide complete natural language instructions such as:
- "Place the petri dish to the right of the microscope"
- "Move the test tube to the upper left corner of the table"
- "Grasp the beaker and put it in the red area"

The system will:
1. Parse the instruction to extract grasp target and placement description
2. Use VLM to segment the target object from the table camera
3. Infer grasp pose using GraspNet
4. Use multi‑camera views (global cameras) to identify the placement position
5. Execute the complete pick‑and‑place task

In fusion mode (mode 3), you will be prompted for a **natural language command** for the target object, then VLM segmentation and grasp inference run.

## Demo

UI snapshot (Mode selection):

![UI](Visual%20results/交互界面.png)

UI snapshot (Intelligent placement mode):

![UI2](Visual%20results/交互界面2.png)

Fused point cloud:

![Fusion](Visual%20results/融合点云效果.png)

Grasping:

![Grasp](Visual%20results/抓取物品.png)

Pre‑place:

![Pre‑place](Visual%20results/准备放置.png)

Placed:

![Placed](Visual%20results/放置完成.png)

Video:

<video src="Visual%20results/video.webm" controls></video>

## Reference
- Environment setup reference: `https://blog.csdn.net/agentssl/article/details/148089323`
