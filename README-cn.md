# VLM_Grasp_Bio

[English](README.md)

基于多模态视觉语言模型（VLM）的机械臂抓取与放置演示工程。项目在 MuJoCo 中模拟 UR5e 机械臂，通过 **文本指令 + 视觉分割 + 点云抓取推理** 完成目标物体的抓取与放置；支持 **单相机** 与 **双相机点云融合** 两种模式。

## 功能概览
- **VLM 目标定位**：输入自然语言指令，VLM 输出目标物体的 2D 边界框。
- **抓取姿态推理**：结合深度图与掩码，在 GraspNet 上推理抓取姿态。
- **双相机点云融合**：桌面相机 + 货架相机融合，提升遮挡场景的抓取成功率。
- **抓取执行与放置**：带自动调平与接近补偿的抓取流程，避免夹爪倾斜碰撞。

## 代码结构
- `main_vlm.py`：主入口，交互式选择单相机/双相机融合模式并执行抓取。
- `vlm_process.py`：VLM 分割与指令解析（Qwen/Gemini）。
- `grasp_process_optimized.py`：抓取推理与执行（含调平与补偿逻辑）。
- `manipulator_grasp/`：UR5e 机械臂仿真环境与运动控制。
- `graspnet-baseline/`：GraspNet 推理代码与依赖。

## 安装与环境配置
> 以下步骤为一套可用参考，若本机驱动/系统不同，可按需调整版本。

1) 创建环境
```bash
conda create -n vlm_graspnet python=3.11
conda activate vlm_graspnet
```

2) 安装 GraspNet 相关依赖
```bash
cd graspnet-baseline
pip install -r requirements.txt
```

3) 安装 PyTorch（示例：CUDA 11.3）
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

4) 安装机器人与仿真依赖
```bash
pip install spatialmath-python==1.1.14
pip install roboticstoolbox-python==1.1.1
pip install modern-robotics==1.1.1
pip install mujoco==3.3.1
```

5) 编译 PointNet++ 与 k-NN 算子
```bash
cd graspnet-baseline/pointnet2
python setup.py install
cd ../knn
python setup.py install
cd ../..
```

6) 安装 GraspNet API
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
# 将 setup.py 中的 sklearn 替换为 scikit-learn
pip install .
cd ..
```

7) 其他依赖
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

8) 准备 GraspNet 权重
将 `checkpoint-rs.tar` 放到以下路径：
```
logs/log_rs/checkpoint-rs.tar
```

9) 配置 VLM Key
- Qwen/OpenAI 兼容接口：在 `vlm_process.py` 中设置 `api_key` 与 `base_url`。
- Gemini：在 `vlm_process.py` 中设置 `genai.Client(api_key=...)`。

## 运行流程
```bash
python main_vlm.py
```
运行后可选择模式：
- `1`：单相机（桌面相机 `cam`）
- `2`：单相机（货架相机 `cam_shelf`）
- `3`：双相机点云融合（推荐）

融合模式下会提示输入目标物体的 **自然语言指令**，随后进行 VLM 分割与抓取推理。

## 演示效果

交互界面：

![交互界面](Visual%20results/交互界面.png)

双相机点云融合效果：

![融合点云效果](Visual%20results/融合点云效果.png)

抓取过程：

![抓取物品](Visual%20results/抓取物品.png)

准备放置：

![准备放置](Visual%20results/准备放置.png)

放置完成：

![放置完成](Visual%20results/放置完成.png)

视频演示：

<video src="Visual%20results/video.webm" controls></video>

## 参考
- 环境配置参考：`https://blog.csdn.net/agentssl/article/details/148089323`
