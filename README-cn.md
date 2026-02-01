# VLM_Grasp_Interactive
参考文章进行环境配置
https://blog.csdn.net/agentssl/article/details/148089323

1.配置环境
conda create -n vlm_graspnet python=3.11
conda activate  vlm_graspnet

创建主文件夹
VLM_Grasp_Interactive

主文件夹中移植下面项目的manipulator_grasp操作包（进去找这个文件夹）
git clone https://gitee.com/chaomingsanhua/manipulator_grasp.git

主文件夹中下载graspnet-baseline项目
git clone https://github.com/graspnet/graspnet-baseline.git

打开graspnet-baseline，修改其中的requirements.txt
cd graspnet-baseline
sudo gedit requirements.txt 

保存后执行安装指令
pip install -r requirements.txt

安装torch等所需的环境包
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install spatialmath-python==1.1.14
pip install roboticstoolbox-python==1.1.1
pip install modern-robotics==1.1.1
pip install mujoco==3.3.1

编译并安装PointNet++自定义算子
（PointNet++是处理3D点云的经典神经网络，需要自定义CUDA算子实现高效采样和特征聚合/VoteNet是基于PointNet++的3D目标检测框架）
cd pointnet2
python setup.py install
cd ../

编译并安装基于PyTorch CUDA实现的k-最近邻（k-NN）算子：
- k-NN算子：用于快速计算数据点之间最近邻关系的算法模块
- CUDA加速：利用NVIDIA显卡进行并行计算加速
- PyTorch扩展：为深度学习框架添加自定义C++/CUDA操作
cd knn
python setup.py install
cd ../

安装graspnetAPI工具包
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI

修改setup.py，将sklearn替换为scikit-learn
sudo gedit setup.py

修改后开始安装
pip install .
cd ../

下载GraspNet-baseline的权重(训练好的)

在VLM_Grasp_Interactive文件夹新建logs/log_rs文件夹，将下载好的权重checkpoint-rs.tar放进log_rs


继续完善环境（可以运行代码，缺啥pip啥，版本兼容情况做对应修改）【每个人有自己的玄学体质，后续考虑docker打包】
pip install opencv-python
pip install --force-reinstall "numpy==1.26.4
pip install ultralytics==8.3.98
pip install "opencv-python==4.5.5.64" --force-reinstall
pip install openai-whisper
pip install soundfile
pip install sounddevice
pip install pydub
pip install openai
#进入 graspnetAPI 目录并安装 pip install -e .
pip install --upgrade transforms3d


