import os
import sys
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import spatialmath as sm

from manipulator_grasp.arm.motion_planning import *

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


# ==================== 点云融合工具 ====================
def fuse_point_clouds(clouds_world, colors_list, T_wc_primary, voxel_size=0.005):
    """
    融合多个世界坐标系点云，并变换回主相机坐标系供 GraspNet 使用。
    
    参数:
    clouds_world: list of np.ndarray, 每个元素是 (N, 3) 的世界坐标点云
    colors_list: list of np.ndarray, 每个元素是 (N, 3) 的颜色
    T_wc_primary: sm.SE3, 主相机的世界到相机变换（用于最终输出）
    voxel_size: 下采样体素大小
    
    返回:
    cloud_cam: np.ndarray, 融合后的相机坐标系点云
    colors_cam: np.ndarray, 对应的颜色
    cloud_o3d: o3d.geometry.PointCloud, 用于可视化的 Open3D 点云
    """
    # 合并所有点云
    all_points = np.vstack(clouds_world)
    all_colors = np.vstack(colors_list)
    
    # 创建 Open3D 点云用于下采样
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 体素下采样去除重复点
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    cloud_world = np.asarray(pcd_down.points)
    colors = np.asarray(pcd_down.colors)
    
    # 变换回主相机坐标系 (T_cw = T_wc^-1)
    T_cw = T_wc_primary.inv()
    R_cw = T_cw.R
    t_cw = T_cw.t
    
    cloud_cam = (R_cw @ cloud_world.T).T + t_cw
    
    # 创建用于可视化的点云（相机坐标系）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_cam)
    cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    
    return cloud_cam, colors, cloud_o3d


def transform_cloud_to_world(cloud_cam, T_wc):
    """将相机坐标系点云变换到世界坐标系"""
    R = T_wc.R
    t = T_wc.t
    return (R @ cloud_cam.T).T + t




# ==================== 网络加载 ====================
def get_net():
    """
    加载训练好的 GraspNet 模型
    """
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.05, 
                   hmin=-0.02, 
                   hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net




# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path, fovy=np.pi/4):
    """
    根据给定的 RGB 图、深度图、掩码图，生成输入点云及其它必要数据
    fovy: 垂直视场角 (弧度)
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    height = color.shape[0]
    width = color.shape[1]
    # fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask = depth < 2.0
    # mask = (workspace_mask > 0) & (depth < 2.0)
    mask = (workspace_mask > 0) & (depth < 2.0) & (depth > 0.1) 
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 5000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    # end_points = {'point_clouds': cloud_sampled}

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d



# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, sam_mask_path=None, T_wc=None, fovy=np.pi/4):
    # 1. 加载网络
    net = get_net()

    # 2. 处理数据，此处使用返回的工作空间掩码路径
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, sam_mask_path, fovy=fovy)
    
    # 2.1 获取相机外参
    if T_wc is None:
        # 默认值 (当外部未传递时使用旧的硬编码值作为回退)
        n_wc = np.array([0.0, -1.0, 0.0]) 
        o_wc = np.array([-1.0, 0.0, -0.5]) 
        t_wc = np.array([0.85, 0.8, 1.6]) 
        T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    
    R_wc = T_wc.R
    R_cw = R_wc.T # 相机坐标系相对于世界坐标系的旋转
    
    # 计算世界坐标系下的“向上”向量在相机坐标系中的投影
    world_up_w = np.array([0, 0, 1])
    world_up_c = R_cw @ world_up_w # 相机视角里“天顶”的方向
    
    # 计算世界坐标系下的“垂直向下”向量在相机坐标系中的投影 (抓取接近方向)
    world_down_w = np.array([0, 0, -1])
    world_down_c = R_cw @ world_down_w # 相机视角里“正下方”的方向

    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 5. 碰撞检测
    COLLISION_THRESH = 0.01
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        collision_thresh = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    # 6. NMS 去重 + 按照得分排序（降序）
    gg.nms().sort_by_score()

    # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
    angle_threshold = np.deg2rad(30)  # 30度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第一列 (approach)
        approach_dir_c = grasp.rotation_matrix[:, 0]
        # 计算夹角：衡量接近方向与“真·世界垂直向下”方向的偏差
        cos_angle = np.dot(approach_dir_c, world_down_c)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    else:
        print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    # # ===== 新增：利用 SAM 生成的目标掩码过滤抓取预测（投影到图像坐标判断） =====
    if sam_mask_path is not None:
        # 加载 SAM 目标掩码
        if isinstance(sam_mask_path, str):
            sam_mask = np.array(Image.open(sam_mask_path))
        elif isinstance(sam_mask_path, np.ndarray):
            sam_mask = sam_mask_path
        else:
            raise TypeError("sam_mask_path 既不是字符串路径也不是 NumPy 数组！")
        # 假定 SAM 掩码与颜色图尺寸一致（640x640）
        height, width = sam_mask.shape[:2]
        # 动态计算相机内参
        focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（像素单位）
        cx = width / 2.0   # 光心 X 坐标（图像中心）
        cy = height / 2.0  # 光心 Y 坐标（图像中心）

        sam_filtered = []
        for grasp in filtered:
            # grasp.translation 为摄像头坐标系下的 3D 坐标 [X, Y, Z]
            X, Y, Z = grasp.translation
            if Z <= 0:
                continue
            u = focal * X / Z + cx
            v = focal * Y / Z + cy
            u_int = int(round(u))
            v_int = int(round(v))
            # 检查投影点是否在图像范围内（640x640）
            if u_int < 0 or u_int >= 640 or v_int < 0 or v_int >= 640:
                continue
            # 若 SAM 掩码中该像素有效（非0），则保留
            if sam_mask[v_int, u_int] > 0:
                sam_filtered.append(grasp)
        if len(sam_filtered) == 0:
            print("\n[Warning] No grasp predictions fall inside the SAM mask. Using previous predictions.")
        else:
            print(f"\n[DEBUG] Filtered {len(sam_filtered)} grasps inside the SAM mask out of {len(filtered)} predictions.")
            filtered = sam_filtered

    # ===== 新增部分：计算物体中心点 =====
    # 使用点云计算物体的中心点
    points = np.asarray(cloud_o3d.points)
    object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    # 计算每个抓取位姿中心点与物体中心点的距离
    distances = []
    for grasp in filtered:
        grasp_center = grasp.translation
        distance = np.linalg.norm(grasp_center - object_center)
        distances.append(distance)

    # 创建一个新的排序列表，包含距离和抓取对象
    grasp_with_distances = [(g, d) for g, d in zip(filtered, distances)]
    
    # 按距离升序排序（距离越小越好）
    grasp_with_distances.sort(key=lambda x: x[1])
    
    # 提取排序后的抓取列表
    filtered = [g for g, d in grasp_with_distances]

    # ===== 新增部分：综合得分和距离进行最终排序 =====
    # 创建一个新的排序列表，包含综合得分和抓取对象
    # 综合得分 = 抓取得分 * 0.7 + (1 - 距离/最大距离) * 0.3
    max_distance = max(distances) if distances else 1.0
    grasp_with_composite_scores = []

    for g, d in grasp_with_distances:
        # 归一化距离分数（距离越小分数越高）
        distance_score = 1 - (d / max_distance)
        
        # 综合得分 = 抓取得分 * 权重1 + 距离得分 * 权重2
        composite_score = g.score * 0.1 + distance_score * 0.9
        # print(f"\n g.score = {g.score}, distance_score = {distance_score}")
        grasp_with_composite_scores.append((g, composite_score))

    # 按综合得分降序排序
    grasp_with_composite_scores.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的抓取列表
    filtered = [g for g, score in grasp_with_composite_scores]


    # # 对过滤后的抓取根据 score 排序（降序）
    # filtered.sort(key=lambda g: g.score, reverse=True)

    # 取第1个抓取
    top_grasps = filtered[:1]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]

    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    
    # ===== 新增：抓取头自动调平 (Auto-leveling) =====
    # 目的：让夹爪的横梁（binormal）在世界坐标系下保持水平，避免倾斜撞击桌面
    # 抓取坐标系定义: x=approach, y=binormal(手指连线)
    a_c = best_grasp.rotation_matrix[:, 0] # 接近方向 (保持不变)
    
    # 我们希望新的 binormal (y_c) 垂直于世界 Z 轴 (world_up_c) 和 接近方向 (a_c)
    # y_c_new = normalize(cross(world_up_c, a_c))
    y_c_new = np.cross(world_up_c, a_c)
    norm = np.linalg.norm(y_c_new)
    if norm > 1e-6:
        y_c_new /= norm
        z_c_new = np.cross(a_c, y_c_new)
        z_c_new /= np.linalg.norm(z_c_new)
        # 更新旋转矩阵，实现“调平”旋转
        best_grasp.rotation_matrix = np.column_stack([a_c, y_c_new, z_c_new])
        print("✅ 已自动执行抓取头调平优化 (Orientation Auto-leveled)")

    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取

    visual = True
    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    return new_gg

    #return best_translation, best_rotation, best_width


# ==================== 多相机点云融合抓取推理 ====================
def run_grasp_inference_fused(camera_data_list, T_wc_primary, fovy_primary):
    """
    多相机点云融合抓取推理。
    
    参数:
    camera_data_list: list of dict, 每个dict包含:
        - 'color': np.ndarray, RGB图像
        - 'depth': np.ndarray, 深度图
        - 'mask': np.ndarray, SAM分割掩码
        - 'T_wc': sm.SE3, 相机的世界到相机变换
        - 'fovy': float, 垂直视场角（弧度）
    T_wc_primary: sm.SE3, 主相机的变换（用于输出抓取姿态）
    fovy_primary: float, 主相机的fovy
    
    返回:
    gg: GraspGroup, 最佳抓取
    """
    print("\n[FUSION] 开始多相机点云融合...")
    
    clouds_world = []
    colors_list = []
    
    # 从每个相机生成点云并变换到世界坐标系
    for i, cam_data in enumerate(camera_data_list):
        color = cam_data['color']
        depth = cam_data['depth']
        mask = cam_data['mask']
        T_wc = cam_data['T_wc']
        fovy = cam_data['fovy']
        
        # 确保color是float格式
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        
        # 计算相机内参
        height, width = depth.shape[:2]
        focal = height / (2.0 * np.tan(fovy / 2.0))
        
        camera_info = CameraInfo(width, height, focal, focal, width/2, height/2, 1.0)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
        
        # 应用mask过滤
        valid_mask = (mask > 0) & (depth < 2.0) & (depth > 0.1)
        cloud_masked = cloud[valid_mask]
        color_masked = color[valid_mask]
        
        # 变换到世界坐标系
        cloud_world = transform_cloud_to_world(cloud_masked, T_wc)
        
        clouds_world.append(cloud_world)
        colors_list.append(color_masked)
        
        print(f"   相机 {i+1}: {len(cloud_masked)} 个点")
    
    # 融合点云
    cloud_fused, colors_fused, cloud_o3d = fuse_point_clouds(
        clouds_world, colors_list, T_wc_primary, voxel_size=0.005
    )
    
    print(f"   融合后: {len(cloud_fused)} 个点")
    
    # 采样点云用于网络输入
    NUM_POINT = 5000
    if len(cloud_fused) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_fused), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_fused))
        idxs2 = np.random.choice(len(cloud_fused), NUM_POINT - len(cloud_fused), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_fused[idxs]
    color_sampled = colors_fused[idxs]
    
    # 转换为网络输入格式
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    
    end_points = dict()
    end_points['point_clouds'] = cloud_tensor
    end_points['cloud_colors'] = color_sampled
    
    # 加载网络并推理
    net = get_net()
    
    # 计算世界坐标系下的"向下"方向在主相机坐标系中的投影
    R_wc = T_wc_primary.R
    R_cw = R_wc.T
    world_down_c = R_cw @ np.array([0, 0, -1])
    
    # 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    
    # 碰撞检测
    if len(gg) > 0:
        mfcdetector = ModelFreeCollisionDetector(cloud_fused, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        gg = gg[~collision_mask]
    
    # NMS
    if len(gg) > 0:
        gg.nms().sort_by_score()
    
    # 过滤垂直抓取
    angle_threshold = np.deg2rad(30)
    filtered = []
    for grasp in list(gg):
        approach_dir_c = grasp.rotation_matrix[:, 0]
        cos_angle = np.clip(np.dot(approach_dir_c, world_down_c), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    
    if len(filtered) == 0:
        print("[Warning] 没有符合垂直角度阈值的抓取，使用所有预测。")
        filtered = list(gg)
    else:
        print(f"[FUSION] 过滤后剩余 {len(filtered)} 个垂直抓取")
    
    # 选择最佳抓取
    if len(filtered) > 0:
        best_grasp = filtered[0]
        new_gg = GraspGroup()
        new_gg.add(best_grasp)
        
        # 可视化
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers], 
            window_name="Fused Point Cloud Grasp")
        
        return new_gg
    else:
        print("[Error] 没有找到有效抓取！")
        return None



# ================= 辅助函数：计算保持物品水平所需的腕部补偿角度 ====================
def _compute_wrist_compensation(current_rotation_matrix, initial_rotation_matrix):
    """
    计算为了保持物品水平所需的腕部（wrist_3）补偿角度。

    原理：
    - 物品在初始抓取时，其X轴在世界坐标系中有一个方向
    - 我们希望物品的X轴在世界X-Y平面上的投影方向保持不变
    - 当机械臂末端旋转时，通过调整 wrist_3 来保持物品方向不变

    参数:
    current_rotation_matrix: 当前末端执行器的旋转矩阵 (3x3)
    initial_rotation_matrix: 初始抓取时的旋转矩阵 (3x3)

    返回:
    compensation_angle: 需要补偿的角度（弧度）
    """
    # 提取初始和当前的X轴方向（物品的前方）
    initial_x = initial_rotation_matrix[:, 0]
    current_x = current_rotation_matrix[:, 0]

    # 将X轴投影到世界X-Y平面（忽略Z分量）
    initial_x_xy = initial_x[:2]
    current_x_xy = current_x[:2]

    # 归一化
    initial_x_xy_norm = np.linalg.norm(initial_x_xy)
    current_x_xy_norm = np.linalg.norm(current_x_xy)

    if initial_x_xy_norm < 1e-6 or current_x_xy_norm < 1e-6:
        # X轴几乎垂直，无法计算水平方向
        return 0.0

    initial_x_xy = initial_x_xy / initial_x_xy_norm
    current_x_xy = current_x_xy / current_x_xy_norm

    # 计算两个方向之间的角度
    cos_angle = np.clip(np.dot(initial_x_xy, current_x_xy), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # 使用叉积判断旋转方向
    # 在2D平面上，叉积的Z分量决定旋转方向
    cross_z = initial_x_xy[0] * current_x_xy[1] - initial_x_xy[1] * current_x_xy[0]

    if cross_z < 0:
        angle = -angle

    # 返回补偿角度（需要反向旋转）
    return -angle


# ================= 辅助函数：执行轨迹规划器序列（带腕部补偿） ====================
def _execute_planner_sequence_with_compensation(env, robot, planner_array, time_array,
                                                 gripper_ctrl=None, keep_level=False,
                                                 initial_grasp_rotation=None):
    """
    执行一组轨迹规划器序列，可选择性地锁定腕部旋转以保持物品水平。

    参数:
    env: 机器人环境对象
    robot: 机器人对象
    planner_array: 规划器列表
    time_array: 时间数组，第一个元素为0.0
    gripper_ctrl: 夹爪控制量（0-255）
    keep_level: 是否启用水平保持（锁定wrist_3）
    initial_grasp_rotation: 初始抓取时的旋转矩阵（用于计算补偿）
    """
    action = np.zeros(7)
    if gripper_ctrl is not None:
        action[-1] = gripper_ctrl

    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)

    # 记录初始的 wrist_3 角度，在整个运动过程中保持不变
    initial_wrist3 = None
    if keep_level:
        initial_wrist3 = robot.get_joint()[5]
        print(f"  [LEVEL-KEEP] 锁定 wrist_3 角度为: {np.degrees(initial_wrist3):.2f}°")

    step_count = 0

    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate

                    # 如果启用水平保持，在调用move_joint之前就锁定 wrist_3 角度
                    if keep_level and initial_wrist3 is not None:
                        joint[5] = initial_wrist3

                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()

                    # 如果启用水平保持，强制锁定 wrist_3 角度
                    if keep_level and initial_wrist3 is not None:
                        joint[5] = initial_wrist3

                        # 每500步打印一次确认
                        if step_count % 500 == 0:
                            print(f"  [LEVEL-KEEP] Step {step_count}: wrist_3 保持在 {np.degrees(joint[5]):.2f}°")

                    step_count += 1

                action[:6] = joint
                if gripper_ctrl is not None:
                    action[-1] = gripper_ctrl
                env.step(action)
                break

    if keep_level:
        print(f"  [LEVEL-KEEP] 完成，wrist_3 全程保持在 {np.degrees(initial_wrist3):.2f}°")


# ================= 辅助函数：执行轨迹规划器序列 ====================
def _execute_planner_sequence(env, robot, planner_array, time_array, gripper_ctrl=None):
    """
    执行一组轨迹规划器序列。
    抽取重复的执行逻辑为独立函数，提高代码可维护性。
    
    参数:
    env: 机器人环境对象
    robot: 机器人对象
    planner_array: 规划器列表
    time_array: 时间数组，第一个元素为0.0
    """
    action = np.zeros(7)
    # 如果指定了夹爪控制量，则在整个执行过程中保持该值（例如 0=完全张开, 255=完全闭合）
    if gripper_ctrl is not None:
        action[-1] = gripper_ctrl
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                if gripper_ctrl is not None:
                    action[-1] = gripper_ctrl
                env.step(action)
                break


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg, T_wc=None):

    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    T_wc (sm.SE3): 世界坐标系到相机坐标系的变换矩阵。
    """
    robot = env.robot
    T_wb = robot.base

    # 0.初始准备阶段
    if T_wc is None:
        # 默认值回退
        n_wc = np.array([0.0, -1.0, 0.0]) 
        o_wc = np.array([-1.0, 0.0, -0.5]) 
        t_wc = np.array([0.85, 0.8, 1.6]) 
        T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    T_wo = T_wc * T_co

    # --- Debug: 打印抓取坐标用于诊断 ---
    print(f"\n[GRASP DEBUG]")
    print(f"  Grasp in Camera Coords: {gg.translations[0]}")
    print(f"  Grasp in World Coords:  {T_wo.t}")
    print(f"  Camera Position:        {T_wc.t}")

    # ===== 架子抓取检测 =====
    # 架子层高度列表 (根据 scene.xml 中的配置)
    SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
    SHELF_LAYER_TOLERANCE = 0.15  # 检测容差
    SHELF_X_MIN = 1.6   # 架子 X 范围起点
    SHELF_X_MAX = 2.0   # 架子 X 范围终点
    SHELF_APPROACH_OFFSET = 0.35  # 对准点与架子前沿的距离
    
    def is_shelf_grasp(grasp_pos):
        """检测目标位置是否在架子层上"""
        x, y, z = grasp_pos
        if x < SHELF_X_MIN or x > SHELF_X_MAX:
            return False, -1
        for i, layer_z in enumerate(SHELF_LAYER_HEIGHTS):
            if abs(z - layer_z) < SHELF_LAYER_TOLERANCE:
                return True, i
        return False, -1
    
    # 检测是否为架子抓取
    grasp_world_pos = T_wo.t
    is_shelf, shelf_layer = is_shelf_grasp(grasp_world_pos)
    if is_shelf:
        print(f"🔔 [SHELF GRASP] 检测到架子抓取 - 层 {shelf_layer + 1}, 使用水平接近策略")
    else:
        print(f"📦 [TABLE GRASP] 检测到桌面抓取，使用标准直线接近策略")

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态
    time1 = 1
    q0 = robot.get_joint()
    
    if is_shelf:
        # 架子抓取：使用专门的水平朝向预抓取姿态
        # 这个姿态让机械臂末端水平朝向架子方向(-X)
        # 关节配置：让机械臂处于一个适合水平接近的姿态
        q1 = np.array([np.pi/2, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])
        print("  [SHELF] 使用水平朝向预抓取姿态...")
    else:
        # 桌面抓取：使用原有的垂直向下预抓取姿态
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    # 执行planner_array = [planner1]
    _execute_planner_sequence(env, robot, [planner1], [0.0, time1])

    # 2.接近抓取位姿 + 3.执行抓取
    # 根据是否为架子抓取，采用不同的路径策略
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    
    if is_shelf:
        # ===== 架子抓取：分段水平接近 =====
        
        # 构建水平抓取姿态：
        # 夹爪接近方向朝向 +X，夹爪开口朝向 -Z（朝下）
        # TwoVectors: x=接近方向, y=侧向
        approach_dir = np.array([1, 0, 0])  # 接近方向 +X
        side_dir = np.array([0, 1, 0])       # 夹爪侧向 +Y
        R_horizontal = sm.SO3.TwoVectors(x=approach_dir, y=side_dir)
        print(f"  [DEBUG] R_horizontal:\n{R_horizontal}")
        

        # 阶段 2A: 先用关节空间规划移动到对准点（避免直线路径碰撞）
        print("  [SHELF] 阶段2A: 移动到架子正前方对准点...")
        # 对准点在架子前方（X值比架子小）
        align_x = SHELF_X_MIN - SHELF_APPROACH_OFFSET  # 架子前方 = 1.6 - 0.35 = 1.25
        align_point = np.array([align_x, grasp_world_pos[1], grasp_world_pos[2]])
        T_align = sm.SE3.Trans(align_point) * sm.SE3(R_horizontal)
        
        print(f"  [DEBUG] 对准点: X={align_x:.2f}, Y={grasp_world_pos[1]:.2f}, Z={grasp_world_pos[2]:.2f}")
        
        # 尝试用IK计算对准点的关节角度
        q_align = robot.ikine(T_align)
        if len(q_align) > 0:
            # IK成功，使用关节空间规划到对准点
            time2a = 1.5
            param_2a = JointParameter(q1, q_align)
            vel_2a = QuinticVelocityParameter(time2a)
            traj_2a = TrajectoryParameter(param_2a, vel_2a)
            planner_2a = TrajectoryPlanner(traj_2a)
            _execute_planner_sequence(env, robot, [planner_2a], [0.0, time2a])
            robot.set_joint(q_align)
        else:
            # IK失败，尝试调整对准点位置（降低高度或拉近）
            print("  [SHELF] 对准点IK失败，尝试调整位置...")
            # 拉近对准点，降低一点高度
            align_point_adjusted = align_point + np.array([0.1, 0, -0.05])
            T_align_adj = sm.SE3.Trans(align_point_adjusted) * sm.SE3(R_horizontal)
            q_align = robot.ikine(T_align_adj)
            if len(q_align) > 0:
                time2a = 1.5
                param_2a = JointParameter(q1, q_align)
                vel_2a = QuinticVelocityParameter(time2a)
                traj_2a = TrajectoryParameter(param_2a, vel_2a)
                planner_2a = TrajectoryPlanner(traj_2a)
                _execute_planner_sequence(env, robot, [planner_2a], [0.0, time2a])
                robot.set_joint(q_align)
                T_align = T_align_adj
            else:
                raise RuntimeError("无法找到架子对准点的有效IK解，请调整目标位置")
        
        print("  [SHELF] 阶段2B: 水平伸入架子到抓取预备点...") 
        
        # 阶段 2B: 从对准点水平伸入到抓取预备点 (使用笛卡尔规划，因为这是简单的水平直线)
        # 预备点在抓取点前方（X较小）0.1m
        T2_pos = grasp_world_pos - np.array([0.01, 0, 0])  # 抓取点前方0.1m（接近方向+X，所以在-X方向偏移）
        T2 = sm.SE3.Trans(T2_pos) * sm.SE3(R_horizontal)
        
        time2b = 1.5
        pos_param_2b = LinePositionParameter(T_align.t, T2.t)
        att_param_2b = OneAttitudeParameter(R_horizontal, R_horizontal)
        cart_param_2b = CartesianParameter(pos_param_2b, att_param_2b)
        vel_param_2b = QuinticVelocityParameter(time2b)
        traj_param_2b = TrajectoryParameter(cart_param_2b, vel_param_2b)
        planner_2b = TrajectoryPlanner(traj_param_2b)
        
        # 执行阶段 2B
        _execute_planner_sequence(env, robot, [planner_2b], [0.0, time2b])
        
        print("  [SHELF] 阶段3: 执行水平抓取...")
        
        # 阶段 3: 执行抓取 - 水平伸入到精确抓取点
        # 目标位置比物体中心多伸入 0.05m（沿+X方向），确保夹爪包住物体
        T3_pos = grasp_world_pos + np.array([0.01, 0, 0])  # 接近方向+X，所以往+X多进0.05
        T3 = sm.SE3.Trans(T3_pos) * sm.SE3(R_horizontal)
        
        time3 = 1.0
        pos_param_3 = LinePositionParameter(T2.t, T3.t)

        att_param_3 = OneAttitudeParameter(R_horizontal, R_horizontal)
        cart_param_3 = CartesianParameter(pos_param_3, att_param_3)
        vel_param_3 = QuinticVelocityParameter(time3)
        traj_param_3 = TrajectoryParameter(cart_param_3, vel_param_3)
        planner_3 = TrajectoryPlanner(traj_param_3)
        
        # 执行阶段 3
        _execute_planner_sequence(env, robot, [planner_3], [0.0, time3])
    else:
        # ===== 桌面抓取：原有直线规划 =====
        time2 = 1
        T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
        position_parameter1 = LinePositionParameter(T1.t, T2.t)

        attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
        cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
        velocity_parameter1 = QuinticVelocityParameter(time2)
        trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
        planner2 = TrajectoryPlanner(trajectory_parameter1)
        
        # 执行阶段 2
        _execute_planner_sequence(env, robot, [planner2], [0.0, time2])
        
        # 阶段 3: 执行抓取
        time3 = 1
        T3 = T_wo * sm.SE3(0.015, 0.0, 0.0)
        position_parameter2 = LinePositionParameter(T2.t, T3.t)
        attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
        cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
        velocity_parameter2 = QuinticVelocityParameter(time3)
        trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
        planner3 = TrajectoryPlanner(trajectory_parameter2)
        
        # 执行阶段 3
        _execute_planner_sequence(env, robot, [planner3], [0.0, time3])
    
    # 使用当前真实末端姿态作为后续搬运的基准姿态
    # （而不是理想的 R_horizontal），这样可以避免 IK 误差导致的突然“自旋”
    T_grasp = robot.get_cartesian()
    grasp_rotation = sm.SO3(T_grasp.R)
    
    # 闭合夹爪抓取
    # 重要：闭合夹爪期间必须保持手臂关节不动（否则 action[:6] 默认 0 会把手臂拉回零位，引发末端乱转/抖动）
    for i in range(1000):
        action[:6] = robot.get_joint()
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # 4.提起物体 (针对架子抓取，需要先垂直抬起再水平退出)
    if is_shelf:
        print("  [SHELF] 阶段4: 先垂直抬起，再水平退出架子...")
        
        # 4A: 先在原地垂直抬起较大一段距离，确保物体完全离开架子层面
        lift_delta_first = 0.12  # 首次抬高 12cm，可按需要微调
        T4_up = sm.SE3.Trans(T3.t[0], T3.t[1], T3.t[2] + lift_delta_first) * sm.SE3(grasp_rotation)
        time4_up = 1.0
        pos_param_4_up = LinePositionParameter(T3.t, T4_up.t)
        att_param_4_up = OneAttitudeParameter(grasp_rotation, grasp_rotation)
        cart_param_4_up = CartesianParameter(pos_param_4_up, att_param_4_up)
        traj_param_4_up = TrajectoryParameter(cart_param_4_up, QuinticVelocityParameter(time4_up))
        planner_4_up = TrajectoryPlanner(traj_param_4_up)
        _execute_planner_sequence(env, robot, [planner_4_up], [0.0, time4_up], gripper_ctrl=255)

        # 4B: 再从抬高后的姿态，沿-X方向水平退出架子（进入方向是+X）
        T4_retreat_pos = T4_up.t - np.array([0.3, 0, 0])  # 退出到架子外 0.3m（向-X方向）
        T4 = sm.SE3.Trans(T4_retreat_pos) * sm.SE3(grasp_rotation)

        time4 = 1.0
        pos_param_4 = LinePositionParameter(T4_up.t, T4.t)
        att_param_4 = OneAttitudeParameter(grasp_rotation, grasp_rotation)  # 保持水平姿态不变
        cart_param_4 = CartesianParameter(pos_param_4, att_param_4)
        traj_param_4 = TrajectoryParameter(cart_param_4, QuinticVelocityParameter(time4))
        planner4 = TrajectoryPlanner(traj_param_4)
        
        # 物体已被抓住，保持夹爪闭合（ctrl≈255）退出架子
        _execute_planner_sequence(env, robot, [planner4], [0.0, time4], gripper_ctrl=255)
    else:

        # 桌面抓取：原有直接上升逻辑
        time4 = 1
        T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3
        position_parameter3 = LinePositionParameter(T3.t, T4.t)
        attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
        cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
        velocity_parameter3 = QuinticVelocityParameter(time4)
        trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
        planner4 = TrajectoryPlanner(trajectory_parameter3)

    # 5. 安全移动策略 (Safe Transit)
    # 5.1 先垂直抬升，避免碰撞和奇异点
    time_lift = 1.0

    # 在当前抓取点 T4 的基础上，垂直抬高 0.1 米（保持抓取姿态）
    T_lift = sm.SE3.Trans(T4.t[0], T4.t[1], T4.t[2] + 0.1) * sm.SE3(grasp_rotation)
    
    pos_lift = LinePositionParameter(T4.t, T_lift.t)
    att_lift = OneAttitudeParameter(grasp_rotation, grasp_rotation)  # 保持姿态不变
    traj_lift = TrajectoryParameter(CartesianParameter(pos_lift, att_lift), QuinticVelocityParameter(time_lift))
    planner_lift = TrajectoryPlanner(traj_lift)


    # 5.2 关节空间安全中转 (Joint Space Transit)
    # 定义最终放置位置 (可随意修改)
    # target_pos = [x, y, place_height]
    # place_height: 放置时松开夹爪的高度（机械臂下降到此高度后松开）
    # target_pos = [1.4, 0.5, 0.02]
    target_pos = [0.2, 0.2, 0.92]  # 测试背后位置，放置高度2cm

    # 放置后抬升高度（单独配置，不在target_pos中）
    lift_height_after_place = 0.35  # 松开后抬升15cm到安全高度

    # 策略判断：是去"背后"还是"侧面"？
    is_going_back = (target_pos[0] < 0.5 and target_pos[1] < 0.5)

    if is_going_back:
        # 【去背后 (0.2, 0.2)】：需要大角度旋转，且容易碰到奇异点
        # 目标姿态：朝下朝后（机械臂翻转，但我们会在执行时动态补偿腕部旋转）
        T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_lift.t[2]) * sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
        use_joint_transit_strategy = True
    else:
        # 【去侧面/前方 (1.4, 0.3)】：不需要转身，直接平移
        # 目标姿态：保持抓取时的姿态 (grasp_rotation)
        T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_lift.t[2]) * sm.SE3(grasp_rotation)
        use_joint_transit_strategy = False # 侧面直接走直线 Cartesian 即可

    
    # 获取当前的关节角度
    q_start = robot.ikine(T_lift)
    if len(q_start) == 0:
        q_start = robot.get_joint()
        
    # 计算目标的关节角度
    # 开始规划路径
    planner_transit = None
    time_transit = 2.0

    if use_joint_transit_strategy:
        # === 策略A：去背后 (复杂模式) ===
        # 优先尝试关节插值，失败则回退到了 Waypoint
        
        # 1. 尝试计算关节目标
        q_target = robot.ikine(T_target_high)
        
        if len(q_target) > 0:
            # IK成功：直接转底座 (这种最顺滑)
            print(f"去背后：IK成功，使用关节空间插值。")
            traj_transit = TrajectoryParameter(JointParameter(q_start, q_target), QuinticVelocityParameter(time_transit))
            planner_transit = TrajectoryPlanner(traj_transit)
        else:
            # IK失败：启用安全中转点 fallback
            print(f"去背后：IK失败，启用安全中转点策略 (0.8, 0.1)。")
            time_transit = 3.0
            T_waypoint = sm.SE3.Trans(0.8, 0.1, T_lift.t[2]) * sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
            
            # Lift -> Waypoint (边走边转)
            pos1 = LinePositionParameter(T_lift.t, T_waypoint.t)
            att1 = TwoAttitudeParameter(sm.SO3(T_lift.R), sm.SO3(T_waypoint.R))
            planner1 = TrajectoryPlanner(TrajectoryParameter(CartesianParameter(pos1, att1), QuinticVelocityParameter(time_transit/2)))

            # Waypoint -> Target (保持姿态)
            pos2 = LinePositionParameter(T_waypoint.t, T_target_high.t)
            att2 = OneAttitudeParameter(sm.SO3(T_waypoint.R), sm.SO3(T_target_high.R))
            planner2 = TrajectoryPlanner(TrajectoryParameter(CartesianParameter(pos2, att2), QuinticVelocityParameter(time_transit/2)))
            
            planner_transit = [planner1, planner2]
    else:
        # === 策略B：去侧面 (简单模式) ===
        # 直接走笛卡尔直线，最稳，保持抓取姿态不变
        print(f"去侧面：直接使用笛卡尔直线规划。")
        pos_transit = LinePositionParameter(T_lift.t, T_target_high.t)
        # 使用 grasp_rotation 保持姿态完全一致（不旋转）
        att_transit = OneAttitudeParameter(grasp_rotation, grasp_rotation)
        traj_transit = TrajectoryParameter(CartesianParameter(pos_transit, att_transit), QuinticVelocityParameter(time_transit))
        planner_transit = TrajectoryPlanner(traj_transit)


    # 5.3 移动到最终目标 (Approach Target) - 此时已经在目标上方了，直接下放即可?
    # 如果使用了 Joint Transit，我们已经到了 T_target_high
    # 所以 planner5 可以省略，或者做微调
    # 这里我们只保留 planner6 (Lower)
    
    # 6. 下降放置 (Lower and Drop)
    time6 = 2.5  # 增加下降时间，实现更平缓的放置
    # 下降到用户指定的放置高度（target_pos[2]）
    T6 = sm.SE3.Trans(target_pos[0], target_pos[1], target_pos[2]) * sm.SE3(grasp_rotation)
    
    # 从 T_target_high 直降到 T6（保持姿态不变）
    pos_drop = LinePositionParameter(T_target_high.t, T6.t)
    att_drop = OneAttitudeParameter(grasp_rotation, grasp_rotation)
    traj_drop = TrajectoryParameter(CartesianParameter(pos_drop, att_drop), QuinticVelocityParameter(time6))
    planner6 = TrajectoryPlanner(traj_drop)


    # 执行 planner_array (后续搬运逻辑)
    # 对于架子抓取：稳定三段式（退出货架 -> 平移到目标上方 -> 下降放置），全程保持抓取姿态不变
    if is_shelf:
        print("  [SHELF] 架子抓取后：平移到目标 -> 下降放置")

        # 注意：planner4（水平退出）和 planner_lift（垂直提升）已经在上面执行过了
        # 这里不需要重复执行，直接继续后续的搬运逻辑

        # 用当前真实位姿作为后续搬运起点，避免用理想的 T_lift 造成突变
        T_after_lift = robot.get_cartesian()

        # === 平移到目标上方 ===
        if is_going_back:
            print("  [SHELF] 放置目标在背后，使用关节空间搬运策略...")
            # 目标姿态：朝下朝后
            T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_after_lift.t[2]) * sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)

            # 使用 IK 计算关节角
            q_now = robot.get_joint()
            q_target = robot.ikine(T_target_high)

            if len(q_target) > 0:
                 time_move = 3.0
                 param_move = JointParameter(q_now, q_target)
                 traj_move = TrajectoryParameter(param_move, QuinticVelocityParameter(time_move))
                 planner_move = TrajectoryPlanner(traj_move)
                 # 使用带补偿的执行函数，保持物品水平
                 _execute_planner_sequence_with_compensation(
                     env, robot, [planner_move], [0.0, time_move],
                     gripper_ctrl=255,
                     keep_level=True,
                     initial_grasp_rotation=sm.SE3(grasp_rotation)
                 )

                 # 保持抓取姿态，不使用翻转后的姿态
                 final_rotation = grasp_rotation
            else:
                 print("  [SHELF] [WARNING] 背后目标点 IK 失败，尝试保持原姿态...")
                 T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_after_lift.t[2]) * sm.SE3(grasp_rotation)
                 time_move = 2.0
                 pos_move = LinePositionParameter(T_after_lift.t, T_target_high.t)
                 att_move = OneAttitudeParameter(grasp_rotation, grasp_rotation)
                 traj_move = TrajectoryParameter(CartesianParameter(pos_move, att_move), QuinticVelocityParameter(time_move))
                 planner_move = TrajectoryPlanner(traj_move)
                 _execute_planner_sequence(env, robot, [planner_move], [0.0, time_move], gripper_ctrl=255)
                 final_rotation = grasp_rotation
        else:
            print("  [SHELF] 放置目标在侧面/前方，使用笛卡尔直线搬运...")
            T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_after_lift.t[2]) * sm.SE3(grasp_rotation)
            time_move = 2.0
            pos_move = LinePositionParameter(T_after_lift.t, T_target_high.t)
            att_move = OneAttitudeParameter(grasp_rotation, grasp_rotation)
            traj_move = TrajectoryParameter(CartesianParameter(pos_move, att_move), QuinticVelocityParameter(time_move))
            planner_move = TrajectoryPlanner(traj_move)
            _execute_planner_sequence(env, robot, [planner_move], [0.0, time_move], gripper_ctrl=255)
            final_rotation = grasp_rotation

        # === 下降放置 ===
        time_drop = 2.5  # 增加下降时间，实现更平缓的放置
        # 下降到用户指定的放置高度（target_pos[2]）
        T_drop = sm.SE3.Trans(target_pos[0], target_pos[1], target_pos[2]) * sm.SE3(final_rotation)

        # 重新获取当前位置作为 LineStart
        T_current_high = robot.get_cartesian()

        pos_drop2 = LinePositionParameter(T_current_high.t, T_drop.t)
        # 关键修复：下降时保持姿态不变（不要插值旋转）
        att_drop2 = OneAttitudeParameter(sm.SO3(T_current_high.R), sm.SO3(T_current_high.R))
        traj_drop2 = TrajectoryParameter(CartesianParameter(pos_drop2, att_drop2), QuinticVelocityParameter(time_drop))
        planner_drop2 = TrajectoryPlanner(traj_drop2)

        # 下降时也启用水平保持
        if is_going_back:
            _execute_planner_sequence_with_compensation(
                env, robot, [planner_drop2], [0.0, time_drop],
                gripper_ctrl=255,
                keep_level=True,
                initial_grasp_rotation=sm.SE3(grasp_rotation)
            )
        else:
            _execute_planner_sequence(env, robot, [planner_drop2], [0.0, time_drop], gripper_ctrl=255)

        # 松开夹爪（缓慢松开，保持手臂关节不动）
        # 增加松开步数，使松开过程更加平缓
        for i in range(2000):
            action[:6] = robot.get_joint()
            action[-1] -= 0.1  # 减小每步松开幅度，从0.2改为0.1，更加温柔
            action[-1] = np.max([action[-1], 0])
            env.step(action)

        # === 架子抓取：自动复原到初始姿态 q0 ===
        print("  [SHELF] 放置完成，执行自动复原到初始姿态...")

        # 1) 先在当前位置基础上抬高到安全高度，避免回程时蹭到货架/物体
        T_cur = robot.get_cartesian()
        T_up = sm.SE3.Trans(T_cur.t[0], T_cur.t[1], T_cur.t[2] + lift_height_after_place) * sm.SE3(sm.SO3(T_cur.R))
        time_up = 1.5  # 增加抬升时间
        pos_up = LinePositionParameter(T_cur.t, T_up.t)
        att_up = OneAttitudeParameter(sm.SO3(T_cur.R), sm.SO3(T_up.R))
        traj_up = TrajectoryParameter(CartesianParameter(pos_up, att_up), QuinticVelocityParameter(time_up))
        planner_up = TrajectoryPlanner(traj_up)
        _execute_planner_sequence(env, robot, [planner_up], [0.0, time_up], gripper_ctrl=255)

        # 2) 从当前关节角用关节空间插值回到初始姿态 q0
        q_now = robot.get_joint()
        time_back = 1.5
        param_back = JointParameter(q_now, q0)
        traj_back = TrajectoryParameter(param_back, QuinticVelocityParameter(time_back))
        planner_back = TrajectoryPlanner(traj_back)
        _execute_planner_sequence(env, robot, [planner_back], [0.0, time_back], gripper_ctrl=255)

        print("  [SHELF] 抓取、放置及复原完成！")
        return
    
    # 桌面抓取：使用原有复杂搬运逻辑
    if isinstance(planner_transit, list):
         # 使用了方案B (中转点)
         time_array = [0.0, time4, time_lift, 1.5, 1.5, time6]
         planner_array = [planner4, planner_lift, planner_transit[0], planner_transit[1], planner6]
    else:
         # 使用了方案A (关节插值)
         time_array = [0.0, time4, time_lift, time_transit, time6]
         planner_array = [planner4, planner_lift, planner_transit, planner6]

    # 判断是否需要启用水平保持补偿（去背后时）
    if is_going_back:
        print("  [TABLE] 桌面抓取去背后，启用水平保持补偿...")
        # 使用带补偿的执行函数
        _execute_planner_sequence_with_compensation(
            env, robot, planner_array, time_array,
            gripper_ctrl=None,
            keep_level=True,
            initial_grasp_rotation=sm.SE3(grasp_rotation)
        )
    else:
        # 侧面/前方：不需要补偿，使用原有逻辑
        total_time = np.sum(time_array)
        time_step_num = round(total_time / 0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)
        for timei in times:
            for j in range(len(time_cumsum)):
                if timei == 0.0:
                    break
                if timei <= time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    if isinstance(planner_interpolate, np.ndarray):
                        joint = planner_interpolate
                        robot.move_joint(joint)
                    else:
                        robot.move_cartesian(planner_interpolate)
                        joint = robot.get_joint()
                    action[:6] = joint
                    env.step(action)
                    break
    # 松开夹爪（缓慢松开，保持手臂关节不动）
    # 增加松开步数，使松开过程更加平缓
    for i in range(2000):
        action[:6] = robot.get_joint()
        action[-1] -= 0.1  # 减小每步松开幅度，从0.2改为0.1，更加温柔
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪到安全高度，避免碰撞物体。
    time7 = 1.5  # 增加抬升时间
    T7 = sm.SE3.Trans(0.0, 0.0, lift_height_after_place) * T6  # 使用用户配置的抬升高度
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    # 执行planner_array = [planner7]
    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    # 执行planner_array = [planner8]
    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
