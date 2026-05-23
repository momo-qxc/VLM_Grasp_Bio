import os
import sys
import numpy as np
import torch
import open3d as o3d

# 设为 True 可跳过 Open3D 弹窗（GUI 模式下由 ui_main.py 设置）
HEADLESS = False

# GUI 模式下注入渲染回调，每隔 _RENDER_INTERVAL 步调用一次以更新相机视图
_render_callback = None
_RENDER_INTERVAL = 40  # 每 40 步渲染一次（约 12fps @ 500Hz sim）

# ==================== 物体来源记忆 ====================
# 记录每次抓取/放置的物体位置历史，用于"放回原处"/"放回货架"等功能
# 格式: {物体名称: {
#   "initial":    {"position": [x,y,z], "is_shelf": bool, "shelf_layer": int},  # 第一次抓取的位置
#   "last_grasp": {"position": [x,y,z], "is_shelf": bool, "shelf_layer": int},  # 最近一次抓取前的位置
#   "last_place": {"position": [x,y,z]},                                        # 最近一次放置的位置
# }}
_object_origin_memory = {}


def _fuzzy_match_name(object_name):
    """模糊匹配物体名称，返回匹配到的 key 或 None"""
    if object_name in _object_origin_memory:
        return object_name
    for key in _object_origin_memory:
        if object_name in key or key in object_name:
            return key
    return None


def record_grasp(object_name, grasp_world_pos, is_shelf, shelf_layer=-1):
    """记录物体被抓取时的位置（抓取前位置）"""
    entry = {"position": list(grasp_world_pos), "is_shelf": is_shelf, "shelf_layer": shelf_layer}
    if object_name not in _object_origin_memory:
        _object_origin_memory[object_name] = {"initial": entry, "last_grasp": entry}
    else:
        _object_origin_memory[object_name]["last_grasp"] = entry
    print(f"[MEMORY] 记录抓取 '{object_name}': pos={grasp_world_pos}, "
          f"is_shelf={is_shelf}, layer={shelf_layer}")


def record_place(object_name, place_world_pos, is_shelf=None, shelf_layer=-1):
    """记录物体被放置后的位置。is_shelf/shelf_layer 可选，不传则自动检测。"""
    matched = _fuzzy_match_name(object_name)
    if matched is None:
        _object_origin_memory[object_name] = {}
        matched = object_name
    pos = list(place_world_pos)
    # 自动检测是否为货架位置
    if is_shelf is None:
        is_shelf, shelf_layer = _auto_detect_shelf(pos)
    _object_origin_memory[matched]["last_place"] = {
        "position": pos, "is_shelf": is_shelf, "shelf_layer": shelf_layer,
    }
    print(f"[MEMORY] 记录放置 '{matched}': pos={place_world_pos}, "
          f"is_shelf={is_shelf}, layer={shelf_layer}")


# 货架判定常量（与 task_executor.py 保持一致）
_SHELF_X_MIN = 1.61
_SHELF_X_MAX = 1.97
_SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
_SHELF_LAYER_TOL = 0.15


def _auto_detect_shelf(pos):
    """根据世界坐标自动判断是否在货架上，返回 (is_shelf, shelf_layer)。"""
    if _SHELF_X_MIN <= pos[0] <= _SHELF_X_MAX:
        for i, lz in enumerate(_SHELF_LAYER_HEIGHTS):
            if abs(pos[2] - lz) < _SHELF_LAYER_TOL:
                return True, i
    return False, -1


def get_position(object_name, which="initial"):
    """按类型查询物体位置。
    which: "initial" / "last_grasp" / "last_place" / "shelf"
    返回对应的记录 dict 或 None。
    """
    matched = _fuzzy_match_name(object_name)
    if matched is None:
        return None
    mem = _object_origin_memory[matched]
    if which == "shelf":
        # 优先从 initial 找货架位置，再从 last_grasp 找
        for key in ("initial", "last_grasp"):
            rec = mem.get(key)
            if rec and rec.get("is_shelf"):
                return rec
        return None
    return mem.get(which)


# 向后兼容旧接口
def record_object_origin(object_name, grasp_world_pos, is_shelf, shelf_layer=-1):
    record_grasp(object_name, grasp_world_pos, is_shelf, shelf_layer)


def get_object_origin(object_name):
    """向后兼容：返回 initial 位置"""
    return get_position(object_name, "initial")


def get_all_origins():
    """返回所有记忆的物体来源（initial 位置）"""
    result = {}
    for name, mem in _object_origin_memory.items():
        if "initial" in mem:
            result[name] = mem["initial"]
    return result
from PIL import Image
import spatialmath as sm

from manipulator_grasp.arm.motion_planning import *

# ============================================================
# ==================== 桌面抓取参数配置区 ==========================
# ============================================================
# 抓取接近方向（世界坐标系）：
#   [0, 0, -1]          = 纯垂直向下
#   [0, 0.5, -0.866]    = 向+Y倾斜30度
#   [0, 0.6428, -0.7660] = 向+Y倾斜40度
#   公式: [0, sin(角度), -cos(角度)]
GRASP_APPROACH_WORLD = np.array([0, 0.6428, -0.7660])  # 40度

# 抓取点回退距离（米）：沿 approach 反方向拉出，避免深入物体
GRASP_PULLBACK = 0.01  # 10mm

# 位置微调偏移（世界坐标系，米）
GRASP_OFFSET_WORLD = np.array([0.0, 0.0, 0.0])

# 是否强制用点云 bounding box 中心作为抓取位置
FORCE_CENTER = True
# ============================================================


# ==================== 抓取居中性检查 ====================
def check_grasp_bilateral(grasp, cloud_points, min_points_per_side=10):
    """
    检查抓取是否双侧对称——即夹爪两侧都有物体点云。
    如果只有一侧有点，说明抓取在物体边缘，手指会穿过物体。

    返回 True 表示抓取居中良好（两侧都有点且比例均衡）。
    """
    R = grasp.rotation_matrix
    t = grasp.translation
    width = grasp.width
    height = grasp.height if grasp.height > 0 else 0.02

    # 将点云变换到抓取坐标系
    local_pts = (cloud_points - t) @ R  # (N, 3)

    # 只看夹爪附近的点（沿 approach 方向在指尖范围内）
    finger_length = 0.06
    near_mask = (
        (local_pts[:, 0] > -finger_length * 0.5) &
        (local_pts[:, 0] < finger_length) &
        (np.abs(local_pts[:, 2]) < max(height, 0.02) / 2 + 0.01) &
        (np.abs(local_pts[:, 1]) < width / 2 + 0.01)
    )

    near_pts = local_pts[near_mask]
    if len(near_pts) < min_points_per_side * 2:
        return False

    # 检查 y 轴两侧是否都有足够的点
    left_count = np.sum(near_pts[:, 1] < -0.003)
    right_count = np.sum(near_pts[:, 1] > 0.003)

    if left_count < min_points_per_side or right_count < min_points_per_side:
        return False

    # 左右比例检查：两侧点数不能差太多（防止严重偏心）
    ratio = min(left_count, right_count) / max(left_count, right_count)
    return ratio > 0.2  # 至少 20% 的比例平衡


def recenter_grasp(grasp, cloud_points):
    """
    沿 binormal（手指连线方向）测量物体实际范围，
    将抓取中心校正到物体正中间。
    """
    R = grasp.rotation_matrix
    t = grasp.translation
    height = grasp.height if grasp.height > 0 else 0.02

    # 变换到抓取坐标系
    local_pts = (cloud_points - t) @ R

    # 只看抓取区域附近的点
    finger_length = 0.06
    near_mask = (
        (local_pts[:, 0] > -finger_length * 0.3) &
        (local_pts[:, 0] < finger_length) &
        (np.abs(local_pts[:, 2]) < max(height, 0.02) / 2 + 0.005)
    )
    near_pts = local_pts[near_mask]

    if len(near_pts) < 20:
        print(f"  [RECENTER] 附近点太少({len(near_pts)})，跳过校正")
        return

    # 沿 y 轴（binormal）测量物体实际范围
    y_vals = near_pts[:, 1]
    y_lo = np.percentile(y_vals, 10)
    y_hi = np.percentile(y_vals, 90)
    y_center = (y_lo + y_hi) / 2.0
    obj_width = y_hi - y_lo

    print(f"  [RECENTER] y_lo={y_lo*1000:.1f}mm, y_hi={y_hi*1000:.1f}mm, "
          f"center_offset={y_center*1000:.1f}mm, obj_width={obj_width*1000:.1f}mm, "
          f"near_pts={len(near_pts)}")

    # 如果偏移量大于 2mm，执行校正
    if abs(y_center) > 0.002:
        shift_cam = R[:, 1] * y_center
        grasp.translation = t + shift_cam
        print(f"  [RECENTER] 校正 {y_center*1000:.1f}mm (shift_cam={shift_cam})")
    else:
        print(f"  [RECENTER] 偏移量 {y_center*1000:.1f}mm < 2mm，无需校正")

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
def fuse_point_clouds(clouds_world, colors_list, T_wc_primary, voxel_size=0.005,
                      remove_outliers=True, nb_neighbors=20, std_ratio=2.0):
    """
    融合多个世界坐标系点云，并变换回主相机坐标系供 GraspNet 使用。

    参数:
    clouds_world: list of np.ndarray, 每个元素是 (N, 3) 的世界坐标点云
    colors_list: list of np.ndarray, 每个元素是 (N, 3) 的颜色
    T_wc_primary: sm.SE3, 主相机的世界到相机变换（用于最终输出）
    voxel_size: 下采样体素大小
    remove_outliers: 是否移除离群点（默认True）
    nb_neighbors: 统计离群点去除的邻居数量（默认20）
    std_ratio: 标准差倍数阈值（默认2.0，越小过滤越严格）

    返回:
    cloud_cam: np.ndarray, 融合后的相机坐标系点云
    colors_cam: np.ndarray, 对应的颜色
    cloud_o3d: o3d.geometry.PointCloud, 用于可视化的 Open3D 点云
    """
    # 合并所有点云
    all_points = np.vstack(clouds_world)
    all_colors = np.vstack(colors_list)

    print(f"  [FUSION] 原始点云: {len(all_points)} 个点")

    # 创建 Open3D 点云用于下采样
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # 体素下采样去除重复点
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  [FUSION] 体素下采样后: {len(pcd_down.points)} 个点")

    # 统计离群点去除（移除噪声点）
    if remove_outliers and len(pcd_down.points) > nb_neighbors:
        pcd_filtered, ind = pcd_down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        print(f"  [FUSION] 离群点去除后: {len(pcd_filtered.points)} 个点 (移除了 {len(pcd_down.points) - len(pcd_filtered.points)} 个噪声点)")
        pcd_down = pcd_filtered

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


def filter_largest_cluster(cloud, colors, eps=0.02, min_points=10):
    """
    使用DBSCAN聚类，只保留最大的点云簇。

    参数:
    cloud: np.ndarray, (N, 3) 点云
    colors: np.ndarray, (N, 3) 颜色
    eps: DBSCAN的邻域半径（米）
    min_points: DBSCAN的最小点数

    返回:
    filtered_cloud: np.ndarray, 过滤后的点云
    filtered_colors: np.ndarray, 过滤后的颜色
    """
    if len(cloud) < min_points:
        return cloud, colors

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    # DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # 找到最大的簇（排除噪声点，label=-1）
    if len(labels) == 0 or np.all(labels == -1):
        return cloud, colors

    # 统计每个簇的点数
    unique_labels = labels[labels >= 0]
    if len(unique_labels) == 0:
        return cloud, colors

    label_counts = np.bincount(unique_labels)
    largest_cluster_label = np.argmax(label_counts)

    # 只保留最大簇的点
    mask = (labels == largest_cluster_label)
    filtered_cloud = cloud[mask]
    filtered_colors = colors[mask]

    removed_count = len(cloud) - len(filtered_cloud)
    if removed_count > 0:
        print(f"    聚类过滤: 保留最大簇 {len(filtered_cloud)} 个点，移除 {removed_count} 个离散点")

    return filtered_cloud, filtered_colors




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
    mask = (workspace_mask > 0) & (depth < 3.5) & (depth > 0.1)
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

    # ===== 智能抓取方向选择 =====
    # 根据物品位置（桌面 vs 货架）自动选择抓取方向

    # 计算点云在世界坐标系中的平均位置
    points_cam = np.asarray(cloud_o3d.points)
    cloud_world_check = transform_cloud_to_world(points_cam, T_wc)
    avg_pos = np.mean(cloud_world_check, axis=0)
    avg_x, avg_y, avg_z = avg_pos

    # 货架判断逻辑（与execute_grasp保持一致）
    # 货架位置：X范围 [1.61, 1.97]，Z高度接近货架层
    # 根据scene.xml: 货架碰撞层中心X=1.79, 半宽=0.18
    SHELF_X_MIN = 1.61  # 货架前沿 = 1.79 - 0.18
    SHELF_X_MAX = 1.97  # 货架后沿 = 1.79 + 0.18
    SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
    SHELF_LAYER_TOLERANCE = 0.15

    is_shelf_object = False
    if SHELF_X_MIN <= avg_x <= SHELF_X_MAX:
        # 检查是否接近某个货架层
        for layer_z in SHELF_LAYER_HEIGHTS:
            if abs(avg_z - layer_z) < SHELF_LAYER_TOLERANCE:
                is_shelf_object = True
                break

    if is_shelf_object:
        print(f"\n[GRASP] 检测到货架物品 (位置: X={avg_x:.2f}, Y={avg_y:.2f}, Z={avg_z:.2f})，使用水平抓取过滤")
        grasp_mode = "horizontal"
    else:
        print(f"\n[GRASP] 检测到桌面物品 (位置: X={avg_x:.2f}, Y={avg_y:.2f}, Z={avg_z:.2f})，使用垂直抓取过滤")
        grasp_mode = "vertical"

    # 计算参考方向
    world_z_c = R_cw @ np.array([0, 0, 1])  # 垂直向上

    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    angle_threshold = np.deg2rad(30)  # 30度的弧度值
    filtered = []

    if grasp_mode == "vertical":
        # 垂直抓取：接近方向应该接近垂直向下
        for grasp in all_grasps:
            approach_dir_c = grasp.rotation_matrix[:, 0]
            cos_angle = np.dot(approach_dir_c, world_down_c)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < angle_threshold:
                filtered.append(grasp)

        if len(filtered) == 0:
            print("\n[Warning] No grasp predictions within vertical angle threshold.")
            print("[FIX] 构造垂直向下抓取姿态替代...")
            # 世界坐标系中垂直向下旋转矩阵，转换到相机坐标系
            R_down_world = np.array([[0, 0, -1],
                                     [0, 1,  0],
                                     [1, 0,  0]], dtype=np.float64)
            R_down_cam = R_cw @ R_down_world
            # 使用点云质心作为抓取位置（比 GraspNet 预测平均值更准确）
            obj_center_cam = np.mean(points_cam, axis=0) if len(points_cam) > 0 \
                else np.mean([g.translation for g in all_grasps], axis=0)
            from graspnetAPI import Grasp
            synth_grasp = Grasp()
            synth_grasp.score = 0.9
            synth_grasp.width = 0.05
            synth_grasp.height = 0.02
            synth_grasp.depth = 0.1
            synth_grasp.rotation_matrix = R_down_cam
            synth_grasp.translation = obj_center_cam
            synth_grasp.object_id = -1
            filtered = [synth_grasp]
            print(f"[FIX] 已构造垂直向下抓取，相机坐标: {obj_center_cam}")
        else:
            print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    else:  # horizontal
        # 水平抓取：接近方向应该与世界Z轴垂直（即在水平面内）
        horizontal_threshold = np.deg2rad(30)  # 允许±30度偏差

        for grasp in all_grasps:
            approach_dir_c = grasp.rotation_matrix[:, 0]
            # 计算与垂直方向的夹角
            cos_angle = np.clip(np.dot(approach_dir_c, world_z_c), -1.0, 1.0)
            angle_from_vertical = np.arccos(np.abs(cos_angle))

            # 水平抓取：角度应该接近90度（π/2）
            deviation_from_horizontal = abs(angle_from_vertical - np.pi/2)

            if deviation_from_horizontal < horizontal_threshold:
                filtered.append(grasp)

        if len(filtered) == 0:
            print("\n[Warning] No grasp predictions within horizontal angle threshold. Using all predictions.")
            filtered = all_grasps
        else:
            print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of horizontal out of {len(all_grasps)} total predictions.")

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

    # ===== 调试信息：显示抓取位置 =====
    grasp_pos_cam = best_grasp.translation
    grasp_pos_world = transform_cloud_to_world(grasp_pos_cam.reshape(1, 3), T_wc)[0]
    print(f"\n[DEBUG] 抓取位置:")
    print(f"  相机坐标系: ({grasp_pos_cam[0]:.3f}, {grasp_pos_cam[1]:.3f}, {grasp_pos_cam[2]:.3f})")
    print(f"  世界坐标系: ({grasp_pos_world[0]:.3f}, {grasp_pos_world[1]:.3f}, {grasp_pos_world[2]:.3f})")
    print(f"  物体中心 (相机): ({object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f})")
    object_center_world = transform_cloud_to_world(object_center.reshape(1, 3), T_wc)[0]
    print(f"  物体中心 (世界): ({object_center_world[0]:.3f}, {object_center_world[1]:.3f}, {object_center_world[2]:.3f})")

    # ===== 货架物品：强制水平化抓取姿态 =====
    if grasp_mode == "horizontal":
        print("🔧 货架物品：强制调整为水平抓取姿态...")

        # 获取当前抓取位置
        grasp_pos_c = best_grasp.translation

        # 计算从抓取点指向物体中心的方向（在相机坐标系中）
        to_center = object_center - grasp_pos_c
        to_center_norm = np.linalg.norm(to_center)

        if to_center_norm > 1e-6:
            # 将这个方向投影到水平面（去除Z分量）
            # 首先转换到世界坐标系
            to_center_world = R_wc @ to_center
            to_center_world[2] = 0  # 去除Z分量，保持水平

            # 归一化
            to_center_world_norm = np.linalg.norm(to_center_world)
            if to_center_world_norm > 1e-6:
                to_center_world = to_center_world / to_center_world_norm

                # 转回相机坐标系
                approach_horizontal_c = R_cw @ to_center_world
            else:
                # 如果投影后长度为0，使用默认水平方向（+X）
                approach_horizontal_c = R_cw @ np.array([1, 0, 0])
        else:
            # 如果抓取点就在中心，使用默认水平方向
            approach_horizontal_c = R_cw @ np.array([1, 0, 0])

        # 归一化接近方向
        approach_horizontal_c = approach_horizontal_c / np.linalg.norm(approach_horizontal_c)

        # 构建新的旋转矩阵
        # x轴 = 水平接近方向
        # y轴 = 垂直于接近方向和世界Z轴
        # z轴 = x × y

        a_c = approach_horizontal_c  # 接近方向（水平）

        # y轴应该垂直于接近方向和世界Z轴
        # y = normalize(world_up_c × a_c)
        y_c = np.cross(world_up_c, a_c)
        y_norm = np.linalg.norm(y_c)

        if y_norm > 1e-6:
            y_c = y_c / y_norm
            # z轴 = x × y
            z_c = np.cross(a_c, y_c)
            z_c = z_c / np.linalg.norm(z_c)

            # 更新旋转矩阵
            best_grasp.rotation_matrix = np.column_stack([a_c, y_c, z_c])
            print("✅ 已强制调整为水平抓取姿态 (Horizontal Grasp Enforced)")
        else:
            print("⚠️ 无法构建水平抓取姿态，保持原姿态")

    # ===== 桌面物品：抓取头自动调平 (Auto-leveling) =====
    elif grasp_mode == "vertical":
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
            # 更新旋转矩阵，实现"调平"旋转
            best_grasp.rotation_matrix = np.column_stack([a_c, y_c_new, z_c_new])
            print("✅ 已自动执行抓取头调平优化 (Orientation Auto-leveled)")

    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取

    if not HEADLESS:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    return new_gg

    #return best_translation, best_rotation, best_width


# ==================== 多相机点云融合抓取推理 ====================
def run_grasp_inference_fused(camera_data_list, T_wc_primary, fovy_primary, desktop_mode=False):
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
    desktop_mode: bool, True=桌面三相机模式（30度倾斜+居中检查），False=货架模式（保持原逻辑）

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

        # 应用mask过滤（深度上限需覆盖 cam_2 等高位相机，z≈2.95 看桌面深度约2.3m）
        valid_mask = (mask > 0) & (depth < 3.5) & (depth > 0.1)
        cloud_masked = cloud[valid_mask]
        color_masked = color[valid_mask]

        print(f"   相机 {i+1}: {len(cloud_masked)} 个点（mask过滤后）")

        # 聚类过滤：只保留最大的点云簇（去除VLM分割的噪声）
        cloud_masked, color_masked = filter_largest_cluster(
            cloud_masked, color_masked, eps=0.02, min_points=10
        )

        # 变换到世界坐标系
        cloud_world = transform_cloud_to_world(cloud_masked, T_wc)

        clouds_world.append(cloud_world)
        colors_list.append(color_masked)

        print(f"   相机 {i+1}: {len(cloud_masked)} 个点（聚类过滤后）")
    
    # 融合点云
    voxel_size_fuse = 0.002 if desktop_mode else 0.005
    cloud_fused, colors_fused, cloud_o3d = fuse_point_clouds(
        clouds_world, colors_list, T_wc_primary, voxel_size=voxel_size_fuse
    )

    print(f"   融合后(原始): {len(cloud_fused)} 个点")

    # 桌面模式：统计离群点移除
    if desktop_mode and len(cloud_fused) > 50:
        cl, inlier_idx = cloud_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        cloud_fused = cloud_fused[inlier_idx]
        colors_fused = colors_fused[inlier_idx]
        cloud_o3d = cl
        print(f"   离群点移除后: {len(cloud_fused)} 个点")

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
        if desktop_mode:
            mfcdetector = ModelFreeCollisionDetector(cloud_fused, voxel_size=0.005)
            collision_mask, empty_mask = mfcdetector.detect(
                gg, approach_dist=0.05, collision_thresh=0.01,
                return_empty_grasp=True, empty_thresh=0.01
            )
            valid_mask = (~collision_mask) & (~empty_mask)
            gg = gg[valid_mask]
            print(f"[FUSION] 碰撞检测: 碰撞={collision_mask.sum()}, 空抓={empty_mask.sum()}, 保留={valid_mask.sum()}")
        else:
            mfcdetector = ModelFreeCollisionDetector(cloud_fused, voxel_size=0.01)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
            gg = gg[~collision_mask]
    
    # NMS
    if len(gg) > 0:
        gg.nms().sort_by_score()

    # ===== 抓取方向选择与最佳抓取 =====
    if desktop_mode:
        # ===== 桌面模式：垂直角度过滤 + 30度倾斜固定姿态 =====
        all_grasps = list(gg)
        angle_threshold = np.deg2rad(30)
        filtered = []
        for grasp in all_grasps:
            approach_dir_c = grasp.rotation_matrix[:, 0]
            cos_angle = np.clip(np.dot(approach_dir_c, world_down_c), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < angle_threshold:
                filtered.append(grasp)

        if len(filtered) > 0:
            print(f"[FUSION] angle filter +-30 deg: {len(filtered)}/{len(all_grasps)} passed")
        else:
            print(f"[FUSION] 0/{len(all_grasps)} passed 30 deg -> 将强制垂直向下抓取")
            filtered = all_grasps

        # 选择最佳抓取：综合得分和距物体中心距离
        if len(filtered) > 0:
            # 计算物体中心（用 bounding box 中心）
            if len(cloud_fused) > 0:
                bbox_min = np.min(cloud_fused, axis=0)
                bbox_max = np.max(cloud_fused, axis=0)
                object_center = (bbox_min + bbox_max) / 2.0
            else:
                object_center = np.zeros(3)

            # 按综合得分排序（距离中心越近越好）
            distances = [np.linalg.norm(g.translation - object_center) for g in filtered]
            max_dist = max(distances) if distances else 1.0
            scored = []
            for g, d in zip(filtered, distances):
                dist_score = 1 - (d / max_dist) if max_dist > 0 else 1.0
                composite = g.score * 0.4 + dist_score * 0.6
                scored.append((g, composite))
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered_sorted = [g for g, s in scored]

            # 双侧居中性检查
            bilateral_filtered = [g for g in filtered_sorted if check_grasp_bilateral(g, cloud_fused)]
            if len(bilateral_filtered) > 0:
                print(f"[FUSION] 双侧检查: {len(bilateral_filtered)}/{len(filtered_sorted)} 个抓取通过居中性检查")
                best_grasp = bilateral_filtered[0]
            else:
                print("[FUSION Warning] 没有抓取通过双侧检查，使用最高综合得分")
                best_grasp = filtered_sorted[0]

            # ===== 固定抓取姿态（用配置区的参数） =====
            R_wc_dt = T_wc_primary.R
            R_cw_dt = R_wc_dt.T
            world_up_c = R_cw_dt @ np.array([0, 0, 1])

            # approach 方向：从配置区读取，转到相机坐标系
            approach_world = GRASP_APPROACH_WORLD / np.linalg.norm(GRASP_APPROACH_WORLD)
            a_c = R_cw_dt @ approach_world
            a_c = a_c / np.linalg.norm(a_c)

            # binormal：垂直于 approach 和世界Z+，保证手指水平
            y_c = np.cross(world_up_c, a_c)
            y_norm = np.linalg.norm(y_c)
            if y_norm > 1e-6:
                y_c = y_c / y_norm
            else:
                y_c = R_cw_dt @ np.array([1, 0, 0])
                y_c = y_c / np.linalg.norm(y_c)
            z_c = np.cross(a_c, y_c)
            z_c = z_c / np.linalg.norm(z_c)

            best_grasp.rotation_matrix = np.column_stack([a_c, y_c, z_c])
            print(f"[FUSION] 固定姿态: approach_world={approach_world}")
            print(f"[FUSION] ★ a_c (cam frame): {a_c}")
            print(f"[FUSION] ★ rotation_matrix col0: {best_grasp.rotation_matrix[:, 0]}")

            # ===== 位置处理 =====
            if FORCE_CENTER:
                bbox_min = np.min(cloud_fused, axis=0)
                bbox_max = np.max(cloud_fused, axis=0)
                best_grasp.translation = (bbox_min + bbox_max) / 2.0
                print(f"  [CENTER] 使用点云中心: {best_grasp.translation}")

            # 回退（相机坐标系，沿 approach 反方向）
            best_grasp.translation = best_grasp.translation - a_c * GRASP_PULLBACK

            # 世界坐标系偏移
            if np.any(GRASP_OFFSET_WORLD != 0):
                offset_c = R_cw_dt @ GRASP_OFFSET_WORLD
                best_grasp.translation = best_grasp.translation + offset_c
                print(f"  [OFFSET] 世界偏移: {GRASP_OFFSET_WORLD} -> cam: {offset_c}")

            print(f"  [FINAL] pullback={GRASP_PULLBACK*1000:.0f}mm, offset={GRASP_OFFSET_WORLD}")

            new_gg = GraspGroup()
            new_gg.add(best_grasp)

            # 可视化
            if not HEADLESS:
                grippers = new_gg.to_open3d_geometry_list()
                o3d.visualization.draw_geometries([cloud_o3d, *grippers],
                    window_name="Fused Point Cloud Grasp (Desktop)")

            return new_gg
        else:
            print("[Error] 没有找到有效抓取！")
            return None

    # ===== 货架模式（desktop_mode=False）：保持原有智能抓取方向选择逻辑 =====
    # ===== 智能抓取方向选择 =====
    cloud_world_check = transform_cloud_to_world(cloud_fused, T_wc_primary)
    avg_pos = np.mean(cloud_world_check, axis=0)
    avg_x, avg_y, avg_z = avg_pos

    # 货架判断逻辑（与execute_grasp保持一致）
    # 根据scene.xml: 货架碰撞层中心X=1.79, 半宽=0.18
    SHELF_X_MIN = 1.61  # 货架前沿 = 1.79 - 0.18
    SHELF_X_MAX = 1.97  # 货架后沿 = 1.79 + 0.18
    SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
    SHELF_LAYER_TOLERANCE = 0.15

    is_shelf_object = False
    if SHELF_X_MIN <= avg_x <= SHELF_X_MAX:
        # 检查是否接近某个货架层
        for layer_z in SHELF_LAYER_HEIGHTS:
            if abs(avg_z - layer_z) < SHELF_LAYER_TOLERANCE:
                is_shelf_object = True
                break

    if is_shelf_object:
        print(f"[FUSION] 检测到货架物品 (位置: X={avg_x:.2f}, Y={avg_y:.2f}, Z={avg_z:.2f})，使用水平抓取过滤")
        grasp_mode = "horizontal"
    else:
        print(f"[FUSION] 检测到桌面物品 (位置: X={avg_x:.2f}, Y={avg_y:.2f}, Z={avg_z:.2f})，使用垂直抓取过滤")
        grasp_mode = "vertical"

    # 计算参考方向
    R_wc = T_wc_primary.R
    R_cw = R_wc.T
    world_down_c = R_cw @ np.array([0, 0, -1])  # 垂直向下
    world_z_c = R_cw @ np.array([0, 0, 1])      # 垂直向上

    # 根据模式过滤抓取
    angle_threshold = np.deg2rad(30)
    filtered = []

    if grasp_mode == "vertical":
        # 垂直抓取：接近方向应该接近垂直向下
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

    else:  # horizontal
        # 水平抓取：接近方向应该与世界Z轴垂直（即在水平面内）
        # 计算接近方向与Z轴的夹角，应该接近90度
        horizontal_threshold = np.deg2rad(30)  # 允许±30度偏差

        for grasp in list(gg):
            approach_dir_c = grasp.rotation_matrix[:, 0]
            # 计算与垂直方向的夹角
            cos_angle = np.clip(np.dot(approach_dir_c, world_z_c), -1.0, 1.0)
            angle_from_vertical = np.arccos(np.abs(cos_angle))  # 使用绝对值，因为可能向上或向下

            # 水平抓取：角度应该接近90度（π/2）
            deviation_from_horizontal = abs(angle_from_vertical - np.pi/2)

            if deviation_from_horizontal < horizontal_threshold:
                filtered.append(grasp)

        if len(filtered) == 0:
            print("[Warning] 没有符合水平角度阈值的抓取，使用所有预测。")
            filtered = list(gg)
        else:
            print(f"[FUSION] 过滤后剩余 {len(filtered)} 个水平抓取")
    
    # 选择最佳抓取
    if len(filtered) > 0:
        best_grasp = filtered[0]

        # 计算物体中心点（用于水平化调整）
        object_center = np.mean(cloud_fused, axis=0) if len(cloud_fused) > 0 else np.zeros(3)

        # ===== 调试信息：显示抓取位置 =====
        grasp_pos_cam = best_grasp.translation
        grasp_pos_world = transform_cloud_to_world(grasp_pos_cam.reshape(1, 3), T_wc_primary)[0]
        print(f"\n[FUSION DEBUG] 抓取位置:")
        print(f"  相机坐标系: ({grasp_pos_cam[0]:.3f}, {grasp_pos_cam[1]:.3f}, {grasp_pos_cam[2]:.3f})")
        print(f"  世界坐标系: ({grasp_pos_world[0]:.3f}, {grasp_pos_world[1]:.3f}, {grasp_pos_world[2]:.3f})")
        print(f"  物体中心 (相机): ({object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f})")
        object_center_world = transform_cloud_to_world(object_center.reshape(1, 3), T_wc_primary)[0]
        print(f"  物体中心 (世界): ({object_center_world[0]:.3f}, {object_center_world[1]:.3f}, {object_center_world[2]:.3f})")

        # ===== 货架物品：强制水平化抓取姿态 =====
        if grasp_mode == "horizontal":
            print("🔧 [FUSION] 货架物品：强制调整为水平抓取姿态...")

            # 获取当前抓取位置
            grasp_pos_c = best_grasp.translation

            # 计算从抓取点指向物体中心的方向（在相机坐标系中）
            to_center = object_center - grasp_pos_c
            to_center_norm = np.linalg.norm(to_center)

            if to_center_norm > 1e-6:
                # 将这个方向投影到水平面（去除Z分量）
                # 首先转换到世界坐标系
                to_center_world = R_wc @ to_center
                to_center_world[2] = 0  # 去除Z分量，保持水平

                # 归一化
                to_center_world_norm = np.linalg.norm(to_center_world)
                if to_center_world_norm > 1e-6:
                    to_center_world = to_center_world / to_center_world_norm

                    # 转回相机坐标系
                    approach_horizontal_c = R_cw @ to_center_world
                else:
                    # 如果投影后长度为0，使用默认水平方向（+X）
                    approach_horizontal_c = R_cw @ np.array([1, 0, 0])
            else:
                # 如果抓取点就在中心，使用默认水平方向
                approach_horizontal_c = R_cw @ np.array([1, 0, 0])

            # 归一化接近方向
            approach_horizontal_c = approach_horizontal_c / np.linalg.norm(approach_horizontal_c)

            # 构建新的旋转矩阵
            a_c = approach_horizontal_c  # 接近方向（水平）

            # 计算world_up_c
            world_up_c = R_cw @ np.array([0, 0, 1])

            # y轴应该垂直于接近方向和世界Z轴
            y_c = np.cross(world_up_c, a_c)
            y_norm = np.linalg.norm(y_c)

            if y_norm > 1e-6:
                y_c = y_c / y_norm
                # z轴 = x × y
                z_c = np.cross(a_c, y_c)
                z_c = z_c / np.linalg.norm(z_c)

                # 更新旋转矩阵
                best_grasp.rotation_matrix = np.column_stack([a_c, y_c, z_c])
                print("✅ [FUSION] 已强制调整为水平抓取姿态 (Horizontal Grasp Enforced)")
            else:
                print("⚠️ [FUSION] 无法构建水平抓取姿态，保持原姿态")

        new_gg = GraspGroup()
        new_gg.add(best_grasp)
        
        # 可视化
        if not HEADLESS:
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

                step_count += 1  # 必须在 if/else 外，确保关节空间插值时也递增

                action[:6] = joint
                if gripper_ctrl is not None:
                    action[-1] = gripper_ctrl
                env.step(action)
                if _render_callback is not None and step_count % _RENDER_INTERVAL == 0:
                    _render_callback()
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
    _step_cnt = 0
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
                _step_cnt += 1
                if _render_callback is not None and _step_cnt % _RENDER_INTERVAL == 0:
                    _render_callback()
                break


# ================= 桌面三相机融合模式的抓取执行（完全照搬 VLM_Grasp_bug） ====================
def _execute_grasp_desktop_fusion(env, gg, T_wc, T_wo_unused, target_pos, object_name):
    """桌面三相机融合模式专用的抓取执行。
    完全复制 VLM_Grasp_bug/grasp_process_optimized.py 的 execute_grasp 函数。
    T_wo_unused 不使用，函数内部自行计算 T_wo。
    仅增加: target_pos 参数支持、object_name 来源记录、_render_callback 回调。
    """
    robot = env.robot
    T_wb = robot.base

    # 0.初始准备阶段 —— 自行计算 T_wo
    if T_wc is None:
        n_wc = np.array([0.0, -1.0, 0.0])
        o_wc = np.array([-1.0, 0.0, -0.5])
        t_wc = np.array([0.85, 0.8, 1.6])
        T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    T_wo = T_wc * T_co

    # --- Debug: 打印抓取姿态诊断 ---
    print(f"\n[GRASP DEBUG - DESKTOP FUSION]")
    print(f"  Grasp in Camera Coords: {gg.translations[0]}")
    print(f"  Grasp rotation col0 (approach_cam): {gg.rotation_matrices[0][:, 0]}")
    print(f"  Grasp rotation col1 (binormal_cam): {gg.rotation_matrices[0][:, 1]}")
    print(f"  Grasp in World Coords:  {T_wo.t}")
    approach_world_actual = T_wo.R[:, 0]
    print(f"  ★ Approach dir (world): {approach_world_actual}")
    print(f"  ★ 期望30度倾斜:         [0, 0.5, -0.866]")
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

    # 记录物体来源位置（用于"放回原处"功能）
    if object_name:
        record_object_origin(object_name, grasp_world_pos, is_shelf, shelf_layer)

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态
    time1 = 1
    q0 = robot.get_joint()
    
    if is_shelf:
        # 架子抓取：使用专门的水平朝向预抓取姿态
        q1 = np.array([np.pi/2, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])
        print("  [SHELF] 使用水平朝向预抓取姿态...")
    else:
        # 桌面抓取：原有的垂直向下预抓取姿态
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

    # ===== 安全展开：先转到空旷区域展开手臂，再直接去接近点 =====
    grasp_near_shelf = (not is_shelf) and grasp_world_pos[0] > 1.2

    if grasp_near_shelf:
        print("  [SAFE DEPLOY] 抓取点靠近货架，先转到空旷区域展开...")
        safe_j0 = np.pi

        # 步骤1: 只转底座到安全方向（手臂收拢，不会碰）
        q_rot = q0.copy()
        q_rot[0] = safe_j0
        print(f"  [SAFE DEPLOY] 步骤1: 转底座 joint0={safe_j0:.2f}")
        time_r1 = 1.0
        _execute_planner_sequence(env, robot,
            [TrajectoryPlanner(TrajectoryParameter(JointParameter(q0, q_rot), QuinticVelocityParameter(time_r1)))],
            [0.0, time_r1])
        robot.set_joint(q_rot)

        # 步骤2: 在空旷侧展开手臂（只改 joint1-5，joint0 不动）
        q1_open = q1.copy()
        q1_open[0] = safe_j0
        print(f"  [SAFE DEPLOY] 步骤2: 在空旷侧展开 q1_open={q1_open}")
        time_r2 = 1.0
        _execute_planner_sequence(env, robot,
            [TrajectoryPlanner(TrajectoryParameter(JointParameter(q_rot, q1_open), QuinticVelocityParameter(time_r2)))],
            [0.0, time_r2])
        robot.set_joint(q1_open)

        # 步骤3: 从空旷区域直接去接近点（不再先转回货架旁的 q1）
        # 在空旷区域完成所有姿态调整，到达接近点时手臂已经就位
        T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
        q_approach = robot.ikine(T2)
        if len(q_approach) > 0:
            print(f"  [SAFE DEPLOY] 步骤3: 从空旷区域直接到接近点 q_approach={q_approach}")
            time_r3 = 2.0
            _execute_planner_sequence(env, robot,
                [TrajectoryPlanner(TrajectoryParameter(JointParameter(q1_open, q_approach), QuinticVelocityParameter(time_r3)))],
                [0.0, time_r3])
            robot.set_joint(q_approach)
            # 跳过后面的阶段2（T1→T2），直接标记已到达接近点
            q1 = q_approach
        else:
            # IK 失败，回退到原来的方式：转回 q1 再走笛卡尔
            print("  [SAFE DEPLOY] 接近点IK失败，回退：先转回q1...")
            time_r3 = 1.5
            _execute_planner_sequence(env, robot,
                [TrajectoryPlanner(TrajectoryParameter(JointParameter(q1_open, q1), QuinticVelocityParameter(time_r3)))],
                [0.0, time_r3])
    else:
        # 不靠近货架，或者是货架抓取，直接走原来的路径
        parameter0 = JointParameter(q0, q1)
        velocity_parameter0 = QuinticVelocityParameter(time1)
        trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
        planner1 = TrajectoryPlanner(trajectory_parameter0)
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
        

        # 阶段 2A: 移动到对准点（机械臂已在空旷区域展开并旋转到位，直接关节规划即可）
        print("  [SHELF] 阶段2A: 移动到架子正前方对准点...")
        align_x = SHELF_X_MIN - SHELF_APPROACH_OFFSET  # 架子前方 = 1.6 - 0.35 = 1.25
        align_point = np.array([align_x, grasp_world_pos[1], grasp_world_pos[2]])
        T_align = sm.SE3.Trans(align_point) * sm.SE3(R_horizontal)
        print(f"  [DEBUG] 对准点: X={align_x:.2f}, Y={grasp_world_pos[1]:.2f}, Z={grasp_world_pos[2]:.2f}")

        q_align = robot.ikine(T_align)
        if len(q_align) > 0:
            time2a = 1.5
            param_2a = JointParameter(q1, q_align)
            vel_2a = QuinticVelocityParameter(time2a)
            traj_2a = TrajectoryParameter(param_2a, vel_2a)
            planner_2a = TrajectoryPlanner(traj_2a)
            _execute_planner_sequence(env, robot, [planner_2a], [0.0, time2a])
            robot.set_joint(q_align)
        else:
            print("  [SHELF] 对准点IK失败，尝试调整位置...")
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
        # ===== 桌面抓取 =====
        T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)

        if grasp_near_shelf:
            # 安全展开路径已经直接到了接近点 T2，只需做最后的抓取动作
            print("  [TABLE] 安全展开已到达接近点，直接执行抓取...")
        else:
            # 原有直线规划：T1→T2
            time2 = 1
            position_parameter1 = LinePositionParameter(T1.t, T2.t)
            attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
            cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
            velocity_parameter1 = QuinticVelocityParameter(time2)
            trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
            planner2 = TrajectoryPlanner(trajectory_parameter1)
            _execute_planner_sequence(env, robot, [planner2], [0.0, time2])

        # 阶段 3: 执行抓取（回退一点，避免推倒物品）
        time3 = 1
        T3 = T_wo * sm.SE3(-0.015, 0.0, 0.0)
        position_parameter2 = LinePositionParameter(T2.t, T3.t)
        attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
        cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
        velocity_parameter2 = QuinticVelocityParameter(time3)
        trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
        planner3 = TrajectoryPlanner(trajectory_parameter2)
        
        # 执行阶段 3
        _execute_planner_sequence(env, robot, [planner3], [0.0, time3])
    
    # 使用当前真实末端姿态作为后续搬运的基准姿态
    # （而不是理想的 R_horizontal），这样可以避免 IK 误差导致的突然"自旋"
    T_grasp = robot.get_cartesian()
    grasp_rotation = sm.SO3(T_grasp.R)
    
    # 闭合夹爪抓取
    # 重要：闭合夹爪期间必须保持手臂关节不动（否则 action[:6] 默认 0 会把手臂拉回零位，引发末端乱转/抖动）
    # 注意：action[-1] 从 0 开始，每步 +0.2，需至少 1275 步才能到达 255（满闭合）
    for i in range(1500):
        action[:6] = robot.get_joint()
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)
        if _render_callback is not None and i % _RENDER_INTERVAL == 0:
            _render_callback()

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
    if target_pos is None:
        target_pos = [1.4, 0.5]
    tp = [target_pos[0], target_pos[1]]
    # target_pos = [0.2, 0.2] # 测试背后位置

    # 策略判断：根据目标位置和当前抓取位置选择合适的路径策略
    # 1. 背后区域：x < 0.5 且 y < 0.5，需要转身
    is_going_back = (tp[0] < 0.5 and tp[1] < 0.5)
    # 2. 货架区域：x > 1.2，需要使用关节空间避障，避免撞到显微镜等障碍物
    is_going_to_shelf = (tp[0] > 1.2)
    # 3. 从显微镜附近（x > 1.0）去货架，需要特别小心
    grasp_near_microscope = (grasp_world_pos[0] > 1.0 and grasp_world_pos[0] < 1.5)
    needs_safe_path = is_going_to_shelf and grasp_near_microscope

    if is_going_back:
        # 【去背后 (0.2, 0.2)】：需要大角度旋转，且容易碰到奇异点
        # 目标姿态：朝下朝后
        T_target_high = sm.SE3.Trans(tp[0], tp[1], T_lift.t[2]) * sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
        use_joint_transit_strategy = True
        print(f"[PLACE] 去背后区域，使用关节空间插值")
    elif needs_safe_path:
        # 【从显微镜附近去货架】：需要使用关节空间避障
        # 目标姿态：保持抓取时的姿态
        T_target_high = sm.SE3.Trans(tp[0], tp[1], T_lift.t[2]) * sm.SE3(grasp_rotation)
        use_joint_transit_strategy = True
        print(f"[PLACE] 从显微镜附近去货架，使用关节空间避障")
    else:
        # 【去侧面/前方 (1.4, 0.3)】：不需要转身，直接平移
        # 目标姿态：保持抓取时的姿态 (grasp_rotation)
        T_target_high = sm.SE3.Trans(tp[0], tp[1], T_lift.t[2]) * sm.SE3(grasp_rotation)
        use_joint_transit_strategy = False # 侧面直接走直线 Cartesian 即可
        print(f"[PLACE] 去侧面/前方，使用笛卡尔直线")

    
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
    time6 = 1.0
    # 下降回原来的高度 (实现轻拿轻放，保持姿态不变)
    T6 = sm.SE3.Trans(tp[0], tp[1], T3.t[2]) * sm.SE3(grasp_rotation)
    
    # 从 T_target_high 直降到 T6（保持姿态不变）
    pos_drop = LinePositionParameter(T_target_high.t, T6.t)
    att_drop = OneAttitudeParameter(grasp_rotation, grasp_rotation)
    traj_drop = TrajectoryParameter(CartesianParameter(pos_drop, att_drop), QuinticVelocityParameter(time6))
    planner6 = TrajectoryPlanner(traj_drop)


    # 执行 planner_array (后续搬运逻辑)
    # 对于架子抓取：稳定三段式（退出货架 -> 平移到目标上方 -> 下降放置），全程保持抓取姿态不变
    if is_shelf:
        print("  [SHELF] 架子抓取后：退出货架 -> 平移到目标 -> 下降放置")

        # 直接执行 planner4 (4B垂直提升)，保持夹爪闭合
        _execute_planner_sequence(env, robot, [planner4], [0.0, time4], gripper_ctrl=255)
        # 继续提升一点，确保离开货架上沿，仍然保持闭合
        _execute_planner_sequence(env, robot, [planner_lift], [0.0, time_lift], gripper_ctrl=255)

        # 用当前真实位姿作为后续搬运起点，避免用理想的 T_lift 造成突变
        T_after_lift = robot.get_cartesian()

        # === 平移到目标上方（保持姿态不变）===
        T_target_high = sm.SE3.Trans(tp[0], tp[1], T_after_lift.t[2]) * sm.SE3(grasp_rotation)
        time_move = 2.0
        pos_move = LinePositionParameter(T_after_lift.t, T_target_high.t)
        att_move = OneAttitudeParameter(grasp_rotation, grasp_rotation)
        traj_move = TrajectoryParameter(CartesianParameter(pos_move, att_move), QuinticVelocityParameter(time_move))
        planner_move = TrajectoryPlanner(traj_move)
        _execute_planner_sequence(env, robot, [planner_move], [0.0, time_move], gripper_ctrl=255)

        # === 下降放置（保持姿态不变）===
        time_drop = 1.2
        T_drop = sm.SE3.Trans(tp[0], tp[1], T3.t[2]) * sm.SE3(grasp_rotation)
        pos_drop2 = LinePositionParameter(T_target_high.t, T_drop.t)
        att_drop2 = OneAttitudeParameter(grasp_rotation, grasp_rotation)
        traj_drop2 = TrajectoryParameter(CartesianParameter(pos_drop2, att_drop2), QuinticVelocityParameter(time_drop))
        planner_drop2 = TrajectoryPlanner(traj_drop2)
        _execute_planner_sequence(env, robot, [planner_drop2], [0.0, time_drop], gripper_ctrl=255)

        # 松开夹爪（同样保持手臂关节不动）
        for i in range(1000):
            action[:6] = robot.get_joint()
            action[-1] -= 0.2
            action[-1] = np.max([action[-1], 0])
            env.step(action)
        if _render_callback is not None and i % _RENDER_INTERVAL == 0:
            _render_callback()

        # === 架子抓取：自动复原到初始姿态 q0 ===
        print("  [SHELF] 放置完成，执行自动复原到初始姿态...")

        # 1) 先在当前位置基础上再抬高 0.1m，避免回程时蹭到货架/物体
        T_cur = robot.get_cartesian()
        T_up = sm.SE3.Trans(T_cur.t[0], T_cur.t[1], T_cur.t[2] + 0.1) * sm.SE3(sm.SO3(T_cur.R))
        time_up = 1.0
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

    # === 特殊处理：桌面抓取后放回货架 ===
    is_shelf_place = (target_pos is not None and len(target_pos) >= 1 and
                      SHELF_X_MIN <= target_pos[0] <= SHELF_X_MAX)
    if is_shelf_place:
        print(f"\n[SHELF PLACE - DESKTOP FUSION] 货架放置策略，目标: {target_pos}")

        # 构建水平放置姿态
        approach_dir = np.array([1, 0, 0])
        side_dir = np.array([0, 1, 0])
        R_horizontal = sm.SO3.TwoVectors(x=approach_dir, y=side_dir)

        # 阶段 A: 先抬起物体到安全高度（保持当前抓取姿态，避免旋转时碰撞）
        T_current = robot.get_cartesian()
        current_rotation = sm.SO3(T_current.R)
        print(f"  [SHELF PLACE] 阶段A: 垂直抬升 (当前x={T_current.t[0]:.2f})...")

        # 垂直抬升 0.3m，保持当前倾斜姿态不变
        lift_h = 0.3
        T_lift_up = sm.SE3.Trans(T_current.t[0], T_current.t[1],
                                  T_current.t[2] + lift_h) * sm.SE3(current_rotation)
        time_lift_sp = 1.0
        pos_lift_sp = LinePositionParameter(T_current.t, T_lift_up.t)
        att_lift_sp = OneAttitudeParameter(current_rotation, current_rotation)
        traj_lift_sp = TrajectoryParameter(
            CartesianParameter(pos_lift_sp, att_lift_sp),
            QuinticVelocityParameter(time_lift_sp))
        planner_lift_sp = TrajectoryPlanner(traj_lift_sp)
        _execute_planner_sequence(env, robot, [planner_lift_sp],
                                  [0.0, time_lift_sp], gripper_ctrl=255)
        print(f"  [SHELF PLACE] 已抬升到 z={T_current.t[2] + lift_h:.2f}")

        # 阶段 A2: 在安全高度，用关节空间旋转到水平姿态（180度）
        # 目标：夹爪水平朝向+X，在当前位置上方
        T_after_lift = robot.get_cartesian()
        T_horizontal_high = sm.SE3.Trans(T_after_lift.t) * sm.SE3(R_horizontal)
        q_now_h = robot.get_joint()
        q_horizontal = robot.ikine(T_horizontal_high)
        if len(q_horizontal) > 0:
            print(f"  [SHELF PLACE] 阶段A2: 旋转到水平姿态...")
            time_rot = 1.5
            param_rot = JointParameter(q_now_h, q_horizontal)
            traj_rot = TrajectoryParameter(param_rot, QuinticVelocityParameter(time_rot))
            planner_rot = TrajectoryPlanner(traj_rot)
            _execute_planner_sequence(env, robot, [planner_rot],
                                      [0.0, time_rot], gripper_ctrl=255)
        else:
            print(f"  [SHELF PLACE] 水平姿态IK失败，尝试用Rz(pi)*Rx(pi)...")
            R_down = sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
            T_down_high = sm.SE3.Trans(T_after_lift.t) * R_down
            q_horizontal = robot.ikine(T_down_high)
            if len(q_horizontal) > 0:
                time_rot = 1.5
                param_rot = JointParameter(q_now_h, q_horizontal)
                traj_rot = TrajectoryParameter(param_rot, QuinticVelocityParameter(time_rot))
                planner_rot = TrajectoryPlanner(traj_rot)
                _execute_planner_sequence(env, robot, [planner_rot],
                                          [0.0, time_rot], gripper_ctrl=255)
            else:
                print(f"  [SHELF PLACE] 旋转IK均失败，继续用当前姿态")

        # 阶段 B: 用关节空间移动到货架前方对准点（此时已是水平姿态）
        align_x = SHELF_X_MIN - SHELF_APPROACH_OFFSET
        align_point = np.array([align_x, target_pos[1], target_pos[2]])
        T_align = sm.SE3.Trans(align_point) * sm.SE3(R_horizontal)

        q_now = robot.get_joint()
        q_align = robot.ikine(T_align)
        if len(q_align) > 0:
            time_align = 2.5
            param_align = JointParameter(q_now, q_align)
            traj_align = TrajectoryParameter(param_align, QuinticVelocityParameter(time_align))
            planner_align = TrajectoryPlanner(traj_align)
            _execute_planner_sequence(env, robot, [planner_align], [0.0, time_align], gripper_ctrl=255)
        else:
            print(f"  [SHELF PLACE] 对准点 IK 失败，尝试调整...")
            align_point_adj = align_point + np.array([0.1, 0, -0.05])
            T_align_adj = sm.SE3.Trans(align_point_adj) * sm.SE3(R_horizontal)
            q_align = robot.ikine(T_align_adj)
            if len(q_align) > 0:
                time_align = 2.5
                param_align = JointParameter(q_now, q_align)
                traj_align = TrajectoryParameter(param_align, QuinticVelocityParameter(time_align))
                planner_align = TrajectoryPlanner(traj_align)
                _execute_planner_sequence(env, robot, [planner_align], [0.0, time_align], gripper_ctrl=255)
                T_align = T_align_adj
            else:
                print(f"  [SHELF PLACE] [ERROR] 无法到达货架对准点")
        print(f"  [SHELF PLACE] 已到达对准点 x={align_x:.2f}")

        # 阶段 C: 水平插入货架到目标位置
        T_align_current = robot.get_cartesian()
        R_current = sm.SO3(T_align_current.R)
        T_insert = sm.SE3.Trans(target_pos) * sm.SE3(R_current)
        time_insert = 1.5
        pos_insert = LinePositionParameter(T_align_current.t, T_insert.t)
        att_insert = OneAttitudeParameter(R_current, R_current)
        traj_insert = TrajectoryParameter(CartesianParameter(pos_insert, att_insert), QuinticVelocityParameter(time_insert))
        planner_insert = TrajectoryPlanner(traj_insert)
        _execute_planner_sequence(env, robot, [planner_insert], [0.0, time_insert], gripper_ctrl=255)
        print(f"  [SHELF PLACE] 已插入货架位置")

        # 阶段 D: 松开夹爪
        for i in range(2000):
            action[:6] = robot.get_joint()
            action[-1] -= 0.1
            action[-1] = np.max([action[-1], 0])
            env.step(action)
            if _render_callback is not None and i % _RENDER_INTERVAL == 0:
                _render_callback()
        print(f"  [SHELF PLACE] 夹爪已松开")

        # 阶段 E: 水平退出货架（沿 -X 方向）
        T_retreat_pos = np.array([align_x, target_pos[1], target_pos[2]])
        T_now = robot.get_cartesian()
        R_now = sm.SO3(T_now.R)
        T_retreat = sm.SE3.Trans(T_retreat_pos) * sm.SE3(R_now)
        time_retreat = 1.0
        pos_retreat = LinePositionParameter(T_now.t, T_retreat.t)
        att_retreat = OneAttitudeParameter(R_now, R_now)
        traj_retreat = TrajectoryParameter(CartesianParameter(pos_retreat, att_retreat), QuinticVelocityParameter(time_retreat))
        planner_retreat = TrajectoryPlanner(traj_retreat)
        _execute_planner_sequence(env, robot, [planner_retreat], [0.0, time_retreat], gripper_ctrl=0)
        print(f"  [SHELF PLACE] 已退出货架")

        # 阶段 F: 回到初始姿态
        q_now = robot.get_joint()
        time_back = 1.5
        param_back = JointParameter(q_now, q0)
        traj_back = TrajectoryParameter(param_back, QuinticVelocityParameter(time_back))
        planner_back = TrajectoryPlanner(traj_back)
        _execute_planner_sequence(env, robot, [planner_back], [0.0, time_back], gripper_ctrl=0)

        print(f"  [SHELF PLACE] 货架放置完成！")
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
    for i in range(1000):
        action[:6] = robot.get_joint()
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)
        if _render_callback is not None and i % _RENDER_INTERVAL == 0:
            _render_callback()

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪，避免碰撞物体。
    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
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

    # 8.安全回到初始位置（避免撞货架）
    # 和抓取时一样的策略：先转底座到空旷区域，收拢手臂，再转回来
    q8 = robot.get_joint()

    if grasp_near_shelf:
        print("  [SAFE RETURN] 靠近货架，安全复原...")
        safe_j0 = np.pi

        # 步骤1: 只转底座到安全方向
        q_rot_back = q8.copy()
        q_rot_back[0] = safe_j0
        time_r1 = 1.0
        _execute_planner_sequence(env, robot,
            [TrajectoryPlanner(TrajectoryParameter(JointParameter(q8, q_rot_back), QuinticVelocityParameter(time_r1)))],
            [0.0, time_r1])
        robot.set_joint(q_rot_back)

        # 步骤2: 在空旷侧收拢到初始姿态（joint0 保持 safe_j0）
        q0_open = q0.copy()
        q0_open[0] = safe_j0
        time_r2 = 1.0
        _execute_planner_sequence(env, robot,
            [TrajectoryPlanner(TrajectoryParameter(JointParameter(q_rot_back, q0_open), QuinticVelocityParameter(time_r2)))],
            [0.0, time_r2])
        robot.set_joint(q0_open)

        # 步骤3: 转回初始 joint0（手臂已收拢，安全）
        time_r3 = 1.0
        _execute_planner_sequence(env, robot,
            [TrajectoryPlanner(TrajectoryParameter(JointParameter(q0_open, q0), QuinticVelocityParameter(time_r3)))],
            [0.0, time_r3])
        robot.set_joint(q0)
        print("  [SAFE RETURN] 安全复原完成")
    else:
        # 不靠近货架，直接回
        time8 = 1
        parameter8 = JointParameter(q8, q0)
        velocity_parameter8 = QuinticVelocityParameter(time8)
        trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
        planner8 = TrajectoryPlanner(trajectory_parameter8)
        _execute_planner_sequence(env, robot, [planner8], [0.0, time8])


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg, T_wc=None, target_pos=None, object_name=None, desktop_fusion=False):

    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    T_wc (sm.SE3): 世界坐标系到相机坐标系的变换矩阵。
    target_pos (list): 放置位置 [x, y, z]，如果为None则使用默认位置。
    object_name (str): 物体名称，用于记录来源位置（放回功能）。
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

    # ===== 桌面三相机融合模式：完全使用 VLM_Grasp_bug 的抓取执行逻辑 =====
    if desktop_fusion:
        return _execute_grasp_desktop_fusion(env, gg, T_wc, T_wo, target_pos, object_name)

    # ===== 架子抓取检测 =====
    # 架子层高度列表 (根据 scene.xml 中的配置)
    SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
    SHELF_LAYER_TOLERANCE = 0.15  # 检测容差
    # 根据scene.xml: 货架碰撞层中心X=1.79, Y=0.6, 半宽X=0.18, 半宽Y=0.6
    SHELF_X_MIN = 1.61   # 架子前沿 = 1.79 - 0.18
    SHELF_X_MAX = 1.97   # 架子后沿 = 1.79 + 0.18
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

    # 记录物体来源位置（用于"放回原处"功能）
    if object_name:
        record_object_origin(object_name, grasp_world_pos, is_shelf, shelf_layer)

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态
    time1 = 1
    q0 = robot.get_joint()

    # 检测物体是否在机器人背后区域（桌面抓取且 x < 0.5）
    grasp_x = T_wo.t[0]
    is_behind_table = (not is_shelf) and (grasp_x < 0.5)
    # 绿色区域等靠近货架的桌面物品需要先后退再接近，避免经过货架
    is_near_shelf_table = (not is_shelf) and (grasp_x > 1.0)

    if is_behind_table or is_near_shelf_table:
        # === 特殊区域桌面抓取：跳过标准预抓取姿态，直接关节空间接近 ===
        region_name = "背后区域" if is_behind_table else "货架附近区域"
        print(f"  [TABLE] 物体在{region_name} (x={grasp_x:.2f})，使用直接接近策略")

        grasp_pos = T_wo.t
        # 背后区域：基座不旋转，用 Rz(π)*Rx(π)
        # 货架附近：基座已旋转180°，只需 Rx(π) 即可垂直向下，避免腕部翻转
        if is_near_shelf_table:
            R_down = sm.SE3.Rx(np.pi)
        else:
            R_down = sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
        T_pre = sm.SE3.Trans(grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.10) * R_down
        T3 = sm.SE3.Trans(grasp_pos[0], grasp_pos[1], grasp_pos[2] - 0.015) * R_down

        # 货架附近区域：先后退到安全位置，避免经过货架
        if is_near_shelf_table:
            q_retract = np.array([np.pi, -np.pi/4, np.pi/2, 0.0, -np.pi/2, 0.0])
            print(f"  [TABLE] 货架附近：先后退到安全位置...")
            time_retract = 1.5
            param_retract = JointParameter(q0, q_retract)
            traj_retract = TrajectoryParameter(param_retract, QuinticVelocityParameter(time_retract))
            planner_retract = TrajectoryPlanner(traj_retract)
            _execute_planner_sequence(env, robot, [planner_retract], [0.0, time_retract])
            robot.set_joint(q_retract)
            q_start = q_retract
        else:
            q_start = q0

        # 直接从起始位置用关节空间规划到预抓取点
        q_pre = robot.ikine(T_pre)
        if len(q_pre) > 0:
            print(f"  [TABLE] 预抓取点 IK 成功，关节空间规划")
            time_move = 2.0
            param_move = JointParameter(q_start, q_pre)
            traj_move = TrajectoryParameter(param_move, QuinticVelocityParameter(time_move))
            planner_move = TrajectoryPlanner(traj_move)
            _execute_planner_sequence(env, robot, [planner_move], [0.0, time_move])
            robot.set_joint(q_pre)
        else:
            print(f"  [TABLE] 预抓取点IK失败，尝试更高位置...")
            for extra_h in [0.20, 0.30, 0.40]:
                T_high = sm.SE3.Trans(grasp_pos[0], grasp_pos[1], grasp_pos[2] + extra_h) * R_down
                q_high = robot.ikine(T_high)
                if len(q_high) > 0:
                    print(f"  [TABLE] 高度+{extra_h:.2f}m IK 成功")
                    time_move = 2.0
                    param_move = JointParameter(q_start, q_high)
                    traj_move = TrajectoryParameter(param_move, QuinticVelocityParameter(time_move))
                    planner_move = TrajectoryPlanner(traj_move)
                    _execute_planner_sequence(env, robot, [planner_move], [0.0, time_move])
                    robot.set_joint(q_high)

                    time_desc = 1.5
                    R_d = sm.SO3(T_high.R)
                    pos_desc = LinePositionParameter(T_high.t, T_pre.t)
                    att_desc = OneAttitudeParameter(R_d, R_d)
                    traj_desc = TrajectoryParameter(CartesianParameter(pos_desc, att_desc),
                                                     QuinticVelocityParameter(time_desc))
                    planner_desc = TrajectoryPlanner(traj_desc)
                    _execute_planner_sequence(env, robot, [planner_desc], [0.0, time_desc])
                    break
            else:
                raise RuntimeError("桌面抓取：无法找到有效IK解")

        # 笛卡尔直线下降到精确抓取点
        T_current = robot.get_cartesian()
        time3 = 1.0
        R_cur = sm.SO3(T_current.R)
        pos3 = LinePositionParameter(T_current.t, T3.t)
        att3 = OneAttitudeParameter(R_cur, R_cur)
        traj3 = TrajectoryParameter(CartesianParameter(pos3, att3),
                                     QuinticVelocityParameter(time3))
        planner3 = TrajectoryPlanner(traj3)
        _execute_planner_sequence(env, robot, [planner3], [0.0, time3])

    else:
        # === 正常流程：先到预抓取姿态 ===
        if is_shelf:
            q1 = np.array([np.pi/2, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])
            print("  [SHELF] 使用水平朝向预抓取姿态...")
        else:
            # 桌面抓取：原有的垂直向下预抓取姿态
            q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

        parameter0 = JointParameter(q0, q1)
        velocity_parameter0 = QuinticVelocityParameter(time1)
        trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
        planner1 = TrajectoryPlanner(trajectory_parameter0)
        _execute_planner_sequence(env, robot, [planner1], [0.0, time1])

    # 2.接近抓取位姿 + 3.执行抓取
    # 特殊区域已在上面处理完毕，这里只处理正常区域
    if not (is_behind_table or is_near_shelf_table):
        robot.set_joint(q1)
        T1 = robot.get_cartesian()

        if is_shelf:
            # ===== 架子抓取：分段水平接近 =====

            # 构建水平抓取姿态：
            approach_dir = np.array([1, 0, 0])
            side_dir = np.array([0, 1, 0])
            R_horizontal = sm.SO3.TwoVectors(x=approach_dir, y=side_dir)
            print(f"  [DEBUG] R_horizontal:\n{R_horizontal}")

            # 阶段 2A: 先用关节空间规划移动到对准点
            print("  [SHELF] 阶段2A: 移动到架子正前方对准点...")
            align_x = SHELF_X_MIN - SHELF_APPROACH_OFFSET
            align_point = np.array([align_x, grasp_world_pos[1], grasp_world_pos[2]])
            T_align = sm.SE3.Trans(align_point) * sm.SE3(R_horizontal)
            print(f"  [DEBUG] 对准点: X={align_x:.2f}, Y={grasp_world_pos[1]:.2f}, Z={grasp_world_pos[2]:.2f}")

            q_align = robot.ikine(T_align)
            if len(q_align) > 0:
                time2a = 1.5
                param_2a = JointParameter(q1, q_align)
                vel_2a = QuinticVelocityParameter(time2a)
                traj_2a = TrajectoryParameter(param_2a, vel_2a)
                planner_2a = TrajectoryPlanner(traj_2a)
                _execute_planner_sequence(env, robot, [planner_2a], [0.0, time2a])
                robot.set_joint(q_align)
            else:
                print("  [SHELF] 对准点IK失败，尝试调整位置...")
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
            T2_pos = grasp_world_pos - np.array([0.01, 0, 0])
            T2 = sm.SE3.Trans(T2_pos) * sm.SE3(R_horizontal)
            time2b = 1.5
            pos_param_2b = LinePositionParameter(T_align.t, T2.t)
            att_param_2b = OneAttitudeParameter(R_horizontal, R_horizontal)
            cart_param_2b = CartesianParameter(pos_param_2b, att_param_2b)
            vel_param_2b = QuinticVelocityParameter(time2b)
            traj_param_2b = TrajectoryParameter(cart_param_2b, vel_param_2b)
            planner_2b = TrajectoryPlanner(traj_param_2b)
            _execute_planner_sequence(env, robot, [planner_2b], [0.0, time2b])

            print("  [SHELF] 阶段3: 执行水平抓取...")
            T3_pos = grasp_world_pos + np.array([0.01, 0, 0])
            T3 = sm.SE3.Trans(T3_pos) * sm.SE3(R_horizontal)
            time3 = 1.0
            pos_param_3 = LinePositionParameter(T2.t, T3.t)
            att_param_3 = OneAttitudeParameter(R_horizontal, R_horizontal)
            cart_param_3 = CartesianParameter(pos_param_3, att_param_3)
            vel_param_3 = QuinticVelocityParameter(time3)
            traj_param_3 = TrajectoryParameter(cart_param_3, vel_param_3)
            planner_3 = TrajectoryPlanner(traj_param_3)
            _execute_planner_sequence(env, robot, [planner_3], [0.0, time3])
        else:
            # ===== 桌面抓取 =====
            T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)

            time2 = 1
            position_parameter1 = LinePositionParameter(T1.t, T2.t)
            attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
            cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
            velocity_parameter1 = QuinticVelocityParameter(time2)
            trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
            planner2 = TrajectoryPlanner(trajectory_parameter1)
            try:
                _execute_planner_sequence(env, robot, [planner2], [0.0, time2])
            except RuntimeError:
                # 笛卡尔接近IK失败，回退到关节空间
                print("  [TABLE] 笛卡尔接近IK失败，回退到关节空间规划...")
                q_pre = robot.ikine(T2)
                if len(q_pre) > 0:
                    q_now = robot.get_joint()
                    param_fb = JointParameter(q_now, q_pre)
                    traj_fb = TrajectoryParameter(param_fb, QuinticVelocityParameter(time2))
                    planner_fb = TrajectoryPlanner(traj_fb)
                    _execute_planner_sequence(env, robot, [planner_fb], [0.0, time2])
                else:
                    # 用干净的垂直向下姿态重试
                    print("  [TABLE] 原始姿态IK也失败，尝试垂直向下姿态...")
                    R_clean = sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
                    T2_clean = sm.SE3.Trans(grasp_world_pos + np.array([0, 0, 0.10])) * R_clean
                    q_pre2 = robot.ikine(T2_clean)
                    if len(q_pre2) > 0:
                        q_now = robot.get_joint()
                        param_fb2 = JointParameter(q_now, q_pre2)
                        traj_fb2 = TrajectoryParameter(param_fb2, QuinticVelocityParameter(time2))
                        planner_fb2 = TrajectoryPlanner(traj_fb2)
                        _execute_planner_sequence(env, robot, [planner_fb2], [0.0, time2])
                        # 更新 T_wo 为干净姿态，后续抓取也用这个
                        T_wo = sm.SE3.Trans(grasp_world_pos) * R_clean
                    else:
                        raise RuntimeError("桌面抓取：无法找到有效的接近姿态IK解")

            # 阶段 3: 执行抓取 - T2→T3
            time3 = 1
            T3 = T_wo * sm.SE3(0.015, 0.0, 0.0)
            position_parameter2 = LinePositionParameter(T2.t, T3.t)
            attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
            cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
            velocity_parameter2 = QuinticVelocityParameter(time3)
            trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
            planner3 = TrajectoryPlanner(trajectory_parameter2)
            _execute_planner_sequence(env, robot, [planner3], [0.0, time3])

    # 使用当前真实末端姿态作为后续搬运的基准姿态
    # （而不是理想的 R_horizontal），这样可以避免 IK 误差导致的突然“自旋”
    T_grasp = robot.get_cartesian()
    grasp_rotation = sm.SO3(T_grasp.R)
    
    # 闭合夹爪抓取
    # 重要：闭合夹爪期间必须保持手臂关节不动（否则 action[:6] 默认 0 会把手臂拉回零位，引发末端乱转/抖动）
    # 注意：action[-1] 从 0 开始，每步 +0.2，需至少 1275 步才能到达 255（满闭合）
    for i in range(1500):
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
    # 定义最终放置位置
    # target_pos = [x, y, place_height]
    # place_height: 放置时松开夹爪的高度（机械臂下降到此高度后松开）
    if target_pos is None:
        # 使用默认放置位置
        target_pos = [0.2, 0.2, 0.92]  # 默认背后位置
        print(f"[PLACE] 使用默认放置位置: {target_pos}")
    else:
        print(f"[PLACE] 使用指定放置位置: {target_pos}")

    # 放置后抬升高度（单独配置，不在target_pos中）
    lift_height_after_place = 0.35  # 松开后抬升15cm到安全高度

    # 策略判断：根据放置位置选择合适的搬运策略
    # 机械臂基座大约在 (0, 0) 位置，货架在 x > 1.6 的位置
    #
    # 区域划分（俯视图）：
    #        Y轴
    #        ^
    #   1.2  +------------------+
    #        |   左后方区域      |  (需要转身，关节空间)
    #        |   x<0.5, y>0.5   |
    #   0.5  +--------+---------+
    #        | 背后   | 中间区域 |  侧面/货架方向
    #        | x<0.5  | 0.5<=x  |  x>=1.2
    #        | y<0.5  | <1.2    |  (保持原姿态)
    #   0    +--------+---------+-----> X轴
    #        0       0.5       1.2
    #
    # 简化逻辑：只要 x < 0.5，就需要转身，使用关节空间插值
    # 额外：如果抓取点在背后区域，放置点在远处，也需要关节空间中转
    grasp_in_behind = (is_behind_table or is_near_shelf_table)

    needs_turn_around = (target_pos[0] < 0.7)  # x < 0.7 需要转身（靠近底座区域）
    is_middle_area = (0.7 <= target_pos[0] < 1.2 and not needs_turn_around)
    is_side_area = (target_pos[0] >= 1.2)  # 货架方向

    # 如果从背后区域抓取，放到侧面/中间，距离太远不适合笛卡尔直线，强制用关节空间
    if grasp_in_behind and (is_middle_area or is_side_area):
        is_middle_area = True
        is_side_area = False

    print(f"[PLACE] 位置分析: x={target_pos[0]:.2f}, y={target_pos[1]:.2f}")
    print(f"[PLACE] 区域判断: needs_turn_around={needs_turn_around}, is_middle_area={is_middle_area}, is_side_area={is_side_area}")

    # 根据目标位置选择正确的垂直向下旋转方向
    # x < 0.7（背后/靠近底座区域）→ Rz(π)*Rx(π)（面向背后的垂直向下）
    # x >= 0.7（前方区域）→ Rx(π)（面向前方的垂直向下）
    def _get_place_rotation(tx):
        if tx < 0.7:
            return sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
        else:
            return sm.SE3.Rx(np.pi)

    # 根据区域选择姿态和策略
    if needs_turn_around:
        print(f"[PLACE] 检测到需要转身的区域（x<0.7），使用关节空间插值")
        place_rotation_se3 = _get_place_rotation(target_pos[0])
        use_joint_transit_strategy = True
    elif is_middle_area:
        print(f"[PLACE] 检测到中间区域放置，将使用垂直向下姿态")
        place_rotation_se3 = _get_place_rotation(target_pos[0])
        use_joint_transit_strategy = True
    else:
        print(f"[PLACE] 检测到侧面/货架方向，保持原姿态")
        place_rotation_se3 = sm.SE3(grasp_rotation)
        use_joint_transit_strategy = False

    if needs_turn_around:
        # 【需要转身】：使用垂直向下姿态
        T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_lift.t[2]) * place_rotation_se3
    elif is_middle_area:
        # 【去中间/远处区域】：使用垂直向下姿态
        T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_lift.t[2]) * place_rotation_se3
    else:
        # 【去侧面/前方 (1.4, 0.3)】：不需要转身，直接平移
        # 目标姿态：保持抓取时的姿态 (grasp_rotation 已经是 SO3)
        T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_lift.t[2]) * sm.SE3(grasp_rotation)

    
    # 获取当前的关节角度
    q_start = robot.ikine(T_lift)
    if len(q_start) == 0:
        q_start = robot.get_joint()
        
    # 计算目标的关节角度
    # 开始规划路径
    planner_transit = None
    time_transit = 2.0

    if use_joint_transit_strategy:
        # === 策略A：去背后或中间区域 (复杂模式) ===
        # 优先尝试关节插值，失败则回退到 Waypoint

        # 1. 尝试计算关节目标
        q_target = robot.ikine(T_target_high)

        if len(q_target) > 0:
            # IK成功：直接使用关节空间插值
            strategy_name = "需要转身" if needs_turn_around else "中间区域"
            print(f"去{strategy_name}：IK成功，使用关节空间插值。")
            traj_transit = TrajectoryParameter(JointParameter(q_start, q_target), QuinticVelocityParameter(time_transit))
            planner_transit = TrajectoryPlanner(traj_transit)
        else:
            # IK失败：启用安全中转点 fallback（纯关节空间）
            strategy_name = "需要转身" if needs_turn_around else "中间区域"
            print(f"去{strategy_name}：IK失败，启用安全中转点策略。")
            time_transit = 3.0
            # 中转点：使用标准预抓取关节角（安全的中间姿态）
            q_waypoint = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

            # Lift -> Waypoint (关节空间)
            traj1 = TrajectoryParameter(JointParameter(q_start, q_waypoint), QuinticVelocityParameter(time_transit/2))
            planner1 = TrajectoryPlanner(traj1)

            # Waypoint -> Target (关节空间)
            # 从中转点重新计算目标 IK
            robot.set_joint(q_waypoint)
            T_wp_cart = robot.get_cartesian()
            # 用中转点高度重新构建目标
            T_target_high2 = sm.SE3.Trans(target_pos[0], target_pos[1], T_wp_cart.t[2]) * place_rotation_se3
            q_target2 = robot.ikine(T_target_high2)
            if len(q_target2) > 0:
                traj2 = TrajectoryParameter(JointParameter(q_waypoint, q_target2), QuinticVelocityParameter(time_transit/2))
                planner2 = TrajectoryPlanner(traj2)
                planner_transit = [planner1, planner2]
                # 更新 T_target_high 供后续下降使用
                T_target_high = T_target_high2
            else:
                print(f"  [WARNING] 中转点后目标IK也失败，仅移动到中转点")
                planner_transit = planner1
                time_transit = time_transit / 2
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
    # 使用关节空间中转时，下降姿态应与目标姿态一致（而非原始抓取姿态）
    drop_rotation = sm.SO3(place_rotation_se3.R) if use_joint_transit_strategy else grasp_rotation
    T6 = sm.SE3.Trans(target_pos[0], target_pos[1], target_pos[2]) * sm.SE3(drop_rotation)

    # 从 T_target_high 直降到 T6（保持姿态不变）
    pos_drop = LinePositionParameter(T_target_high.t, T6.t)
    att_drop = OneAttitudeParameter(drop_rotation, drop_rotation)
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
        if needs_turn_around:
            print("  [SHELF] 放置目标需要转身（x<0.7），使用关节空间搬运策略...")
            # 目标姿态：垂直向下
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
                 print("  [SHELF] [WARNING] 目标点 IK 失败，尝试使用中转点...")
                 # 使用安全中转点
                 T_waypoint = sm.SE3.Trans(0.5, 0.5, T_after_lift.t[2]) * sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
                 q_waypoint = robot.ikine(T_waypoint)

                 if len(q_waypoint) > 0:
                     # 先移动到中转点
                     time_move = 2.0
                     param_move1 = JointParameter(q_now, q_waypoint)
                     traj_move1 = TrajectoryParameter(param_move1, QuinticVelocityParameter(time_move))
                     planner_move1 = TrajectoryPlanner(traj_move1)
                     _execute_planner_sequence_with_compensation(
                         env, robot, [planner_move1], [0.0, time_move],
                         gripper_ctrl=255,
                         keep_level=True,
                         initial_grasp_rotation=sm.SE3(grasp_rotation)
                     )

                     # 再从中转点移动到目标
                     q_now2 = robot.get_joint()
                     q_target2 = robot.ikine(T_target_high)
                     if len(q_target2) > 0:
                         param_move2 = JointParameter(q_now2, q_target2)
                         traj_move2 = TrajectoryParameter(param_move2, QuinticVelocityParameter(time_move))
                         planner_move2 = TrajectoryPlanner(traj_move2)
                         _execute_planner_sequence_with_compensation(
                             env, robot, [planner_move2], [0.0, time_move],
                             gripper_ctrl=255,
                             keep_level=True,
                             initial_grasp_rotation=sm.SE3(grasp_rotation)
                         )
                 final_rotation = grasp_rotation

        elif is_middle_area:
            print("  [SHELF] 放置目标在中间区域，使用关节空间搬运策略（垂直向下姿态）...")
            # 目标姿态：垂直向下（与 needs_turn_around 相同策略，使用 keep_level 补偿）
            T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_after_lift.t[2]) * place_rotation_se3

            # 使用 IK 计算关节角
            q_now = robot.get_joint()
            q_target = robot.ikine(T_target_high)

            if len(q_target) > 0:
                 time_move = 3.0
                 param_move = JointParameter(q_now, q_target)
                 traj_move = TrajectoryParameter(param_move, QuinticVelocityParameter(time_move))
                 planner_move = TrajectoryPlanner(traj_move)
                 _execute_planner_sequence_with_compensation(
                     env, robot, [planner_move], [0.0, time_move],
                     gripper_ctrl=255,
                     keep_level=True,
                     initial_grasp_rotation=sm.SE3(grasp_rotation)
                 )
                 final_rotation = place_rotation_se3.R
            else:
                 print("  [SHELF] [WARNING] 中间区域目标点 IK 失败，尝试使用中转点...")
                 # 使用安全中转点（与 needs_turn_around 相同的中转策略）
                 T_waypoint = sm.SE3.Trans(0.8, 0.5, T_after_lift.t[2]) * place_rotation_se3
                 q_waypoint = robot.ikine(T_waypoint)

                 if len(q_waypoint) > 0:
                     # 先移动到中转点
                     time_move = 2.0
                     param_move1 = JointParameter(q_now, q_waypoint)
                     traj_move1 = TrajectoryParameter(param_move1, QuinticVelocityParameter(time_move))
                     planner_move1 = TrajectoryPlanner(traj_move1)
                     _execute_planner_sequence_with_compensation(
                         env, robot, [planner_move1], [0.0, time_move],
                         gripper_ctrl=255,
                         keep_level=True,
                         initial_grasp_rotation=sm.SE3(grasp_rotation)
                     )

                     # 再从中转点移动到目标
                     T_target_high = sm.SE3.Trans(target_pos[0], target_pos[1], T_after_lift.t[2]) * place_rotation_se3
                     q_now2 = robot.get_joint()
                     q_target2 = robot.ikine(T_target_high)
                     if len(q_target2) > 0:
                         param_move2 = JointParameter(q_now2, q_target2)
                         traj_move2 = TrajectoryParameter(param_move2, QuinticVelocityParameter(time_move))
                         planner_move2 = TrajectoryPlanner(traj_move2)
                         _execute_planner_sequence_with_compensation(
                             env, robot, [planner_move2], [0.0, time_move],
                             gripper_ctrl=255,
                             keep_level=True,
                             initial_grasp_rotation=sm.SE3(grasp_rotation)
                         )
                     final_rotation = place_rotation_se3.R
                 else:
                     print("  [SHELF] [ERROR] 中转点 IK 也失败，使用默认放置位置...")
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
        # final_rotation 可能是 SO3 对象或旋转矩阵，需要统一处理
        if isinstance(final_rotation, np.ndarray):
            T_drop = sm.SE3.Trans(target_pos[0], target_pos[1], target_pos[2]) * sm.SE3(sm.SO3(final_rotation))
        else:
            T_drop = sm.SE3.Trans(target_pos[0], target_pos[1], target_pos[2]) * sm.SE3(final_rotation)

        # 重新获取当前位置作为 LineStart
        T_current_high = robot.get_cartesian()

        pos_drop2 = LinePositionParameter(T_current_high.t, T_drop.t)
        # 关键修复：下降时保持姿态不变（不要插值旋转）
        att_drop2 = OneAttitudeParameter(sm.SO3(T_current_high.R), sm.SO3(T_current_high.R))
        traj_drop2 = TrajectoryParameter(CartesianParameter(pos_drop2, att_drop2), QuinticVelocityParameter(time_drop))
        planner_drop2 = TrajectoryPlanner(traj_drop2)

        # 下降时也启用水平保持
        if needs_turn_around or is_middle_area:
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
        _execute_planner_sequence(env, robot, [planner_up], [0.0, time_up], gripper_ctrl=0)  # 保持夹爪打开

        # 2) 从当前关节角用关节空间插值回到初始姿态 q0
        q_now = robot.get_joint()
        time_back = 1.5
        param_back = JointParameter(q_now, q0)
        traj_back = TrajectoryParameter(param_back, QuinticVelocityParameter(time_back))
        planner_back = TrajectoryPlanner(traj_back)
        _execute_planner_sequence(env, robot, [planner_back], [0.0, time_back], gripper_ctrl=0)  # 保持夹爪打开

        print("  [SHELF] 抓取、放置及复原完成！")
        return
    
    # 桌面抓取：使用原有复杂搬运逻辑
    # === 特殊处理：桌面抓取后放回货架 ===
    is_shelf_place = (target_pos is not None and
                      SHELF_X_MIN <= target_pos[0] <= SHELF_X_MAX)
    if is_shelf_place:
        print(f"\n[SHELF PLACE] 货架放置策略，目标: {target_pos}")

        # 桌面抓取是垂直姿态，放货架需要水平姿态
        # 构建水平放置姿态（与货架抓取相同）
        approach_dir = np.array([1, 0, 0])   # 接近方向 +X
        side_dir = np.array([0, 1, 0])        # 夹爪侧向 +Y
        R_horizontal = sm.SO3.TwoVectors(x=approach_dir, y=side_dir)
        print(f"  [SHELF PLACE] 检测到垂直抓取姿态，强制改为水平放置姿态...")

        # 阶段 A: 先抬起物体到安全高度
        T_current = robot.get_cartesian()
        current_x = T_current.t[0]

        # 阶段 A: 抬升物体到安全高度（统一使用关节空间，避免非标准姿态导致IK失败）
        print(f"  [SHELF PLACE] 先关节空间抬升避免碰撞 (x={current_x:.2f})...")
        R_down = sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)
        lift_heights = [0.35, 0.45, 0.55] if current_x < 0.5 else [0.15, 0.25, 0.35]
        lifted = False
        for lift_h in lift_heights:
            lift_pos = np.array([T_current.t[0], T_current.t[1],
                                 T_current.t[2] + lift_h])
            T_lift = sm.SE3.Trans(lift_pos) * R_down
            q_lift = robot.ikine(T_lift)
            if len(q_lift) > 0:
                print(f"  [SHELF PLACE] 抬升 +{lift_h:.2f}m IK 成功")
                q_now_lift = robot.get_joint()
                time_lift = 1.0
                param_lift = JointParameter(q_now_lift, q_lift)
                traj_lift = TrajectoryParameter(param_lift,
                                                 QuinticVelocityParameter(time_lift))
                planner_lift = TrajectoryPlanner(traj_lift)
                _execute_planner_sequence_with_compensation(env, robot, [planner_lift],
                                          [0.0, time_lift], gripper_ctrl=255,
                                          keep_level=True)
                lifted = True
                break
        if not lifted:
            print(f"  [SHELF PLACE] 抬升IK均失败，直接转向对准点")

        # 阶段 B: 用关节空间移动到货架前方对准点（水平姿态）
        # 对准点 Z 比目标层高 0.04m，避免运送途中碰到货架边缘
        align_x = SHELF_X_MIN - SHELF_APPROACH_OFFSET  # 1.61 - 0.35 = 1.26
        align_point = np.array([align_x, target_pos[1], target_pos[2] + 0.06])
        T_align = sm.SE3.Trans(align_point) * sm.SE3(R_horizontal)

        q_now = robot.get_joint()
        q_align = robot.ikine(T_align)
        if len(q_align) > 0:
            time_align = 2.5
            param_align = JointParameter(q_now, q_align)
            traj_align = TrajectoryParameter(param_align, QuinticVelocityParameter(time_align))
            planner_align = TrajectoryPlanner(traj_align)
            _execute_planner_sequence_with_compensation(env, robot, [planner_align], [0.0, time_align], gripper_ctrl=255,
                                      keep_level=True)
        else:
            # IK 失败，尝试调整对准点
            print(f"  [SHELF PLACE] 对准点 IK 失败，尝试调整...")
            align_point_adj = align_point + np.array([0.1, 0, -0.05])
            T_align_adj = sm.SE3.Trans(align_point_adj) * sm.SE3(R_horizontal)
            q_align = robot.ikine(T_align_adj)
            if len(q_align) > 0:
                time_align = 2.5
                param_align = JointParameter(q_now, q_align)
                traj_align = TrajectoryParameter(param_align, QuinticVelocityParameter(time_align))
                planner_align = TrajectoryPlanner(traj_align)
                _execute_planner_sequence_with_compensation(env, robot, [planner_align], [0.0, time_align], gripper_ctrl=255,
                                          keep_level=True)
                T_align = T_align_adj
            else:
                print(f"  [SHELF PLACE] [ERROR] 无法到达货架对准点")
        print(f"  [SHELF PLACE] 已到达对准点 x={align_x:.2f}")

        # 阶段 C: 水平插入货架到目标位置
        # 使用当前实际姿态（而非理论R_horizontal），避免keep_level后突然翻转180°
        T_align_current = robot.get_cartesian()
        R_current = sm.SO3(T_align_current.R)
        T_insert = sm.SE3.Trans(target_pos) * sm.SE3(R_current)
        time_insert = 1.5
        pos_insert = LinePositionParameter(T_align_current.t, T_insert.t)
        att_insert = OneAttitudeParameter(R_current, R_current)
        traj_insert = TrajectoryParameter(CartesianParameter(pos_insert, att_insert), QuinticVelocityParameter(time_insert))
        planner_insert = TrajectoryPlanner(traj_insert)
        _execute_planner_sequence(env, robot, [planner_insert], [0.0, time_insert], gripper_ctrl=255)
        print(f"  [SHELF PLACE] 已插入货架位置")

        # 阶段 D: 松开夹爪
        for i in range(2000):
            action[:6] = robot.get_joint()
            action[-1] -= 0.1
            action[-1] = np.max([action[-1], 0])
            env.step(action)
        print(f"  [SHELF PLACE] 夹爪已松开")

        # 阶段 E: 水平退出货架（沿 -X 方向）
        T_retreat_pos = np.array([align_x, target_pos[1], target_pos[2]])
        T_now = robot.get_cartesian()
        R_now = sm.SO3(T_now.R)
        T_retreat = sm.SE3.Trans(T_retreat_pos) * sm.SE3(R_now)
        time_retreat = 1.0
        pos_retreat = LinePositionParameter(T_now.t, T_retreat.t)
        att_retreat = OneAttitudeParameter(R_now, R_now)
        traj_retreat = TrajectoryParameter(CartesianParameter(pos_retreat, att_retreat), QuinticVelocityParameter(time_retreat))
        planner_retreat = TrajectoryPlanner(traj_retreat)
        _execute_planner_sequence(env, robot, [planner_retreat], [0.0, time_retreat], gripper_ctrl=0)
        print(f"  [SHELF PLACE] 已退出货架")

        # 阶段 F: 回到初始姿态
        q_now = robot.get_joint()
        time_back = 1.5
        param_back = JointParameter(q_now, q0)
        traj_back = TrajectoryParameter(param_back, QuinticVelocityParameter(time_back))
        planner_back = TrajectoryPlanner(traj_back)
        _execute_planner_sequence(env, robot, [planner_back], [0.0, time_back], gripper_ctrl=0)

        print(f"  [SHELF PLACE] 货架放置完成！")
        return

    if isinstance(planner_transit, list):
         # 使用了方案B (中转点)
         time_array = [0.0, time4, time_lift, 1.5, 1.5, time6]
         planner_array = [planner4, planner_lift, planner_transit[0], planner_transit[1], planner6]
    else:
         # 使用了方案A (关节插值)
         time_array = [0.0, time4, time_lift, time_transit, time6]
         planner_array = [planner4, planner_lift, planner_transit, planner6]

    # 判断是否需要启用水平保持补偿（需要转身时）
    if needs_turn_around:
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
    # 注意：保持夹爪打开
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
                action[-1] = 0  # 保持夹爪打开
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    # 注意：保持夹爪打开，避免重新抓起物品
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
                action[-1] = 0  # 保持夹爪打开
                env.step(action)
                break
