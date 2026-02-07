import os
import sys
import cv2
import mujoco
import matplotlib.pyplot as plt 
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from vlm_process import segment_image, parse_instruction, detect_place_position, pixel_to_world
# from grasp_process import run_grasp_inference, execute_grasp
from grasp_process_optimized import run_grasp_inference, execute_grasp
import spatialmath as sm
import numpy as np


# 全局变量
global color_img, depth_img, env
color_img = None
depth_img = None
env = None


#获取彩色和深度图像数据
def get_image(env, camera_name=None):
    global color_img, depth_img
     # 从环境渲染获取图像数据
    imgs = env.render(camera_name=camera_name)

    # 提取彩色和深度图像数据
    color_img = imgs['img']   # 这是RGB格式的图像数据
    depth_img = imgs['depth'] # 这是深度数据

    # 将RGB图像转换为OpenCV常用的BGR格式
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    return color_img, depth_img

#构造回调函数，不断调用
def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()


def test_grasp():
    global color_img, depth_img, env

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # --- 动态获取相机参数 ---
    cam_name = "cam"
    cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    
    # 获取相机外部参数 (World-to-Camera Transform)
    # data.cam_xpos: 相机在世界坐标系下的位置
    # data.cam_xmat: 相机在世界坐标系下的旋转矩阵 (3x3)
    t_wc = env.mj_data.cam_xpos[cam_id]
    # 获取旋转矩阵 (MuJoCo 默认为 3x3)
    R_mj = env.mj_data.cam_xmat[cam_id].reshape(3, 3)
    
    # [核心修复] 直接通过列向量变换将 MuJoCo 坐标系转为 CV 标准坐标系
    # CV_X = MuJoCo_X (第一列)
    # CV_Y = -MuJoCo_Y (第二列取反)
    # CV_Z = -MuJoCo_Z (第三列取反)
    R_cv = np.column_stack([
        R_mj[:, 0], 
        -R_mj[:, 1], 
        -R_mj[:, 2]
    ])
    
    T_wc = sm.SE3.Rt(R_cv, t_wc)
    
    # 获取相机内部参数 (fovy)
    # model.cam_fovy: 垂直视场角 (角度制)，转为弧度
    fovy_deg = env.mj_model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy_deg)

    # --- Debug: 对比动态提取值与原始硬编码值 ---
    print(f"\n[DEBUG] Camera '{cam_name}' (ID: {cam_id})")
    print(f"  Position (MuJoCo): {t_wc}")
    print(f"  FOVY (Deg): {fovy_deg:.2f}")
    
    # 计算原始硬编码的 T_wc 用于对比
    n_wc_orig = np.array([0.0, -1.0, 0.0])
    o_wc_orig = np.array([-1.0, 0.0, -0.5])
    t_wc_orig = np.array([0.85, 0.8, 1.6])
    T_wc_orig = sm.SE3.Trans(t_wc_orig) * sm.SE3(sm.SO3.TwoVectors(x=n_wc_orig, y=o_wc_orig))
    
    print(f"\n  --- 原始硬编码 T_wc.R ---")
    print(T_wc_orig.R)
    print(f"\n  --- 动态提取 R_cv ---")
    print(R_cv)
    print(f"\n  --- 差异 (应该接近0如果正确) ---")
    print(np.abs(T_wc_orig.R - R_cv).max())

    # 图像处理部分
    masks = segment_image(color_img)  

    # 传入动态提取的相机参数
    gg = run_grasp_inference(color_img, depth_img, masks, T_wc=T_wc, fovy=fovy_rad)

    execute_grasp(env, gg, T_wc=T_wc)



if __name__ == '__main__':
    
    env = UR5GraspEnv()
    env.reset()
    
    # 相机配置
    CAMERA_TABLE = "cam"        # 观察桌面
    CAMERA_SHELF = "cam_shelf"  # 观察货架
    CAMERA_GLOBAL_1 = "cam_global_1"  # 全局相机1
    CAMERA_GLOBAL_2 = "cam_global_2"  # 全局相机2
    current_mode = "single"     # single, fusion, 或 smart
    current_camera = CAMERA_TABLE

    # 导入融合函数
    from grasp_process_optimized import run_grasp_inference_fused

    # 辅助函数：获取相机参数
    def get_cam_params(env, cam_name):
        cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        t_wc = env.mj_data.cam_xpos[cam_id].copy()
        R_mj = env.mj_data.cam_xmat[cam_id].reshape(3, 3)
        R_cv = np.column_stack([R_mj[:, 0], -R_mj[:, 1], -R_mj[:, 2]])
        T_wc = sm.SE3.Rt(R_cv, t_wc)
        fovy = np.deg2rad(env.mj_model.cam_fovy[cam_id])
        return T_wc, fovy

    while True:

        for i in range(500):
            env.step()

        # 选择模式
        print(f"\n📷 当前模式: {current_mode.upper()}")
        print("   输入 '1' 单相机模式 - 桌面相机 (cam)")
        print("   输入 '2' 单相机模式 - 货架相机 (cam_shelf)")
        print("   输入 '3' 融合模式 - 双相机点云融合")
        print("   输入 '4' 智能放置模式 - 自然语言指定放置位置 🔥")
        print("   直接按回车继续...")

        choice = input("选择: ").strip()

        if choice == '1':
            current_mode = "single"
            current_camera = CAMERA_TABLE
            print(f"✅ 单相机模式: {current_camera}")
        elif choice == '2':
            current_mode = "single"
            current_camera = CAMERA_SHELF
            print(f"✅ 单相机模式: {current_camera}")
        elif choice == '3':
            current_mode = "fusion"
            print("✅ 融合模式: 将使用双相机点云融合")
        elif choice == '4':
            current_mode = "smart"
            print("✅ 智能放置模式: 支持自然语言指定放置位置")

        if current_mode == "single":
            # 单相机模式
            color_img, depth_img = get_image(env, camera_name=current_camera)
            callback(color_img, depth_img)

        elif current_mode == "smart":
            # 智能放置模式 - 支持自然语言指定放置位置
            print("\n" + "="*60)
            print("🤖 智能放置模式")
            print("="*60)
            print("请输入完整的自然语言指令，例如：")
            print("  - 把培养皿放置到显微镜的右边")
            print("  - 把试管移到桌子左上角")
            print("  - 抓取烧杯放到红色区域")
            print("="*60)

            user_input = input("\n请输入指令: ").strip()
            if not user_input:
                print("⚠️ 未输入指令，跳过")
                continue

            # 1. 解析用户指令
            print("\n[Step 1] 解析用户指令...")
            instruction = parse_instruction(user_input)
            grasp_target = instruction.get("grasp_target", user_input)
            place_description = instruction.get("place_description", "")
            has_place = instruction.get("has_place_instruction", False)

            print(f"  抓取目标: {grasp_target}")
            print(f"  放置位置: {place_description if place_description else '(未指定，使用默认位置)'}")

            # 2. 获取桌面相机图像进行抓取识别
            print("\n[Step 2] 获取桌面相机图像...")
            color_img, depth_img = get_image(env, camera_name=CAMERA_TABLE)
            T_wc_table, fovy_table = get_cam_params(env, CAMERA_TABLE)

            # 3. VLM分割目标物体
            print("\n[Step 3] VLM识别抓取目标...")
            masks = segment_image(color_img, command_text=grasp_target)

            # 4. 抓取推理
            print("\n[Step 4] 抓取姿态推理...")
            gg = run_grasp_inference(color_img, depth_img, masks, T_wc=T_wc_table, fovy=fovy_table)

            if gg is None:
                print("❌ 未能找到有效的抓取姿态")
                continue

            # 5. 确定放置位置
            target_pos = None
            if has_place and place_description:
                print("\n[Step 5] 识别放置位置...")

                # 获取多个全局相机图像
                print("  获取多视角全局相机图像...")

                # 主相机 (用于坐标计算)
                imgs_global_2 = env.render(camera_name=CAMERA_GLOBAL_2)
                color_global = cv2.cvtColor(imgs_global_2['img'], cv2.COLOR_RGB2BGR)
                depth_global = imgs_global_2['depth']
                T_wc_global, fovy_global = get_cam_params(env, CAMERA_GLOBAL_2)

                # 额外相机 (用于辅助识别)
                extra_images = []
                try:
                    imgs_global_1 = env.render(camera_name=CAMERA_GLOBAL_1)
                    color_global_1 = cv2.cvtColor(imgs_global_1['img'], cv2.COLOR_RGB2BGR)
                    extra_images.append(color_global_1)
                    cv2.imwrite("debug_global_view_1.jpg", color_global_1)
                except:
                    print("  cam_global_1 不可用")

                # 如果有 cam_global_3
                try:
                    imgs_global_3 = env.render(camera_name="cam_global_3")
                    color_global_3 = cv2.cvtColor(imgs_global_3['img'], cv2.COLOR_RGB2BGR)
                    extra_images.append(color_global_3)
                    cv2.imwrite("debug_global_view_3.jpg", color_global_3)
                except:
                    pass  # cam_global_3 可能不存在

                print(f"  使用 {1 + len(extra_images)} 个相机视角")

                # 保存主相机图像用于调试
                cv2.imwrite("debug_global_view.jpg", color_global)
                print("  全局视图已保存: debug_global_view.jpg")

                # VLM识别放置位置 (传入多相机图像)
                place_result = detect_place_position(place_description, color_global, extra_images=extra_images)

                if place_result and "place_point" in place_result:
                    pixel_x, pixel_y = place_result["place_point"]
                    print(f"  VLM识别的放置位置: 像素坐标 ({pixel_x}, {pixel_y})")
                    print(f"  置信度: {place_result.get('confidence', 'N/A')}")
                    print(f"  原因: {place_result.get('reason', 'N/A')}")

                    # 在全局图像上标注放置位置和参考物体
                    debug_img = color_global.copy()

                    # 标注参考物体位置（如果有）
                    if "reference_position" in place_result:
                        ref_x, ref_y = place_result["reference_position"]
                        cv2.circle(debug_img, (int(ref_x), int(ref_y)), 12, (255, 0, 0), -1)  # 蓝色圆点
                        cv2.circle(debug_img, (int(ref_x), int(ref_y)), 16, (255, 255, 0), 2)  # 青色边框
                        cv2.putText(debug_img, "Reference", (int(ref_x)+20, int(ref_y)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        print(f"  参考物体位置: ({ref_x}, {ref_y})")

                    # 标注放置位置
                    cv2.circle(debug_img, (pixel_x, pixel_y), 15, (0, 0, 255), -1)  # 红色圆点
                    cv2.circle(debug_img, (pixel_x, pixel_y), 20, (0, 255, 0), 3)   # 绿色边框
                    cv2.putText(debug_img, "Place Here", (pixel_x+25, pixel_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    cv2.imwrite("debug_place_position.jpg", debug_img)
                    print("  标注图像已保存: debug_place_position.jpg")
                    print("    - 蓝色圆点: 参考物体位置")
                    print("    - 红色圆点: 放置位置")

                    # 像素坐标转世界坐标
                    world_pos = pixel_to_world(
                        pixel_x, pixel_y, depth_global,
                        T_wc_global, fovy_global, color_global.shape
                    )

                    # 设置放置高度（桌面高度约0.74m，加上一点余量）
                    target_pos = [world_pos[0], world_pos[1], 0.76]
                    print(f"  原始世界坐标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

                    # 检查并调整到机械臂工作空间内
                    # UR5e机械臂基座在原点附近，工作半径约0.85m
                    # 安全工作范围：x: [0.1, 1.0], y: [0.1, 0.9]
                    WORKSPACE_X_MIN, WORKSPACE_X_MAX = 0.1, 1.0
                    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = 0.1, 0.9

                    original_pos = target_pos.copy()
                    target_pos[0] = max(WORKSPACE_X_MIN, min(WORKSPACE_X_MAX, target_pos[0]))
                    target_pos[1] = max(WORKSPACE_Y_MIN, min(WORKSPACE_Y_MAX, target_pos[1]))

                    if original_pos[0] != target_pos[0] or original_pos[1] != target_pos[1]:
                        print(f"  ⚠️ 原始位置超出工作空间，已调整!")
                        print(f"  调整后坐标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
                    else:
                        print(f"  世界坐标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
                else:
                    print("  ⚠️ 未能识别放置位置，使用默认位置")

            # 6. 执行抓取和放置
            print("\n[Step 6] 执行抓取和放置...")
            execute_grasp(env, gg, T_wc=T_wc_table, target_pos=target_pos)

            print("\n✅ 任务完成!")

        elif current_mode == "fusion":
            # 融合模式
            print("\n[FUSION] 采集双相机图像...")
            
            # 从两个相机采集
            imgs_cam = env.render(camera_name=CAMERA_TABLE)
            imgs_shelf = env.render(camera_name=CAMERA_SHELF)
            
            # 获取相机参数
            def get_cam_params(env, cam_name):
                cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                t_wc = env.mj_data.cam_xpos[cam_id].copy()
                R_mj = env.mj_data.cam_xmat[cam_id].reshape(3, 3)
                R_cv = np.column_stack([R_mj[:, 0], -R_mj[:, 1], -R_mj[:, 2]])
                T_wc = sm.SE3.Rt(R_cv, t_wc)
                fovy = np.deg2rad(env.mj_model.cam_fovy[cam_id])
                return T_wc, fovy
            
            T_wc_table, fovy_table = get_cam_params(env, CAMERA_TABLE)
            T_wc_shelf, fovy_shelf = get_cam_params(env, CAMERA_SHELF)
            
            # VLM 分割 - 对两个相机图像都进行目标分割
            color_img_table = cv2.cvtColor(imgs_cam['img'], cv2.COLOR_RGB2BGR)
            color_img_shelf = cv2.cvtColor(imgs_shelf['img'], cv2.COLOR_RGB2BGR)
            
            # 获取用户指令（只询问一次）
            print("\n📝 [FUSION] 请通过文字描述目标物体及抓取指令...")
            user_command = input("请输入: ").strip()
            
            print("\n[FUSION] VLM 分割桌面相机图像...")
            masks_table = segment_image(color_img_table, command_text=user_command)
            
            print("\n[FUSION] VLM 分割货架相机图像...")
            masks_shelf = segment_image(color_img_shelf, command_text=user_command)

            
            # 准备融合数据 - 两个相机都使用各自的分割结果
            camera_data_list = [
                {
                    'color': imgs_cam['img'],
                    'depth': imgs_cam['depth'],
                    'mask': masks_table,
                    'T_wc': T_wc_table,
                    'fovy': fovy_table
                },
                {
                    'color': imgs_shelf['img'],
                    'depth': imgs_shelf['depth'],
                    'mask': masks_shelf,  # 使用 VLM 分割的掩码
                    'T_wc': T_wc_shelf,
                    'fovy': fovy_shelf
                }
            ]

            
            # 融合推理
            gg = run_grasp_inference_fused(camera_data_list, T_wc_table, fovy_table)
            
            if gg is not None:
                execute_grasp(env, gg, T_wc=T_wc_table)


    env.close()


    