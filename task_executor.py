"""
共享任务执行引擎 — 从 ui_main_qt.py 提取的核心抓取/放置逻辑。
UI (PyQt5) 和 CLI (mujoco_vlm.py) 均通过回调适配使用本模块。
"""
import re
import os
import cv2
import numpy as np
import mujoco
import spatialmath as sm
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from vlm_process import (segment_image, parse_instruction,
                          detect_place_position, pixel_to_world,
                          check_place_ambiguity)
import grasp_process_optimized
from grasp_process_optimized import (run_grasp_inference, execute_grasp,
                                      run_grasp_inference_fused,
                                      get_object_origin, get_all_origins,
                                      get_position, record_place)
from config import Config

# ── 相机名称常量 ──────────────────────────────────────────
CAMERA_TABLE    = "cam"
CAMERA_TABLE_2  = "cam_2"
CAMERA_TABLE_3  = "cam_3"
CAMERA_SHELF    = "cam_shelf"
CAMERA_LEFT     = "cam_left"
CAMERA_RIGHT    = "cam_right"
CAMERA_GLOBAL_1 = "cam_global_1"
CAMERA_GLOBAL_2 = "cam_global_2"

# ── 货架判定常量 ──────────────────────────────────────────
SHELF_X_MIN = 1.61
SHELF_X_MAX = 1.97
SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
SHELF_LAYER_TOL = 0.15


class TaskExecutor:
    """
    共享任务执行引擎。

    Parameters
    ----------
    env : UR5GraspEnv
        MuJoCo 仿真环境实例。
    log_fn : callable(level: str, msg: str)
        日志回调，level 为 "INFO"/"WARN"/"ERR"/"STEP"/"SUCCESS"。
    ask_fn : callable(question: str) -> str
        向用户提问并阻塞等待回答（CLI 用 input，GUI 用弹窗）。
    image_fn : callable(key: str, img: ndarray) | None
        可选，用于向 UI 推送相机图像（key 如 "cam1"）。
    headless : bool
        True 时跳过 Open3D 弹窗。
    render_callback : callable | None
        GUI 模式下注入的渲染回调。
    """

    def __init__(self, env, log_fn, ask_fn,
                 image_fn=None, headless=True, render_callback=None):
        self.env = env
        self.log = log_fn
        self.ask = ask_fn
        self.image_fn = image_fn or (lambda k, img: None)
        self.prediction_results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "PredictionResults")
        # 设置 grasp_process_optimized 全局状态
        grasp_process_optimized.HEADLESS = headless
        grasp_process_optimized._render_callback = render_callback

    def _draw_utf8_text(self, image_bgr, text, org, color=(255, 255, 255), size=26):
        """使用 Pillow 绘制 UTF-8 文本（支持中文），输入输出均为 BGR。"""
        if not text:
            return image_bgr
        try:
            candidates = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            font = None
            for fp in candidates:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, size=size)
                    break
            if font is None:
                font = ImageFont.load_default()

            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            # Pillow 绘制失败时回退 OpenCV（中文可能退化为 ?）
            cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, size / 36.0), color, 2)
            return image_bgr

    def _save_prediction_visualization(self, image_bgr, display_desc,
                                       place_result, source_camera):
        """将放置预测结果可视化后保存到 PredictionResults 目录。"""
        if image_bgr is None or place_result is None:
            return
        if "place_point" not in place_result:
            return

        try:
            os.makedirs(self.prediction_results_dir, exist_ok=True)

            vis = image_bgr.copy()
            place_x, place_y = map(int, place_result["place_point"])
            ref_pos = place_result.get("reference_position")

            # 图中仅保留更小的实心点，避免遮挡主体。
            marker_radius = 10
            if ref_pos and len(ref_pos) == 2:
                ref_x, ref_y = map(int, ref_pos)
                cv2.circle(vis, (ref_x, ref_y), marker_radius, (255, 0, 0), -1)

            cv2.circle(vis, (place_x, place_y), marker_radius, (0, 0, 255), -1)

            # 左上角图例：说明点含义与相机来源。
            legend_x = 14
            legend_y = 22
            legend_gap = 28
            legend_radius = 6
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2

            line_idx = 0
            if ref_pos and len(ref_pos) == 2:
                y = legend_y + legend_gap * line_idx
                cv2.circle(vis, (legend_x, y), legend_radius, (255, 0, 0), -1)
                cv2.putText(vis, "Reference", (legend_x + 14, y + 6),
                            font, font_scale, (255, 0, 0), thickness)
                line_idx += 1

            y = legend_y + legend_gap * line_idx
            cv2.circle(vis, (legend_x, y), legend_radius, (0, 0, 255), -1)
            cv2.putText(vis, "Place", (legend_x + 14, y + 6),
                        font, font_scale, (0, 255, 0), thickness)

            y = legend_y + legend_gap * (line_idx + 1)
            cv2.putText(vis, f"Camera: {source_camera}", (legend_x, y + 6),
                        font, font_scale, (255, 255, 255), thickness)

            # 左上角保留中文指令，避免图内标签遮挡主体。
            desc_text = (display_desc or "").strip()
            if desc_text:
                max_chars = 24
                max_lines = 2
                desc_lines = []
                for i in range(0, len(desc_text), max_chars):
                    desc_lines.append(desc_text[i:i + max_chars])
                if len(desc_lines) > max_lines:
                    desc_lines = desc_lines[:max_lines]
                    if len(desc_lines[-1]) >= 2:
                        desc_lines[-1] = desc_lines[-1][:-2] + "..."
                    else:
                        desc_lines[-1] += "..."

                base_y = y + legend_gap
                vis = self._draw_utf8_text(vis, "Desc:", (legend_x, base_y),
                                           color=(255, 255, 255), size=26)
                for idx, line in enumerate(desc_lines):
                    vis = self._draw_utf8_text(
                        vis, line, (legend_x + 72, base_y + idx * legend_gap),
                        color=(255, 255, 255), size=26
                    )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(
                self.prediction_results_dir,
                f"place_prediction_{timestamp}.jpg")
            ok = cv2.imwrite(out_path, vis)
            if ok:
                self.log("INFO", f"[Step 5] 预测标注图已保存: {out_path}")
            else:
                self.log("WARN", "[Step 5] 预测标注图保存失败: cv2.imwrite 返回 False")
        except Exception as e:
            self.log("WARN", f"[Step 5] 保存预测标注图失败: {e}")

    # ── 相机操作 ──────────────────────────────────────────
    def get_cam_params(self, cam_name):
        """返回 (T_wc: SE3, fovy: float) — MuJoCo→OpenCV 坐标变换 + 垂直视场角(rad)。"""
        env = self.env
        cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        t_wc = env.mj_data.cam_xpos[cam_id].copy()
        R_mj = env.mj_data.cam_xmat[cam_id].reshape(3, 3)
        R_cv = np.column_stack([R_mj[:, 0], -R_mj[:, 1], -R_mj[:, 2]])
        T_wc = sm.SE3.Rt(R_cv, t_wc)
        fovy = np.deg2rad(env.mj_model.cam_fovy[cam_id])
        return T_wc, fovy

    def get_image(self, cam_name):
        """渲染指定相机，返回 (color_bgr, depth, raw_rgb)。"""
        imgs = self.env.render(camera_name=cam_name)
        color = cv2.cvtColor(imgs['img'], cv2.COLOR_RGB2BGR)
        return color, imgs['depth'], imgs['img']

    # ── 主入口 ────────────────────────────────────────────
    def execute_smart_task(self, instruction):
        """解析自然语言指令并执行完整的抓取/放置流程。"""
        env = self.env

        for _ in range(500):
            env.step()

        # Step 1: 解析指令
        parsed = parse_instruction(instruction)
        grasp_target = parsed.get("grasp_target", instruction)
        place_desc   = parsed.get("place_description", "")
        has_place    = parsed.get("has_place_instruction", False)
        # 兼容新旧字段：优先使用 return_type，回退到 is_return
        return_type  = parsed.get("return_type", "none")
        if return_type == "none":
            is_return_old = parsed.get("is_return", False) or parsed.get("is_putback", False)
            if is_return_old:
                return_type = "initial"
        self.log("STEP",
            f"[Step 1] 抓取目标: {grasp_target} | 放置: {place_desc or '默认'}"
            + (f" | 放回指令({return_type})" if return_type != "none" else ""))

        # ===== 放回流程 =====
        if return_type != "none":
            self._do_return_task(grasp_target, return_type)
            return

        # 检查放置描述是否模糊
        if has_place and place_desc:
            ambiguity = check_place_ambiguity(place_desc)
            if ambiguity.get("is_ambiguous"):
                question = (ambiguity.get("clarification_question")
                            or "放置描述中的距离不够明确，请问具体是多少厘米？")
                self.log("WARN", f"[!] 放置描述模糊: {question}")
                user_clarification = self.ask(question)
                if user_clarification:
                    ambiguous_part = ambiguity.get("ambiguous_part", "")
                    if ambiguous_part and ambiguous_part in place_desc:
                        place_desc = place_desc.replace(ambiguous_part, user_clarification)
                    else:
                        place_desc = f"{place_desc}（距离：{user_clarification}）"
                    self.log("INFO", f"[澄清] 更新放置描述: {place_desc}")

        T_wc_table, fovy_table = self.get_cam_params(CAMERA_TABLE)

        # Step 2: 多相机搜索物品
        found_camera, masks_cam, imgs_cam, T_wc_table, fovy_table = \
            self._search_object(grasp_target, T_wc_table, fovy_table)
        if found_camera is None:
            return

        # 判断货架/桌面
        is_on_shelf = self._detect_shelf_or_table(
            masks_cam, imgs_cam, T_wc_table, fovy_table)

        # Step 3: 抓取推理
        if found_camera == CAMERA_TABLE_2:
            self.log("STEP", f"[cam_2] 物品在 cam_2 视野，使用单相机抓取")
            color_cam = cv2.cvtColor(imgs_cam['img'], cv2.COLOR_RGB2BGR)
            gg = run_grasp_inference(color_cam, imgs_cam['depth'], masks_cam,
                                     T_wc=T_wc_table, fovy=fovy_table)
        elif is_on_shelf:
            self.log("STEP", "[SHELF] 物品在货架上，使用 cam 单相机抓取")
            color_cam = cv2.cvtColor(imgs_cam['img'], cv2.COLOR_RGB2BGR)
            gg = run_grasp_inference(color_cam, imgs_cam['depth'], masks_cam,
                                     T_wc=T_wc_table, fovy=fovy_table)
        else:
            gg = self._fused_table_grasp(grasp_target, imgs_cam, masks_cam,
                                          T_wc_table, fovy_table)

        if gg is None:
            self.log("ERR", "[X] 未找到有效抓取姿态")
            return

        # Step 5: 放置位置
        target_pos = None
        if has_place and place_desc:
            # 优先检测坐标格式，如 (0.5, 0.5) 或 0.5,0.5
            coord_match = re.search(
                r'\(?\s*(-?[\d.]+)\s*[,，]\s*(-?[\d.]+)\s*\)?', place_desc)
            if coord_match:
                cx, cy = float(coord_match.group(1)), float(coord_match.group(2))
                target_pos = self._clamp_to_workspace([cx, cy, 0.76])
                self.log("SUCCESS", f"[坐标模式] 直接使用坐标: ({cx}, {cy}) → {target_pos}")
            else:
                target_pos = self._detect_place_position(place_desc, instruction)

        # Step 6: 执行
        use_desktop_fusion = (found_camera == CAMERA_TABLE and not is_on_shelf)
        execute_grasp(env, gg, T_wc=T_wc_table, target_pos=target_pos,
                      object_name=grasp_target,
                      desktop_fusion=use_desktop_fusion)

        # 记录放置位置
        if target_pos and grasp_target:
            record_place(grasp_target, target_pos)

    # ── 多相机搜索物品 ────────────────────────────────────
    def _search_object(self, grasp_target, T_wc_table, fovy_table):
        """依次尝试 cam → cam_2 搜索目标物品。
        返回 (found_camera, masks, imgs_raw, T_wc, fovy) 或全 None。"""
        env = self.env

        def _mask_valid(m):
            return m is not None and int(np.sum(m > 0)) > 100

        self.log("STEP", "[Step 2] 多相机搜索物品...")

        # 尝试 cam
        self.log("INFO", "  [搜索] 尝试 cam 相机...")
        imgs_cam = env.render(camera_name=CAMERA_TABLE)
        c_cam = cv2.cvtColor(imgs_cam['img'], cv2.COLOR_RGB2BGR)
        masks_cam = segment_image(c_cam, command_text=grasp_target, auto_only=True)

        if _mask_valid(masks_cam):
            self.log("SUCCESS", f"  [搜索] cam 相机找到 '{grasp_target}'")
            return CAMERA_TABLE, masks_cam, imgs_cam, T_wc_table, fovy_table

        # 尝试 cam_2
        self.log("WARN", f"  [搜索] cam 相机未找到 '{grasp_target}'，尝试 cam_2...")
        imgs_cam = env.render(camera_name=CAMERA_TABLE_2)
        c_cam = cv2.cvtColor(imgs_cam['img'], cv2.COLOR_RGB2BGR)
        T_wc_2, fovy_2 = self.get_cam_params(CAMERA_TABLE_2)
        masks_cam = segment_image(c_cam, command_text=grasp_target, auto_only=True)

        if _mask_valid(masks_cam):
            self.log("SUCCESS", f"  [搜索] cam_2 相机找到 '{grasp_target}'")
            return CAMERA_TABLE_2, masks_cam, imgs_cam, T_wc_2, fovy_2

        self.log("ERR", f"[X] 所有相机均未找到 '{grasp_target}'，该实验场景未找到该物品")
        return None, None, None, None, None

    # ── 判断货架/桌面 ─────────────────────────────────────
    def _detect_shelf_or_table(self, masks, imgs, T_wc, fovy):
        """根据 mask 中心点的世界坐标判断物品在货架还是桌面。返回 bool。"""
        is_on_shelf = False
        if masks is not None and np.any(masks > 0):
            ys, xs = np.where(masks > 0)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            depth_cam = imgs['depth']
            h, w = depth_cam.shape[:2]
            focal = h / (2.0 * np.tan(fovy / 2.0))
            z_val = depth_cam[cy, cx]
            if z_val > 0.01:
                x_c = (cx - w / 2.0) * z_val / focal
                y_c = (cy - h / 2.0) * z_val / focal
                pt_cam = np.array([x_c, y_c, z_val])
                pt_world = (T_wc.R @ pt_cam) + T_wc.t
                self.log("INFO",
                    f"[DEBUG] 物品世界坐标: X={pt_world[0]:.3f}, "
                    f"Y={pt_world[1]:.3f}, Z={pt_world[2]:.3f}")
                if SHELF_X_MIN <= pt_world[0] <= SHELF_X_MAX:
                    for lz in SHELF_LAYER_HEIGHTS:
                        if abs(pt_world[2] - lz) < SHELF_LAYER_TOL:
                            is_on_shelf = True
                            self.log("INFO",
                                f"[DEBUG] 匹配货架层 Z={lz:.3f}, "
                                f"差值={abs(pt_world[2]-lz):.3f}")
                            break
            else:
                self.log("WARN", f"[DEBUG] 深度值无效: z_val={z_val:.4f}")
        else:
            self.log("WARN", "[DEBUG] mask为空，无法判断物品位置")

        self.log("INFO",
            f"[DEBUG] >>> 判断结果: {'货架' if is_on_shelf else '桌面'} <<<")
        return is_on_shelf

    # ── 桌面三相机融合抓取 ────────────────────────────────
    def _fused_table_grasp(self, grasp_target, imgs_cam, masks_cam,
                            T_wc_table, fovy_table):
        """cam + cam_left + cam_right 三相机融合推理。返回 gg 或 None。"""
        self.log("STEP", "[TABLE] 物品在桌面，使用三相机融合")
        env = self.env
        camera_data_list = [
            {'color': imgs_cam['img'], 'depth': imgs_cam['depth'],
             'mask': masks_cam, 'T_wc': T_wc_table, 'fovy': fovy_table},
        ]
        for side_name, side_cam in [("左", CAMERA_LEFT), ("右", CAMERA_RIGHT)]:
            try:
                side_imgs = env.render(camera_name=side_cam)
                side_color = cv2.cvtColor(side_imgs['img'], cv2.COLOR_RGB2BGR)
                side_T, side_fovy = self.get_cam_params(side_cam)
                side_mask = segment_image(side_color, command_text=grasp_target,
                                          auto_only=True)
                if side_mask is not None and np.any(side_mask > 0):
                    camera_data_list.append({
                        'color': side_imgs['img'], 'depth': side_imgs['depth'],
                        'mask': side_mask, 'T_wc': side_T, 'fovy': side_fovy
                    })
                    self.log("INFO", f"  {side_name}相机分割成功")
                else:
                    self.log("WARN", f"  {side_name}相机未检测到目标，跳过")
            except Exception as e:
                self.log("WARN", f"  {side_name}相机分割失败: {e}，跳过")

        self.log("INFO", f"  融合相机数: {len(camera_data_list)}")
        return run_grasp_inference_fused(camera_data_list, T_wc_table,
                                          fovy_table, desktop_mode=True)

    # ── 放置位置检测 ──────────────────────────────────────
    def _detect_place_position(self, place_desc, display_desc=None):
        """使用全局相机 + VLM 识别放置位置，返回 [x, y, z] 或 None。"""
        env = self.env
        self.log("STEP", "[Step 5] 识别放置位置...")

        # 俯视相机
        try:
            imgs_g4 = env.render(camera_name="cam_global_4")
            color_top = cv2.cvtColor(imgs_g4['img'], cv2.COLOR_RGB2BGR)
            depth_top = imgs_g4['depth']
            T_wc_top, fovy_top = self.get_cam_params("cam_global_4")
            self.image_fn("cam1", color_top)
        except Exception:
            color_top = depth_top = T_wc_top = fovy_top = None

        # 主全局相机
        imgs_g2 = env.render(camera_name=CAMERA_GLOBAL_2)
        color_g = cv2.cvtColor(imgs_g2['img'], cv2.COLOR_RGB2BGR)
        depth_g = imgs_g2['depth']
        T_wc_g, fovy_g = self.get_cam_params(CAMERA_GLOBAL_2)

        # 额外视角
        extra = []
        try:
            imgs_g1 = env.render(camera_name=CAMERA_GLOBAL_1)
            extra.append(cv2.cvtColor(imgs_g1['img'], cv2.COLOR_RGB2BGR))
        except Exception:
            pass

        place_result = detect_place_position(
            place_desc, color_g, extra_images=extra, top_view_image=color_top)

        if place_result and "place_point" in place_result:
            px, py = place_result["place_point"]
            vis_img = color_top if color_top is not None else color_g
            vis_cam = "cam_global_4" if color_top is not None else CAMERA_GLOBAL_2
            self._save_prediction_visualization(
                vis_img, display_desc or place_desc, place_result, vis_cam)
            if color_top is not None:
                world_pos = pixel_to_world(px, py, depth_top, T_wc_top,
                                            fovy_top, color_top.shape)
            else:
                world_pos = pixel_to_world(px, py, depth_g, T_wc_g,
                                            fovy_g, color_g.shape)
            return self._clamp_to_workspace(world_pos)

        self.log("WARN", "  未能识别放置位置，使用默认位置")
        return None

    # ── 工作空间裁剪 ──────────────────────────────────────
    def _clamp_to_workspace(self, world_pos):
        """将世界坐标限制在桌面 + 环形工作空间内，返回 [x, y, 0.76]。"""
        tx = max(Config.TABLE_X_MIN, min(Config.TABLE_X_MAX, world_pos[0]))
        ty = max(Config.TABLE_Y_MIN, min(Config.TABLE_Y_MAX, world_pos[1]))
        BASE_X, BASE_Y = Config.ROBOT_BASE_X, Config.ROBOT_BASE_Y
        R_MIN, R_MAX   = Config.WORKSPACE_R_MIN, Config.WORKSPACE_R_MAX
        dx, dy = tx - BASE_X, ty - BASE_Y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            tx, ty = BASE_X + R_MIN, BASE_Y
        elif dist < R_MIN:
            tx = BASE_X + dx / dist * R_MIN
            ty = BASE_Y + dy / dist * R_MIN
        elif dist > R_MAX:
            tx = BASE_X + dx / dist * R_MAX
            ty = BASE_Y + dy / dist * R_MAX
        target_pos = [tx, ty, 0.76]
        self.log("SUCCESS", f"放置坐标: ({tx:.3f}, {ty:.3f}, 0.76)")
        return target_pos

    # ── cam 前方三相机融合抓取（用于放回任务） ─────────────
    def _grasp_via_cam_front(self, grasp_target):
        """cam + cam_left + cam_right 三相机融合推理。
        返回 (gg, T_wc, CAMERA_TABLE) 或 (None, None, None)。"""
        self.log("INFO", f"  [cam前方] 尝试在 cam 中搜索 '{grasp_target}'...")
        color, depth, raw_rgb = self.get_image(CAMERA_TABLE)
        T_wc, fovy = self.get_cam_params(CAMERA_TABLE)
        try:
            masks = segment_image(color, command_text=grasp_target)
        except Exception as e:
            self.log("WARN", f"  [cam前方] 分割异常: {e}")
            return None, None, None

        mask_pixels = int(np.sum(masks > 0)) if masks is not None else 0
        if mask_pixels <= 100:
            self.log("WARN",
                f"  [cam前方] 未找到 '{grasp_target}' (mask像素: {mask_pixels})")
            return None, None, None

        self.log("SUCCESS",
            f"  [cam前方] 找到 '{grasp_target}' (mask像素: {mask_pixels})")
        try:
            camera_data_list = [
                {'color': raw_rgb, 'depth': depth, 'mask': masks,
                 'T_wc': T_wc, 'fovy': fovy},
            ]
            for side_name, side_cam in [("左", CAMERA_LEFT), ("右", CAMERA_RIGHT)]:
                try:
                    sc, sd, sr = self.get_image(side_cam)
                    sT, sf = self.get_cam_params(side_cam)
                    sm_ = segment_image(sc, command_text=grasp_target)
                    if sm_ is not None and np.any(sm_ > 0):
                        camera_data_list.append({
                            'color': sr, 'depth': sd,
                            'mask': sm_, 'T_wc': sT, 'fovy': sf
                        })
                        self.log("INFO", f"  [cam前方] {side_name}相机分割成功")
                    else:
                        self.log("WARN",
                            f"  [cam前方] {side_name}相机未检测到目标，跳过")
                except Exception as e:
                    self.log("WARN",
                        f"  [cam前方] {side_name}相机分割失败: {e}，跳过")
            self.log("INFO", f"  [cam前方] 融合相机数: {len(camera_data_list)}")
            gg = run_grasp_inference_fused(camera_data_list, T_wc, fovy,
                                            desktop_mode=True)
        except Exception as e:
            self.log("WARN", f"  [cam前方] 抓取推理异常: {e}")
            gg = None

        if gg is not None:
            return gg, T_wc, CAMERA_TABLE
        self.log("WARN", "  [cam前方] 找到物体但未生成有效抓取姿态")
        return None, None, None

    # ── cam_2 背部单相机抓取（用于放回任务） ───────────────
    def _grasp_via_cam_back(self, grasp_target):
        """cam_2 单相机推理 + pixel_to_world 回退。
        返回 (gg, T_wc, CAMERA_TABLE_2) 或 (None, None, None)。"""
        self.log("INFO", f"  [cam_2背部] 尝试在 cam_2 中搜索 '{grasp_target}'...")
        color, depth, raw_rgb = self.get_image(CAMERA_TABLE_2)
        T_wc, fovy = self.get_cam_params(CAMERA_TABLE_2)
        try:
            masks = segment_image(color, command_text=grasp_target)
        except Exception as e:
            self.log("WARN", f"  [cam_2背部] 分割异常: {e}")
            return None, None, None

        mask_pixels = int(np.sum(masks > 0)) if masks is not None else 0
        if mask_pixels <= 100:
            self.log("WARN",
                f"  [cam_2背部] 未找到 '{grasp_target}' (mask像素: {mask_pixels})")
            return None, None, None

        self.log("SUCCESS",
            f"  [cam_2背部] 找到 '{grasp_target}' (mask像素: {mask_pixels})")

        # 先尝试单相机 GraspNet 推理
        try:
            gg = run_grasp_inference(color, depth, masks,
                                     T_wc=T_wc, fovy=fovy)
        except Exception as e:
            self.log("WARN", f"  [cam_2背部] 抓取推理异常: {e}")
            gg = None

        if gg is not None:
            return gg, T_wc, CAMERA_TABLE_2

        # pixel_to_world 回退构造垂直向下抓取
        self.log("WARN",
            "  [cam_2背部] 抓取推理未生成有效姿态，使用深度图定位 + 垂直抓取回退...")
        mask_ys, mask_xs = np.where(masks > 0)
        cx, cy = int(np.mean(mask_xs)), int(np.mean(mask_ys))
        obj_world = pixel_to_world(cx, cy, depth, T_wc, fovy, color.shape[:2])
        if obj_world is not None:
            self.log("SUCCESS",
                f"  [cam_2背部] 物体世界坐标: "
                f"({obj_world[0]:.3f}, {obj_world[1]:.3f}, {obj_world[2]:.3f})")
            from graspnetAPI import GraspGroup
            T_cw = T_wc.inv()
            obj_cam = T_cw.R @ obj_world + T_cw.t
            R_down_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            R_cam = T_cw.R @ R_down_world
            grasp_array = np.zeros((1, 17))
            grasp_array[0, 0] = 0.8       # score
            grasp_array[0, 1] = 0.02      # width
            grasp_array[0, 2] = 0.02      # height
            grasp_array[0, 3] = 0.1       # depth
            grasp_array[0, 4:13] = R_cam.flatten()
            grasp_array[0, 13:16] = obj_cam
            grasp_array[0, 16] = -1       # object_id
            gg = GraspGroup(grasp_array)
            self.log("SUCCESS",
                "  [cam_2背部] 已构造垂直向下抓取姿态（基于深度定位）")
            return gg, T_wc, CAMERA_TABLE_2

        self.log("WARN", "  [cam_2背部] pixel_to_world 也失败")
        return None, None, None

    # ── 放回任务 ──────────────────────────────────────────
    def _do_return_task(self, grasp_target, return_type="initial"):
        """处理"放回"类型的指令，根据 return_type 查询不同的位置记录。"""
        env = self.env

        # 1. 根据 return_type 查询目标位置
        type_label = {"initial": "最初位置", "shelf": "货架位置",
                      "last_grasp": "上次取物位置"}
        label = type_label.get(return_type, "最初位置")

        origin = get_position(grasp_target, return_type)
        if origin is None and return_type == "shelf":
            # shelf 查询失败时提示
            self.log("ERR", f"[X] 没有找到 '{grasp_target}' 的货架位置记录")
            return
        if origin is None:
            # 回退到 initial
            origin = get_position(grasp_target, "initial")
            if origin is not None:
                self.log("WARN", f"[!] 未找到 '{grasp_target}' 的{label}，回退到最初位置")
                label = "最初位置"
        if origin is None:
            all_origins = get_all_origins()
            if all_origins:
                known = "、".join(all_origins.keys())
                msg = f"没有找到 '{grasp_target}' 的来源记录。已记录的物体有：{known}"
            else:
                msg = f"没有找到 '{grasp_target}' 的来源记录，还没有抓取过任何物体。"
            self.log("ERR", f"[X] {msg}")
            return

        origin_pos = origin["position"]
        is_shelf_origin = origin.get("is_shelf", False)
        shelf_layer = origin.get("shelf_layer", -1)

        self.log("STEP",
            f"[Step 2] 找到 '{grasp_target}' 的{label}: "
            f"({origin_pos[0]:.3f}, {origin_pos[1]:.3f}, {origin_pos[2]:.3f})"
            f"{' (货架第' + str(shelf_layer+1) + '层)' if is_shelf_origin else ' (桌面)'}")

        # 2. 依次尝试 cam前方 和 cam_2背部 搜索物体
        self.log("STEP", "[Step 3] 搜索物体...")
        gg, T_wc_grasp, grasp_cam = self._grasp_via_cam_front(grasp_target)
        if gg is None:
            gg, T_wc_grasp, grasp_cam = self._grasp_via_cam_back(grasp_target)

        if gg is None:
            self.log("ERR",
                f"[X] 在所有相机中均未找到 '{grasp_target}'")
            return

        # 3. 确定放回位置
        target_pos = list(origin_pos)
        if is_shelf_origin:
            self.log("STEP",
                f"[Step 4] 放回货架第{shelf_layer+1}层: "
                f"({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        else:
            target_pos[2] = 0.76
            self.log("STEP",
                f"[Step 4] 放回桌面: "
                f"({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

        # 4. 执行抓取和放回
        use_desktop_fusion = (grasp_cam == CAMERA_TABLE)
        self.log("STEP",
            f"[Step 5] 执行抓取和放回 "
            f"(cam={grasp_cam}, desktop_fusion={use_desktop_fusion})")
        execute_grasp(env, gg, T_wc=T_wc_grasp, target_pos=target_pos,
                      object_name=grasp_target,
                      desktop_fusion=use_desktop_fusion)

        # 记录放置位置
        if grasp_target:
            record_place(grasp_target, target_pos)
