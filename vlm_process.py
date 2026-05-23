import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

import json
import re
import base64
import textwrap
import time
import os  # 导入 os 用于处理环境变量

# 强制清除可能导致错误的代理环境变量
for key in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

from openai import OpenAI  # 导入OpenAI客户端
import httpx  # 导入 httpx 用于处理代理问题

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from config import Config, get_config_manager


# ----------------------- 指令解析与放置位置识别 -----------------------

def parse_instruction(user_input, image_input=None):
    """
    解析用户的自然语言指令，分离抓取目标和放置位置描述。

    输入: "把培养皿放置到显微镜的右边红色区域"
    输出: {
        "grasp_target": "培养皿",
        "place_description": "显微镜的右边红色区域",
        "has_place_instruction": True
    }
    """
    client = get_config_manager().create_client()

    system_prompt = textwrap.dedent("""\
    你是一个机器人指令解析系统。请分析用户的自然语言指令，提取以下信息：

    1. 要抓取的物体名称（grasp_target）
    2. 放置位置的描述（place_description）- 如果用户没有指定放置位置，则为空字符串
    3. 放回类型（return_type）：
       - "none"       — 普通放置指令，不是放回
       - "initial"    — 放回最初位置/放回原处/归还
       - "shelf"      — 明确说放回货架
       - "last_grasp" — 放回上次取物品时的位置

    【示例】
    输入: "把培养皿放置到显微镜的右边"
    输出: {"grasp_target": "培养皿", "place_description": "显微镜的右边", "has_place_instruction": true, "return_type": "none"}

    输入: "抓取红色的试管"
    输出: {"grasp_target": "红色的试管", "place_description": "", "has_place_instruction": false, "return_type": "none"}

    输入: "把烧杯移到桌子左上角的红色区域"
    输出: {"grasp_target": "烧杯", "place_description": "桌子左上角的红色区域", "has_place_instruction": true, "return_type": "none"}

    输入: "把培养皿放回货架"
    输出: {"grasp_target": "培养皿", "place_description": "货架", "has_place_instruction": true, "return_type": "shelf"}

    输入: "把培养皿放回原处"
    输出: {"grasp_target": "培养皿", "place_description": "原处", "has_place_instruction": true, "return_type": "initial"}

    输入: "把培养皿放回最初的位置"
    输出: {"grasp_target": "培养皿", "place_description": "最初的位置", "has_place_instruction": true, "return_type": "initial"}

    输入: "把培养皿还回去"
    输出: {"grasp_target": "培养皿", "place_description": "原处", "has_place_instruction": true, "return_type": "initial"}

    输入: "把培养皿归还"
    输出: {"grasp_target": "培养皿", "place_description": "原处", "has_place_instruction": true, "return_type": "initial"}

    输入: "把培养皿放回上次的位置"
    输出: {"grasp_target": "培养皿", "place_description": "上次的位置", "has_place_instruction": true, "return_type": "last_grasp"}

    【注意】
    - 只返回JSON对象，不要有其他文字
    - 如果指令中包含"放到"、"放置到"、"移到"、"移动到"等词语，说明有放置指令
    - 如果指令中包含"放回货架"、"还到货架"，return_type 为 "shelf"
    - 如果指令中包含"放回原处"、"放回原位"、"放回最初"、"还回去"、"归还"、"归位"、"放回去"，return_type 为 "initial"
    - 如果指令中包含"放回上次"、"放回之前"、"放到上次"，return_type 为 "last_grasp"
    - 其他普通放置指令，return_type 为 "none"
    """)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = [{"type": "text", "text": f"用户指令：{user_input}"}]

    # 如果提供了图像，也可以帮助理解上下文
    if image_input is not None:
        base64_img = encode_np_array(image_input)
        user_content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })

    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=get_config_manager().get_active_model_name(),
            messages=messages,
            temperature=0.1,
        )

        content = completion.choices[0].message.content
        print(f"[指令解析] 原始响应: {content}")

        # 解析JSON
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            return result
        else:
            # 如果解析失败，假设整个输入都是抓取目标
            return {
                "grasp_target": user_input,
                "place_description": "",
                "has_place_instruction": False
            }

    except Exception as e:
        print(f"[指令解析] 失败: {e}")
        return {
            "grasp_target": user_input,
            "place_description": "",
            "has_place_instruction": False
        }


def check_place_ambiguity(place_description):
    """
    使用VLM判断放置位置描述中的距离是否模糊，需要用户进一步澄清。

    返回:
        {
            "is_ambiguous": bool,
            "ambiguous_part": str,       # 模糊的词语，如 "一点"
            "clarification_question": str # 向用户提问的内容
        }
    """
    client = get_config_manager().create_client()

    system_prompt = textwrap.dedent("""\
    你是一个机器人指令分析系统。请判断用户的放置位置描述是否足够明确，可以直接执行。

    【需要询问的情况】
    - 距离词语主观模糊：如"一点"、"一点点"、"稍微"、"远一点"、"近一点"、"偏一点"
    - 缺少方向：如"显微镜旁边"、"显微镜附近"、"显微镜边上"（不知道放哪个方向）

    【不需要询问的情况】
    - 有方向但没有距离：如"显微镜右边"（可以用默认距离执行）
    - 有方向和具体距离：如"显微镜右边5厘米"
    - 颜色区域描述：如"绿色区域"、"红色区域左边"（有明确参考点）

    【示例】
    输入: "显微镜右边一点"
    输出: {"is_ambiguous": true, "ambiguous_part": "一点", "clarification_question": "您说的'一点'具体是多少厘米？"}

    输入: "显微镜右边5厘米"
    输出: {"is_ambiguous": false, "ambiguous_part": "", "clarification_question": ""}

    输入: "显微镜右边"
    输出: {"is_ambiguous": false, "ambiguous_part": "", "clarification_question": ""}

    输入: "显微镜旁边"
    输出: {"is_ambiguous": true, "ambiguous_part": "旁边", "clarification_question": "请问放在显微镜的哪个方向？（如左边、右边、上面、下面）"}

    输入: "显微镜附近"
    输出: {"is_ambiguous": true, "ambiguous_part": "附近", "clarification_question": "请问放在显微镜的哪个方向，距离多少厘米？"}

    输入: "绿色区域"
    输出: {"is_ambiguous": false, "ambiguous_part": "", "clarification_question": ""}

    只返回JSON对象，不要有其他文字。
    """)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"放置描述：{place_description}"}
    ]

    try:
        completion = client.chat.completions.create(
            model=get_config_manager().get_active_model_name(),
            messages=messages,
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        print(f"[模糊性检测] 响应: {content}")

        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        print(f"[模糊性检测] 失败: {e}")

    return {"is_ambiguous": False, "ambiguous_part": "", "clarification_question": ""}


def detect_place_position(place_description, global_image, depth_image=None, extra_images=None, top_view_image=None):
    """
    使用VLM在全局图像中识别放置位置。
    支持多相机图像输入，提供更全面的场景理解。

    参数:
        place_description: 放置位置描述
        global_image: 主相机图像 (用于坐标计算)
        depth_image: 深度图像 (可选)
        extra_images: 额外的相机图像列表 (可选，用于辅助识别)
        top_view_image: 俯视视角图像 (可选，用于精确定位放置点)
    """
    client = get_config_manager().create_client()

    # 如果有俯视图，使用俯视图作为主图像进行坐标预测
    if top_view_image is not None:
        main_image = top_view_image
        print(f"[放置位置识别] 使用俯视视角作为主图像")
    else:
        main_image = global_image

    h, w = main_image.shape[:2]

    # 【修改】只使用俯视图，不拼接其他视角，避免混淆VLM
    # 如果有俯视图，直接使用俯视图；否则使用主图像
    print(f"[放置位置识别] 只使用俯视图模式（不拼接其他视角）")
    combined_image = main_image
    combined_h, combined_w = h, w
    first_image_width = w

    # 强制使用单视图模式的提示词
    force_single_view = True

    # 使用VLM解析放置描述，提取结构化信息（替代硬编码正则）
    color_region = None
    reference_object = None
    desc_type = "direct"

    parse_prompt = (
        f'你是机器人指令解析系统。请分析放置位置描述，提取结构化信息。\n\n'
        f'描述："{place_description}"\n\n'
        f'请判断描述属于哪种类型并提取关键信息：\n'
        f'- "color_region"：描述中提到了颜色区域（如"绿色区域"、"红色区域"）\n'
        f'- "reference_object"：描述中提到了具体物体作为参考（如"显微镜"、"机械臂"）\n'
        f'- "direct"：直接描述位置（如"桌面中央"、"左上角"）\n\n'
        f'返回JSON（只返回JSON，不要其他文字）：\n'
        f'{{"type": "color_region或reference_object或direct", "color_region": "颜色名称中文如绿色或null", "reference_object": "物体名称或null"}}\n\n'
        f'示例：\n'
        f'"显微镜右边5厘米" → {{"type": "reference_object", "color_region": null, "reference_object": "显微镜"}}\n'
        f'"绿色区域偏左" → {{"type": "color_region", "color_region": "绿色", "reference_object": null}}\n'
        f'"桌面中央" → {{"type": "direct", "color_region": null, "reference_object": null}}'
    )
    try:
        parse_completion = client.chat.completions.create(
            model=get_config_manager().get_active_model_name(),
            messages=[{"role": "user", "content": parse_prompt}],
            temperature=0.1)
        parse_content = parse_completion.choices[0].message.content
        print(f"[放置位置识别] 描述解析: {parse_content}")
        parse_match = re.search(r'(\{.*\})', parse_content, re.DOTALL)
        if parse_match:
            parsed = json.loads(parse_match.group(1))
            color_region = parsed.get("color_region")
            reference_object = parsed.get("reference_object")
            desc_type = parsed.get("type", "direct")
    except Exception as e:
        print(f"[放置位置识别] 描述解析失败，使用单阶段识别: {e}")

    print(f"[放置位置识别] 解析结果: 类型='{desc_type}', 参考物体='{reference_object}', 颜色区域='{color_region}'")

    # ===== 颜色区域识别（VLM智能解析版本）=====
    if color_region:
        print(f"[放置位置识别] 使用颜色区域识别模式（VLM智能解析）...")

        # 颜色映射（中文到英文）
        color_map = {
            "红色": "red",
            "绿色": "green",
            "蓝色": "blue",
            "黄色": "yellow",
            "白色": "white",
            "黑色": "black",
            "橙色": "orange",
            "紫色": "purple",
        }
        color_en = color_map.get(color_region, color_region)

        # 判断是否使用多视角图像（强制使用单视图）
        use_multi_view = False if force_single_view else ((top_view_image is not None) or (extra_images and len(extra_images) > 0))

        # 第一步：让VLM找到颜色区域的中心
        color_prompt = f"""请在这张俯视图中找到 {color_region}/{color_en} 颜色的区域，并返回该区域的中心点像素坐标。

【俯视图场景布局】
- 绿色区域是一大块矩形区域，覆盖了桌面的上半部分大部分面积（图像上方）
- 红色区域是一小块正方形区域，位于桌面下方偏右的角落（图像下方偏右）
- 图像尺寸: {w} x {h} 像素
- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大

请返回JSON格式：
{{"found": true, "center": [x, y], "reason": "找到{color_region}区域的原因"}}
如果找不到{color_region}区域，返回：
{{"found": false, "reason": "未找到的原因"}}"""

        messages = [{"role": "system", "content": color_prompt}]
        base64_img = encode_np_array(combined_image)
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": f"请找到图像中的{color_region}区域"}
        ]})

        try:
            completion = client.chat.completions.create(
                model=get_config_manager().get_active_model_name(), messages=messages, temperature=0.1)
            content = completion.choices[0].message.content
            print(f"[放置位置识别] 颜色区域中心识别响应: {content}")

            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                if result.get("found") and "center" in result:
                    center_x, center_y = result["center"]
                    center_x = max(0, min(w-1, int(center_x)))
                    center_y = max(0, min(h-1, int(center_y)))
                    print(f"[放置位置识别] 颜色区域中心: ({center_x}, {center_y})")

                    # 第二步：让VLM理解用户的完整指令，计算最终位置
                    offset_prompt = f"""你是机器人视觉系统。用户的完整指令是："{place_description}"

【当前状态】
- 已找到{color_region}区域的中心点：({center_x}, {center_y})
- 图像尺寸: {w} x {h} 像素
- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大

【任务】
请理解用户指令中的方向和距离要求，计算最终的放置位置：
1. 如果指令只是"{color_region}区域"或"中心"，返回中心点
2. 如果指令包含方向（如"左边"、"右边"、"偏左"、"左侧"等），从中心点向该方向偏移
3. 如果指令包含距离描述（如"一点"、"稍微"、"不要太远"），使用小偏移（30-50像素）
4. 如果没有距离描述或说"远一点"，使用正常偏移（80-120像素）

【方向对应关系】
- 左/左边/左侧/偏左 → x坐标减小
- 右/右边/右侧/偏右 → x坐标增大
- 上/上面/上方/偏上 → y坐标减小
- 下/下面/下方/偏下 → y坐标增大

请返回JSON格式：
{{"place_point": [x, y], "offset_direction": "方向描述", "offset_distance": 偏移像素数, "reason": "计算原因"}}

如果只是中心位置，返回：
{{"place_point": [x, y], "offset_direction": "center", "offset_distance": 0, "reason": "使用区域中心"}}"""

                    messages2 = [{"role": "system", "content": offset_prompt}]
                    messages2.append({"role": "user", "content": f"用户指令：{place_description}"})

                    completion2 = client.chat.completions.create(
                        model=get_config_manager().get_active_model_name(), messages=messages2, temperature=0.1)
                    content2 = completion2.choices[0].message.content
                    print(f"[放置位置识别] 偏移计算响应: {content2}")

                    json_match2 = re.search(r'(\{.*\})', content2, re.DOTALL)
                    if json_match2:
                        result2 = json.loads(json_match2.group(1))
                        if "place_point" in result2:
                            pixel_x, pixel_y = result2["place_point"]
                            pixel_x = max(0, min(w-1, int(pixel_x)))
                            pixel_y = max(0, min(h-1, int(pixel_y)))

                            offset_dir = result2.get("offset_direction", "unknown")
                            offset_dist = result2.get("offset_distance", 0)
                            reason = result2.get("reason", "")

                            print(f"[放置位置识别] 最终位置: ({pixel_x}, {pixel_y})")
                            print(f"[放置位置识别] 偏移: {offset_dir}, 距离: {offset_dist}像素")

                            return {
                                "place_point": [pixel_x, pixel_y],
                                "reference_position": [center_x, center_y],
                                "confidence": 0.85,
                                "reason": f"{color_region}区域中心({center_x},{center_y})，{reason}"
                            }
        except Exception as e:
            print(f"[放置位置识别] 颜色区域识别失败: {e}")

    # 两阶段识别
    if desc_type == "reference_object" and reference_object:
        print(f"[放置位置识别] 第一阶段：识别参考物体 '{reference_object}'...")

        # 为常见物体添加英文和描述性提示
        object_hints = {
            "显微镜": "显微镜/microscope - 俯视图中是位于工作台**最底部**的白色/浅灰色小矩形（y坐标>500），在机械臂正下方，不要把机械臂误认为显微镜",
            "机械臂": "机械臂/robot arm (银色或灰色的机械手臂，位于图像中央)",
            "桌子": "桌子/table (工作台面)",
        }
        search_hint = object_hints.get(reference_object, reference_object)

        # 判断是否使用多视角图像（强制使用单视图）
        use_multi_view = False if force_single_view else ((top_view_image is not None) or (extra_images and len(extra_images) > 0))

        if use_multi_view:
            view_desc = "俯视视角" if top_view_image is not None else "主视角"
            # 使用拼接图，但明确告诉VLM每个视角的作用
            stage1_prompt = f"""这是从4个不同角度拍摄的机器人工作场景图像（水平拼接）：
- 第1张（最左边）：俯视视角 - 从正上方往下看，用于最终坐标定位
- 第2张：正对桌面视角 - 从桌子正前方看
- 第3张：面对桌面左侧视角 - 从桌子左边看
- 第4张：面对桌面右侧视角 - 从桌子右边看

【任务】
请先通过第2、3、4张图片认识 "{search_hint}" 的外观特征，然后在第1张图（俯视图）中找到它并返回像素坐标。

【物体识别提示 - 非常重要！】
- 显微镜在侧视图（第2、3、4张）中是黑色的光学设备，有圆柱形镜筒
- 但在俯视图（第1张）中，显微镜是白色/浅灰色的矩形物体，位于图像底部（y坐标很大，约在500-600范围）
- 机械臂是银色/白色的，有多个关节，位于图像中央区域
- 千万不要把中央的机械臂误认为显微镜！显微镜是底部的白色矩形物体

【重要提示】
- 第2、3、4张图片只用于帮助你认识物体的外观，不要在这些图上标注坐标
- 最终坐标必须是在第1张图（俯视图，最左边，宽度{w}像素）中的位置
- 俯视图尺寸: {w} x {h} 像素
- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大

只返回JSON：{{"found": true, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}}
如果确实找不到返回：{{"found": false}}"""
            image_for_vlm = combined_image
        else:
            stage1_prompt = f"""你是机器人视觉系统。请在这张俯视图中找到 "{search_hint}" 并返回其边界框和中心点像素坐标。

【俯视图场景布局】
- 图像尺寸: {w} x {h} 像素
- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大

【物体位置 - 非常重要！】
- 显微镜：白色/浅灰色的矩形物体，位于图像**最底部**（y坐标 > 500，通常在520-580范围）
- 机械臂：银色/白色的，有多个关节，位于图像**中央区域**（y坐标约300-420）
- 绿色区域在图像上方（y坐标较小）
- 红色区域在图像右下角

【识别要求】
- 如果要找显微镜，必须找图像底部的白色矩形物体（y > 500）
- 不要把中央的机械臂误认为显微镜！
- 机械臂的y坐标在300-420之间，显微镜的y坐标在520-580之间
- 请仔细检查y坐标是否符合要求

只返回JSON：{{"found": true, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}}
如果确实找不到返回：{{"found": false}}"""
            image_for_vlm = main_image

        messages = [{"role": "system", "content": stage1_prompt}]
        base64_img = encode_np_array(image_for_vlm)
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": f"请找到图像中的 {search_hint}"}
        ]})

        try:
            completion = client.chat.completions.create(
                model=get_config_manager().get_active_model_name(), messages=messages, temperature=0.1)
            content = completion.choices[0].message.content
            print(f"[放置位置识别] 第一阶段响应: {content}")

            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                if result.get("found") and "center" in result:
                    ref_x, ref_y = result["center"]
                    print(f"[放置位置识别] 找到 '{reference_object}' 在像素坐标 ({ref_x}, {ref_y})")

                    # 第二步：让VLM根据完整指令计算最终放置位置（与颜色区域路径一致）
                    offset_prompt = (
                        f'你是机器人视觉系统。用户的完整指令是："{place_description}"\n\n'
                        f'【当前状态】\n'
                        f'- 已找到{reference_object}的中心点：({ref_x}, {ref_y})\n'
                        f'- 图像尺寸: {w} x {h} 像素\n'
                        f'- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大\n\n'
                        f'【任务】\n'
                        f'请理解用户指令中的方向和距离要求，计算最终的放置位置：\n'
                        f'1. 如果指令包含方向（如"左边"、"右边"），从参考物体中心向该方向偏移\n'
                        f'2. 如果指令包含具体距离（如"5厘米"、"0.5cm"），按比例换算（1cm ≈ 10像素）\n'
                        f'3. 如果没有距离描述，使用默认偏移80像素\n\n'
                        f'【方向对应关系】\n'
                        f'- 左/左边/左侧 → x坐标减小\n'
                        f'- 右/右边/右侧 → x坐标增大\n'
                        f'- 上/上面/上方 → y坐标减小\n'
                        f'- 下/下面/下方 → y坐标增大\n\n'
                        f'请返回JSON格式：\n'
                        f'{{"place_point": [x, y], "offset_direction": "方向", "offset_distance": 偏移像素数, "reason": "计算原因"}}'
                    )

                    messages2 = [{"role": "system", "content": offset_prompt}]
                    messages2.append({"role": "user", "content": f"用户指令：{place_description}"})

                    try:
                        completion2 = client.chat.completions.create(
                            model=get_config_manager().get_active_model_name(), messages=messages2, temperature=0.1)
                        content2 = completion2.choices[0].message.content
                        print(f"[放置位置识别] 偏移计算响应: {content2}")

                        json_match2 = re.search(r'(\{.*\})', content2, re.DOTALL)
                        if json_match2:
                            result2 = json.loads(json_match2.group(1))
                            if "place_point" in result2:
                                place_x, place_y = result2["place_point"]
                                place_x = max(0, min(w-1, int(place_x)))
                                place_y = max(0, min(h-1, int(place_y)))
                                print(f"[放置位置识别] 放置位置像素坐标: ({place_x}, {place_y})")
                                return {
                                    "place_point": [place_x, place_y],
                                    "confidence": 0.9,
                                    "reason": result2.get("reason", f"'{reference_object}'在({ref_x},{ref_y})"),
                                    "reference_position": [int(ref_x), int(ref_y)]
                                }
                    except Exception as e2:
                        print(f"[放置位置识别] 偏移计算失败: {e2}")
        except Exception as e:
            print(f"[放置位置识别] 第一阶段失败: {e}")

    # 回退到单阶段识别
    print(f"[放置位置识别] 使用单阶段识别...")

    # 判断是否使用多视角图像（强制使用单视图）
    use_multi_view = False if force_single_view else ((top_view_image is not None) or (extra_images and len(extra_images) > 0))

    if use_multi_view:
        view_desc = "俯视视角" if top_view_image is not None else "主视角"
        system_prompt = f"""你是机器人视觉系统。用户想把物体放到：{place_description}

【图像说明】
这是从4个不同角度拍摄的机器人工作场景图像（水平拼接）：
- 第1张（最左边）：俯视视角 - 从正上方往下看，用于最终坐标定位
- 第2张：正对桌面视角 - 从桌子正前方看
- 第3张：面对桌面左侧视角 - 从桌子左边看
- 第4张：面对桌面右侧视角 - 从桌子右边看

【场景布局】
- 绿色区域是一大块矩形，覆盖了桌面的上半部分（在俯视图的上方）
- 红色区域是一小块正方形，在桌面的下方偏右角落（在俯视图的下方偏右）
- 显微镜：在俯视图中是白色/浅灰色的矩形物体，位于图像底部（y坐标很大，约500-600）
- 机械臂是银色/白色的，有多个关节，位于图像中央，不要把机械臂误认为显微镜

【重要提示】
- 第2、3、4张图片只用于帮助你理解场景，不要在这些图上标注坐标
- 最终坐标必须是在第1张图（俯视图，最左边，宽度{w}像素）中的位置
- 俯视图尺寸: {w} x {h} 像素
- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大

【任务要求】
1. 理解用户的放置描述："{place_description}"
2. 如果描述中提到了参考物体（如显微镜、机械臂等），请标注出参考物体的中心位置
3. 标注出最终的放置位置

请返回JSON格式：
- 如果有参考物体：{{"place_point": [x, y], "reference_position": [ref_x, ref_y], "confidence": 0.9, "reason": "原因"}}
- 如果没有参考物体：{{"place_point": [x, y], "confidence": 0.9, "reason": "原因"}}"""
        image_for_vlm = combined_image
    else:
        system_prompt = f"""你是机器人视觉系统。用户想把物体放到：{place_description}

【图像信息】
- 图像尺寸: {w} x {h} 像素
- 像素坐标系: 图像左上角为(0,0)，x向右增大，y向下增大

【俯视图场景布局】
- 绿色区域是一大块矩形区域，覆盖了桌面的上半部分大部分面积（图像上方）
- 红色区域是一小块正方形区域，位于桌面下方偏右的角落（图像下方偏右）
- 显微镜：白色/浅灰色的矩形物体，位于图像底部（y坐标很大，约500-600）
- 机械臂：银色/白色的，有多个关节，位于图像中央

【任务要求】
1. 理解用户的放置描述："{place_description}"
2. 如果描述中提到了参考物体（如显微镜、机械臂等），请标注出参考物体的中心位置
3. 标注出最终的放置位置

请返回JSON格式：
- 如果有参考物体：{{"place_point": [x, y], "reference_position": [ref_x, ref_y], "confidence": 0.9, "reason": "原因"}}
- 如果没有参考物体：{{"place_point": [x, y], "confidence": 0.9, "reason": "原因"}}"""
        image_for_vlm = main_image

    messages = [{"role": "system", "content": system_prompt}]
    base64_img = encode_np_array(image_for_vlm)
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        {"type": "text", "text": f"放置位置：{place_description}"}
    ]})

    try:
        completion = client.chat.completions.create(
            model=get_config_manager().get_active_model_name(), messages=messages, temperature=0.1)
        content = completion.choices[0].message.content
        print(f"[放置位置识别] 响应: {content}")

        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            if "place_point" in result:
                pixel_x, pixel_y = result["place_point"]
                pixel_x = max(0, min(w-1, int(pixel_x)))
                pixel_y = max(0, min(h-1, int(pixel_y)))
                print(f"[放置位置识别] 像素坐标: ({pixel_x}, {pixel_y})")
                result["place_point"] = [pixel_x, pixel_y]
            return result
    except Exception as e:
        print(f"[放置位置识别] 失败: {e}")

    return None


def pixel_to_world(pixel_x, pixel_y, depth_img, T_wc, fovy, img_shape):
    """
    将像素坐标转换为世界坐标。

    参数:
        pixel_x, pixel_y: 像素坐标
        depth_img: 深度图像
        T_wc: 相机到世界的变换矩阵 (spatialmath.SE3)
        fovy: 垂直视场角 (弧度)
        img_shape: 图像尺寸 (height, width)

    返回:
        world_point: [x, y, z] 世界坐标
    """
    height, width = img_shape[:2]

    # 计算相机内参
    focal = height / (2.0 * np.tan(fovy / 2.0))
    cx = width / 2.0
    cy = height / 2.0

    # 获取深度值
    depth = depth_img[int(pixel_y), int(pixel_x)]

    print(f"[像素转世界] 像素({pixel_x}, {pixel_y}), 深度={depth:.3f}m")

    if depth <= 0 or np.isnan(depth) or np.isinf(depth):
        print(f"[警告] 深度值无效: {depth}，使用默认桌面高度")
        # 假设桌面高度为0.74m，相机高度约4.4m
        depth = 3.6  # 俯视相机到桌面的距离
        print(f"[像素转世界] 使用估计深度: {depth:.3f}m")

    # 反投影到相机坐标系
    x_c = (pixel_x - cx) * depth / focal
    y_c = (pixel_y - cy) * depth / focal
    z_c = depth

    # 转换到世界坐标系
    point_camera = np.array([x_c, y_c, z_c, 1.0])
    point_world = T_wc.A @ point_camera

    return point_world[:3]


# ----------------------- 基础工具函数 -----------------------

def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64



# ----------------------- 多模态模型调用（Qwen） -----------------------

def generate_robot_actions(user_command, image_input=None):
    """
    使用 base64 的方式将 numpy 图像和用户文本指令传给 Qwen 多模态模型，
    要求模型返回两部分：
      - 模型返回内容中，第一部分为自然语言响应（说明为何选择该物体），
      - 紧跟其后的部分为纯 JSON 对象，格式如下：

        {
          "name": "物体名称",
          "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
        }

    返回一个 dict，包含 "response" 和 "coordinates"。
    参数 image_input 为 numpy 数组（BGR 格式）。
    """
    client = get_config_manager().create_client()
    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点和右下角点）及其类别名称。

    【指令解析阶段】
    2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体。

    【响应生成阶段】
    3. 输出格式必须严格如下：
    - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
    - 紧跟其后，从下一行开始返回 **标准 JSON 对象**,但是不要返回json本体,格式如下：

    {
      "name": "物体名称",
      "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
    }

    【注意事项】
    - JSON 必须从下一行开始；
    - 自然语言响应与 JSON 之间无其他额外文本;
    - JSON 对象不能有任何注释、额外文本或解释,包括不能有辅助标识为json文本的内容,不要有json;
    - 坐标 bbox 必须为整数；
    - 只允许使用 "bbox" 作为坐标格式。
    """)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    if image_input is not None:
        base64_img = encode_np_array(image_input)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    try:
        # 使用OpenAI客户端调用API
        completion = client.chat.completions.create(
            model=get_config_manager().get_active_model_name(), 
            # model="gpt-5.2-2025-12-11",  # 指定模型名称，请确认服务提供商支持的模型名
            # qwen3-omni-flash"
            # model="qwen-vl-plus",
            # model="qwen-vl-max", 
            # model="gpt-5",
            # model="qwen2.5-vl-32b-instruct",
            messages=messages,
            # max_tokens=4096,  # 可根据需要调整
            temperature=0.1,   # 降低温度以提高输出的确定性，对结构化输出有益
        )
        
        content = completion.choices[0].message.content
        print("原始响应：", content)

        # 使用正则表达式查找 JSON 部分
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                coord = json.loads(json_str)
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                coord = {}
            natural_response = content[:match.start()].strip()
        else:
            natural_response = content.strip()
            coord = {}

        return {
            "response": natural_response,
            "coordinates": coord
        }

    except Exception as e:
        print(f"请求失败：{e}")
        return {"response": "处理失败", "coordinates": {}}

# ----------------------- SAM 分割相关 -----------------------
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)

def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# ----------------------- 主流程：图像分割 -----------------------
def segment_image(image_input, output_mask='mask1.png', command_text=None, auto_only=False):
    # 1. 使用文字获取目标指令
    if command_text is None:
        print("📝 请通过文字描述目标物体及抓取指令...")
        command_text = input("请输入: ").strip()
    
    if not command_text:
        print("⚠️ 未输入指令，请重试。")
        # 返回黑色的全零掩码，防止程序崩溃
        h, w = image_input.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    print(f"✅ 输入指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    # --- Prompt Enhancing: 自动补充视觉描述以提高识别率 ---
    enhanced_command = command_text
    if "培养皿" in command_text:
        enhanced_command = f"{command_text} (green cylinder, small container, cup)"
    
    print(f"[DEBUG] VLM 增强提示词: {enhanced_command}")

    result = generate_robot_actions(enhanced_command, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("自然语言回应：", natural_response)
    print("检测到的物体信息：", detection_info)

    bbox = detection_info.get("bbox") if detection_info and "bbox" in detection_info else None

    # 过滤无效 bbox：空列表、全0、全负数、或面积过小都视为未找到
    if bbox is not None:
        if len(bbox) != 4:
            print(f"⚠️ VLM 返回无效 bbox {bbox}（长度不为4），视为未检测到目标")
            bbox = None
        else:
            x1, y1, x2, y2 = bbox
        if (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0) or \
           all(v <= 0 for v in bbox) or \
           (x2 - x1) * (y2 - y1) <= 4:
            print(f"⚠️ VLM 返回无效 bbox {bbox}，视为未检测到目标")
            bbox = None

    # 3. 准备图像供 SAM 使用（转换为 RGB）
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    # 4. 初始化 SAM，并设置图像
    predictor = choose_model()
    predictor.set_image(image_rgb)

    if bbox:
        results = predictor(bboxes=[bbox])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标,bbox:{bbox}")
    elif auto_only:
        # 自动搜索模式：不弹窗，直接返回空 mask
        print("⚠️ 未检测到目标（自动搜索模式，跳过手动选择）")
        h, w = image_input.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
    else:
        print("⚠️ 未检测到目标，请点击图像选择对象")
        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_input)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 点击坐标：{x}, {y}")
                cv2.setMouseCallback('Select Object', lambda *args: None)

        cv2.setMouseCallback('Select Object', click_handler)
        while True:
            key = cv2.waitKey(100)
            if point:
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:
                print("❌ 窗口被关闭，未进行点击")
                return None
        cv2.destroyAllWindows()
        results = predictor(points=[point], labels=[1])
        center, mask = process_sam_results(results)

    # 5. 保存分割掩码
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 分割掩码已保存：{output_mask}")
        return mask
    else:
        print("⚠️ 分割失败，未生成掩码")
        # 返回黑色的全零掩码，防止程序崩溃
        h, w = image_input.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)



# ----------------------- 主程序入口 -----------------------
if __name__ == '__main__':
    seg_mask = segment_image('color_img_path.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
