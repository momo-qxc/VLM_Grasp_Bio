import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

# import whisper
import json
import re
import base64
import textwrap
import queue
import time
import io
import os  # 导入 os 用于处理环境变量

# 强制清除可能导致错误的代理环境变量
for key in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

# import soundfile as sf  
# import sounddevice as sd
# from scipy.io.wavfile import write
# from pydub import AudioSegment

from openai import OpenAI  # 导入OpenAI客户端
import httpx  # 导入 httpx 用于处理代理问题

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from google import genai
from google.genai import types

# 导入全局配置
from config import Config


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
    client = OpenAI(
        api_key=Config.QWEN_API_KEY,
        base_url=Config.QWEN_BASE_URL,
        http_client=httpx.Client(trust_env=False)
    )

    system_prompt = textwrap.dedent("""\
    你是一个机器人指令解析系统。请分析用户的自然语言指令，提取以下信息：

    1. 要抓取的物体名称（grasp_target）
    2. 放置位置的描述（place_description）- 如果用户没有指定放置位置，则为空字符串

    【示例】
    输入: "把培养皿放置到显微镜的右边"
    输出: {"grasp_target": "培养皿", "place_description": "显微镜的右边", "has_place_instruction": true}

    输入: "抓取红色的试管"
    输出: {"grasp_target": "红色的试管", "place_description": "", "has_place_instruction": false}

    输入: "把烧杯移到桌子左上角的红色区域"
    输出: {"grasp_target": "烧杯", "place_description": "桌子左上角的红色区域", "has_place_instruction": true}

    【注意】
    - 只返回JSON对象，不要有其他文字
    - 如果指令中包含"放到"、"放置到"、"移到"、"移动到"等词语，说明有放置指令
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
            model=Config.QWEN_MODEL,
            messages=messages,
            temperature=Config.DEFAULT_TEMPERATURE,
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


def detect_place_position(place_description, global_image, depth_image=None, extra_images=None):
    """
    使用VLM在全局图像中识别放置位置。
    支持多相机图像输入，提供更全面的场景理解。

    参数:
        place_description: 放置位置描述
        global_image: 主相机图像 (用于坐标计算)
        depth_image: 深度图像 (可选)
        extra_images: 额外的相机图像列表 (可选，用于辅助识别)
    """
    client = OpenAI(
        api_key=Config.QWEN_API_KEY,
        base_url=Config.QWEN_BASE_URL,
        http_client=httpx.Client(trust_env=False)
    )

    h, w = global_image.shape[:2]

    # 如果有多个相机图像，拼接成一张大图用于识别
    if extra_images and len(extra_images) > 0:
        print(f"[放置位置识别] 使用多相机融合模式 ({1 + len(extra_images)} 个视角)")
        # 创建拼接图像用于VLM识别
        all_images = [global_image] + extra_images
        # 水平拼接所有图像
        # 先调整所有图像到相同高度
        target_h = min(img.shape[0] for img in all_images)
        resized_images = []
        for img in all_images:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized = cv2.resize(img, (new_w, target_h))
            resized_images.append(resized)
        combined_image = np.hstack(resized_images)
        combined_h, combined_w = combined_image.shape[:2]
        print(f"[放置位置识别] 拼接图像尺寸: {combined_w} x {combined_h}")
        # 保存拼接图像用于调试
        cv2.imwrite("debug_combined_views.jpg", combined_image)
    else:
        combined_image = global_image
        combined_h, combined_w = h, w

    # 解析放置描述，提取参考物体和方向
    reference_object = None
    direction = None
    color_region = None  # 新增：颜色区域

    # 检查是否是颜色区域描述
    color_pattern = r'(红色|绿色|蓝色|黄色|白色|黑色|橙色|紫色)(的)?(区域|地方|位置|部分)'
    color_match = re.search(color_pattern, place_description)
    if color_match:
        color_region = color_match.group(1)
        print(f"[放置位置识别] 检测到颜色区域描述: {color_region}")

    patterns = [
        (r'(.+?)的(左边|右边|上面|下面|前面|后面|旁边)', lambda m: (m.group(1), m.group(2))),
        (r'(左边|右边|上面|下面|前面|后面)的(.+)', lambda m: (m.group(2), m.group(1))),
    ]

    for pattern, extractor in patterns:
        match_result = re.search(pattern, place_description)
        if match_result:
            reference_object, direction = extractor(match_result)
            break

    # 解析距离描述，计算偏移量
    def parse_distance_offset(description):
        """
        根据描述中的距离信息计算像素偏移量。
        相机视角下，大约 1cm ≈ 8-12 像素（取决于深度）
        """
        # 检查明确的厘米数值
        cm_match = re.search(r'(\d+)[-~到]?(\d*)(?:cm|厘米)', description)
        if cm_match:
            cm_min = int(cm_match.group(1))
            cm_max = int(cm_match.group(2)) if cm_match.group(2) else cm_min
            avg_cm = (cm_min + cm_max) / 2
            # 大约 10 像素/厘米
            offset = int(avg_cm * 10)
            print(f"[距离解析] 检测到距离: {cm_min}-{cm_max}cm → 偏移 {offset} 像素")
            return max(20, min(200, offset))  # 限制在合理范围内

        # 检查相对距离描述
        close_keywords = ['紧挨', '紧贴', '贴着', '挨着', '很近', '近一点', '不要太远', '不要离.*太远', '靠近']
        medium_keywords = ['旁边', '边上', '附近']
        far_keywords = ['远一点', '远些', '离远', '稍远']

        for keyword in close_keywords:
            if re.search(keyword, description):
                print(f"[距离解析] 检测到近距离关键词: '{keyword}' → 偏移 40 像素")
                return 40

        for keyword in medium_keywords:
            if re.search(keyword, description):
                print(f"[距离解析] 检测到中等距离关键词: '{keyword}' → 偏移 70 像素")
                return 70

        for keyword in far_keywords:
            if re.search(keyword, description):
                print(f"[距离解析] 检测到远距离关键词: '{keyword}' → 偏移 120 像素")
                return 120

        # 默认偏移
        print(f"[距离解析] 未检测到距离描述，使用默认偏移 80 像素")
        return 80

    # 计算偏移量
    pixel_offset = parse_distance_offset(place_description)

    print(f"[放置位置识别] 解析结果: 参考物体='{reference_object}', 方向='{direction}', 颜色区域='{color_region}', 偏移={pixel_offset}像素")

    # ===== 新增：颜色区域识别 =====
    if color_region:
        print(f"[放置位置识别] 使用颜色区域识别模式...")

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

        color_prompt = f"""请在图像中找到 {color_region}/{color_en} 颜色的区域，并返回该区域的中心点坐标。

【重要提示】
- 仔细观察桌面/工作台上的颜色标记区域
- {color_region}区域通常是桌面上的彩色标记或贴纸
- 图像尺寸: {w} x {h} 像素
- 坐标系: 左上角(0,0)，x向右增大，y向下增大

请返回JSON格式：
{{"found": true, "center": [x, y], "reason": "找到{color_region}区域的原因"}}
如果找不到{color_region}区域，返回：
{{"found": false, "reason": "未找到的原因"}}"""

        messages = [{"role": "system", "content": color_prompt}]
        base64_img = encode_np_array(global_image)
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": f"请找到图像中的{color_region}区域"}
        ]})

        try:
            completion = client.chat.completions.create(
                model=Config.QWEN_MODEL, messages=messages, temperature=Config.DEFAULT_TEMPERATURE)
            content = completion.choices[0].message.content
            print(f"[放置位置识别] 颜色区域识别响应: {content}")

            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                if result.get("found") and "center" in result:
                    place_x, place_y = result["center"]
                    place_x = max(0, min(w-1, int(place_x)))
                    place_y = max(0, min(h-1, int(place_y)))

                    return {
                        "place_point": [place_x, place_y],
                        "confidence": 0.85,
                        "reason": f"找到{color_region}区域在({place_x},{place_y})"
                    }
        except Exception as e:
            print(f"[放置位置识别] 颜色区域识别失败: {e}")

    # 两阶段识别
    if reference_object and direction:
        print(f"[放置位置识别] 第一阶段：识别参考物体 '{reference_object}'...")

        # 为常见物体添加英文和描述性提示
        object_hints = {
            "显微镜": "显微镜/microscope (黑色的光学设备，有目镜和物镜，通常在桌面上)",
            "机械臂": "机械臂/robot arm (银色或灰色的机械手臂)",
            "桌子": "桌子/table (工作台面)",
        }
        search_hint = object_hints.get(reference_object, reference_object)

        # 使用多视角图像（如果有）
        if extra_images and len(extra_images) > 0:
            stage1_prompt = f"""这是从多个角度拍摄的场景图像（水平拼接）。
请在图像中找到 "{search_hint}" 并返回其在【第一张图（最左边）】中的位置。

【重要提示】
- 图像是多个视角的拼接，请综合所有视角来识别物体
- 显微镜通常是黑色的光学设备，有圆柱形的镜筒
- 返回的坐标必须是在第一张图（最左边，宽度约{w}像素）中的位置

第一张图尺寸: {w} x {h} 像素。
只返回JSON：{{"found": true, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}}
如果确实找不到返回：{{"found": false}}"""
            image_for_vlm = combined_image
        else:
            stage1_prompt = f"""请在图像中找到 "{search_hint}" 并返回其边界框和中心点。

【重要提示】
- 仔细观察整个图像
- 显微镜通常是黑色的光学设备，有圆柱形的镜筒
- 如果看到类似的设备，请标记它的位置

图像尺寸: {w} x {h} 像素。
只返回JSON：{{"found": true, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}}
如果确实找不到返回：{{"found": false}}"""
            image_for_vlm = global_image

        messages = [{"role": "system", "content": stage1_prompt}]
        base64_img = encode_np_array(image_for_vlm)
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": f"请找到图像中的 {search_hint}"}
        ]})

        try:
            completion = client.chat.completions.create(
                model=Config.QWEN_MODEL, messages=messages, temperature=Config.DEFAULT_TEMPERATURE)
            content = completion.choices[0].message.content
            print(f"[放置位置识别] 第一阶段响应: {content}")

            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                if result.get("found") and "center" in result:
                    ref_x, ref_y = result["center"]
                    print(f"[放置位置识别] 找到 '{reference_object}' 在 ({ref_x}, {ref_y})")

                    # 使用解析出的偏移量
                    if direction == "左边":
                        place_x, place_y = ref_x - pixel_offset, ref_y
                    elif direction == "右边":
                        place_x, place_y = ref_x + pixel_offset, ref_y
                    elif direction in ["上面", "前面"]:
                        place_x, place_y = ref_x, ref_y - pixel_offset
                    elif direction in ["下面", "后面"]:
                        place_x, place_y = ref_x, ref_y + pixel_offset
                    else:
                        place_x, place_y = ref_x + pixel_offset, ref_y

                    place_x = max(0, min(w-1, int(place_x)))
                    place_y = max(0, min(h-1, int(place_y)))

                    return {
                        "place_point": [place_x, place_y],
                        "confidence": 0.9,
                        "reason": f"'{reference_object}'在({ref_x},{ref_y})，{direction}偏移{pixel_offset}像素",
                        "reference_position": [ref_x, ref_y]
                    }
        except Exception as e:
            print(f"[放置位置识别] 第一阶段失败: {e}")

    # 回退到单阶段识别
    print(f"[放置位置识别] 使用单阶段识别...")
    system_prompt = f"""你是机器人视觉系统。用户想把物体放到：{place_description}

【图像信息】
- 图像尺寸: {w} x {h} 像素
- 坐标系: 左上角(0,0)，x向右增大，y向下增大

【识别提示】
- 如果描述中包含颜色（红色、绿色、蓝色等），请找到桌面上对应颜色的标记区域
- 如果描述中包含参考物体（如显微镜），请先找到该物体，再确定相对位置
- 显微镜通常是黑色的光学设备，有圆柱形的镜筒
- 桌面上可能有彩色的标记区域（红色、绿色等）

请返回JSON格式：{{"place_point": [x, y], "confidence": 0.9, "reason": "原因"}}"""

    messages = [{"role": "system", "content": system_prompt}]
    base64_img = encode_np_array(global_image)
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        {"type": "text", "text": f"放置位置：{place_description}"}
    ]})

    try:
        completion = client.chat.completions.create(
            model=Config.QWEN_MODEL, messages=messages, temperature=0.1)
        content = completion.choices[0].message.content
        print(f"[放置位置识别] 响应: {content}")

        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            if "place_point" in result:
                x, y = result["place_point"]
                result["place_point"] = [max(0,min(w-1,int(x))), max(0,min(h-1,int(y)))]
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

    if depth <= 0 or np.isnan(depth) or np.isinf(depth):
        print(f"[警告] 深度值无效: {depth}，使用默认桌面高度")
        # 假设桌面高度为0.74m，相机高度约3m
        depth = 2.5  # 估计深度

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
    # 初始化OpenAI客户端，彻底禁用环境代理 (trust_env=False)
    # 替换为自己的模型调用，没有本地部署的，可以参考该网站 https://sg.uiuiapi.com/v1
    client = OpenAI(
        api_key=Config.QWEN_API_KEY,
        base_url=Config.QWEN_BASE_URL,
        http_client=httpx.Client(trust_env=False)
    )       
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
            model=Config.QWEN_MODEL, 
            # model="gpt-5.2-2025-12-11",  # 指定模型名称，请确认服务提供商支持的模型名
            # qwen3-omni-flash"
            # model="qwen-vl-plus",
            # model="qwen-vl-max", 
            # model="gpt-5",
            # model="qwen2.5-vl-32b-instruct",
            messages=messages,
            # max_tokens=4096,  # 可根据需要调整
            temperature=Config.DEFAULT_TEMPERATURE,   # 降低温度以提高输出的确定性，对结构化输出有益
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


def generate_robot_actions_gemini(user_command, image_input=None):
    """
    使用 Google Gemini Robotics-ER 1.5 模型处理图像和指令。
    使用与原 Qwen/OpenAI 相同的提示词逻辑，但适配 Gemini 的输入输出。
    """
    # 替换为用户的 API Key
    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    MODEL_ID = Config.GEMINI_MODEL

    if image_input is None:
        return {"response": "需要图像输入", "coordinates": {}}

    # 将 numpy BGR 图像转为 RGB 并编码为 bytes
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    success, encoded_image = cv2.imencode('.jpg', image_rgb)
    if not success:
         return {"response": "图像编码失败", "coordinates": {}}
    image_bytes = encoded_image.tobytes()

    # 复用原有的 System Prompt 逻辑，保持实验一致性
    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体。

    【指令解析阶段】
    2. 根据用户的自然语言指令 ({user_command})，从识别的物体中筛选出最匹配的目标物体。

    【响应生成阶段】
    3. 输出格式必须严格如下：
    - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
    - 紧跟其后，从下一行开始返回 **标准 JSON 对象**, 格式如下：

    [
      {{
        "box_2d": [ymin, xmin, ymax, xmax],
        "label": "物体名称"
      }}
    ]

    【注意事项】
    - 坐标 box_2d 必须为 0-1000 的归一化整数 (Gemini 标准)；
    - 自然语言响应与 JSON 之间无其他额外文本;
    - JSON 对象不能有任何注释。
    """)

    # 组合 Prompt
    full_prompt = system_prompt.format(user_command=user_command)

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                full_prompt
            ],
            config = types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=1024) 
            )
        )
        
        content = response.text
        print("Gemini 原始响应：", content)

        # 解析响应
        # 寻找 JSON 部分
        match = re.search(r'(\[.*\])', content, re.DOTALL)
        natural_response = content
        coord = {}

        if match:
            json_str = match.group(1)
            natural_response = content[:match.start()].strip()
            try:
                items = json.loads(json_str)
                if items and len(items) > 0:
                    item = items[0]
                    #处理坐标转换：Gemini (0-1000) -> 像素
                    h, w = image_input.shape[:2]
                    box_2d = item.get("box_2d")
                    
                    if box_2d:
                        ymin, xmin, ymax, xmax = box_2d
                        x1 = int(xmin / 1000 * w)
                        y1 = int(ymin / 1000 * h)
                        x2 = int(xmax / 1000 * w)
                        y2 = int(ymax / 1000 * h)
                        
                        coord = {
                            "name": item.get("label", "target"),
                            "bbox": [x1, y1, x2, y2]
                        }
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                # 尝试稍微清洗下 json 再解析
        
        return {
            "response": natural_response,
            "coordinates": coord
        }

    except Exception as e:
        print(f"Gemini 请求失败：{e}")
        return {"response": f"处理失败: {e}", "coordinates": {}}


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


# ----------------------- 语音识别与 TTS -----------------------

# 初始化全局模型变量
_global_models = {}


def load_models():
    """在需要时加载模型，避免启动时全部加载占用资源"""
    if not _global_models:
        print("🔄 正在加载离线语音模型...")
        # 加载Whisper小型模型 (适合你的6GB显存)
        # _global_models['asr'] = whisper.load_model("small")
        # _global_models['asr'] = whisper.load_model("tiny")
        # _global_models['asr'] = whisper.load_model("base")
        print("✅ Whisper的base模型加载完毕")

        try:
            import pyttsx3
            _global_models['tts_backup'] = pyttsx3.init()
            # 配置TTS
            _global_models['tts_backup'].setProperty('rate', 160)  # 语速
            voices = _global_models['tts_backup'].getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    _global_models['tts_backup'].setProperty('voice', voice.id)
                    break
            print("✅ TTS (pyttsx3) 初始化完毕")
        except Exception as e:
            print(f"⚠️  TTS初始化失败: {e}")
            _global_models['tts_backup'] = None

    return _global_models


# 音频参数配置
samplerate = 16000
channels = 1
dtype = 'int16'
frame_duration = 0.2
frame_samples = int(frame_duration * samplerate)
silence_threshold = 250
silence_max_duration = 2.0
q = queue.Queue()


def rms(audio_frame):
    samples = np.frombuffer(audio_frame, dtype=np.int16)
    if samples.size == 0:
        return 0
    mean_square = np.mean(samples.astype(np.float32) ** 2)
    if np.isnan(mean_square) or mean_square < 1e-5:
        return 0
    return np.sqrt(mean_square)

def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 状态警告：", status)
    q.put(bytes(indata))

def recognize_speech():
    """录音并返回音频数据（numpy 数组）"""
    print("🎙️ 启动录音，请说话...")
    # print("💡 调试信息：正在监测实时音量（RMS），请观察不说话时的基础噪音值")
    audio_buffer = []
    is_speaking = False
    last_voice_time = time.time()

    with sd.RawInputStream(samplerate=samplerate, blocksize=frame_samples,
                           dtype=dtype, channels=channels, callback=callback):
        while True:
            frame = q.get()
            volume = rms(frame)
            current_time = time.time()

            # print(f"实时音量（RMS）: {volume}") 

            if volume > silence_threshold:
                if not is_speaking:
                    print("🎤 检测到语音，开始录音...")
                    is_speaking = True
                    audio_buffer = []
                audio_np = np.frombuffer(frame, dtype=np.int16)
                audio_buffer.append(audio_np)
                last_voice_time = current_time
            elif is_speaking and (current_time - last_voice_time > silence_max_duration):
                print("🛑 停止录音，准备识别...")
                full_audio = np.concatenate(audio_buffer, axis=0)
                return full_audio
            elif not is_speaking and (current_time - last_voice_time > 10.0):
                print("🛑 超时：未检测到语音输入")
                return np.array([], dtype=np.int16)

def speech_to_text_offline(audio_data):
    """
    使用离线Whisper模型将录音数据转换为文本
    """
    print("📡 正在进行离线语音识别...")
    models = load_models()
    asr_model = models['asr']

    # 保存临时音频文件
    temp_wav = "temp_audio.wav"
    write(temp_wav, samplerate, audio_data.astype(np.int16))

    try:
        # 使用Whisper进行识别，指定语言为中文以提高精度和速度
        result = asr_model.transcribe(temp_wav, language="zh", fp16=torch.cuda.is_available())
        return result["text"].strip()
    except Exception as e:
        print(f"❌ 离线语音识别失败: {e}")
        return ""

def play_tts_offline(text):
    """
    使用离线TTS模型将文本转换为语音并播放
    """
    if not text:
        return
        
    print(f"📢 离线TTS播放: {text}")
    models = load_models()

    try:
        if models['tts_backup'] is not None:
            models['tts_backup'].say(text)
            models['tts_backup'].runAndWait()

    except Exception as e:
        print("❌ 无可用TTS引擎")


def voice_command_to_keyword():
    """
    获取语音命令并转换为文本。
    直接返回识别的文本指令。
    """
    audio_data = recognize_speech()
    text = speech_to_text_offline(audio_data) # 改为调用离线ASR
    if not text:
        print("⚠️ 没有识别到文本")
        return ""
    print("📝 识别文本：", text)
    # play_tts_offline(f"已收到指令: {text}") # 改为调用离线TTS
    return text


# ----------------------- 主流程：图像分割 -----------------------
def segment_image(image_input, output_mask='mask1.png', command_text=None):
    # 1. 使用文字获取目标指令
    if command_text is None:
        print("📝 请通过文字描述目标物体及抓取指令...")
        command_text = input("请输入: ").strip()
    
    if not command_text:
        print("⚠️ 未识别到语音指令，请重试。")
        # 返回黑色的全零掩码，防止程序崩溃
        h, w = image_input.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
        
    print(f"✅ 识别的语音指令：{command_text}")

    # # 1. 使用语音获取目标指令
    # print("🎙️ 请通过语音描述目标物体及抓取指令...")
    # command_text = voice_command_to_keyword()
    # if not command_text:
    #     print("⚠️ 未识别到语音指令，请重试。")
    #     return None
    # print(f"✅ 识别的语音指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    # 2. 通过多模态模型获取检测框
    # --- Prompt Enhancing: 自动补充视觉描述以提高识别率 ---
    enhanced_command = command_text
    if "培养皿" in command_text:
        enhanced_command = f"{command_text} (green cylinder, small container, cup)"
    
    print(f"[DEBUG] VLM 增强提示词: {enhanced_command}")
    
    result = generate_robot_actions(enhanced_command, image_input)
    # 切换为 Gemini 模型
    # result = generate_robot_actions_gemini(command_text, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("自然语言回应：", natural_response)
    print("检测到的物体信息：", detection_info)

    # 仅对模型返回的自然语言回应播报
    # play_tts_offline(natural_response)
    
    bbox = detection_info.get("bbox") if detection_info and "bbox" in detection_info else None
    
    # 3. 准备图像供 SAM 使用（转换为 RGB）
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    # 4. 初始化 SAM，并设置图像
    predictor = choose_model()
    predictor.set_image(image_rgb)

    if bbox:
        results = predictor(bboxes=[bbox])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标,bbox:{bbox}")
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
