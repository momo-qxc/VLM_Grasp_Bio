# 金属架子和显微镜集成指南

本文档记录了如何将 `model/lab_equipment_scene.xml` 中的金属架子和显微镜模型集成到 VLM_Grasp_Interactive 项目的 `scene.xml` 中。

## 一、模型文件复制

首先需要将模型文件复制到项目中：

```bash
# 复制金属架子模型
cp -r /home/robot/robot/model/metal_shelf_01 /home/robot/robot/VLM_Grasp_Interactive/manipulator_grasp/assets/

# 复制显微镜模型
cp -r /home/robot/robot/model/microscope /home/robot/robot/VLM_Grasp_Interactive/manipulator_grasp/assets/
```

## 二、在 scene.xml 中添加资产定义

在 `<asset>` 标签内添加以下内容：

```xml
<!-- Metal shelf and microscope -->
<mesh name="metal_shelf_mesh" file="../metal_shelf_01/eb_metal_shelf_01.obj" scale="0.01 0.01 0.01"/>
<material name="shelf_metal" rgba="0.7 0.7 0.75 1" specular="0.8"/>

<mesh name="microscope_mesh" file="../microscope/microscope_textured.obj" scale="0.7 0.7 0.7"/>
<mesh name="microscope_convex" file="../microscope/microscope_convex.obj" scale="0.7 0.7 0.7"/>
<texture name="microscope_albedo" type="2d" file="../microscope/textures/Microscope_AlbedoTransparency.png"/>
<material name="microscope_material" texture="microscope_albedo" texrepeat="1 1" specular="0.3" shininess="0.5"/>
```

## 三、在 worldbody 中添加物体

### 3.1 架子视觉模型

```xml
<!-- Metal Shelf -->
<body name="metal_shelf" pos="X Y 0" euler="1.5708 Y_ROT 0">
    <geom name="shelf_visual" type="mesh" mesh="metal_shelf_mesh" material="shelf_metal" 
          contype="0" conaffinity="0" group="2"/>
</body>
```

### 3.2 架子碰撞体（5层）

```xml
<!-- Shelf collision - 5层货架碰撞体 -->
<body name="shelf_collisions" pos="COLLISION_X COLLISION_Y 0" quat="1 0 0 0">
    <geom name="shelf_layer_1" type="box" size="0.18 0.5 0.02" pos="0 0 0.10" 
          conaffinity="1" contype="1" condim="6" friction="2 0.5 0.001" rgba="0.5 0.5 0.5 0.2"/>
    <geom name="shelf_layer_2" type="box" size="0.18 0.5 0.02" pos="0 0 0.46"
          conaffinity="1" contype="1" condim="6" friction="2 0.5 0.001" rgba="0.5 0.5 0.5 0.2"/>
    <geom name="shelf_layer_3" type="box" size="0.18 0.5 0.02" pos="0 0 0.82"
          conaffinity="1" contype="1" condim="6" friction="2 0.5 0.001" rgba="0.5 0.5 0.5 0.2"/>
    <geom name="shelf_layer_4" type="box" size="0.18 0.5 0.02" pos="0 0 1.17"
          conaffinity="1" contype="1" condim="6" friction="2 0.5 0.001" rgba="0.5 0.5 0.5 0.2"/>
    <geom name="shelf_layer_5" type="box" size="0.18 0.5 0.02" pos="0 0 1.53"
          conaffinity="1" contype="1" condim="6" friction="2 0.5 0.001" rgba="0.5 0.5 0.5 0.2"/>
</body>
```

### 3.3 显微镜

```xml
<!-- Microscope - 放在架子顶层 -->
<body name="microscope" pos="MICROSCOPE_X MICROSCOPE_Y 1.55" euler="1.5708 Y_ROT 0">
    <joint name="microscope_free" type="free" damping="1.0"/>
    <geom name="microscope_visual" type="mesh" mesh="microscope_mesh" material="microscope_material" 
          contype="0" conaffinity="0" group="2"/>
    <geom name="microscope_col" type="mesh" mesh="microscope_convex"
          contype="1" conaffinity="1" condim="6" friction="2 0.5 0.001" rgba="0 0 0 0"/>
</body>
```

## 四、pos 属性详解

`pos="X Y Z"` 表示物体在世界坐标系中的位置：

| 分量 | 含义 | 说明 |
|-----|------|------|
| X | 左右位置 | 正值向右，负值向左 |
| Y | 前后位置 | 正值向前（屏幕外），负值向后 |
| Z | 高度 | 正值向上，0 表示地面 |

### 桌子坐标参考

桌子碰撞体定义：`pos="0.8 0.6 0.69"` `size="0.8 0.6 0.05"`

| 桌子角落 | X | Y | Z（桌面高度）|
|---------|---|---|-------------|
| 左后 | 0 | 0 | 0.74 |
| 右后 | 1.6 | 0 | 0.74 |
| 左前 | 0 | 1.2 | 0.74 |
| 右前 | 1.6 | 1.2 | 0.74 |

## 五、euler 属性详解

`euler="X Y Z"` 表示依次绕三个轴旋转的角度（单位：弧度）：

| 值 | 角度 |
|----|------|
| 0 | 0° |
| 1.5708 | 90° |
| 3.1416 | 180° |
| -1.5708 | -90° |

### 架子原始模型

架子原始模型是"躺着"的，需要旋转才能正确站立。

### 架子旋转设置

| 开放面朝向 | euler 设置 | 说明 |
|-----------|-----------|------|
| -Y（后方） | `euler="1.5708 0 0"` | 绕X轴转90°站立 |
| -X（左侧） | `euler="1.5708 1.5708 0"` | 先绕X轴转90°，再绕Y轴转90° |
| +X（右侧） | `euler="1.5708 -1.5708 0"` | 先绕X轴转90°，再绕Y轴转-90° |
| +Y（前方） | `euler="1.5708 3.1416 0"` | 先绕X轴转90°，再绕Y轴转180° |

**重要：** 不要使用 `euler="1.5708 0 Z_ROT"` 的形式来改变朝向，这会导致架子一半在地上一半在地下！正确的做法是修改 **第二个值（Y轴旋转）**。

## 六、碰撞体位置计算

碰撞体需要偏移到架子的开放面方向，偏移量约为 **0.19**。

### 开放面朝向 -Y 时

```
碰撞体 Y = 架子 Y - 0.19
碰撞体 X = 架子 X
```

### 开放面朝向 -X 时

```
碰撞体 X = 架子 X + 0.19
碰撞体 Y = 架子 Y
```

### 开放面朝向 +X 时

```
碰撞体 X = 架子 X - 0.19
碰撞体 Y = 架子 Y
```

## 七、当前配置示例

架子放在桌子右侧，开放面朝向 -X（朝向桌子）：

```xml
<!-- 架子视觉模型 -->
<body name="metal_shelf" pos="1.6 0.6 0" euler="1.5708 1.5708 0">
    <geom name="shelf_visual" type="mesh" mesh="metal_shelf_mesh" material="shelf_metal" 
          contype="0" conaffinity="0" group="2"/>
</body>

<!-- 碰撞体，X 偏移 +0.19 -->
<body name="shelf_collisions" pos="1.79 0.6 0" quat="1 0 0 0">
    <!-- 5层碰撞盒... -->
</body>

<!-- 显微镜，与碰撞体同位置，Z=1.55 -->
<body name="microscope" pos="1.79 0.6 1.55" euler="1.5708 1.5708 0">
    <!-- ... -->
</body>
```

## 八、常见问题

### Q1: 架子一半在地上一半在地下？

**原因：** 使用了错误的 euler 角度，比如 `euler="1.5708 0 1.5708"`

**解决：** 改变朝向时修改**第二个值**，而不是第三个值：
- ❌ `euler="1.5708 0 1.5708"` 
- ✅ `euler="1.5708 1.5708 0"`

### Q2: 架子是斜的或者倒着的？

**原因：** 第一个值（X轴旋转）不是 1.5708

**解决：** 确保第一个值始终是 `1.5708`，只修改第二个值来改变朝向

### Q3: 物体放在架子上会穿透？

**原因：** 碰撞体位置没有正确偏移

**解决：** 根据架子朝向调整碰撞体的 X 或 Y 偏移量

## 九、调试技巧

1. **使用 free joint 调试位置：**
   ```xml
   <body name="metal_shelf" pos="X Y Z" euler="1.5708 1.5708 0">
       <joint name="shelf_free" type="free" damping="100"/>
       <geom .../>
   </body>
   ```
   这样可以在 MuJoCo 中拖动架子找到合适位置。

2. **使用 Copy state 获取坐标：**
   调整好位置后，点击 MuJoCo 界面的 "Copy state" 按钮，可以获取当前物体的精确位置。

3. **碰撞体可视化：**
   将碰撞体的 `rgba` 设置为半透明（如 `rgba="0.5 0.5 0.5 0.2"`）可以看到碰撞盒的位置。

## 十、架子尺寸调整

### 10.1 修改架子宽度

架子的宽度通过 mesh 的 `scale` 属性控制。

**重要发现：** 当架子使用 `euler="1.5708 1.5708 0"` 朝向时，**第一个值**（原始模型的X方向）控制实际宽度！

```xml
<!-- scale: 宽度 高度 深度（当euler="1.5708 1.5708 0"时） -->
<mesh name="metal_shelf_mesh" file="../metal_shelf_01/eb_metal_shelf_01.obj" scale="0.012 0.01 0.01"/>
```

| scale 第一个值 | 效果 |
|---------------|------|
| 0.01 | 原始宽度（约1米） |
| 0.012 | 增加20%（约1.2米，与桌子同宽） |
| 0.015 | 增加50% |
| 0.02 | 翻倍（约2米） |

### 10.2 同步更新碰撞体

修改架子宽度后，需要同步更新碰撞体的 `size` 属性：

```xml
<!-- 碰撞体 size: 深度 宽度 厚度 -->
<geom name="shelf_layer_1" type="box" size="0.18 0.6 0.02" .../>
```

| 架子 scale 第一个值 | 碰撞体 size 第二个值 |
|--------------------|---------------------|
| 0.01 | 0.5 |
| 0.012 | 0.6 |
| 0.015 | 0.75 |
| 0.02 | 1.0 |

**计算公式：** `碰撞体宽度 = 0.5 × (scale / 0.01)`

### 10.3 当前配置

```xml
<!-- 架子宽度：原始的120%，与桌子同宽 -->
<mesh name="metal_shelf_mesh" file="../metal_shelf_01/eb_metal_shelf_01.obj" scale="0.012 0.01 0.01"/>

<!-- 碰撞体宽度：从0.5增加到0.6 -->
<geom name="shelf_layer_1" type="box" size="0.18 0.6 0.02" pos="0 0 0.10" .../>
<geom name="shelf_layer_2" type="box" size="0.18 0.6 0.02" pos="0 0 0.46" .../>
<geom name="shelf_layer_3" type="box" size="0.18 0.6 0.02" pos="0 0 0.82" .../>
<geom name="shelf_layer_4" type="box" size="0.18 0.6 0.02" pos="0 0 1.17" .../>
<geom name="shelf_layer_5" type="box" size="0.18 0.6 0.02" pos="0 0 1.53" .../>
```

### 10.4 修改架子高度（2026-01-17 更新）

架子的高度缩放由 `scale` 的 **第二个值** 控制。

**注意：** 修改高度后，必须同步调整所有 5 层碰撞板的 `pos.z` 和 `size.z`，以及放在架子上的物体（如显微镜）的高度。

| 缩放比例 | 物理高度系数 | 显微镜新 Z 轴 | 碰撞层厚度 (size.z) |
|---------|------------|--------------|-------------------|
| 0.01 (原始) | 1.0 | 1.55 | 0.02 |
| 0.009 (当前) | 0.9 | 1.395 | 0.018 |

**同步公式：** `新坐标 = 原始坐标 × 0.9`

## 十一、培养皿集成 (Petri Dish)

培养皿集成采用了“显微镜风格”的高性能模式：

1. **碰撞优化**：使用 `collision_convex.obj`（凸包网格）代替原始的 17 个几何体，极大提升了仿真运行速度（Real-time Factor）。
2. **物理参数**：
    - `damping="0.01"`：保证翻转动作灵敏。
    - `density="2000"`：增加分量感，防止过于轻飘。

## 十二、排错记录与经验沉淀 (Debugging History)

为了防止后续出现类似问题，以下记录了在集成培养皿过程中的关键报错及修复逻辑。

### 12.1 OpenAI 客户端代理报错 (2026-01-17)

**问题现象：** 运行 `main_vlm.py` 时报 `ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7890/')`。
**原因：** 系统中设置了 `socks://` 全局代理，而 `httpx` 库不支持此协议头。

```python
# 【旧代码】(直接初始化，受环境代理干扰)
client = OpenAI(api_key='...', base_url="...")

# 【新代码】(强制清除环境代理，设置 trust_env=False)
import os, httpx
for key in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
    os.environ.pop(key, None)
client = OpenAI(..., http_client=httpx.Client(trust_env=False))
```

## 十二、模型集成排错经验 (Modeling Debugging Notes)

### 12.1 模型穿模与吸附感问题 (核心：中心点对齐)

**问题现象：** 模型落到桌面后陷入一半，拖动时有严重的粘滞吸附感。
**本质原因：** MuJoCo 加载 Mesh 时会以模型的 AABB 中心为 Body 原点，导致模型中线下压。

```xml
<!-- 【错误】pos="0 0 0" 会导致模型底部被埋进桌面 -->
<!-- 【正确】通过 pos="0 0 0.02" (高度一半) 手动补偿偏移 -->
<geom name="dish" type="mesh" pos="0 0 0.02" .../>
```

### 12.2 尺寸对齐与协作冲突

当物体（如培养皿）直径过大（>8.5cm）时，会与机器人夹爪产生结构性碰撞。
- **解决方案**：将 `scale` 调整为 `0.0008` (80%)，为夹爪留出安全余量。

---

## 十三、更多记录

关于 **VLM 抓取算法、代理配置、自动调平及放置逻辑** 的详细排错记录，请参阅：
👉 [README_vlm_grasping.md](file:///home/robot/robot/VLM_Grasp_Interactive/README_vlm_grasping.md)

---
## 十四、MuJoCo 核心参数释义 (Geom)

为了方便后续调整架子（Shelf）或障碍物，下表详细说明了 `scene.xml` 中 `geom` 标签各参数的物理含义：

| 参数 | 示例值 | 详细释义 |
| :--- | :--- | :--- |
| **`type`** | `box` | 几何形状。`box` 表示长方体，在计算碰撞时速度最快。 |
| **`size`** | `0.18 0.5 0.02` | **半长 (Half-lengths)**。实际的长宽高需乘以 2。例如该值代表宽 0.36m, 深 1.0m, 厚 0.04m。 |
| **`pos`** | `0 0 0.1` | **相对位置**。相对于父节点（body）中心的偏移量 `[X, Y, Z]`。 |
| **`contype`** | `1` | **碰撞类型码**。定义“我是谁”。 |
| **`conaffinity`** | `1` | **碰撞亲和力**。定义“我想撞谁”。当 A 的 type 与 B 的 affinity 匹配时发生物理碰撞。 |
| **`condim`** | `6` | **接触维度**。`1`:无摩擦, `3`:普通滑动摩擦, `6`:包含滑动、扭转和滚动摩擦（用于精密操作）。 |
| **`friction`** | `2 0.5 0.001` | **摩擦系数**。顺序为 `[滑动, 扭转, 滚动]`。值越大物体越不容易滑走。 |
| **`rgba`** | `0.5 0.5 0.5 0.2` | **颜色与透明度**。`[R, G, B, Alpha]`。Alpha=0.2 表示 20% 不透明，视觉上呈半透明灰色。 |

> [!TIP]
> **关于碰撞层**：在仿真中，我们经常将视觉模型（group 1）设为 `contype="0" conaffinity="0"`，而将简化的几何体（group 0）作为实际的碰撞体，这样既能保证视觉精细度，又能让仿真运行得飞快。

---
*记录人：GG-bond*
*最后更新：2026-01-17 21:00*

