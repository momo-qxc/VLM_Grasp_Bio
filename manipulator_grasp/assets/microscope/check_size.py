#!/usr/bin/env python3
"""分析 OBJ 模型的实际尺寸"""
import re

def get_obj_bounds(filepath):
    """读取 OBJ 文件并计算边界框"""
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
    
    return {
        'width': max_x - min_x,
        'depth': max_y - min_y,
        'height': max_z - min_z,
        'center': ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
    }

# 分析新模型
new_model = '/home/robot/robot/model/microscope/microscope_textured.obj'
bounds = get_obj_bounds(new_model)

print("=== 新模型 (microscope_textured.obj) 尺寸 ===")
print(f"宽度 (X): {bounds['width']:.4f} m")
print(f"深度 (Y): {bounds['depth']:.4f} m")  
print(f"高度 (Z): {bounds['height']:.4f} m")
print(f"中心点: {bounds['center']}")

# 目标尺寸：显微镜大约 30cm 高
target_height = 0.30  # 30cm
scale_factor = target_height / bounds['height']
print(f"\n=== 推荐配置 ===")
print(f"如果目标高度是 {target_height*100}cm，推荐缩放比例: {scale_factor:.4f}")
