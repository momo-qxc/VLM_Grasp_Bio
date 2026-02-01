import os

def get_obj_bounds(filepath):
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
        'min_z': min_z,
        'max_z': max_z,
        'height': max_z - min_z,
        'center_z': (max_z + min_z) / 2
    }

def main():
    petri_obj = "/home/robot/robot/VLM_Grasp_Interactive/manipulator_grasp/assets/container/cell_dish_100_vis/visual.obj"
    if os.path.exists(petri_obj):
        bounds = get_obj_bounds(petri_obj)
        print(f"Mesh bounds: {bounds}")
        
        # Current scales in scene.xml are 0.001 0.001 0.002
        scale_z = 0.002
        actual_min_z = bounds['min_z'] * scale_z
        actual_max_z = bounds['max_z'] * scale_z
        actual_height = bounds['height'] * scale_z
        
        print(f"Scaled Min Z: {actual_min_z}")
        print(f"Scaled Max Z: {actual_max_z}")
        print(f"Scaled Height: {actual_height}")
        
        # Table top is at 0.74
        # We want Scaled Min Z + Body_Z = 0.74
        # So Body_Z = 0.74 - Scaled Min Z
        required_body_z = 0.74 - actual_min_z
        print(f"Required Body Z to sit on table: {required_body_z}")

if __name__ == "__main__":
    main()
