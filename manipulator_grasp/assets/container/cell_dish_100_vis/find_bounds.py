import sys

def get_z_bounds(file_path):
    z_values = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                z_values.append(float(parts[3]))
    if not z_values:
        return None
    return min(z_values), max(z_values)

if __name__ == "__main__":
    file_path = "/home/robot/robot/VLM_Grasp_Interactive/manipulator_grasp/assets/container/cell_dish_100_vis/visual.obj"
    bounds = get_z_bounds(file_path)
    if bounds:
        min_z, max_z = bounds
        print(f"Min Z: {min_z}")
        print(f"Max Z: {max_z}")
        print(f"Center Z: {(min_z + max_z) / 2}")
        print(f"Height: {max_z - min_z}")
    else:
        print("No vertices found.")
