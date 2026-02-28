import os
import trimesh

def generate_convex_hull(input_path, output_path, scale=1.0):
    print(f"Loading mesh from: {input_path}")
    mesh = trimesh.load(input_path)
    
    if scale != 1.0:
        mesh.apply_scale(scale)
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    convex_hull = mesh.convex_hull
    print(f"Convex hull: {len(convex_hull.vertices)} vertices, {len(convex_hull.faces)} faces")
    
    convex_hull.export(output_path)
    print(f"Saved convex hull to: {output_path}")

def main():
    petri_dir = "/home/robot/robot/VLM_Grasp_Interactive/manipulator_grasp/assets/container/cell_dish_100_vis"
    petri_input = os.path.join(petri_dir, "visual.obj")
    petri_output = os.path.join(petri_dir, "collision_convex.obj")
    
    if os.path.exists(petri_input):
        generate_convex_hull(petri_input, petri_output)
    else:
        print(f"Input file not found: {petri_input}")

if __name__ == "__main__":
    main()
