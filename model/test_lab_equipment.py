#!/usr/bin/env python3
"""
Test script to visualize the lab equipment models (metal shelf and microscope) in MuJoCo.

Usage:
    python test_lab_equipment.py

This script loads the lab_equipment_scene.xml and opens the MuJoCo viewer 
for interactive visualization.
"""

import mujoco
import mujoco.viewer
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the scene XML
    scene_path = os.path.join(script_dir, "lab_equipment_scene.xml")
    
    print(f"Loading MuJoCo model from: {scene_path}")
    
    # Check if file exists
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at {scene_path}")
        return
    
    try:
        # Load the model
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print("Model loaded successfully!")
        print(f"  - Number of bodies: {model.nbody}")
        print(f"  - Number of geoms: {model.ngeom}")
        print(f"  - Number of meshes: {model.nmesh}")
        
        # Launch the interactive viewer
        print("\nLaunching MuJoCo viewer...")
        print("Controls:")
        print("  - Left mouse button: Rotate view")
        print("  - Right mouse button: Pan view")
        print("  - Scroll wheel: Zoom")
        print("  - Ctrl + Right mouse: Drag objects")
        print("  - Double-click on object: Select for perturbation")
        print("  - ESC or close window: Exit")
        
        import time
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera to look at microscope (top shelf)
            viewer.cam.lookat[0] = 0.0   # x
            viewer.cam.lookat[1] = -0.2  # y  
            viewer.cam.lookat[2] = 1.5   # z (top shelf height)
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            
            while viewer.is_running():
                step_start = time.time()
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # Control simulation speed (60 Hz)
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTip: Check if the mesh files are in the correct location.")
        print("Expected mesh paths relative to the XML file:")
        print("  - metal_shelf_01/eb_metal_shelf_01.obj")
        print("  - microscope/microscope_textured.obj")

if __name__ == "__main__":
    main()
