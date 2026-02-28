#!/usr/bin/env python3
"""
Generate a convex hull mesh from the microscope OBJ file for use as collision geometry.
This creates a simpler collision mesh that is more stable in MuJoCo.

Usage:
    pip install trimesh numpy
    python generate_convex_hull.py
"""

import os

try:
    import trimesh
    import numpy as np
except ImportError:
    print("Please install required packages:")
    print("  pip install trimesh numpy")
    exit(1)

def generate_convex_hull(input_path, output_path, scale=1.0):
    """
    Generate a convex hull mesh from an OBJ file.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file for convex hull
        scale: Scale factor to apply
    """
    print(f"Loading mesh from: {input_path}")
    mesh = trimesh.load(input_path)
    
    # Apply scale if needed
    if scale != 1.0:
        mesh.apply_scale(scale)
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Generate convex hull
    convex_hull = mesh.convex_hull
    
    print(f"Convex hull: {len(convex_hull.vertices)} vertices, {len(convex_hull.faces)} faces")
    
    # Export convex hull
    convex_hull.export(output_path)
    print(f"Saved convex hull to: {output_path}")
    
    return convex_hull

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Microscope - use the new textured model
    microscope_input = os.path.join(script_dir, "microscope", "microscope_textured.obj")
    microscope_output = os.path.join(script_dir, "microscope", "microscope_convex.obj")
    
    if os.path.exists(microscope_input):
        generate_convex_hull(microscope_input, microscope_output)
    else:
        print(f"Input file not found: {microscope_input}")

if __name__ == "__main__":
    main()
