#!/usr/bin/env python
import sys
import os

# Get arguments
if len(sys.argv) != 3:
    print("Usage: python run_pose_refinement.py <source_dir> <output_dir>")
    sys.exit(1)

source_dir = sys.argv[1]
output_dir = sys.argv[2]

# Set Python path
project_root = os.path.dirname(os.getcwd())  # Go back to parent directory
sys.path.insert(0, project_root)

try:
    from rag.transformed_grasp_angles import transformed_grasp_angles
    transformed_grasp_angles(source_dir, output_dir)
except Exception as e:
    print(f"3D pose refinement error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 