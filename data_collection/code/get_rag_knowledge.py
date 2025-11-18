import os
import json
import shutil

# Define source and target directories
source_root = "../data/test_data_ori"
target_root = "../data/rag_knowledge_base_1"

# Ensure target directory exists
os.makedirs(target_root, exist_ok=True)

def process_directory(src_dir, dst_dir):
    """
    Process single directory:
- Keep contact_point.png and viz_pull_pose.png, rename to rag_example.png and rag_result.png
- Process result.json file, keep specified fields and rename field names
    """
    # Ensure target subdirectory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Process contact_point.png -> rag_example.png
    src_contact_point = os.path.join(src_dir, "contact_point.png")
    dst_contact_point = os.path.join(dst_dir, "rag_example.png")
    if os.path.exists(src_contact_point):
        shutil.copy(src_contact_point, dst_contact_point)

    # Process viz_pull_pose.png -> rag_result.png
    src_viz_pull_pose = os.path.join(src_dir, "viz_pull_pose.png")
    dst_viz_pull_pose = os.path.join(dst_dir, "rag_result.png")
    if os.path.exists(src_viz_pull_pose):
        shutil.copy(src_viz_pull_pose, dst_viz_pull_pose)

    # Process result.json file
    src_json = os.path.join(src_dir, "result.json")
    dst_json = os.path.join(dst_dir, "result.json")
    if os.path.exists(src_json):
        with open(src_json, "r") as f:
            data = json.load(f)

        # Extract specified fields and rename
        filtered_data = {
            "rag_pixel_locs": data.get("pixel_locs"),
            "rag_gripper_forward_direction_camera": data.get("gripper_forward_direction_camera"),
            "rag_gripper_up_direction_camera": data.get("gripper_up_direction_camera")
        }
        # Write new JSON file
        with open(dst_json, "w") as f:
            json.dump(data, f, indent=4)

# Traverse all subdirectories in source directory
for subdir in os.listdir(source_root):
    source_subdir = os.path.join(source_root, subdir)
    target_subdir = os.path.join(target_root, subdir)

    if os.path.isdir(source_subdir):
        process_directory(source_subdir, target_subdir)

print("Processing completed successfully!")