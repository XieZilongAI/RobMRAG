import os
import shutil

# Specify target directory
base_dir = "../data/balanced_rag_knowledge_base_1/"

# Traverse all subdirectories in target directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # Get list of files and directories in subdirectory
        items = os.listdir(subdir_path)

        # Count files in subdirectory (ignore subdirectories)
        file_count = sum(1 for item in items if os.path.isfile(os.path.join(subdir_path, item)))

        # Check if there are any images with 0 size
        has_empty_image = any(
            os.path.isfile(os.path.join(subdir_path, item)) and
            item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and
            os.path.getsize(os.path.join(subdir_path, item)) == 0
            for item in items
        )

        # If there are fewer than 3 files or any 0-size images, delete the subdirectory
        if file_count < 3 or has_empty_image:
            shutil.rmtree(subdir_path)
            print(f"Deleted: {subdir_path}")
        else:
            print(f"Kept: {subdir_path} (contains {file_count} files)")

def check_cat_balance(base_dir):
    data_list = os.listdir(base_dir)
    cat_cal = dict()
    for data_name in data_list:
        cat = data_name.split('_')[1]
        if cat not in list(cat_cal.keys()):
            cat_cal[cat] =  1
        else:
            cat_cal[cat] +=  1
    return cat_cal

cat_cal = check_cat_balance(base_dir)
print(cat_cal)