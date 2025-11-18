# -*- coding: utf-8 -*-
"""
@Time: 2024/12/19 PM 1:30
@Auth: ***
@File: balanced_test_data.py
@IDE: PyCharm
@Motto: YES(Always Be Coding)
"""
import os
import random
import shutil

# Set paths
base_dir = '../data/rag_knowledge_base_1/'  # test_data/ directory
output_dir = '../data/balanced_rag_knowledge_base_1/'  # Output directory for balanced data
sample_num = 800
# Get all subfolders (object examples)
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Count the number of each object
objects = {}
for subfolder in subfolders:
    parts = subfolder.split('_')
    object_id = parts[1]  # Get object name (assuming format id_name_type_number)

    if object_id not in objects:
        objects[object_id] = []
    objects[object_id].append(subfolder)

# Calculate number of samples to keep for each object type
class_num = len(objects)  # Number of object types
samples_per_class = sample_num // class_num  # Number of samples to keep for each object type

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Count all samples
all_samples = []
for object_id, samples in objects.items():
    all_samples.extend([(object_id, sample) for sample in samples])

# Balance samples
final_samples = []
remaining_samples = []

# Iterate through object categories and randomly select samples
for object_id, samples in objects.items():
    # If this object has fewer samples than required, select all samples and add to remaining samples list
    if len(samples) < samples_per_class:
        selected_samples = samples  # Select all samples
        remaining_samples.extend([(object_id, sample) for sample in samples])
    else:
        # Randomly select samples_per_class samples
        selected_samples = random.sample(samples, samples_per_class)

    # Add selected samples to final sample list
    final_samples.extend([(object_id, sample) for sample in selected_samples])

# Ensure final sample count is 800
num_needed = sample_num - len(final_samples)

    # Supplement from remaining samples, ensuring supplemented samples come from other object categories
if num_needed > 0:
    # Shuffle remaining samples and randomly select from them
    random.shuffle(remaining_samples)
    additional_samples = remaining_samples[:num_needed]
    final_samples.extend(additional_samples)

# Copy samples to final directory
for object_id, sample in final_samples:
    sample_path = os.path.join(base_dir, sample)
    dest_path = os.path.join(output_dir, sample)  # Store samples directly in output directory

    # If target path already exists, skip this sample
    if os.path.exists(dest_path):
        print(f"Skipping {sample} because it already exists.")
        continue

    # Copy entire subfolder
    shutil.copytree(sample_path, dest_path)

print(f"Data balancing completed. {len(final_samples)} samples have been evenly distributed to {output_dir} directory.")
