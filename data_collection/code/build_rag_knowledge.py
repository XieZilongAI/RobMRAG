# -*- coding: utf-8 -*-
"""
RAG Knowledge Base Construction Script
Integrates data extraction, category-balanced sampling, and quality checking
Merges from train_data_ori and test_data_ori source directories, extracting N samples per category
"""
import os
import json
import shutil
import random
import argparse


def process_single_sample(src_dir, dst_dir):
    """
    Process a single sample directory:
    - Copy and rename contact_point.png -> rag_example.png
    - Copy and rename viz_pull_pose.png -> rag_result.png
    - Process result.json, keep specified fields
    """
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

    # Process result.json
    src_json = os.path.join(src_dir, "result.json")
    dst_json = os.path.join(dst_dir, "result.json")
    if os.path.exists(src_json):
        with open(src_json, "r") as f:
            data = json.load(f)
        
        # Keep original data (Note: original code writes data instead of filtered_data at line 48)
        with open(dst_json, "w") as f:
            json.dump(data, f, indent=4)


def collect_samples_by_category(source_dir, source_name=""):
    """
    Collect samples by category from source directory
    Returns: {category: [(sample_dir_name, source_dir), ...]}
    """
    if not os.path.exists(source_dir):
        print(f"  Warning: Source directory does not exist {source_dir}")
        return {}
    
    category_samples = {}
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            # Assume directory format is id_category_type_number
            parts = subdir.split('_')
            if len(parts) >= 2:
                category = parts[1]
                if category not in category_samples:
                    category_samples[category] = []
                # Save sample name and source directory
                category_samples[category].append((subdir, source_dir))
    
    return category_samples


def merge_category_samples(train_samples, test_samples):
    """
    Merge category samples from train and test
    Returns: {category: [(sample_dir_name, source_dir), ...]}
    """
    merged_samples = {}
    
    # Merge samples from train
    for category, samples in train_samples.items():
        if category not in merged_samples:
            merged_samples[category] = []
        merged_samples[category].extend(samples)
    
    # Merge samples from test
    for category, samples in test_samples.items():
        if category not in merged_samples:
            merged_samples[category] = []
        merged_samples[category].extend(samples)
    
    return merged_samples


def sample_and_process(merged_category_samples, target_dir, samples_per_class):
    """
    Sample and process samples for each category
    merged_category_samples: {category: [(sample_name, source_dir), ...]}
    """
    processed_count = 0
    
    for category, samples in merged_category_samples.items():
        # If sample count is less than required, keep all; otherwise randomly sample
        if len(samples) <= samples_per_class:
            selected_samples = samples
            print(f"  Category {category}: Total {len(samples)} samples (train+test), keeping all")
        else:
            selected_samples = random.sample(samples, samples_per_class)
            print(f"  Category {category}: Randomly sampling {samples_per_class} from {len(samples)} samples (train+test)")
        
        # Process selected samples
        for sample_name, source_dir in selected_samples:
            src_sample_path = os.path.join(source_dir, sample_name)
            dst_sample_path = os.path.join(target_dir, sample_name)
            
            if os.path.exists(dst_sample_path):
                print(f"    Skipping {sample_name} (already exists)")
                continue
            
            process_single_sample(src_sample_path, dst_sample_path)
            processed_count += 1
    
    return processed_count


def check_and_clean(target_dir):
    """
    Check and clean target directory:
    - Delete directories with file count < 3
    - Delete directories containing 0-byte images
    """
    deleted_count = 0
    kept_count = 0
    
    for subdir in os.listdir(target_dir):
        subdir_path = os.path.join(target_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
        
        items = os.listdir(subdir_path)
        file_count = sum(1 for item in items if os.path.isfile(os.path.join(subdir_path, item)))
        
        # Check if there are 0-byte images
        has_empty_image = any(
            os.path.isfile(os.path.join(subdir_path, item)) and
            item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and
            os.path.getsize(os.path.join(subdir_path, item)) == 0
            for item in items
        )
        
        # Delete directory if file count < 3 or has 0-byte images
        if file_count < 3 or has_empty_image:
            shutil.rmtree(subdir_path)
            print(f"  Deleted: {subdir} (file_count={file_count}, has_empty_image={has_empty_image})")
            deleted_count += 1
        else:
            kept_count += 1
    
    return kept_count, deleted_count


def check_category_balance(target_dir):
    """
    Count sample numbers for each category in the final knowledge base
    """
    if not os.path.exists(target_dir):
        return {}
    
    category_count = {}
    for data_name in os.listdir(target_dir):
        data_path = os.path.join(target_dir, data_name)
        if os.path.isdir(data_path):
            parts = data_name.split('_')
            if len(parts) >= 2:
                category = parts[1]
                category_count[category] = category_count.get(category, 0) + 1
    
    return category_count


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='RAG Knowledge Base Construction Tool - Extract samples from train and test data to build knowledge base'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='../data/train_data_ori',
        help='Training data source directory (default: ../data/train_data_ori)'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../data/test_data_ori',
        help='Test data source directory (default: ../data/test_data_ori)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/rag_knowledge_base',
        help='Output directory (default: ../data/rag_knowledge_base)'
    )
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=50,
        help='Number of samples per category (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Use parameters
    train_source_dir = args.train_dir
    test_source_dir = args.test_dir
    target_dir = args.output_dir
    samples_per_class = args.samples_per_class
    
    print("=" * 60)
    print("RAG Knowledge Base Construction Tool")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Train data source: {train_source_dir}")
    print(f"  Test data source: {test_source_dir}")
    print(f"  Output directory: {target_dir}")
    print(f"  Samples per class: {samples_per_class}")
    print()
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Step 1: Collect samples from training data
    print("Step 1: Collecting training data...")
    print(f"  Source directory: {train_source_dir}")
    train_category_samples = collect_samples_by_category(train_source_dir, "train")
    train_categories = len(train_category_samples)
    train_total_samples = sum(len(samples) for samples in train_category_samples.values())
    print(f"  Found {train_categories} categories, {train_total_samples} samples in total")
    print()
    
    # Step 2: Collect samples from test data
    print("Step 2: Collecting test data...")
    print(f"  Source directory: {test_source_dir}")
    test_category_samples = collect_samples_by_category(test_source_dir, "test")
    test_categories = len(test_category_samples)
    test_total_samples = sum(len(samples) for samples in test_category_samples.values())
    print(f"  Found {test_categories} categories, {test_total_samples} samples in total")
    print()
    
    # Step 3: Merge samples from both sources
    print("Step 3: Merging train and test samples...")
    merged_samples = merge_category_samples(train_category_samples, test_category_samples)
    
    # Count overlap
    train_only = set(train_category_samples.keys()) - set(test_category_samples.keys())
    test_only = set(test_category_samples.keys()) - set(train_category_samples.keys())
    overlap = set(train_category_samples.keys()) & set(test_category_samples.keys())
    
    print(f"  Total categories: {len(merged_samples)}")
    print(f"  Train only: {len(train_only)} categories")
    print(f"  Test only: {len(test_only)} categories")
    print(f"  Train and test overlap: {len(overlap)} categories")
    print()
    
    # Step 4: Sample and process merged samples
    print("Step 4: Sampling from merged samples...")
    processed_count = sample_and_process(merged_samples, target_dir, samples_per_class)
    print(f"  Processing completed: {processed_count} samples")
    print()
    
    # Step 5: Quality check and cleanup
    print("Step 5: Quality check and cleanup...")
    kept_count, deleted_count = check_and_clean(target_dir)
    print(f"  Kept: {kept_count} samples")
    print(f"  Deleted: {deleted_count} samples")
    print()
    
    # Step 6: Final statistics
    print("Step 6: Final statistics...")
    category_balance = check_category_balance(target_dir)
    total_samples = sum(category_balance.values())
    
    print(f"  Total categories: {len(category_balance)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Target directory: {target_dir}")
    print()
    print("  Sample count by category:")
    for category in sorted(category_balance.keys()):
        print(f"    {category}: {category_balance[category]}")
    
    print()
    print("=" * 60)
    print("RAG Knowledge Base Construction Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

