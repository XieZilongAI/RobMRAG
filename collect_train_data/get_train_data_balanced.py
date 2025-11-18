import json
import os
import re
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import argparse
from tqdm import tqdm
import torch
import random
from random import randrange
from rag.multi_rag_main import multimodal_rag
import uform
from sentence_transformers import SentenceTransformer
from torchvision.models import vgg19
from collections import defaultdict, Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print('Start generating balanced training json..............')
count = 0
parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', type=str, help='dataset dir')
parser.add_argument('--rag_knowledge_base_dir', type=str,
                    default='../data_collection/data/balanced_rag_knowledge_base_1',
                    help='rag knowledge base dir')
parser.add_argument('--output_dir', type=str, help='training json dir')
parser.add_argument('--num_point', type=int, help='training json dir')
parser.add_argument('--mlm', default='True', type=str, help='if use mask language model')
parser.add_argument('--bins', default='True', type=str, help='if use bin in orientation')
parser.add_argument('--aff_prior', default='True', type=str, help='if learn from affordance')
parser.add_argument('--target_samples_per_object', type=int, default=500, 
                    help='target number of samples per object')
parser.add_argument('--balancing_strategy', type=str, default='target_based',
                    choices=['target_based', 'dynamic_sampling', 'stratified'],
                    help='sample balancing strategy')
parser.add_argument('--max_total_samples', type=int, default=20000,
                    help='maximum total number of samples')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else "cpu"
# Load text encoding model
text_model_path = 'all-MiniLM-L6-v2'
text_model = SentenceTransformer(text_model_path).eval().to(device)

# Load image encoding model
image_model = uform.get_model('unum-cloud/uform-vl-english').eval().to(device)
# Load VGG19 feature extractor
vgg19_model = vgg19(pretrained=True).features.eval().to(device)
rag_knowledge_base_dir = args.rag_knowledge_base_dir
folder_dir = args.folder_dir
folder_names = os.listdir(folder_dir)
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Variables related to sample balancing
object_sample_counts = defaultdict(int)  # Record current sample count for each object
target_samples_per_object = args.target_samples_per_object
max_total_samples = args.max_total_samples
balancing_strategy = args.balancing_strategy

def create_answer_mask(answer):
    # Parse contact point and gripper direction using regex
    contact_match = re.search(r'The contact point is \(([^)]+)\)', answer)
    up_match = re.search(r'the gripper up direction is \[([^\]]+)\]', answer)
    forward_match = re.search(r'the gripper forward direction is \[([^\]]+)\]', answer)

    # Convert values to lists
    contact_point = contact_match.group(1).split(', ') if contact_match else []
    up_direction = up_match.group(1).split(', ') if up_match else []
    forward_direction = forward_match.group(1).split(', ') if forward_match else []

    # Define masking method
    def mask_random_value(values):
        idx = random.randint(0, len(values) - 1)
        values[idx] = '<mask>'
        return values

    # Randomly select two directions
    directions = ['contact', 'up', 'forward']
    selected_directions = random.sample(directions, 3)

    # Mask values based on selection
    if 'contact' in selected_directions and contact_point:
        contact_point = mask_random_value(contact_point)
    if 'up' in selected_directions and up_direction:
        up_direction = mask_random_value(up_direction)
    if 'forward' in selected_directions and forward_direction:
        forward_direction = mask_random_value(forward_direction)

    # Recombine masked content
    masked_answer = (
        f"The contact point is ({', '.join(contact_point)}), "
        f"the gripper up direction is [{', '.join(up_direction)}], "
        f"the gripper forward direction is [{', '.join(forward_direction)}]."
    )

    return masked_answer

def create_token_answer_mask(answer):
    # Optimized regular expression (key fix)
    contact_match = re.search(r'The contact point is \(([^)]+)\)', answer)
    up_match = re.search(r'the gripper up direction is \[([^\]]+)\]', answer)
    forward_match = re.search(r'the gripper forward direction is \[([^\]]+)\]', answer)

    # Improved numerical parsing (handling negative signs and spaces)
    def parse_values(value_str):
        if not value_str: return []
        # First split by comma for value blocks, then by space for individual values
        return [s.strip().replace(' ', '').split() for s in value_str.split(',')]

    contact_values = parse_values(contact_match.group(1)) if contact_match else []
    up_values = parse_values(up_match.group(1)) if up_match else []
    forward_values = parse_values(forward_match.group(1)) if forward_match else []

    # Ensure at least two directions have values that can be masked
    directions = []
    if len(contact_values) >= 1: directions.append('contact')
    if len(up_values) >= 1: directions.append('up')
    if len(forward_values) >= 1: directions.append('forward')

    if len(directions) < 2:
        raise ValueError("At least two directions need values for masking")

    # Randomly select two directions for masking
    selected_directions = random.sample(directions, 2)

    # Masking operation (mask one value for each direction)
    def mask_direction(values):
        if not values: return
        idx = random.randint(0, len(values) - 1)
        if len(values[idx]) == 0: return
        part_idx = random.randint(0, len(values[idx]) - 1)
        values[idx][part_idx] = '<mask>'

    for direction in selected_directions:
        if direction == 'contact':
            mask_direction(contact_values)
        elif direction == 'up':
            mask_direction(up_values)
        elif direction == 'forward':
            mask_direction(forward_values)

    # Reorganize format (handle empty lists)
    def format_section(values):
        if not values: return ""
        return ', '.join([' '.join(parts) for parts in values])

    masked_answer = (
        f"Contact point: ({format_section(contact_values)}); "
        f"Up direction: [{format_section(up_values)}]; "
        f"Forward direction: [{format_section(forward_values)}]."
    )
    return masked_answer

def tokenize_scaled(num, scale=100):
    scaled = int(round(num * scale))
    return ' '.join(list(str(scaled)))

def should_sample_item(cat, current_total_samples):
    """
    Decide whether to sample the current item based on balancing strategy
    """
    if current_total_samples >= max_total_samples:
        return False
    
    if balancing_strategy == 'target_based':
        # Target-based balancing: prioritize collecting samples for objects with insufficient samples
        return object_sample_counts[cat] < target_samples_per_object
    
    elif balancing_strategy == 'dynamic_sampling':
        # Dynamic sampling rate adjustment: dynamically adjust sampling probability based on current sample count
        if object_sample_counts[cat] >= target_samples_per_object:
            return False
        
        # Calculate dynamic sampling probability
        current_count = object_sample_counts[cat]
        if current_count == 0:
            sampling_prob = 1.0  # Objects with no samples yet, 100% sampling
        else:
            # The fewer samples, the higher the sampling probability
            sampling_prob = min(1.0, target_samples_per_object / (current_count * 2))
        
        return random.random() < sampling_prob
    
    elif balancing_strategy == 'stratified':
        # Stratified sampling: ensure each object type has a chance to be sampled
        # If an object type has few samples, increase its sampling probability
        if object_sample_counts[cat] >= target_samples_per_object:
            return False
        
        # Calculate average sample count for all objects
        if len(object_sample_counts) == 0:
            return True
        
        avg_samples = sum(object_sample_counts.values()) / len(object_sample_counts)
        current_count = object_sample_counts[cat]
        
        # If current object sample count is below average, increase sampling probability
        if current_count < avg_samples * 0.5:
            return random.random() < 0.8
        elif current_count < avg_samples:
            return random.random() < 0.6
        else:
            return random.random() < 0.3
    
    return True

def get_object_info_from_folder_name(folder_name):
    """
    Extract object information from folder name
    """
    parts = folder_name.split('_')
    if len(parts) >= 4:
        object_id = parts[0]
        cat = parts[1]
        action = parts[3]
        return object_id, cat, action
    return None, None, None

# Preprocessing: analyze all available object types
print("Analyzing available object types...")
available_objects = set()
for item in folder_names:
    _, cat, _ = get_object_info_from_folder_name(item)
    if cat:
        available_objects.add(cat)

print(f"Found {len(available_objects)} object types: {sorted(available_objects)}")

cal_cat = dict()
data_list = []

# Randomly shuffle folder order to ensure fairness
random.shuffle(folder_names)

for item in tqdm(folder_names, desc="Processing training data"):
    object_id, cat, action = get_object_info_from_folder_name(item)
    
    if not cat:
        continue
    
    # Check if this item should be sampled
    if not should_sample_item(cat, len(data_list)):
        continue
    
    NUM_OF_POINTS = args.num_point
    cur_dir = os.path.join(folder_dir, str(item))

    rgb_path = os.path.join(cur_dir, 'original_rgb.png')
    if not os.path.exists(rgb_path):
        continue
    
    input_instruction = action + ' ' + cat

    rag_path = multimodal_rag(input_instruction, object_id, rgb_path, rag_knowledge_base_dir, text_model, image_model,
                              vgg19_model, device)
    
    # Check if RAG path exists and contains required files
    if not rag_path or not os.path.exists(rag_path):
        continue
        
    rag_example_path = os.path.join(rag_path, 'rag_example.png')
    rag_result_path = os.path.join(rag_path, 'rag_result.png')
    
    # Verify that essential files exist
    if not os.path.exists(rag_example_path):
        continue
        
    if not os.path.exists(os.path.join(rag_path, 'result.json')):
        continue
        
    # If rag_result.png doesn't exist, use rag_example.png as fallback
    if not os.path.exists(rag_result_path):
        rag_result_path = rag_example_path
        
    with open(os.path.join(rag_path, 'result.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)
    numbers = data['pixel_locs']
    try:
        point = tuple(tokenize_scaled(round(x / 336, 2)) for x in numbers)
    except Exception as e:
        continue
    forward_num = data['gripper_forward_direction_camera']
    rag_forward = [tokenize_scaled(round(x, 1), 10) for x in forward_num]
    up_num = data['gripper_up_direction_camera']
    rag_up = [tokenize_scaled(round(x, 1), 10) for x in up_num]
    
    if os.path.exists(os.path.join(cur_dir, 'result.json')):
        with open(os.path.join(cur_dir, 'result.json'), 'r') as fin:
            data_inf = json.load(fin)
            if data_inf['mani_succ'] != 'True':
                continue

            aff_gt_dir = os.path.join(cur_dir, 'aff_gt_all.png')
            if not os.path.exists(aff_gt_dir):
                continue
            img_pil = Image.open(rgb_path)
            intermask_pil = np.array(Image.open(os.path.join(cur_dir, 'interaction_mask.png')))
            gray_image = ImageOps.grayscale(img_pil)
            threshold = 200
            object_mask = gray_image.point(lambda p: p < threshold and 255)
            object_mask.save(os.path.join(cur_dir, 'object_mask.png'))

            object_mask = np.array(object_mask) / 255

            aff_gt_pil = Image.open(aff_gt_dir)
            aff_gt = np.array(aff_gt_pil) / 255
            result_mask = np.where(aff_gt < 0.2, intermask_pil, 0).astype(np.uint8)
            object_mask = np.where(aff_gt < 0.2, object_mask, 0).astype(np.uint8)
            Image.fromarray((result_mask).astype(np.uint8)).save(os.path.join(cur_dir, 'result_mask.png'))
            Image.fromarray((object_mask * 255).astype(np.uint8)).save(os.path.join(cur_dir, 'object_mask.png'))

            row_indices_pos, col_indices_pos = np.where(aff_gt > 0.8)
            if NUM_OF_POINTS > len(row_indices_pos):
                NUM_OF_POINTS = len(row_indices_pos)

            row_indices_neg1, col_indices_neg1 = np.where(result_mask > 0.8)

            if NUM_OF_POINTS > len(row_indices_neg1) and len(row_indices_neg1) != 0:
                NUM_OF_POINTS = len(row_indices_neg1)

            if NUM_OF_POINTS == 0:
                continue

            if len(row_indices_neg1) != 0:
                indices_neg = np.random.choice(len(row_indices_neg1), size=NUM_OF_POINTS // 2, replace=False)
                selected_row_indices_neg = row_indices_neg1[indices_neg].reshape(-1, 1)
                selected_col_indices_neg = col_indices_neg1[indices_neg].reshape(-1, 1)
                top_indices_neg1 = np.hstack((selected_row_indices_neg, selected_col_indices_neg))
                top_indices_neg1_gt = np.zeros(top_indices_neg1.shape[0])

            row_indices_neg, col_indices_neg = np.where(object_mask > 0.8)

            if len(row_indices_neg) != 0 and len(row_indices_neg1) != 0:
                indices_neg = np.random.choice(len(row_indices_neg), size=NUM_OF_POINTS // 2, replace=False)
                selected_row_indices_neg = row_indices_neg[indices_neg].reshape(-1, 1)
                selected_col_indices_neg = col_indices_neg[indices_neg].reshape(-1, 1)
                top_indices_neg2 = np.hstack((selected_row_indices_neg, selected_col_indices_neg))
                top_indices_neg2_gt = np.zeros(top_indices_neg2.shape[0])
            else:
                try:
                    indices_neg = np.random.choice(len(row_indices_neg), size=NUM_OF_POINTS, replace=False)
                    selected_row_indices_neg = row_indices_neg[indices_neg].reshape(-1, 1)
                    selected_col_indices_neg = col_indices_neg[indices_neg].reshape(-1, 1)
                    top_indices_neg2 = np.hstack((selected_row_indices_neg, selected_col_indices_neg))
                    top_indices_neg2_gt = np.zeros(top_indices_neg2.shape[0])
                except:
                    continue

            indices_pos = np.random.choice(len(row_indices_pos), size=NUM_OF_POINTS, replace=False)
            selected_row_indices_pos = row_indices_pos[indices_pos].reshape(-1, 1)
            selected_col_indices_pos = col_indices_pos[indices_pos].reshape(-1, 1)
            top_indices_pos = np.hstack((selected_row_indices_pos, selected_col_indices_pos))
            top_indices_pos_gt = np.ones(top_indices_pos.shape[0])

            if len(row_indices_neg1) == 0:
                select_indices = np.vstack((top_indices_neg2, top_indices_pos))
                select_indices_gt = np.concatenate((top_indices_neg2_gt, top_indices_pos_gt))
            else:
                select_indices = np.vstack((top_indices_neg1, top_indices_neg2, top_indices_pos))
                select_indices_gt = np.concatenate((top_indices_neg1_gt, top_indices_neg2_gt, top_indices_pos_gt))

            permutation = np.random.permutation(len(select_indices_gt))
            select_indices = select_indices[permutation] / 336
            select_indices_gt = select_indices_gt[permutation]

            mapping = {0: "no", 1: "yes"}
            if len(select_indices_gt) == 0:
                continue
            select_string_gt = np.vectorize(mapping.get)(select_indices_gt)

            select_token_string = np.array2string(select_indices, separator=', ', formatter={
                'all': lambda x: ' '.join(str(int(round(x * 100))))
            })[1:-1].strip().replace("\n", " ")
            select_string_gt = np.array2string(select_string_gt, separator=',',
                                               formatter={'all': lambda x: str(x)})[
                               1:-1].strip().replace("\n", " ")
            
            aff_question = 'Determine if operating on each following point can effectively manipulate the object within the image: {}'.format(
                select_token_string)

            aff_gt = select_string_gt

            # draw the selected point in the image
            draw = ImageDraw.Draw(img_pil)
            if len(row_indices_neg1) != 0:
                for index in range(top_indices_neg1.shape[0]):
                    draw.point((top_indices_neg1[index][1], top_indices_neg1[index][0]), 'blue')
            for index in range(top_indices_neg2.shape[0]):
                draw.point((top_indices_neg2[index][1], top_indices_neg2[index][0]), 'blue')
            for index in range(top_indices_pos.shape[0]):
                draw.point((top_indices_pos[index][1], top_indices_pos[index][0]), 'red')
            img_pil.save(os.path.join(cur_dir, 'select_point.png'))

            up_cam = [tokenize_scaled(round(x, 1), 10) for x in data_inf['gripper_up_direction_camera']]
            forward_cam = [tokenize_scaled(round(x, 1), 10) for x in data_inf['gripper_forward_direction_camera']]
            x, y = data_inf['pixel_locs']
            x_str, y_str = tokenize_scaled(round(float(x) / 336, 2)), tokenize_scaled(round(float(y) / 336, 2))
            data_item = {
                "messages": [
                    {
                     "content": f'''The current robot is a Franka Panda Robot equipped with a Flying Suction Gripper as its actuator. The task is: {action} the {cat}. <image>The first image shows what you are currently observing. <image>The second image is a reference, where the red dot marks the grasping point. This reference image is similar to your current observation, with the grasp point at ({point[0]},{point[1]}), the Gripper Up Direction set to [{','.join(rag_up)}], and Gripper Forward Direction set to [{','.join(rag_forward)}]. If a third image is present, <image>it displays the final outcome based on the second image.
Your responsibility is to predict precise grasp points and grasp directions of observed image in the camera coordinate system according to the task requirements. Specifically, you need to provide:
1. **Grasp Point**: Two integer values formatted as `(x, y)`, where each value is a sequence of numbers from 0 to 100 separated by spaces (e.g., `3 2` represents 32).
2. **Grasp Direction**:
  - **Gripper Up Direction**: Three integer components formatted as `[x, y, z]`, each component being a number from -10 to 10 separated by spaces (e.g., `- 4` represents -4).
  - **Gripper Forward Direction**: Three integer components formatted as `[x, y, z]`, following the same rule.''',
                        "role": "user"
                    },
                    {
                        "content": f"The contact point is ({x_str},{y_str}), the gripper up direction is [{','.join(up_cam)}], the gripper forward direction is [{','.join(forward_cam)}].",
                        "role": "assistant"
                    }

                ],
                "images": [
                    rgb_path,
                    rag_example_path,
                    rag_result_path
                ],
                'cat_prompt': 'What is the category of the object in the image?',
                'cat_ans': item.split('_')[1],
                "instruction": "Specify the contact point and gripper direction of manipulating the object.",
                "input": os.path.join(cur_dir, 'original_rgb.png'),
                'aff_question': aff_question,
                'aff_gt': aff_gt.strip()

            }
            answer = data_item['messages'][1]['content']
            
            if args.bins == 'True' and args.mlm == 'True' and args.aff_prior:
                i = random.randint(0, 3)
                if i % 4 == 0:
                    pass
                elif i % 4 == 1:
                    answer_mask = create_token_answer_mask(answer)
                    data_item['messages'][0]['content'] = 'Predict the masked value: ' + answer_mask + '\n' + \
                                                          data_item['messages'][0]['content']
                elif i % 4 == 2:
                    pass
                elif i % 4 == 3:
                    pass
            
            data = {key: data_item[key] for key in ['messages', 'images']}
            data_list.append(data)
            
            # Update object sample count
            object_sample_counts[cat] += 1
            
            if cat not in list(cal_cat.keys()):
                cal_cat[cat] = 1
            else:
                cal_cat[cat] += 1

# Output balanced statistics
print("\n" + "="*60)
print("Sample Balancing Results Statistics")
print("="*60)
print(f"Total samples: {len(data_list)}")
print(f"Balancing strategy used: {balancing_strategy}")
print(f"Target samples per object: {target_samples_per_object}")
print(f"Maximum total samples: {max_total_samples}")

print("\nSample count per object:")
print("-"*40)
sorted_objects = sorted(object_sample_counts.items(), key=lambda x: x[1], reverse=True)
for obj, count in sorted_objects:
    print(f"{obj:<20} : {count:>6} samples")

# Calculate balancing metrics
if len(object_sample_counts) > 0:
    counts = list(object_sample_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    std_count = np.std(counts)
    
    print(f"\nBalancing metrics:")
    print(f"Minimum samples: {min_count}")
    print(f"Maximum samples: {max_count}")
    print(f"Average samples: {avg_count:.2f}")
    print(f"Standard deviation: {std_count:.2f}")
    print(f"Coefficient of variation: {std_count/avg_count:.3f}")

# Save results
output_filename = f'train_balanced_{balancing_strategy}_{len(data_list)}_samples.json'
with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f'\nBalanced training data saved to: {os.path.join(output_dir, output_filename)}')
print('Finish generating balanced training json..............')
