import os
import sys
import random
import torch
import torchvision.transforms as transforms
from sentence_transformers import util
import numpy as np
import h5py


# Extract image feature maps (through VGG19 intermediate layers)
def extract_feature_map(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.squeeze(0)  # Return feature maps


# Calculate Instance Matching Distance (IMD)
def compute_instance_matching_distance(feature_map1, feature_map2):
    # Use Euclidean distance as IMD calculation method
    return torch.norm(feature_map1 - feature_map2, p=2).item()


# Sparse retrieval: extract object ID, action and object name from subdirectory names
def sparse_text_search(input_instruction, object_id, local_dir):
    # Decompose input instruction information
    input_action, input_object = input_instruction.split()  # Assume instruction format is "action object"
    input_object = input_object.lower()

    matching_dirs = []
    for subdir in os.listdir(local_dir):
        subdir_path = os.path.join(local_dir, subdir)
        if os.path.isdir(subdir_path):
            # Extract subdirectory information: object ID, object name, action
            # print(subdir)
            parts = subdir.split('_')
            sub_object_id, sub_object, sub_action = parts[0], parts[1].lower(), parts[3].lower()
            # print(sub_object_id, sub_object, sub_action)

            # Check if action and object type match, but object ID is not completely identical
            if sub_action == input_action and sub_object == input_object and sub_object_id != object_id:
                matching_dirs.append(subdir_path)
        # Randomly keep 50 results
        if len(matching_dirs) > 50:
            matching_dirs = random.sample(matching_dirs, 50)
    return matching_dirs


# Optimized dense retrieval: perform semantic similarity calculation based on unique action+object combinations
def dense_text_search(input_instruction, local_dir, text_model):
    unique_action_objects = {}  # Used to store unique action+object combinations
    subdir_mapping = {}  # Used to store subdirectory lists corresponding to action+object combinations

    # Extract all subdirectory information
    for subdir in os.listdir(local_dir):
        subdir_path = os.path.join(local_dir, subdir)
        if os.path.isdir(subdir_path):
            # Extract subdirectory information: object ID, object name, action
            parts = subdir.strip('/').split('_')
            if len(parts) < 3:
                continue  # Skip directories that don't conform to naming convention
            sub_object, sub_action = parts[1].lower(), parts[3].lower()

            # Build action+object combination
            action_object_key = f"{sub_action} {sub_object}"

            # If this combination is new, add to unique_action_objects
            if action_object_key not in unique_action_objects:
                unique_action_objects[action_object_key] = subdir_path

            # Add subdirectory path to the corresponding action+object combination list
            if action_object_key not in subdir_mapping:
                subdir_mapping[action_object_key] = []
            subdir_mapping[action_object_key].append(subdir_path)

    # Perform semantic embedding for unique action+object combinations
    embeddings = []
    action_object_keys = list(unique_action_objects.keys())
    for action_object_key in action_object_keys:
        embeddings.append(text_model.encode(action_object_key, convert_to_tensor=True))

    # Calculate semantic embedding of input instruction
    input_embedding = text_model.encode(input_instruction, convert_to_tensor=True)
    # Convert embedding list to 2D tensor
    embeddings = torch.stack(embeddings)
    # Calculate similarity between input instruction and all action+object combinations
    similarities = util.cos_sim(input_embedding, embeddings).squeeze(0)

    # Sort by similarity in descending order
    sorted_indices = similarities.argsort(descending=True)

    # Get subdirectories corresponding to the action+object combination with highest similarity
    most_similar_action_object = action_object_keys[sorted_indices[0]]
    matching_dirs = subdir_mapping[most_similar_action_object]
    
    # Return highest similarity score
    max_similarity = similarities[sorted_indices[0]].item()

    # print(f"Most similar action-object: {most_similar_action_object}")
    # print(f"Maximum similarity score: {max_similarity}")
    
    # Randomly keep 50 results
    if len(matching_dirs) > 50:
        matching_dirs = random.sample(matching_dirs, 50)
    return matching_dirs, max_similarity  # Return matching subdirectory list and highest similarity score


def get_global_position_from_camera(camera, depth, x, y):
    """
    This function is provided only to show how to convert camera observation to world space coordinates.
    It can be removed if not needed.

    camera: an camera agent
    depth: the depth obsrevation
    x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
    """
    cm = camera.get_metadata()
    proj, model = cm['projection_matrix'], cm['model_matrix']
    print('proj:', proj)
    print('model:', model)
    w, h = cm['width'], cm['height']

    # get 0 to 1 coordinate for (x, y) coordinates
    xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

    # get 0 to 1 depth value at (x,y)
    zf = depth[int(y), int(x)]

    # get the -1 to 1 (x,y,z) coordinate
    ndc = np.array([xf, yf, zf, 1]) * 2 - 1

    # transform from image space to view space
    v = np.linalg.inv(proj) @ ndc
    v /= v[3]

    # transform from view space to world space
    v = model @ v

    return v


def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()


def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x


def get_random_number(l, r):
    return np.random.rand() * (r - l) + l
