from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rag.utils import extract_feature_map, compute_instance_matching_distance
from PIL import Image
import os


def instance_matching_distance(input_image_path, rag_images_path, feature_extractor, device, top_k=5, tau_imd=95):
    input_image = Image.open(input_image_path).convert('RGB')
    input_feature_map = extract_feature_map(input_image, feature_extractor, device)
    # print("input_feature_map: ", input_feature_map.device)
    images = []
    rag_images_path_list = [os.path.join(rag_images_path, subdir) for subdir in os.listdir(rag_images_path) if
                       os.path.isdir(os.path.join(rag_images_path, subdir))]
    for dir_path in rag_images_path_list:
        image_path = os.path.join(dir_path, 'rag_example.png')  # Assume each subdirectory has a rag_example.png
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            images.append((img, image_path))
    imd_distances = []
    for i, (image, path) in enumerate(images):
        # print(i, image, path)
        feature_map = extract_feature_map(image, feature_extractor, device)
        distance = compute_instance_matching_distance(input_feature_map, feature_map)
        imd_distances.append((distance, path, image))

    imd_distances.sort(key=lambda t: t[0])
    top_k_imd_results = imd_distances[:top_k]

    
    # Get lowest IMD score
    min_imd_score = top_k_imd_results[0][0]
    
    # According to paper description: if IMD score is below threshold τ_IMD, use directly as final reference
    # Otherwise, use as geometric prior for subsequent pose refinement
    if min_imd_score < tau_imd:
        print(f"IMD score ({min_imd_score:.2f}) below threshold ({tau_imd}). Using directly as final reference.")
        dir_path = os.path.dirname(top_k_imd_results[0][1])
        return dir_path, True  # Return path and flag for direct use
    else:
        print(f"IMD score ({min_imd_score:.2f}) above threshold ({tau_imd}). Using as geometric prior for pose refinement.")
        dir_path = os.path.dirname(top_k_imd_results[0][1])
        return dir_path, False  # Return path and flag for further refinement needed


def imd_filtering(input_image, images, feature_extractor, device, top_k=5):
    """
    Filter top k images based on IMD geometric matching.
    
    Args:
        input_image: Input image for comparison
        images: List of candidate images
        feature_extractor: Feature extraction model
        device: Device for computation
        top_k: Number of top images to return
        
    Returns:
        tuple: (top_k_results, min_imd_score)
    """
    # print("Performing IMD filtering...")
    input_feature_map = extract_feature_map(input_image, feature_extractor, device)
    # print("input_feature_map: ", input_feature_map.device)
    imd_distances = []
    for i, (image, path) in enumerate(images):
        # print(i, image, path)
        feature_map = extract_feature_map(image, feature_extractor, device)
        distance = compute_instance_matching_distance(input_feature_map, feature_map)
        imd_distances.append((distance, path, image))

    imd_distances.sort(key=lambda t: t[0])
    top_k_imd_results = imd_distances[:top_k]

    
    # Return results and minimum IMD value
    min_imd_score = top_k_imd_results[0][0]
    return top_k_imd_results, min_imd_score


# 2. Filter based on Cosine Similarity
def cosine_similarity_filtering(input_image, top_k_imd_results, image_model, device, top_n=2):
    """
    Filter top n images based on Cosine Similarity.
    
    Args:
        input_image: Input image for comparison
        top_k_imd_results: Results from IMD filtering
        image_model: Image encoding model
        device: Device for computation
        top_n: Number of top images to return
        
    Returns:
        list: Top n results sorted by cosine similarity
    """
    print("Performing cosine similarity filtering...")
    image_data = image_model.preprocess_image(input_image).to(device)
    # print("image_data: ", image_data.device)
    search_image_embedding = image_model.encode_image(image_data).squeeze(0).to(device)
    cosine_similarities = []
    for _, path, image in top_k_imd_results:
        image_data = image_model.preprocess_image(image).to(device)
        image_embedding = image_model.encode_image(image_data).squeeze(0).to(device)
        sim = F.cosine_similarity(search_image_embedding, image_embedding, dim=0).item()
        cosine_similarities.append((sim, path, image))

    cosine_similarities.sort(reverse=True, key=lambda t: t[0])
    top_n_results = cosine_similarities[:top_n]

    return top_n_results


def image_to_image_search(input_image, images, image_model, feature_extractor, device):
    """
    Integrate IMD and Cosine Similarity filtering, return path and IMD value of most similar image.
    
    Combines Instance Matching Distance (IMD) and cosine similarity metrics to find
    the most visually similar image from a candidate set.
    
    Args:
        input_image: Input image for comparison
        images: List of candidate images with their paths
        image_model: Pre-trained image model for feature extraction
        feature_extractor: Feature extraction model (e.g., VGG19)
        device: Device for model inference ('cuda' or 'cpu')
        
    Returns:
        tuple: (best_match_path, min_imd_score) - path to best match and its IMD score
    """
    # IMD filtering
    top_k_results, min_imd_score = imd_filtering(input_image, images, feature_extractor, device, top_k=5)

    # Cosine Similarity filtering
    # top_k_results = cosine_similarity_filtering(input_image, top_k_results, image_model, device,
    #                                             top_n=2)
    return top_k_results[0][1], min_imd_score
