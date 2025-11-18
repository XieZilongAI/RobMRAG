import os
import shutil

from PIL import Image
import numpy as np
from rag.image_search import image_to_image_search
from rag.utils import sparse_text_search, dense_text_search


def multimodal_rag(input_instruction, object_id, input_image_path, local_dir, text_model, image_model,
                   feature_extractor, device, tau_den=0.7, tau_imd=135):
    """
    Multimodal Retrieval-Augmented Generation (RAG) system for task planning.
    
    Performs a multi-step retrieval process using sparse text search, dense text search,
    and image-to-image matching to find relevant knowledge for manipulation tasks.
    
    Args:
        input_instruction (str): Natural language instruction for the manipulation task
        object_id (str): Unique identifier for the target object
        input_image_path (str): Path to the input image for visual matching
        local_dir (str): Directory containing the knowledge base
        text_model: Pre-trained text embedding model for dense search
        image_model: Pre-trained image model for visual similarity
        feature_extractor: Feature extraction component for image processing
        device (str): Device for model inference ('cuda' or 'cpu')
        tau_den (float): Threshold for dense text search similarity (default: 0.7)
        tau_imd (int): Threshold for image-to-image matching (default: 135)
    
    Returns:
        tuple: (rag_instruction, contact_file_path) or (None, None) if no match found
    """
    # Load input image
    input_image = Image.open(input_image_path).convert('RGB')

    # Step 1: Sparse retrieval
    # print("Performing sparse text search...")
    matching_dirs = sparse_text_search(input_instruction, object_id, local_dir)
    # print('matching_dirs: ', len(matching_dirs))
    
    # Step 2: If sparse retrieval returns no results, perform dense retrieval
    if not matching_dirs:
        print("No results from sparse text search. Performing dense text search...")
        matching_dirs, max_similarity = dense_text_search(input_instruction, local_dir, text_model)
        
        # Step 3: If dense retrieval's highest similarity is below threshold, switch knowledge base for sparse retrieval
        if max_similarity < tau_den:
            print(f"Dense search max similarity ({max_similarity:.3f}) below threshold ({tau_den}). Switching to alternative knowledge base...")
            # Switch to balanced_rag_knowledge_base_2/
            alternative_local_dir = local_dir.replace('balanced_rag_knowledge_base_1', 'balanced_rag_knowledge_base_2')
            if os.path.exists(alternative_local_dir):
                matching_dirs = sparse_text_search(input_instruction, object_id, alternative_local_dir)
            else:
                print(f"Alternative knowledge base not found: {alternative_local_dir}")

    if not matching_dirs:
        print("No relevant directories found. Exiting...")
        return None

    # Step 3: Load images from matching subdirectories
    images = []
    for dir_path in matching_dirs:
        image_path = os.path.join(dir_path, 'rag_example.png')
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            images.append((img, image_path))

    if not images:
        print("No images found in matching directories. Exiting...")
        return None
    # Step 4: Find most relevant examples based on IMD and cosine similarity
    # print("Performing image-to-image search with IMD and cosine similarity...")
    most_similar_dir, min_imd_score = image_to_image_search(input_image, images, image_model, feature_extractor,device)
    directory = os.path.dirname(most_similar_dir)
    
    # Step 5: IMD threshold judgment and 3D pose refinement
    if min_imd_score >= tau_imd:
        print(f"IMD score ({min_imd_score:.2f}) above threshold ({tau_imd}). Performing 3D pose refinement...")
        
        try:
            import subprocess
            import sys
            import time
            import uuid
            
            start_time = time.time()
            
            # Create temporary directory for 3D pose refinement
            temp_rag_path = '../data_collection/data/temp/'
            
            # Create permanent storage directory for refined results
            permanent_rag_path = '../data_collection/data/refined_rag_results/'
            os.makedirs(permanent_rag_path, exist_ok=True)
            
            # Generate unique identifier for this refinement session
            session_id = str(uuid.uuid4())[:8]
            
            # If temp directory exists and has content, clear it first
            if os.path.exists(temp_rag_path):
                for item in os.listdir(temp_rag_path):
                    item_path = os.path.join(temp_rag_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            os.makedirs(temp_rag_path, exist_ok=True)
            
            # Use subprocess directly, but optimize paths and parameters
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_file_dir, 'run_pose_refinement.py')
            cmd = ['xvfb-run', '-a', sys.executable, script_path, directory, temp_rag_path]
            print("Running 3D pose refinement with virtual display...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=300)

            
            # Select best match from refined results and save permanently
            best_refined_dir = None
            if os.path.exists(temp_rag_path):
                refined_dirs = [os.path.join(temp_rag_path, subdir) for subdir in os.listdir(temp_rag_path) 
                              if os.path.isdir(os.path.join(temp_rag_path, subdir)) and subdir != '__pycache__']
                if refined_dirs:
                    for refined_dir in refined_dirs:
                        if os.path.exists(os.path.join(refined_dir, 'result.json')):
                            best_refined_dir = refined_dir
                            # print(f"Using refined pose from: {refined_dir}")
                            break
                    print(f"3D pose refinement completed. Generated {len(refined_dirs)} refined poses.")
                    
                    # Save the best refined result permanently
                    if best_refined_dir:
                        permanent_dir = os.path.join(permanent_rag_path, f"refined_{session_id}")
                        shutil.copytree(best_refined_dir, permanent_dir)
                        directory = permanent_dir
                        # print(f"Best refined result saved permanently to: {permanent_dir}")
                else:
                    print("No refined poses generated, using original directory.")
            
            
        except subprocess.TimeoutExpired:
            print("3D pose refinement timed out. Using original directory.")
        except ImportError as ie:
            print(f"3D pose refinement import failed: {ie}. Using original directory.")
        except Exception as e:
            print(f"3D pose refinement failed: {e}. Using original directory.")
    else:
        pass
        # print(f"IMD score ({min_imd_score:.2f}) below threshold ({tau_imd}). Using directly as final reference.")
    
    # print(f"Most similar example found in: {directory}")
    return directory
