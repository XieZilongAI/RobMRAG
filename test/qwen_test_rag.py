# -*- coding: utf-8 -*-
"""
@Time: 2024/12/10 PM 3:25
@Auth: ***
@File: qwen_test_rag.py
@IDE: PyCharm
@Motto: YES(Always Be Coding)
"""
import os
import json
import torch
import uform
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import Qwen2VLForConditionalGeneration,AutoProcessor, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torchvision.models import vgg19
from qwen_vl_utils import process_vision_info
from rag.multi_rag_main import multimodal_rag
import shutil
from peft import PeftModel,PeftConfig

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--adapter_dir', type=str, default="none")
parser.add_argument('--action', type=str, help='llama directory')
conf = parser.parse_args()
# Configuration parameters
device = 'cuda' if torch.cuda.is_available() else "cpu"
data_dir = conf.data_dir  # Input data directory
out_dir = conf.out_dir  # Output directory
action = conf.action

model_path = "/data2/huggingface_pretrained_llms/Qwen2-VL-7B-Instruct"
# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
# )
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/pretrained_model/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
# lora_model_path = "../LLaMA-Factory/saves/qwen2_vl-7b/lora33/sft"  # Replace with your fine-tuned model path
# model = PeftModel.from_pretrained(model, lora_model_path)
model.to(device)
model.eval()
# Load text encoding model
text_model_path = 'all-MiniLM-L6-v2'
text_model = SentenceTransformer(text_model_path).eval().to(device)  # Use lightweight semantic model
# Load image encoding model
image_model = uform.get_model('unum-cloud/uform-vl-english').eval().to(device)
# Load VGG19 feature extractor
vgg19_model = vgg19(pretrained=True).features.eval().to(device)
processor = AutoProcessor.from_pretrained(model_path)
rag_knowledge_base_path = '../data_collection/data/balanced_rag_knowledge_base_1'
# Set different prompt text based on conditions
if '-ori' in conf.adapter_dir:
    prompt_text = "Specify the contact point and orientation of pushing the object."
else:
    prompt_text = '''You are a robot operation instructor, specialized in guiding robots to complete complex tasks.'''
def tokenize_scaled(num, scale=100):
    # return num
    scaled = int(round(num * scale))
    return ' '.join(list(str(scaled)))
# Traverse input data directory
record_names = os.listdir(data_dir)
for record_name in tqdm(record_names):
    record_dir = os.path.join(data_dir, record_name)
    # Input image path
    rgb_path = os.path.join(record_dir, 'original_rgb.png')
    object_id = record_name.split('_')[0]
    object_name = record_name.split('_')[1]
    input_instruction = action + ' ' + object_name
    if not os.path.exists(rgb_path):
        continue
    rag_path = multimodal_rag(input_instruction, object_id, rgb_path, rag_knowledge_base_path, text_model,
                                    image_model,
                                    vgg19_model, device)
    rag_example_path = os.path.join(rag_path, 'rag_example.png')
    with open(os.path.join(rag_path, 'result.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    numbers = data['pixel_locs']
    try:
        point = tuple(tokenize_scaled(round(x / 336, 2)) for x in numbers)
    except Exception as e:
        print(numbers)
        print(rag_path)
    forward_num = data['gripper_forward_direction_camera']
    rag_forward = [tokenize_scaled(round(x, 1), 10) for x in forward_num]
    up_num = data['gripper_up_direction_camera']
    rag_up = [tokenize_scaled(round(x, 1), 10) for x in up_num]


#     prompt_text_v1 = f'''The current robot is a Franka Panda Robot equipped with a Flying Suction Gripper as its actuator. The task is: {action} the {object_name}. The first image shows what you are currently observing. The second image is a reference, where the red dot marks the grasping point. This reference image is similar to your current observation, with the grasp point at {point}, the the Gripper Up Direction set to {rag_up}, and Gripper Forward Direction set to {rag_forward}. If a third image is present, it displays the final outcome based on the second image.
# Your responsibility is to predict precise grasp points and grasp directions of observed image in the camera coordinate system according to the task requirements. Specifically, you need to provide:
# 1. Grasp Point: A 2D floating-point value representing the normalized pixel coordinates of the grasp point on the observed image, formatted as (x, y) and accurate to two decimal places.
# 2. Grasp Direction: A 3D vector including:
#   # Gripper Up Direction
#   # Gripper Forward Direction
# Each direction must be represented as a normalized 3D vector (x, y, z).
# '''
#     prompt_text_v2 = f'''The current robot is a Franka Panda Robot equipped with a Flying Suction Gripper as its actuator. The task is: {action} the {object_name}. The first image shows what you are currently observing. The second image is a reference, where the red dot marks the grasping point. This reference image is similar to your current observation, with the grasp point at ({point[0]},{point[1]}), the Gripper Up Direction set to [{','.join(rag_up)}], and Gripper Forward Direction set to [{','.join(rag_forward)}]. If a third image is present, it displays the final outcome based on the second image.
# Your responsibility is to predict precise grasp points and grasp directions of observed image in the camera coordinate system according to the task requirements. Specifically, you need to provide:
# 1. **Grasp Point**: Two integer values formatted as `[x_tokens],[y_tokens]`, where each value is a sequence of numbers from 0 to 100 separated by spaces (e.g., `3 2` represents 32).
# 2. **Grasp Direction**:
#   - **Gripper Up Direction**: Three integer components formatted as `[x],[y],[z]`, each component being a number from -10 to 10 separated by spaces (e.g., `- 4` represents -4).
#   - **Gripper Forward Direction**: Three integer components formatted as `[x],[y],[z]`, following the same rule.
# Example response format: Contact point: (3 2,9 5); Up direction: [5,0,1]; Forward direction: [- 1 0,9,0] '''
    prompt_text_v3 = f'''The current robot is a Franka Panda Robot equipped with a Flying Suction Gripper as its actuator. The task is: {action} the {object_name}. The first image shows what you are currently observing. The second image is a reference, where the red dot marks the grasping point. This reference image is similar to your current observation, with the grasp point at ({point[0]},{point[1]}), the Gripper Up Direction set to [{','.join(rag_up)}], and Gripper Forward Direction set to [{','.join(rag_forward)}]. If a third image is present, it displays the final outcome based on the second image.
    Your responsibility is to predict precise grasp points and grasp directions of observed image in the camera coordinate system according to the task requirements. Specifically, you need to provide:
    1. **Grasp Point**: Two integer values formatted as `(x, y)`, where each value is a sequence of numbers from 0 to 100 separated by spaces (e.g., `3 2` represents 32).
    2. **Grasp Direction**:
      - **Gripper Up Direction**: Three integer components formatted as `[x, y, z]`, each component being a number from -10 to 10 separated by spaces (e.g., `- 4` represents -4).
      - **Gripper Forward Direction**: Three integer components formatted as `[x, y, z]`, following the same rule. '''
    prompt_text_zero_shot = f'''The current robot is a Franka Panda Robot equipped with a Flying Suction Gripper as its actuator. The task is: {action} the {object_name}. The first image shows what you are currently observing. The second image is a reference, where the red dot marks the grasping point. This reference image is similar to your current observation, with the grasp point at {point}, the the Gripper Up Direction set to {rag_up}, and Gripper Forward Direction set to {rag_forward}. If a third image is present, it displays the final outcome based on the second image.
Your observed image certainly includes the {object_name}, even if it may appear too blurry to identify clearly.
Your responsibility is to provide precise grasp points and grasp directions in the camera coordinate system according to the task requirements. Specifically, you need to provide:
1. Grasp Point: A 2D floating-point value representing the normalized pixel coordinates of the grasp point on the image, formatted as (x, y) and accurate to two decimal places.
2. Grasp Direction: A 3D vector including:
  # Gripper Up Direction
  # Gripper Forward Direction
Each direction must be represented as a normalized 3D vector (x, y, z).
### Output Format Requirements:
The final result must strictly follow this format, without any additional output:
Grasp Point: Contact point is (x, y)
Gripper Up Direction: Gripper up direction is [x, y, z]
Gripper Forward Direction: Gripper forward direction is [x, y, z]
### Example:
Input: An observed image and several similar example operation images.
Output: Contact point is (0.65, 0.52), gripper up direction is [0.75, -0.43, 0.68], gripper forward direction is [-0.27, 0.71, 0.60].
'''
    # print(prompt_text_zero_shot)
    record_out_dir = os.path.join(out_dir, record_name)
    if not os.path.exists(record_out_dir):
        os.makedirs(record_out_dir)
    exist = os.path.exists(os.path.join(rag_path, 'rag_result.png'))
    # exist = False
    if exist:
        rag_result_path = os.path.join(rag_path, 'rag_result.png')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_path},
                    {"type": "image", "image": rag_example_path},
                    {"type": "image", "image": rag_result_path},
                    {"type": "text", "text": prompt_text_zero_shot},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_path},
                    {"type": "image", "image": rag_example_path},
                    {"type": "text", "text": prompt_text_zero_shot},
                ],
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None,
                                       top_k=None)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]  # only get the firts result
    # print(result)
    # Save result as JSON file
    output_path = os.path.join(record_out_dir, 'prediction.json')
    with open(output_path, 'w') as fout:
        json.dump(result, fout)
