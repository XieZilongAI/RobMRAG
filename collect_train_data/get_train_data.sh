#step1: generate training json
export PYTHONPATH=../../RobMRAG
JSON_DIR='./data/train_json'
CUDA_VISIBLE_DEVICES=7 python ./get_train_data.py --folder_dir ../data_collection/data/train_data_ori_depth --output_dir "$JSON_DIR" --num_point 20
# Go back to parent directory cd ../, download and install LLaMA-Factory, git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory
# pip install -e ".[torch,metrics]" --no-build-isolation
# Configure the data/dataset_info.json file, modify the examples/train_lora/qwen2vl_lora_sft.yaml configuration, then run the following command
# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml


