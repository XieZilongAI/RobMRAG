# RobMRAG
The official codebase for **RobMRAG: 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation for Zero-Shot Robotic Manipulation**

## Abstract

Existing end-to-end approaches of robotic manipulation often lack generalization to unseen objects or tasks due to limited data and poor interpretability. While recent Multimodal Large Language Models (MLLMs) demonstrate strong commonsense reasoning, they struggle with geometric and spatial understanding required for pose prediction. In this paper, we propose RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation.

## Acknowledgement
This repo benefits from [ManipLLM](https://github.com/clorislili/ManipLLM), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and [TRELLIS](https://github.com/microsoft/TRELLIS). Thanks for their wonderful works.

## Setup
1) conda create --name robmrag python=3.8

2) conda activate robmrag

3) pip install -r requirements.txt

## Model Download
Download the required multimodal large language models using huggingface-cli:

```bash
# Download Qwen2-VL-7B-Instruct (primary model)
huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir ./models/Qwen2-VL-7B-Instruct

# Download Qwen2.5-VL-7B-Instruct or meta-llama/Llama-3.2-11B-Vision-Instruct (for ablation studies)
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen2.5-VL-7B-Instruct
```

**Note:** Make sure you have sufficient disk space as these models are several GB in size.

            
## Data Collection

### Knowledge Base Construction
RobMRAG constructs a multi-source manipulation knowledge base containing object contact frames, task completion frames, and pose parameters.

- Download [PartNet Mobility](https://sapien.ucsd.edu/downloads) URDF files from the official website and place under `./data_collection/asset`:
  ```bash
  ./asset/original_sapien_dataset
    ├── 148
    |   └── mobility.urdf
    ├── 149
    |   └── mobility.urdf
    ├── ...
    │   ...
    └── ...
  ```

- Generate training and testing datasets:
  ```bash
  cd ./data_collection/code
  bash scripts/run_gen_offline_data.sh
  ```

This command will first generate the training dataset and then generate the testing dataset for the retrieval-augmented generation framework.

## Model Training

### Training Data Preparation
Generate training JSON files for the MLLM:

```bash
cd ./collect_train_data
bash get_train_data.sh
```

### Fine-tuning with LLaMA-Factory
RobMRAG uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning of multimodal large language models.

1. **Install LLaMA-Factory:**
   ```bash
   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]" --no-build-isolation
   ```

2. **Configure Training:**
   - Configure the `data/dataset_info.json` file
   - Modify the `examples/train_lora/qwen2vl_lora_sft.yaml` configuration file

3. **Start Training:**
   ```bash
   WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
   ```

**Note:** Training requires at least 24GB GPU memory. Multi-GPU training is recommended for better performance.

## Model Testing

### Test Data Preparation
- Download the [test data](https://disk.pku.edu.cn/link/AA103C5B00398E4E4089903CB06AC09D8C) and unzip under `./data_collection/data/`
- Download [PartNet Mobility](https://sapien.ucsd.edu/downloads) URDF files and place under `./data_collection/asset/`

### Zero-Shot Robotic Manipulation Testing
RobMRAG enables zero-shot manipulation on unseen objects through the multimodal retrieval-augmented generation framework.

#### Configuration Options

**For Fine-tuned Model Testing:**
- Load LoRA fine-tuned model weights
- Use standard prompt templates

**For Zero-Shot Testing:**
- Skip LoRA model loading (use base model directly)
- Switch to `prompt_text_zero_shot` for zero-shot prompts

```bash
cd ./test
bash test.sh
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
