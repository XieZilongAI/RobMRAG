# RobMRAG
The official codebase for **Zero-Shot Robotic Manipulation via 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation**

## Abstract

Existing end-to-end approaches of robotic manipulation often lack generalization to unseen objects or tasks due to limited data and poor interpretability. While recent Multimodal Large Language Models (MLLMs) demonstrate strong commonsense reasoning, they struggle with geometric and spatial understanding required for pose prediction. In this paper, we propose RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation.

**Key Contributions:**
- We propose a Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation, which enables manipulation on unseen objects through a multi-source knowledge base.
- We integrate a 3D-Aware Pose Refinement module into the MRAG framework, enabling precise pose alignment between reference and target objects, thereby enhancing the geometric consistency of the retrieved results.
- Experimental results demonstrate that, on a test set comprising 30 categories of household objects, the proposed method achieves a 7.76% improvement in success rate compared with the SOTA zero-shot baseline, and a 6.54% improvement compared with the SOTA supervised baseline.

## Acknowledgement
This repo benefits from [ManipLLM](https://github.com/ToyotaResearchInstitute/ManipLLM), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their wonderful works.

## Setup
1) conda create --name robmrag python=3.8

2) conda activate robmrag

3) pip install -r requirements.txt

## Model Download
Download the required multimodal large language models using huggingface-cli:

```bash
# Download Qwen2-VL-7B-Instruct (primary model)
huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir ./models/Qwen2-VL-7B-Instruct

# Download Qwen2.5-VL-7B-Instruct (for ablation studies)
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

- Build balanced RAG knowledge base:
  ```bash
  cd ./data_collection/code
  bash scripts/build_rag_knowledge_base.sh
  ```

Example with custom parameters:
```bash
cd ./data_collection/code
bash scripts/build_rag_knowledge_base.sh --samples_per_class 100 --output_dir ../data/rag_knowledge_base_100
```

The final balanced knowledge base will be saved at the specified output directory and can be used for training data generation and testing.

## Model Training

### Training Data Preparation
Generate training JSON files for the MLLM. There are two methods available:

#### Method 1: Standard Training Data Generation
Generate training data using the standard method:

```bash
cd ./collect_train_data
bash get_train_data.sh
```

This will generate training JSON files using `get_train_data.py` with default settings.

#### Method 2: Balanced Training Data Generation
Generate balanced training data with sample balancing strategies:

```bash
cd ./collect_train_data
bash test_balancing.sh
```

Or manually run with custom parameters:

```bash
cd ./collect_train_data
python get_train_data_balanced.py \
    --folder_dir ../data_collection/data/train_data_ori_depth \
    --rag_knowledge_base_dir ../data_collection/data/balanced_rag_knowledge_base_1 \
    --output_dir ./data/train_json \
    --num_point 20 \
    --target_samples_per_object 1000 \
    --balancing_strategy target_based \
    --max_total_samples 20000 \
    --mlm True \
    --bins True \
    --aff_prior True
```

**Balancing Strategies:**
- `target_based`: Target-based balancing - prioritizes collecting samples for objects with insufficient samples
- `dynamic_sampling`: Dynamic sampling rate adjustment - dynamically adjusts sampling probability based on current sample count
- `stratified`: Stratified sampling - ensures each object type has a chance to be sampled

## Model Testing

### Test Data Preparation
- Download the [test data](https://disk.pku.edu.cn/link/AA103C5B00398E4E4089903CB06AC09D8C) and unzip under `./data_collection/data/`
- Download [PartNet Mobility](https://sapien.ucsd.edu/downloads) URDF files and place under `./data_collection/asset/`

### Zero-Shot Robotic Manipulation Testing
The default `test.sh` script performs zero-shot testing using the base model without any fine-tuning.

```bash
cd ./test
bash test.sh
```

This script will:
1. Run model inference on test data using the zero-shot base model
2. Test the entire process in SAPIEN simulation
3. Calculate the success rate

**Note:** The test script uses `prompt_text_zero_shot` for zero-shot prompts and does not load any LoRA weights.

### Fine-tuned Model Testing
To test with a LoRA fine-tuned model, you need to modify the test script (`test/qwen_test_rag.py`):

1. **Enable LoRA loading:**
   - Uncomment lines 44-45 to load the LoRA adapter weights:
   ```python
   lora_model_path = "../LLaMA-Factory/saves/qwen2_vl-7b/lora33/sft"  # Replace with your fine-tuned model path
   model = PeftModel.from_pretrained(model, lora_model_path)
   ```
   - Update `lora_model_path` to point to your trained LoRA checkpoint directory

2. **Switch prompt template:**
   - Change `prompt_text_zero_shot` to `prompt_text_v3` in the messages (lines 148 and 159)
   - The `prompt_text_v3` uses tokenized format suitable for fine-tuned models

3. **Run the test:**
   ```bash
   cd ./test
   bash test.sh
   ```

**Testing Pipeline:**
The test process includes three steps:
- **Step 1:** Model inference - Generate predictions for test samples
- **Step 2:** SAPIEN simulation - Test predictions in physics simulation
- **Step 3:** Success rate calculation - Evaluate manipulation success rate

### Fine-tuning with LLaMA-Factory
RobMRAG uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient LoRA fine-tuning of multimodal large language models.

#### Step 1: Install LLaMA-Factory
```bash
cd ../
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

#### Step 2: Configure Dataset
1. Configure the `data/dataset_info.json` file to include your training dataset:
   ```json
   {
     "your_dataset_name": {
       "file_name": "train_balanced_target_based_20000_samples.json",
       "file_encoding": "utf-8",
       "columns": {
         "messages": "messages",
         "images": "images"
       }
     }
   }
   ```

2. Update the dataset path in the configuration to point to your generated training JSON file.

#### Step 3: Configure Training Parameters
Modify the `examples/train_lora/qwen2vl_lora_sft.yaml` configuration file:
- Set `dataset` to your dataset name
- Configure `model_name_or_path` to point to your Qwen2-VL-7B-Instruct model
- Adjust `output_dir` for saving checkpoints
- Set `per_device_train_batch_size`, `gradient_accumulation_steps`, and other training hyperparameters

#### Step 4: Start LoRA Training
```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
```

**Note:** 
- Training requires at least 24GB GPU memory per GPU. Multi-GPU training is recommended for better performance.
- The trained LoRA weights will be saved in the `output_dir` specified in the configuration file.

