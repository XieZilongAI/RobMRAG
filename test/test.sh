export PYTHONPATH=../../RobMRAG
OUPUT_DIR='./test_results/result_ori_qwen2_zero_shot'

# step1: model inference
CUDA_VISIBLE_DEVICES=7 python qwen_test_rag.py \
 --data_dir ../data_collection/data/test_data \
 --out_dir "$OUPUT_DIR" \
 --action pulling

 # step2: test the entire process in sapien simulation
python test_entireprocess_in_sapien.py \
 --data_dir ../data_collection/data/test_data \
 --num_processes 10 \
 --out_dir "$OUPUT_DIR" \
 --no_gui \
 --use_mask False

# step3: calculate success rate
python cal_test_mani_succ_rate.py \
    --primact_type pulling \
    --data_dir "$OUPUT_DIR"
