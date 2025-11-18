#!/bin/bash
export PYTHONPATH=../../RobMRAG

# Script for testing sample balancing strategies

echo "Starting sample balancing strategy test..."

# Set parameters
FOLDER_DIR="../data_collection/data/train_data_ori_depth"
RAG_KNOWLEDGE_BASE_DIR="../data_collection/data/balanced_rag_knowledge_base_1"
OUTPUT_DIR="./data/train_json"
NUM_POINT=20
TARGET_SAMPLES_PER_OBJECT=1000
MAX_TOTAL_SAMPLES=20000

echo "Parameter settings:"
echo "  Folder directory: $FOLDER_DIR"
echo "  RAG knowledge base directory: $RAG_KNOWLEDGE_BASE_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Target samples per object: $TARGET_SAMPLES_PER_OBJECT"
echo "  Maximum total samples: $MAX_TOTAL_SAMPLES"
echo ""

# Test strategy 1: Target-based balancing
echo "Testing strategy 1: Target-based balancing (target_based)"
python get_train_data_balanced.py \
    --folder_dir "$FOLDER_DIR" \
    --rag_knowledge_base_dir "$RAG_KNOWLEDGE_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_point $NUM_POINT \
    --target_samples_per_object $TARGET_SAMPLES_PER_OBJECT \
    --balancing_strategy target_based \
    --max_total_samples $MAX_TOTAL_SAMPLES \
    --mlm True \
    --bins True \
    --aff_prior True

echo ""
echo "Strategy 1 test completed"
echo ""

# Test strategy 2: Dynamic sampling rate adjustment
# echo "Testing strategy 2: Dynamic sampling rate adjustment (dynamic_sampling)"
# python get_train_data_balanced.py \
#     --folder_dir "$FOLDER_DIR" \
#     --rag_knowledge_base_dir "$RAG_KNOWLEDGE_BASE_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --num_point $NUM_POINT \
#     --target_samples_per_object $TARGET_SAMPLES_PER_OBJECT \
#     --balancing_strategy dynamic_sampling \
#     --max_total_samples $MAX_TOTAL_SAMPLES \
#     --mlm True \
#     --bins True \
#     --aff_prior True

# echo ""
# echo "Strategy 2 test completed"
# echo ""

# # Test strategy 3: Stratified sampling
# echo "Testing strategy 3: Stratified sampling (stratified)"
# python get_train_data_balanced.py \
#     --folder_dir "$FOLDER_DIR" \
#     --rag_knowledge_base_dir "$RAG_KNOWLEDGE_BASE_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --num_point $NUM_POINT \
#     --target_samples_per_object $TARGET_SAMPLES_PER_OBJECT \
#     --balancing_strategy stratified \
#     --max_total_samples $MAX_TOTAL_SAMPLES \
#     --mlm True \
#     --bins True \
#     --aff_prior True

# echo ""
# echo "Strategy 3 test completed"
# echo ""

# echo "All balancing strategy tests completed!"
# echo "Please check the result files in the output directory to compare the balancing effects of different strategies."
