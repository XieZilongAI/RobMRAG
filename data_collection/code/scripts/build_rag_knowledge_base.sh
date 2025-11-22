#!/bin/bash

TRAIN_DIR="/data0/xzl/ManipLLM/data_collection/data/train_data_dist4.0"
TEST_DIR="/data0/xzl/ManipLLM/data_collection/data/test_data_dist4.0"
OUTPUT_DIR="../data/rag_knowledge_base_dist4.0"
SAMPLES_PER_CLASS=50


while [[ $# -gt 0 ]]; do
    case $1 in
        --train_dir)
            TRAIN_DIR="$2"
            shift 2
            ;;
        --test_dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --samples_per_class)
            SAMPLES_PER_CLASS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done


python build_rag_knowledge.py \
    --train_dir "$TRAIN_DIR" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --samples_per_class "$SAMPLES_PER_CLASS"

