#!/bin/bash
# Script for building balanced RAG knowledge base
# This script performs three steps:
# 1. Extract RAG knowledge from test data
# 2. Balance the knowledge base (select a smaller subset)
# 3. Check and clean the balanced knowledge base

echo "=========================================="
echo "Building Balanced RAG Knowledge Base"
echo "=========================================="
echo ""

# Step 1: Extract RAG knowledge from test data
echo "Step 1: Extracting RAG knowledge from test data..."
echo "Source: ../data/test_data_ori"
echo "Target: ../data/rag_knowledge_base_1"
python get_rag_knowledge.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract RAG knowledge"
    exit 1
fi
echo "Step 1 completed successfully!"
echo ""

# Step 2: Balance the knowledge base
echo "Step 2: Balancing knowledge base..."
echo "Source: ../data/rag_knowledge_base_1"
echo "Target: ../data/balanced_rag_knowledge_base_1"
echo "Sample number: 800"
python balanced_test_data.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to balance knowledge base"
    exit 1
fi
echo "Step 2 completed successfully!"
echo ""

