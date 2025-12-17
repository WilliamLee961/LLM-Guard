#!/bin/bash

# ==========================================
# Guard Pipeline 批量后台运行脚本
# ==========================================
python apply_guard_to_pipeline.py \
    --input_detect_mode strict \
    --output_detect_mode normal \
    --enable_input_guard \
    --enable_output_guard \

python apply_guard_to_pipeline.py \
    --input_detect_mode normal \
    --output_detect_mode normal \
    --enable_input_guard \
    --enable_output_guard \

python apply_guard_to_pipeline.py \
    --input_detect_mode normal \
    --output_detect_mode strict \
    --enable_input_guard \
    --enable_output_guard \

python apply_guard_to_pipeline.py \
    --input_detect_mode strict \
    --output_detect_mode strict \
    --enable_input_guard \
    --enable_output_guard \


python apply_guard_to_pipeline.py \
    --input_detect_mode strict \
    --output_detect_mode normal \
    --enable_input_guard \
    --enable_output_guard \
    --use_rag

python apply_guard_to_pipeline.py \
    --input_detect_mode normal \
    --output_detect_mode normal \
    --enable_input_guard \
    --enable_output_guard \
    --use_rag

python apply_guard_to_pipeline.py \
    --input_detect_mode normal \
    --output_detect_mode strict \
    --enable_input_guard \
    --enable_output_guard \
    --use_rag

python apply_guard_to_pipeline.py \
    --input_detect_mode strict \
    --output_detect_mode strict \
    --enable_input_guard \
    --enable_output_guard \
    --use_rag


