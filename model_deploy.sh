CUDA_VISIBLE_DEVICES=2 \
vllm serve ../models/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --max_model_len 4096 \
    --gpu-memory-utilization 0.8 \
    --port 8004 \