#!/usr/bin/env bash
set -e

export VLLM_USE_V1=${VLLM_USE_V1:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_DIR=${MODEL_DIR:-/models/qwen_lora_merged_v1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen_lora_merged_v1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
DTYPE=${DTYPE:-float16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

echo "Starting vLLM server"
echo "MODEL_DIR=$MODEL_DIR"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "PORT=$PORT"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager
