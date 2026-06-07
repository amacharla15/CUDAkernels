#!/usr/bin/env bash
set -e

IMAGE_NAME=${IMAGE_NAME:-qwen-vllm:phase3}

MODEL_DIR_HOST=${MODEL_DIR_HOST:-$(pwd)/toy-llm-prod-flow/02_vllm_serving/models/qwen_lora_merged_v1}
MODEL_DIR_CONTAINER=${MODEL_DIR_CONTAINER:-/models/qwen_lora_merged_v1}

HOST_PORT=${HOST_PORT:-8000}
CONTAINER_PORT=${CONTAINER_PORT:-8000}

echo "Running vLLM Docker container"
echo "IMAGE_NAME=$IMAGE_NAME"
echo "MODEL_DIR_HOST=$MODEL_DIR_HOST"
echo "MODEL_DIR_CONTAINER=$MODEL_DIR_CONTAINER"
echo "HOST_PORT=$HOST_PORT"
echo "CONTAINER_PORT=$CONTAINER_PORT"

docker run --rm --gpus all \
  --ipc=host \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${MODEL_DIR_HOST}:${MODEL_DIR_CONTAINER}:ro" \
  "${IMAGE_NAME}"
