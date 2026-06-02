#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models/qwen_lora_merged_v1"

echo "Serving model from: $MODEL_DIR"

vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.85
