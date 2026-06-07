FROM vllm/vllm-openai:v0.8.5.post1

WORKDIR /app

ENV MODEL_DIR=/models/qwen_lora_merged_v1
ENV SERVED_MODEL_NAME=qwen_lora_merged_v1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DTYPE=float16
ENV MAX_MODEL_LEN=1024
ENV GPU_MEMORY_UTILIZATION=0.85
ENV VLLM_USE_V1=0

COPY docker/entrypoint.sh /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
