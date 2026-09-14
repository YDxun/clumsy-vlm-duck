#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER="${ADAPTER:-duck_vlm/train/output}"
PORT="${PORT:-8000}"
exec vllm serve "$MODEL" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "duck-vlm-lora=$ADAPTER" \
  --served-model-name duck-vlm-lora