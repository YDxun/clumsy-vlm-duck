#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
CONFIG="${CONFIG:-duck_vlm/train/configs/qwen3_5_2b_lora.yaml}"
exec "$CLI" train "$CONFIG" "$@"