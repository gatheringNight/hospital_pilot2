#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
GPU_LAYERS="${GPU_LAYERS:-0}"
PORT="${PORT:-18080}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" "$PACKAGE_DIR/tools/model_parts.py" restore
exec "$LLAMA_SERVER" -m "$PACKAGE_DIR/models/qwen2.5-3b-instruct-q4_k_m.gguf" \
  --host 127.0.0.1 --port "$PORT" -ngl "$GPU_LAYERS" -c 4096 --no-webui
