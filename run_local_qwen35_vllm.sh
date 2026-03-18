#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
VLLM_BIN=/home/saisamarth/dev/chartQA/.venv/bin/vllm
MODEL_ID=Qwen/Qwen3.5-9B
SERVED_NAME=Qwen3.5-9B-local
PORT=7999
HOST=127.0.0.1

exec "$VLLM_BIN" serve "$MODEL_ID"   --host "$HOST"   --port "$PORT"   --served-model-name "$SERVED_NAME"   --dtype bfloat16   --kv-cache-dtype fp8   --gpu-memory-utilization 0.92   --max-model-len 32768   --max-num-seqs 8   --enable-prefix-caching   --trust-remote-code   --reasoning-parser qwen3   --default-chat-template-kwargs '{"enable_thinking": true}'
