#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-0.8B}
DATASET_DIR=${DATASET_DIR:-data/exports/phase1_sft_v2/action_only/hf_dataset}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen35-0.8b-browser-action-unsloth}
EVAL_JSON=${EVAL_JSON:-${OUTPUT_DIR}/eval_after_conditional_256.json}

python3 scripts/train_unsloth_sft_generic.py   --dataset-dir "$DATASET_DIR"   --model-name "$BASE_MODEL"   --output-dir "$OUTPUT_DIR"   --max-length 2048   --num-train-epochs 1   --per-device-train-batch-size 4   --gradient-accumulation-steps 4   --learning-rate 2e-4

python3 scripts/eval_action_model.py   --dataset-dir "$DATASET_DIR"   --model-name "$BASE_MODEL"   --adapter-dir "$OUTPUT_DIR"   --output-json "$EVAL_JSON"   --max-new-tokens 256
