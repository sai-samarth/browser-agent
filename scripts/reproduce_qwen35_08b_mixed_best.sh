#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-0.8B}
SOURCE_DATASET_DIR=${SOURCE_DATASET_DIR:-data/exports/phase1_sft_v2/action_only/hf_dataset}
BASE_ADAPTER_DIR=${BASE_ADAPTER_DIR:-outputs/qwen35-0.8b-browser-action-unsloth}
MIXED_DATASET_DIR=${MIXED_DATASET_DIR:-data/exports/phase1_sft_v2_action_weak3_mixed50_1000/hf_dataset}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft}
EVAL_JSON=${EVAL_JSON:-${OUTPUT_DIR}/eval_after_conditional_256.json}

python3 scripts/build_mixed_weak_subset.py   --source-dataset-dir "$SOURCE_DATASET_DIR"   --output-dataset-dir "$MIXED_DATASET_DIR"   --weak-tasks click-checkboxes-large find-word enter-text-2   --weak-train-size 500   --other-train-size 500   --seed 3407   --val-mode full

python3 scripts/train_qwen35_continuation_sft.py   --dataset-dir "$MIXED_DATASET_DIR"   --base-model "$BASE_MODEL"   --adapter-dir "$BASE_ADAPTER_DIR"   --output-dir "$OUTPUT_DIR"   --max-length 2048   --num-train-epochs 2.0   --per-device-train-batch-size 4   --gradient-accumulation-steps 4   --learning-rate 1e-4   --seed 3407

python3 scripts/eval_action_model.py   --dataset-dir "$SOURCE_DATASET_DIR"   --model-name "$BASE_MODEL"   --adapter-dir "$OUTPUT_DIR"   --output-json "$EVAL_JSON"   --max-new-tokens 256
