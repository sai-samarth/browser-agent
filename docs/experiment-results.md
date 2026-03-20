# Browser-Agent Experiment Results

## Overview

This file tracks the major experiments, datasets, and model results for the browser-use fine-tuning project.

## Data generation summary

### Local teacher
- Model: `Qwen/Qwen3.5-9B`
- Serving stack: local vLLM on RTX 4090
- Endpoint: `http://127.0.0.1:7999/v1`

### Production data campaign
- Production task subset: 30 curated MiniWoB tasks
- Batch 1: 8 repeated runs
- Batch 2: 6 repeated runs
- Distinct seed offsets used between repeated runs

### Corpus totals
- Episodes: 4200
- Step-level examples collected: 10451
- Successful episodes: 4023
- Overall episode success rate: 95.79%

## SFT export summary

### Export version
- Export directory: `data/exports/phase1_sft_v2`

### Filters
- Successful episodes only
- Max action errors: 0
- Max repeated loops: 0
- Max sparse observations: 2
- Max root-only observations: 0
- Max fallback count: 0

### Exported rows
- Episodes seen: 4200
- Episodes kept: 3415
- Action-only: 6508 train / 240 val
- Reasoning+action: 6508 train / 240 val

### Dataset links
- Action-only dataset: https://huggingface.co/datasets/saital/browser-agent-phase1-sft-action-only
- Reasoning+action dataset: https://huggingface.co/datasets/saital/browser-agent-phase1-sft-reasoning-action

## Model fine-tuning experiment 1

### Setup
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: PEFT LoRA
- Dataset: `data/exports/phase1_sft_v2/action_only/hf_dataset`
- Epochs: 1
- Max length: 2048
- Batch size: 4
- Grad accumulation: 4
- Learning rate: 2e-4
- Output dir: `outputs/qwen25-1.5b-browser-action-lora`

### Training result
- Train runtime: ~3197s (~53.3 min)
- Final train loss: 0.1059
- Final eval loss: 0.05004

### Before/after evaluation
Validation set size: 240

Before fine-tuning:
- Parseable action rate: 100%
- Exact-match action accuracy: 17.08%

After fine-tuning:
- Parseable action rate: 100%
- Exact-match action accuracy: 79.58%

### Interpretation
- The dataset is teaching the model something real.
- The first PEFT run produced a large improvement over baseline.
- Remaining failures are often close misses or formatting-equivalent action variants.

### Model artifact
- Local artifact dir: `outputs/qwen25-1.5b-browser-action-lora`


## Model fine-tuning experiment 2

### Setup
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: Unsloth LoRA
- Dataset: `data/exports/phase1_sft_v2/reasoning_action/hf_dataset`
- Epochs: 1
- Max length: 3072
- Batch size: 4
- Grad accumulation: 4
- Learning rate: 2e-4
- Output dir: `outputs/qwen25-1.5b-browser-reasoning-unsloth`

### Training result
- Train runtime: ~2717s (~45.3 min)
- Final train loss: 27.31
- Final eval loss: 8.766

### Before/after evaluation
Validation set size: 240

Before fine-tuning:
- Parseable action rate: 57.50%
- Exact-match action accuracy: 12.50%

After fine-tuning with low eval budget (`max_new_tokens=64`):
- Parseable action rate: 83.75%
- Exact-match action accuracy: 69.58%

After fine-tuning with corrected eval budget (`max_new_tokens=512`):
- Parseable action rate: 100.00%
- Exact-match action accuracy: 81.67%

### Interpretation
- The initial 69.58% exact-match result materially undercounted the model because long reasoning traces were truncated by the evaluation budget.
- Once evaluation was rerun with a larger generation budget, the reasoning-action model slightly outperformed the previous action-only Unsloth baseline on raw exact-match.
- Remaining errors after the high-budget rerun are mostly genuine action-choice failures on checkbox-heavy tasks, plus a small amount of canonicalization noise.

### Model artifact
- Local artifact dir: `outputs/qwen25-1.5b-browser-reasoning-unsloth`
