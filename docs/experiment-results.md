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
- Judge-equivalent accuracy on saved low-budget outputs: 70.83%

After fine-tuning with corrected eval budget (`max_new_tokens=512`):
- Parseable action rate: 100.00%
- Exact-match action accuracy: 81.67%
- Judge-equivalent accuracy: 83.33%

### Interpretation
- The initial 69.58% exact-match result materially undercounted the model because long reasoning traces were truncated by the evaluation budget.
- LLM-as-judge slightly increased the measured score relative to raw exact-match by forgiving harmless formatting differences like quote style and omitted `button='left'`.
- The remaining gap after high-budget evaluation is mostly genuine action-selection error on checkbox-heavy tasks, not parser failure.
- One judge disagreement was clearly a spurious false negative, so the 83.33% judge score should be treated as slightly noisy but directionally useful.

### Model artifact
- Local artifact dir: `outputs/qwen25-1.5b-browser-reasoning-unsloth`
- Judge summary: `outputs/qwen25-1.5b-browser-reasoning-unsloth/judge_512_summary.json`

## Model fine-tuning experiment 3

### Setup
- Base model: `Qwen/Qwen3.5-0.8B`
- Method: baseline eval only so far
- Dataset: `data/exports/phase1_sft_v2/action_only/hf_dataset`
- Validation rows: 240
- Eval budget: `max_new_tokens=256`
- Output dir: `outputs/qwen35-0.8b-browser-action-unsloth`

### Baseline evaluation
- Parseable action rate: 100.00%
- Exact-match action accuracy: 17.92%

### Initial interpretation
- The model loads cleanly in the same 4-bit eval path used for the other comparisons.
- Baseline exact-match is close to the Qwen2.5-1.5B action-only baseline despite the much smaller parameter count, making it a good next SFT candidate.
- Failure analysis shows a strong bias toward unnecessary `modifiers=[Control]` clicks plus some real checkbox-target mistakes, which is a plausible SFT-fixable pattern.
- The first Unsloth training attempt failed before training because `Qwen/Qwen3.5-0.8B` loads through a `Qwen3VLProcessor` path under Unsloth rather than a plain text tokenizer.
- Root cause: tokenizing through the processor wrapper in the dataset map produced the wrong shape for text-only SFT and triggered the multimodal/image parsing path.
- Confirmed fix: render chat text with `apply_chat_template(...)`, then tokenize with the underlying text tokenizer (`processor.tokenizer`) for padded text-only batches.
- Fine-tuning on the action-only dataset is the next step.
