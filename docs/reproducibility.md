# Reproducibility Guide

This document explains how to reproduce the key project results without digging through the full experiment log.

## Reproduction levels

### Level 1 — inspect the final tracked evidence

Use this if you mainly want to audit the project:

- read `docs/final-summary.md`
- read `docs/experiment-results.md`
- inspect tracked eval artifacts under `outputs/`
- inspect dataset manifests under `data/exports/`

### Level 2 — reproduce the final 0.8B mixed continuation from released datasets

This is the recommended path for most readers.

### Level 3 — rebuild exports from local rollout traces

Use this only if you already have compatible BrowserGym rollout directories locally.

## 1. Lightweight setup

For export, manifests, and helper scripts:

```bash
uv sync
```

This is enough for:
- dataset download helpers
- export scripts
- manifest inspection
- most non-training utilities

## 2. Fine-tuning environment

Training scripts require a GPU environment with at least:
- `torch`
- `datasets`
- `transformers`
- `peft`
- `bitsandbytes`
- `unsloth`

The repo’s later Qwen3.5 RL work also required:
- `transformers==5.3.0`
- `weave==0.52.35`

The earlier RL env pinned to `transformers==4.57.3` is not the final recommended environment for Qwen3.5 work.

## 3. Download the released base dataset from Hugging Face

Action-only dataset:

```bash
uv run python3 scripts/download_hf_dataset.py   --repo-id saital/browser-agent-phase1-sft-action-only   --output-dir data/exports/from_hub/phase1_action_only/hf_dataset
```

Reasoning+action dataset:

```bash
uv run python3 scripts/download_hf_dataset.py   --repo-id saital/browser-agent-phase1-sft-reasoning-action   --output-dir data/exports/from_hub/phase1_reasoning_action/hf_dataset
```

## 4. Rebuild the base export from local rollout traces instead of downloading

Only do this if you have the local rollout corpus.

```bash
uv run python3 scripts/export_sft_dataset.py   --rollout-glob 'miniwob_phase1_prod_local_qwen35_batch*_r*_w*_202603*'   --output-dir data/exports/phase1_sft_v2
```

The exact matched rollout set used for the canonical export is recorded in:

- `data/exports/phase1_sft_v2/manifest.json`

## 5. Train the Qwen3.5-0.8B action-only baseline

This is the warm-start adapter used by the final 0.8B continuation.

```bash
bash scripts/train_qwen35_08b_action_baseline.sh
```

Equivalent core command:

```bash
python3 scripts/train_unsloth_sft_generic.py   --dataset-dir data/exports/phase1_sft_v2/action_only/hf_dataset   --model-name Qwen/Qwen3.5-0.8B   --output-dir outputs/qwen35-0.8b-browser-action-unsloth   --max-length 2048   --num-train-epochs 1   --per-device-train-batch-size 4   --gradient-accumulation-steps 4   --learning-rate 2e-4
```

Then evaluate:

```bash
python3 scripts/eval_action_model.py   --dataset-dir data/exports/phase1_sft_v2/action_only/hf_dataset   --model-name Qwen/Qwen3.5-0.8B   --adapter-dir outputs/qwen35-0.8b-browser-action-unsloth   --output-json outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json   --max-new-tokens 256
```

Expected result:
- parseable: 100.00%
- exact-match: 80.83%

## 6. Build the mixed weak-task continuation dataset

```bash
python3 scripts/build_mixed_weak_subset.py   --source-dataset-dir data/exports/phase1_sft_v2/action_only/hf_dataset   --output-dataset-dir data/exports/phase1_sft_v2_action_weak3_mixed50_1000/hf_dataset   --weak-tasks click-checkboxes-large find-word enter-text-2   --weak-train-size 500   --other-train-size 500   --seed 3407   --val-mode full
```

The final tracked manifest for this dataset is in:
- `data/exports/phase1_sft_v2_action_weak3_mixed50_1000/manifest.json`

## 7. Run the final mixed continuation SFT

```bash
bash scripts/reproduce_qwen35_08b_mixed_best.sh
```

Equivalent core command:

```bash
python3 scripts/train_qwen35_continuation_sft.py   --dataset-dir data/exports/phase1_sft_v2_action_weak3_mixed50_1000/hf_dataset   --base-model Qwen/Qwen3.5-0.8B   --adapter-dir outputs/qwen35-0.8b-browser-action-unsloth   --output-dir outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft   --max-length 2048   --num-train-epochs 2.0   --per-device-train-batch-size 4   --gradient-accumulation-steps 4   --learning-rate 1e-4   --seed 3407
```

Then evaluate:

```bash
python3 scripts/eval_action_model.py   --dataset-dir data/exports/phase1_sft_v2/action_only/hf_dataset   --model-name Qwen/Qwen3.5-0.8B   --adapter-dir outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft   --output-json outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft/eval_after_conditional_256.json   --max-new-tokens 256
```

Expected result:
- parseable: 100.00%
- exact-match: 83.33%

## 8. Why the Qwen3.5 evaluation path matters

Do not evaluate Qwen3.5 adapters through the old causal-LM-only path.

The correct evaluator is already implemented in `scripts/eval_action_model.py`, which now auto-selects the conditional-generation loader for Qwen3.5 models and adapters.

That fix is required for valid reproduction of the 0.8B and 2B Qwen3.5 results.

## 9. Reference artifacts

### Key manifests
- `data/exports/phase1_sft_v2/manifest.json`
- `data/exports/phase1_sft_v2_action_weak3_exact1000/manifest.json`
- `data/exports/phase1_sft_v2_action_weak3_mixed50_1000/manifest.json`
- `data/exports/phase1_sft_v3/reasoning_action_reinforced/manifest.json`

### Key eval artifacts
- `outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json`
- `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft/eval_after_conditional_256.json`
- `outputs/qwen35-2b-browser-action-unsloth/` tracked summary files and notes in `docs/`

## 10. What is not required for the main conclusion

You do not need to rerun GRPO to reproduce the central project takeaway.

The main finding is already visible from the SFT path:
- action-only works reliably
- mixed weak-task continuation improves the 0.8B line
- the final small-model checkpoint is the 0.8B mixed continuation at 83.33%
