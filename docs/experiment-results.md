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

After fine-tuning with the original saved eval (too-small generation budget):
- Parseable action rate: 83.75%
- Exact-match action accuracy: 69.58%

After fine-tuning with corrected high-budget eval (`max_new_tokens=512`):
- Parseable action rate: 100.00%
- Exact-match action accuracy: 81.67%
- Canonicalized local equivalence accuracy: 83.75%

### Interpretation
- The original 69.58% score materially undercounted the model because long reasoning traces were truncated by the evaluation budget.
- After fixing truncation, parser strictness becomes a small residual issue rather than the main problem.
- The grounded numbers to keep are 81.67% strict exact-match and 83.75% canonicalized local equivalence on 240 validation examples.
- Remaining misses are mostly genuine action-selection failures concentrated in checkbox-heavy families, especially `click-checkboxes-large` and then `click-checkboxes-transfer`.

### Model artifact
- Local artifact dir: `outputs/qwen25-1.5b-browser-reasoning-unsloth`
- Corrected high-budget eval rows: `outputs/qwen25-1.5b-browser-reasoning-unsloth/eval_after_512.json`

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

### Post-train status (debugged)
- Training finished successfully and adapter artifacts are present in `outputs/qwen35-0.8b-browser-action-unsloth`.
- The originally saved post-train eval at 17.92% exact-match was invalid.
- Root cause: the old shared eval path loaded this model through `AutoModelForCausalLM`, while the adapter was trained against the `Qwen3_5ForConditionalGeneration` / processor-backed path. Under the wrong path, PEFT emitted missing-adapter-key warnings and the adapted model behaved like baseline.
- Corrected post-train eval with the proper conditional-generation loader: parseable 100.00%, exact-match 80.83% on 240 validation rows.
- Direct prompt-by-prompt comparison confirmed that the corrected loader exposes learned behavior immediately; for example, several baseline `click(..., modifiers=[Control])` outputs become plain `click(...)` after adaptation.
- Conclusion: Qwen3.5-0.8B fine-tuning worked. The failure was in evaluation plumbing, not in optimization.


## Model fine-tuning experiment 4

### Setup
- Base model: `Qwen/Qwen3.5-0.8B`
- Dataset: `data/exports/phase1_sft_v2/reasoning_action/hf_dataset`
- Method: Unsloth LoRA
- Output dir: `outputs/qwen35-0.8b-browser-reasoning-unsloth`
- Max length: 4096
- Validation rows: 240

### Training result
- Training finished successfully.
- Final step: 407 / 407
- Best checkpoint: `outputs/qwen35-0.8b-browser-reasoning-unsloth/checkpoint-400`
- Best eval loss: 0.03082
- Final logged train loss near the end: 0.19615

### Before/after evaluation
Before fine-tuning:
- Parseable action rate: 100.00%
- Exact-match action accuracy: 10.83%
- Loader: `conditional_generation`
- Eval budget: 512 generation tokens

After fine-tuning with strict parser scoring and corrected loader:
- At 512 generation tokens: parseable 11.25%, exact-match 8.75%
- At 1536 generation tokens: parseable 11.25%, exact-match 8.75%
- Loader: `conditional_generation`

### Semantic judging
- OpenRouter judging was stopped because the key hit spending limits.
- A full local judge pass with the local `Qwen3.5-9B-judge` server on the 1536-token eval file estimated 60.00% action-equivalent outputs.
- This suggests many outputs are semantically near the right action but are not emitted in valid BrowserGym action syntax.

### Interpretation
- Increasing `max_new_tokens` from 512 to 1536 did not recover the parser score, so truncation is not the primary cause of failure here.
- The dominant post-train failure is format collapse: the model often explains the next action in prose but omits the final executable BrowserGym action line.
- Because baseline and post-train evals both used the corrected conditional-generation loader, this does not look like the old Qwen3.5 action-only eval-path bug.
- The current reasoning-action adapter therefore appears to have learned partial task semantics more than the output contract, making it weak for one-step action execution.
- Immediate next work should focus on enforcing final-action emission and inspecting the reasoning-action export/format, not on scaling this run.

### Prompt-enforcement ablation
- A 40-row inference-only ablation tested whether prompt shaping alone could recover executable outputs from the reasoning-action adapter.
- Baseline on that 40-row sample: parseable 7.50%, exact-match 7.50%.
- Reinforced prompt with a strict format rule plus one worked `<think>...</think>` → final action example: parseable 100.00%, exact-match 67.50%.
- This indicates the adapter's internal task knowledge is much better than the default strict eval score suggests; the main bottleneck is prompt-sensitive action-format emission.
- Artifact: `outputs/qwen35-0.8b-browser-reasoning-unsloth/prompt_enforcement_ablation_40.json`

## Model fine-tuning experiment 5

### Setup
- Base model: `Qwen/Qwen3.5-0.8B`
- Source dataset: `data/exports/phase1_sft_v2/reasoning_action/hf_dataset`
- New dataset variant: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Method: Unsloth LoRA
- Output dir: `outputs/qwen35-0.8b-browser-reasoning-reinforced-unsloth`
- Max length: 4096
- Process: `proc_b4a7b2545e93`

### Dataset change
- Added a stricter reasoning-action system instruction requiring exactly one `<think>...</think>` block and a final non-empty BrowserGym action line.
- Added one worked in-context example showing the required reasoning-to-action shape.
- Added a short user-side reminder repeating the exact output format.
- Preserved the original assistant targets so the ablation isolates prompt-format changes rather than target rewriting.

### Why this ablation exists
- The prior Qwen3.5-0.8B reasoning-action run finished training but mostly emitted prose-only answers at inference time.
- A 40-row inference-only prompt-enforcement ablation improved that finished adapter from 7.50% parseable / 7.50% exact-match to 100.00% parseable / 67.50% exact-match under a reinforced prompt.
- This suggests the model retains useful task knowledge but is highly sensitive to prompt format.

### Post-train evaluation
- Default reasoning prompt, corrected conditional-generation loader, 1536-token budget: parseable 10.00%, exact-match 3.75% on 240 validation rows.
- Reinforced evaluation prompt with strict format rule + one-shot example + user reminder, same loader and same 1536-token budget: parseable 92.92%, exact-match 66.67% on 240 validation rows.

### Interpretation
- The reinforced training run did not make the model robust under the default reasoning prompt.
- It did, however, preserve a large amount of useful behavior when the inference prompt reinforces the final-action contract.
- Compared with the prior reasoning-action run, this ablation demonstrates that prompt-format alignment matters at both train time and inference time, but train-time reinforcement alone was not sufficient to remove prompt sensitivity.
- The model remains strongly conditional on explicit output-shape scaffolding.
- A few reinforced-prompt outputs still omit the opening `<think>` while emitting the correct final action, so the model is learning the action-line requirement more strongly than the full XML-like wrapper.

## Model fine-tuning experiment 6

### Setup
- Base model: `Qwen/Qwen3.5-2B`
- Dataset: `data/exports/phase1_sft_v2/action_only/hf_dataset`
- Method: Unsloth LoRA
- Output dir: `outputs/qwen35-2b-browser-action-unsloth`
- Max length: 2048
- Process: `proc_f88e7d47f6b9`

### Baseline evaluation
- Parseable action rate: 100.00%
- Exact-match action accuracy: 58.33%
- Loader: `conditional_generation`
- Validation rows: 240
- Eval budget: 256 generation tokens

### Interpretation
- Qwen3.5-2B starts from a far stronger action-only baseline than Qwen3.5-0.8B.
- This makes it the right next capacity ablation before deciding whether reasoning-action failure at 0.8B was mostly a size issue or mostly a format issue.

### Training result
- Training finished successfully.
- Final train loss: 0.2382
- Final eval loss: 0.04981

### Before/after evaluation
- Baseline: parseable 100.00%, exact-match 58.33% on 240 validation rows.
- Post-train: parseable 100.00%, exact-match 87.50% on 240 validation rows.
- Loader: `conditional_generation`
- Eval budget: 256 generation tokens

### Interpretation
- Qwen3.5-2B starts from a much stronger base than 0.8B and fine-tunes cleanly on action-only.
- This is now the strongest small-model action-only result in the current line of work.
- Qwen3.5-2B looks like the right next base for a reinforced reasoning-action follow-up.

## Model fine-tuning experiment 7

### Setup
- Base model: `Qwen/Qwen3.5-2B`
- Source eval baseline: `data/exports/phase1_sft_v2/reasoning_action/hf_dataset`
- Training dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Method: Unsloth LoRA
- Output dir: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Max length: 4096
- Process: `proc_26665d1b1e2d`

### Baseline evaluation
- Default reasoning prompt baseline at 1536 generation tokens: parseable 96.67%, exact-match 53.33% on 240 validation rows.
- Loader: `conditional_generation`

### Interpretation
- Qwen3.5-2B starts from a far stronger reasoning baseline than 0.8B, suggesting capacity likely matters materially for reasoning-action behavior.
- This makes it a much better test of whether the reinforced prompt format can be internalized without extreme prompt fragility.

### Training / relaunch status
- The first training attempt failed at the first eval/checkpoint boundary around step 100 with a CUDA driver error followed by open-file exhaustion during cleanup.
- To remove that failure path, the trainer was patched to allow disabling mid-training eval and save.
- The run was relaunched with raised open-file limit, no mid-training eval, and no mid-training checkpoint saves.
- The relaunched training finished successfully and final adapter artifacts are present in `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`.

### Post-train evaluation
- Default reasoning prompt, corrected conditional-generation loader, 1536-token budget: parseable 41.67%, exact-match 29.17% on 240 validation rows.
- Reinforced reasoning prompt with strict format rule + one-shot example + user reminder, same loader and same 1536-token budget: parseable 95.42%, exact-match 84.17% on 240 validation rows.

### Failure inspection
- On the default prompt, 170 / 240 rows were non-matches and 140 of those were unparseable under the strict BrowserGym parser.
- The largest failure buckets were prose-only action descriptions (81 rows) and other natural-language completions without a valid final action line (59 rows).
- A smaller but real residual bucket emitted near-correct non-canonical syntax such as `click(bid='18')` (14 rows).
- Genuine action-selection mistakes still appear on checkbox-heavy tasks, but they are no longer the dominant error mode under the default prompt.

### Interpretation
- Relative to the default-prompt baseline (96.67% parseable, 53.33% exact), reinforced reasoning training did not improve plain default-prompt robustness; it made it worse.
- Relative to the 0.8B reinforced run, however, the 2B model is much stronger when inference uses the reinforced contract, reaching 84.17% exact-match instead of 66.67%.
- The main conclusion is that scaling to 2B improved recoverable task knowledge, but did not solve prompt-conditional formatting collapse.
- This remains a format-internalization problem, not a simple lack-of-capacity problem.

## Qwen3.5-2B RL stack validation (2026-03-23)

### Root-cause summary
- Reproduced the Qwen3.5-2B GRPO failure inside `apply_rotary_pos_emb`.
- The underlying cause was stale `rope_deltas` state in `Qwen3_5Model` during GRPO's generation-to-logprob transition, which produced zero-batch RoPE tensors.

### Fix
- Added a guard in both RL training scripts to clear or batch-align stale `rope_deltas` before `compute_3d_position_ids` reuses them.

### Validation after fix
- One-step GRPO smoke completed cleanly with action adapter `outputs/qwen35-2b-browser-action-unsloth`.
- One-step GRPO smoke also completed cleanly with reasoning adapter `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`.
- Multi-turn GRPO smokes completed cleanly for both the action and reinforced-reasoning adapters.

### Interpretation
- This removes the main infrastructure blocker for trying Qwen3.5-2B in RL.
- The remaining open question is now experimental rather than infrastructural: whether Qwen3.5-2B action or reasoning warm starts produce better reward variance and downstream policy improvement on harder tasks.

