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

## 2026-03-23 Qwen3.5-2B action-only small multi-turn RL run

- Ran a short action-only multi-turn RL check with the fixed Qwen3.5-2B path using `configs/grpo_multiturn_qwen35_2b_action_phase_small.yaml`.
- Startpoint: `outputs/qwen35-2b-browser-action-unsloth`
- Task subset: `enter-password`, `click-option`, `enter-text-2`
- Sampling settings: `rollout_temperature=0.9`, `rollout_top_p=0.95`
- Runtime: ~62.7s for 12 steps.
- The run completed cleanly with no Qwen3.5 rotary / rope-delta crash.
- Observed reward stayed in a narrow band around 3.08 to 3.12.
- `reward_std` remained exactly 0.0 throughout the run.
- Interpretation: this small curriculum is still too saturated / tied for the Qwen3.5-2B action-only warm start, so GRPO is not receiving usable preference signal even with higher rollout temperature.
- Current recommendation: move to a harder action-only curriculum centered more aggressively on unsaturated tasks instead of mixing in already-near-solved tasks like `click-option` and `enter-text-2`.

## 2026-03-23 Qwen3.5-2B action-only hard-task multi-turn RL run

- Ran a harder action-only multi-turn RL check with `configs/grpo_multiturn_qwen35_2b_action_phase_hard.yaml`.
- Startpoint: `outputs/qwen35-2b-browser-action-unsloth`
- Task subset: `click-checkboxes-large`, `click-checkboxes-transfer`, `find-word`
- Sampling settings: `rollout_temperature=0.9`, `rollout_top_p=0.95`
- Runtime: ~124.7s for 12 steps.
- The run completed cleanly with no Qwen3.5 rope-delta / rotary crash.
- This harder subset finally produced non-zero GRPO preference signal on multiple steps.
- Notable batches:
  - step 1-equivalent log point: `reward≈1.71`, `reward_std≈1.994`
  - later strong batch: `reward≈1.81`, `reward_std≈2.135`
  - late weak batches still collapsed, including `reward≈0.4`, `reward_std=0.0` and `reward≈0.35`, `reward_std≈0.0707`
- Interpretation: this is meaningfully better than the earlier saturated action-only subset. The Qwen3.5-2B action warm start can produce useful GRPO signal when the curriculum is centered on harder tasks.
- Current read: the fixed Qwen3.5-2B RL path is now both infrastructurally stable and experimentally promising on the harder action-only subset, though reward variance is still intermittent rather than consistently strong across the entire run.
- Recommended next move: either scale this same harder action-only curriculum modestly, or run the matched reinforced-reasoning warm start on the same task subset for a clean comparison.

## 2026-03-23 Qwen3.5-2B reinforced-reasoning hard-task multi-turn RL run

- Ran the matched reinforced-reasoning multi-turn RL comparison with `configs/grpo_multiturn_qwen35_2b_reasoning_phase_hard.yaml`.
- Startpoint: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Task subset matched the action-only comparison exactly: `click-checkboxes-large`, `click-checkboxes-transfer`, `find-word`
- Sampling settings matched the action-only run: `rollout_temperature=0.9`, `rollout_top_p=0.95`
- Runtime: ~199.5s for 12 steps, noticeably slower than the matched action-only run (~124.7s).
- The run completed cleanly with no Qwen3.5 rope-delta / rotary crash.
- This run did produce non-zero GRPO signal on multiple batches, with strong peaks:
  - `reward≈1.45`, `reward_std≈2.305`
  - `reward≈1.38`, `reward_std≈2.15`
- However, the run also showed substantially less stable behavior than the matched action-only warm start:
  - completion lengths were much longer, often 100+ tokens
  - several batches hit near-max-length generations and clipped completions
  - multiple batches had negative mean reward, including about `-0.16`, `-0.18`, and `-0.14`
  - one late batch still collapsed to `reward_std=0`
- Comparison read:
  - action-only hard run: slightly weaker peak variance, but cleaner / faster / more stable reward profile
  - reinforced-reasoning hard run: can produce strong variance, but pays for it with much longer completions, clipping, slower runtime, and more negative-reward instability
- Current recommendation: for RL continuation, prefer the Qwen3.5-2B action-only warm start over the reinforced-reasoning warm start on this stack.

## 2026-03-23 Qwen3.5-2B action-only hard-subset scaled RL run

- Ran a scaled continuation with `configs/grpo_multiturn_qwen35_2b_action_phase_hard_scaled.yaml`.
- Startpoint: `outputs/qwen35-2b-browser-action-unsloth`
- Task subset stayed fixed: `click-checkboxes-large`, `click-checkboxes-transfer`, `find-word`
- Scaling changes relative to the earlier hard run:
  - `samples_per_task: 8` (up from 4)
  - `max_steps: 24` (up from 12)
  - same sampling: `rollout_temperature=0.9`, `rollout_top_p=0.95`
- Runtime: ~183.3s for 24 steps.
- The run completed cleanly with no Qwen3.5 rope-delta / rotary failure.
- Signal quality remained mixed but real:
  - strongest late batch reached `reward≈1.9`, `reward_std≈2.121`
  - several other batches still collapsed to tied rewards with `reward_std=0`
  - examples of collapsed high-reward batches: `reward≈3.08`, `3.16`, `3.04` with `reward_std=0`
  - examples of collapsed weak batches: `reward≈0.08`, `0.12` with `reward_std=0`
  - there were also a few low-but-nonzero-variance batches like `reward≈0.35`, `reward_std≈0.0707`
- Interpretation: scaling the action-only hard subset did not destroy the useful RL signal; it preserved intermittent strong GRPO batches. But the reward landscape is still alternating between informative batches and fully tied batches.
- Current read: the Qwen3.5-2B action-only hard subset remains the best RL continuation path so far, but it still needs curriculum or reward refinement if we want more consistently nonzero `reward_std` across the run.

## 2026-03-23 Qwen3.5-2B action-only task-signal replay analysis

- Ran a focused replay analysis using the current Qwen3.5-2B action-only warm start to estimate task-level reward variance under the same multi-turn action-only policy style.
- Analysis artifact: `outputs/qwen35-2b-browser-action-grpo-task-signal-analysis.json`
- Candidate tasks checked:
  - `click-checkboxes-large`
  - `click-checkboxes-transfer`
  - `find-word`
  - `enter-password`
  - `enter-text-2`
  - `click-option`
- Summary from the replay analysis:
  - `enter-text-2`: strongest dense signal among the checked tasks, with `avg_reward_std≈0.765`, `max_reward_std≈1.53`, and non-zero variance on 3/4 prompts.
  - `enter-password`: also strong, with `avg_reward_std≈0.448`, `max_reward_std≈1.5`, and non-zero variance on 3/4 prompts.
  - `click-checkboxes-large`: moderate signal, `avg_reward_std≈0.088`, non-zero variance on 4/4 prompts, but weaker than the two text-entry tasks.
  - `find-word`: non-zero variance on 4/4 prompts with `avg_reward_std≈0.197`, but low mean reward (`avg_reward≈0.1025`), suggesting it is hard in a less productive way right now.
  - `click-checkboxes-transfer`: weak signal, `avg_reward_std≈0.0375`, only 2/4 prompts with non-zero variance, and high average reward (`avg_reward≈3.10`), so it is already close to saturation.
  - `click-option`: weakest and most saturated, `avg_reward_std≈0.0125`, only 2/4 prompts with non-zero variance, and high average reward (`avg_reward≈3.10`).
- Interpretation:
  - The earlier hard-subset run was directionally useful, but the best RL signal is not actually centered on the checkbox-transfer task family alone.
  - The best next curriculum should be anchored on `enter-text-2` and `enter-password`, with `click-checkboxes-large` as the third task if we want diversity.
  - `click-option` should be dropped from further RL curricula for now.
  - `click-checkboxes-transfer` should be deprioritized relative to `click-checkboxes-large`.
  - `find-word` is worth revisiting later, but right now it looks more like low-reward difficulty than high-leverage signal.
- Refined recommendation: next action-only RL curriculum should be `enter-text-2`, `enter-password`, and `click-checkboxes-large`.

## 2026-03-23 Qwen3.5-2B action-only refined-curriculum RL run

- Ran the refined action-only curriculum suggested by the task-signal replay analysis with `configs/grpo_multiturn_qwen35_2b_action_phase_refined.yaml`.
- Startpoint: `outputs/qwen35-2b-browser-action-unsloth`
- Refined task mix: `enter-text-2`, `enter-password`, `click-checkboxes-large`
- Sampling settings matched the earlier action runs: `rollout_temperature=0.9`, `rollout_top_p=0.95`
- Runtime: ~191.4s for 24 steps.
- The run completed cleanly with no Qwen3.5 rope-delta / rotary failure.
- Contrary to the replay-analysis expectation, the live RL run was mostly saturated / tied:
  - most logged batches had `reward_std = 0`
  - many high-reward tied batches around `reward≈3.08`, `3.12`, `3.36`
  - several low-reward tied batches around `reward≈0.4`
- There was only a small non-zero variance blip early in the run, e.g. `reward≈3.32`, `reward_std≈0.0566`, which is much weaker than the best signal seen in the hard checkbox/find-word curriculum.
- Interpretation:
  - the replay analysis was directionally useful for identifying candidate tasks, but it overestimated how much dense GRPO signal the refined enter-text / enter-password curriculum would produce under full training dynamics.
  - In actual multi-step GRPO training, this refined mix collapsed more than the harder checkbox/find-word curriculum.
- Revised recommendation after the live test:
  - keep `enter-text-2` and `enter-password` as potentially useful tasks,
  - but do not replace the harder signal-bearing tasks entirely.
  - The best observed live-training signal still came from the harder action-only curriculum centered on `click-checkboxes-large`, with additional support from harder tasks rather than a mostly-text-entry curriculum.

## 2026-03-23 Qwen3.5-2B exact live-signal slice run

- Added explicit `task_seed_pairs_file` support to `scripts/train_browsergym_grpo_multiturn.py` so multi-turn RL runs can be built from exact task+seed slices rather than only broad task families.
- Built a focused action-only run from the exact strongest previously observed live-signal slices:
  - `click-checkboxes-large` @ seed `935000`
  - `click-checkboxes-transfer` @ seed `935004`
  - `find-word` @ seed `937020`
- Repeated those exact slices as a small targeted curriculum using `configs/grpo_multiturn_qwen35_2b_action_signal_slices.yaml` and `configs/qwen35_action_signal_slices.tsv`.
- The run finished cleanly in ~97.8s with no Qwen3.5 rope-delta / rotary failure.
- Result: the exact-slice curriculum did **not** preserve the earlier strong live-signal behavior.
  - almost all batches still collapsed to `reward_std = 0`
  - most rewards were tied high (`~3.08`) or tied low (`~0.4`, `~0.08`)
  - only one small non-zero variance blip appeared (`reward≈0.05`, `reward_std≈0.0424`)
- Interpretation:
  - the earlier strong-signal batches were not explained solely by exact prompt identity.
  - They appear to depend on broader rollout/training dynamics rather than a small set of individually magical prompts.
  - This rules out a simple “just replay the best seeds” strategy as the next scaling path.
- Updated recommendation after this test:
  - stop curriculum-only narrowing as the primary knob,
  - treat reward formulation / generation diversity / group sampling as the likely bottleneck,
  - if continuing RL next, change the signal mechanism rather than only swapping task slices.

## 2026-03-23 Qwen3.5-2B action-only num_generations=4 RL run

- Ran the recommended action-only hard-curriculum check with `num_generations=4` using `configs/grpo_multiturn_qwen35_2b_action_phase_hard_gen4.yaml`.
- Curriculum remained the stronger hard subset:
  - `click-checkboxes-large`
  - `click-checkboxes-transfer`
  - `find-word`
- Key config change relative to the prior hard run:
  - `num_generations: 4`
  - `generation_batch_size: 4`
- Runtime: ~98.3s for 12 steps.
- The run completed cleanly with no Qwen3.5 rope-delta / rotary failure.
- Results:
  - there was a meaningful non-zero variance batch with `reward≈1.835`, `reward_std≈1.657`
  - there was also an earlier small but real non-zero variance batch with `reward≈3.15`, `reward_std≈0.02`
  - however, many batches still collapsed, including tied low-reward and tied high-reward phases.
- Interpretation:
  - raising `num_generations` helped somewhat by restoring at least one clearly useful GRPO batch without changing the curriculum.
  - but it did not solve the main issue; reward variance is still intermittent rather than consistently dense.
- Current recommendation after this test:
  - treat this as partial evidence that generation diversity matters,
  - but also treat it as the point where reward shaping should become the main next knob if we want stronger and more consistent RL signal.

## 2026-03-23 Reward-design analysis for Qwen3.5-2B BrowserGym RL

- Inspected the current one-step and multi-turn reward implementations directly.
- Current one-step reward in `scripts/train_browsergym_grpo.py` is effectively:
  - parseability bonus / penalty
  - immediate environment reward
  - done bonus
  - action-error penalty
- Current multi-turn reward in `scripts/train_browsergym_grpo_multiturn.py` is effectively:
  - per-step penalty
  - parseability bonus / invalid-action penalty
  - per-step environment reward
  - action-error penalty
  - final success bonus
- Empirical finding from repeated action-only runs:
  - strong signal appears occasionally, but many batches remain tied with `reward_std = 0`
  - curriculum changes alone do not reliably fix this
  - exact seed slicing does not reliably reproduce the strongest batches
  - increasing `num_generations` to 4 helps somewhat, but not enough to make the signal consistently dense

### External guidance synthesized
- GRPO/TRL guidance emphasizes composing multiple reward functions, using incremental partial credit, and testing each reward component independently rather than relying on a single coarse scalar.
- Policy-invariant reward shaping literature argues for shaping via progress-style potentials rather than arbitrary bonuses, to speed learning without changing the true optimum.
- Recent web-agent training work highlights sparse feedback and delayed rewards as central challenges, and recommends curriculum plus more informative intermediate signals for multi-step web tasks.

### Diagnosis of the current reward
- The current reward is too coarse and too frequently tied for MiniWoB-style browser tasks.
- Many tasks only separate trajectories at the end state, so two different rollouts often receive the same scalar reward until success/failure is fully resolved.
- Parseability bonus is useful as a floor, but once most completions are parseable it no longer creates within-group ordering.
- Final success bonus is important, but by itself it makes successful short trajectories tie too often.
- This explains why we observe repeated high-reward tied batches (`reward_std = 0`) even when the agent is doing something sensible.

### Recommended reward redesign direction
- Keep the current core reward as the outer scaffold.
- Add task-specific partial-progress shaping so rollouts can diverge before final success.
- The shaping should be progress-based, not just format-based.

### First shaping candidates
1. `enter-text-2`
- Reward progress when the filled textbox moves closer to the target transformed string.
- Example signals:
  - exact typed string match bonus
  - prefix / edit-distance improvement bonus
  - submit only after correct fill bonus

2. `enter-password`
- Reward partial completion by counting how many of the two required fields match the target password.
- Example signals:
  - first field correct
  - second field correct
  - both correct before submit
  - penalty for submitting with only one correct

3. `click-checkboxes-large` / checkbox tasks
- Reward overlap between currently selected checkboxes and target set.
- Example signals:
  - + for each newly correct checkbox selected
  - - for each incorrect checkbox selected
  - + bonus when submit is used with the exact target set

4. `find-word`
- Reward the typed textbox content if it moves closer to the target extracted word.
- Example signals:
  - exact answer in textbox
  - prefix / normalized string similarity gain
  - submit with correct textbox bonus

### Design principle for implementation
- Compute a progress score from observation state after each action.
- Use reward based on delta-progress between consecutive states, not only absolute final state.
- This should create more pairwise separation between sampled rollouts inside the same GRPO group.

### Recommended next implementation
- Implement a minimal modular task-shaping layer in the multi-turn reward path for the three best action-only tasks:
  - `enter-text-2`
  - `enter-password`
  - `click-checkboxes-large`
- Keep the existing base reward components, then add a bounded task-specific progress delta term.
- Validate with a short action-only run before any broader scaling.

## 2026-03-23 Reward-shaping implementation and validation

- Implemented the first task-specific progress-shaping layer in `scripts/train_browsergym_grpo_multiturn.py`.
- Added a shared shaping scaffold on top of the existing base reward:
  - `progress_shaping_scale`
  - `premature_submit_penalty`
- Added task-family-specific progress estimators for:
  - `enter-text-2`
  - `enter-password`
  - `click-checkboxes-*`
- Progress is computed as a bounded state score and applied via delta shaping:
  - `reward += progress(next_state) - progress(prev_state)`
- Also added a small premature-submit penalty when the agent clicks Submit before the task-specific progress is near complete.

### Validation run
- Validation config: `configs/grpo_multiturn_qwen35_2b_action_phase_refined_shaped.yaml`
- Task mix:
  - `enter-text-2`
  - `enter-password`
  - `click-checkboxes-large`
- Runtime: ~184.1s for 24 steps.
- The run completed cleanly.

### Result versus the previous unshaped refined run
- Unshaped refined run was mostly tied, with only a tiny non-zero variance blip (`reward_std≈0.0566`).
- The shaped run produced multiple clearly non-zero variance batches:
  - early strong batch: `reward≈2.447`, `reward_std≈2.31`
  - later strong batch: `reward≈2.685`, `reward_std≈1.994`
  - additional medium-signal batches around `reward_std≈0.226`, `0.129`, and `0.064`
- Some tied batches still remain, especially in high-reward solved phases, but the signal is now substantially denser than before.

### Interpretation
- This is the first clean validation that reward redesign materially improves within-group separation for the Qwen3.5-2B action-only RL path.
- The shaping does not eliminate tied batches, but it turns the previously almost-dead refined curriculum into a run with multiple useful GRPO batches.
- Current read: reward shaping was the right next knob, and it helps more than additional curriculum-only tuning did.

### Recommended next step
- Keep the shaping layer and extend it carefully.
- Re-run the stronger action-only curriculum under the shaped reward to see whether we can combine:
  - the better task mix from the harder curriculum
  - with the denser signal from task-specific progress shaping.

## 2026-03-23 Reward-shaped hard-curriculum validation

- Ran the stronger action-only hard curriculum under the new progress-shaped reward using `configs/grpo_multiturn_qwen35_2b_action_phase_hard_shaped.yaml`.
- Task mix:
  - `click-checkboxes-large`
  - `click-checkboxes-transfer`
  - `find-word`
- Runtime: ~94.6s for 12 steps.
- The run completed cleanly.
- Result:
  - shaping produced a strong non-zero variance batch with `reward≈2.401`, `reward_std≈2.044`
  - but many other batches still collapsed to tied rewards with `reward_std=0`
  - several tied high-reward shaped batches reached roughly `4.02`, `4.14`, and `4.18`
- Interpretation:
  - reward shaping does help on the harder curriculum too, not just on the refined text-entry curriculum.
  - however, the gain is still intermittent rather than uniform.
  - The current shaping layer improves the signal landscape, but does not yet fully solve the tied-reward problem across the harder action-only mix.
- Current read:
  - shaping is worth keeping,
  - but the next refinement likely needs either stronger task-specific shaping for the checkbox and extraction tasks, or a better combination of shaping + generation diversity.

## 2026-03-23 Qwen3.5-0.8B action-only weak-task failure analysis

- Pivoted back to Qwen3.5-0.8B action-only because it has a much larger improvement margin to demonstrate than Qwen3.5-2B.
- Verified the correct post-SFT file for 0.8B action-only is `outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json`.
- Real summary for 0.8B action-only:
  - baseline exact-match: 17.92%
  - post-SFT exact-match: 80.83%

### Weakest post-SFT tasks
- `click-checkboxes-large`: after 0.386, before 0.088, delta +0.298, n=57
- `find-word`: after 0.750, before 0.000, delta +0.750, n=8
- `enter-text-2`: after 0.786, before 0.000, delta +0.786, n=14
- `click-checkboxes-transfer`: after 0.870, before 0.304, delta +0.565, n=23
- `enter-password`: after 0.917, before 0.167, delta +0.750, n=24

### Failure-mode read
1. `click-checkboxes-large`
- This is the highest-leverage weak task.
- The target is a sequence of checkbox clicks ending with Submit.
- Failure pattern: the model often locks onto a single visible checkbox and repeats a positional shortcut rather than selecting the correct target subset in sequence.
- Example failures show systematic wrong-click bias such as repeatedly predicting `click('27')` when the correct next action is a different target checkbox.
- Interpretation: this is not just formatting failure; it is a state-tracking / set-progress failure.

2. `find-word`
- Failure pattern is usually semantic extraction error rather than formatting.
- The model fills the textbox with the wrong word from the paragraph, e.g. `non` instead of `interdum`, or `sem` instead of `nec`.
- The action form itself is correct; the content is wrong.
- Interpretation: reward should focus on textbox content closeness to the target extracted word.

3. `enter-text-2`
- Failure pattern is mostly near-miss transformation error.
- Example: `fill('14', 'kase')` instead of `fill('14', 'kasie')`.
- Another miss is only a quoting/canonicalization difference, and another is equivalent submit formatting (`click('15')` vs `click('15', button='left')`).
- Interpretation: most remaining real headroom is in exact transformed text fidelity, not basic action formatting.

### Reward-design implications
- `click-checkboxes-large` should get the strongest task-specific shaping first.
  - Reward overlap between selected checkbox set and target set.
  - Reward positive delta when a newly selected checkbox belongs to the target set.
  - Penalize selecting non-target boxes.
  - Penalize premature submit unless the selected set matches the target set.
- `find-word` should use textbox-content shaping.
  - Reward textbox string similarity / exact match to the target word.
  - Reward improvement in textbox content after fill.
  - Penalize submit before the textbox matches the target.
- `enter-text-2` should use transformed-text shaping.
  - Reward textbox similarity to the target transformed string.
  - Reward exact fill strongly.
  - Treat submit before exact match as premature.

### Recommended next focus subset for 0.8B
- Primary 3-task subset:
  - `click-checkboxes-large`
  - `find-word`
  - `enter-text-2`
- Optional 4th later:
  - `click-checkboxes-transfer`
- Recommendation: improve task-specific SFT/data for these tasks first, then test shaped rewards on them before broader GRPO.

