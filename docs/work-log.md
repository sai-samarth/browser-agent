# Browser-Agent Work Log

## Major milestones completed

1. Turned the scratch workspace into a proper repo baseline.
2. Added rollout baseline analysis and failure taxonomy.
3. Defined a Phase 1 core task subset and a production task subset.
4. Added rollout diagnostics for sparse observations and repeated loops.
5. Set up local vLLM serving for `Qwen/Qwen3.5-9B` on the RTX 4090.
6. Validated local smoke runs and fixed local action-format issues.
7. Ran repeated local production collection campaigns.
8. Exported clean SFT-ready datasets in action-only and reasoning+action formats.
9. Uploaded both datasets to Hugging Face.
10. Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` with PEFT LoRA and evaluated before/after.

## Important fixes we made

### 1. Numeric bid normalization
Local teacher sometimes emitted unquoted BrowserGym bids such as:
- `focus(12)`
- `fill(15, '8')`

We normalized them to:
- `focus('12')`
- `fill('15', '8')`

This fixed local smoke failures.

### 2. Bare fill-value normalization
Local teacher also sometimes emitted:
- `fill('31', Taoism)`

We normalized that to:
- `fill('31', 'Taoism')`

This fixed `read-table`-type failures.

### 3. Export prompt correction
The first SFT export accidentally carried over generation-time think-format instructions into the action-only system prompt.
We corrected the exporter to:
- reuse the detailed generation prompt content
- strip generation-only output formatting instructions
- append a small variant-specific suffix for action-only vs reasoning+action

## Notes for future experiments

- Action-only is the recommended default training format for small models.
- Reasoning+action should be treated as an ablation.
- Canonicalized evaluation would likely improve apparent exact-match scores by treating formatting-equivalent actions as equal.
- Weak task families worth separate tracking include `identify-shape`, `click-dialog-2`, and `navigate-tree`.
- A proper Unsloth environment is the next infra task for follow-up SFT comparisons.

## Follow-up fine-tuning work

11. Established an Unsloth fine-tuning environment for small-model BrowserGym SFT comparisons.
12. Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` on the reasoning+action dataset.
13. Identified that low `max_new_tokens` in evaluation was undercounting long reasoning outputs.
14. Re-ran reasoning-model evaluation with a larger generation budget and recovered the true score.
15. Re-ran the reasoning-action model with a corrected high eval budget and confirmed that most of the earlier undercount came from low `max_new_tokens`, not parser failure.
16. Logged the corrected reasoning-model result at 81.67% strict exact-match and 83.75% canonicalized local equivalence on 240 validation examples.
17. Broke down the remaining misses and found they are now mostly real checkbox-planning failures rather than evaluation artifacts.
18. Measured `Qwen/Qwen3.5-0.8B` action-only baseline on the shared evaluation path before fine-tuning.

## Current active experiment

- `Qwen/Qwen3.5-0.8B` action-only baseline is complete at 17.92% exact-match on 240 validation examples with 100% parseable outputs.
- Baseline failures are mostly structurally consistent: many unnecessary `modifiers=[Control]` clicks, some checkbox-task target mistakes, and a few harmless `button=left` omissions.
- The first Qwen3.5 Unsloth training attempt failed before optimization because the `Qwen3VLProcessor` path interpreted the prompt string incorrectly during tokenization.
- Root cause: using the processor call directly in the dataset map produced an incompatible batched/multimodal-shaped token structure for single-example text-only SFT.
- Confirmed fix: render prompts with `apply_chat_template(...)`, then tokenize with the underlying text tokenizer (`processor.tokenizer`) rather than the VL processor wrapper.
- A smoke test on real training rows now yields sane non-pad token counts (~573) with no image-source error.
- Next step is action-only Unsloth fine-tuning into `outputs/qwen35-0.8b-browser-action-unsloth`.
- Notes/results should continue to be written into `docs/experiment-results.md` and `docs/work-log.md` for every experiment update.


## 2026-03-20 session continuity update

- Added `docs/current-status.md` as the canonical short status file for active runs and latest validated metrics.
- Qwen2.5-1.5B reasoning-action Unsloth run completed.
- Reasoning-action metrics after correcting eval budget:
  - saved low-budget eval: parseable 83.75%, exact 69.58%
  - 256-token eval: parseable 99.58%, exact 81.25%
  - 512-token eval: parseable 100.00%, exact 81.67%
  - canonicalized local equivalence at 512 tokens: 83.75%
- Conclusion: the earlier 69.58% score was depressed by generation truncation; after fixing that, the remaining gap is mostly real checkbox-planning error.
- Active monitored run is now Qwen3.5-0.8B action-only Unsloth (`proc_8d6ef4acc03b`) on `outputs/qwen35-0.8b-browser-action-unsloth`.
- Project hygiene rule: keep `current-status.md`, `work-log.md`, and `experiment-results.md` updated so experiment state does not depend on chat memory.

- Re-read `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` before continuing, because project-state confusion had accumulated.
- Confirmed the active debugging target is `Qwen/Qwen3.5-0.8B` action-only only; no paid-judge or unrelated model work should continue from here.
- Confirmed the saved Qwen3.5-0.8B post-train eval is still 17.92% exact-match, identical to baseline despite finished training and present adapter artifacts.
- Compared saved baseline vs post-train samples and found identical `modifiers=[Control]` click behavior on the same examples, so the current failure does not look like a simple low-`max_new_tokens` truncation artifact.
- Next debugging goal is to determine whether the eval path is ignoring the adapter, whether the adapter failed to train effectively despite nontrivial loss movement, or whether Qwen3.5-0.8B needs model-specific inference configuration to expose the learned behavior.

- Debugged the Qwen3.5-0.8B post-train mismatch by comparing generation behavior across loading paths.
- Found the key architectural mismatch: adapter config targets `Qwen3_5ForConditionalGeneration`, but the old eval used `AutoModelForCausalLM`.
- Under the wrong causal-LM path, PEFT emitted large missing-adapter-key warnings and baseline/adapted generations stayed identical.
- Under the correct conditional-generation path using the processor-backed interface, adapted generations changed immediately on the same validation prompts (for example removing the unnecessary `modifiers=[Control]` clicks seen in baseline outputs).
- Re-ran the full 240-row action-only validation eval with the corrected conditional-generation loader and recovered the real Qwen3.5-0.8B post-train score: 100.00% parseable, 80.83% exact-match.
- Conclusion: the earlier saved 17.92% post-train score was an evaluation bug, not a failed fine-tune.

- Fixed the shared `scripts/eval_action_model.py` so it chooses the proper loader path: causal-LM for standard text models and conditional-generation / processor-backed loading for `Qwen/Qwen3.5-0.8B`.
- Validated the fixed shared eval on smoke runs for both Qwen2.5-1.5B and Qwen3.5-0.8B; the Qwen3.5 path now reports `loader=conditional_generation` and no longer falls back to the broken causal-LM adapter path.
- Measured `Qwen/Qwen3.5-0.8B` on the reasoning-action validation split with the fixed shared eval: parseable 100.00%, exact 10.83% on 240 rows at `max_new_tokens=512`.
- Started Qwen3.5-0.8B reasoning-action Unsloth fine-tune on `outputs/qwen35-0.8b-browser-reasoning-unsloth` with process `proc_d740bb1d5398`, `max_length=4096`, 1 epoch, LoRA, and 4-bit NF4.

## 2026-03-21 reasoning-action completion update

- Confirmed the Qwen3.5-0.8B reasoning-action Unsloth run (`proc_d740bb1d5398`) finished successfully; `checkpoint-407` and final adapter artifacts are present under `outputs/qwen35-0.8b-browser-reasoning-unsloth`.
- Re-ran post-train evaluation with the corrected shared eval path at both 512 and 1536 generation tokens using the same conditional-generation / processor-backed loader as the baseline.
- Result stayed unchanged on 240 validation rows: parseable 11.25%, exact 8.75%.
- This rules out low `max_new_tokens` truncation as the main reason for the weak strict parser score.
- Inspected raw generations and found the dominant failure mode is still prose-only answers such as “I need to click the submit button” without the final BrowserGym action string like `click('18')`.
- Switched off OpenRouter judging because the key hit spending limits.
- Ran a full local semantic judge pass with the local `Qwen3.5-9B-judge` server on all 240 rows; estimated action-equivalent rate was 60.00%.
- Practical conclusion: the reasoning-action run completed, but it currently degrades executable output formatting badly enough that higher generation budget does not recover the strict metric.

- Ran a 40-row prompt-enforcement ablation on the completed Qwen3.5-0.8B reasoning-action adapter.
- Added a stronger system rule plus a one-shot example showing `<think>...</think>` followed by a final BrowserGym action line.
- The same adapter improved from parseable/exact 7.50% / 7.50% on that sample to 100.00% parseable and 67.50% exact-match under the reinforced prompt.
- This is strong evidence that the model still contains usable action knowledge and that the dominant failure is prompt-sensitive output formatting.
- Saved comparison artifact: `outputs/qwen35-0.8b-browser-reasoning-unsloth/prompt_enforcement_ablation_40.json`

## 2026-03-21 reinforced reasoning-action ablation launch

- Built a fresh dataset variant at `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset` from the original reasoning-action export.
- The reinforced variant adds: a stricter system output contract, a one-shot worked example, and a user-side format reminder while preserving the original reasoning+action targets.
- Inspecting a sample confirms the message layout is now: system, example user, example assistant, real user with reminder, real assistant target.
- This design was chosen because the inference-only prompt-enforcement ablation on the completed reasoning adapter recovered a large amount of structure-sensitive performance.
- Launched a fresh Qwen3.5-0.8B Unsloth run on the reinforced dataset.
- Process: `proc_b4a7b2545e93`
- Output dir: `outputs/qwen35-0.8b-browser-reasoning-reinforced-unsloth`
- Goal: test whether the model can internalize the final-action contract during training rather than depending on inference-time prompt scaffolding alone.

## 2026-03-21 reinforced reasoning-action evaluation update

- Evaluated the finished reinforced reasoning-action adapter on the original reasoning-action validation split using the corrected conditional-generation loader and a 1536-token generation budget.
- Under the default reasoning prompt, the reinforced model was still poor: parseable 10.00%, exact 3.75% on 240 validation rows.
- Under the reinforced evaluation prompt mirroring the training ablation structure (strict format rule, one-shot example, user reminder), the same adapter improved to parseable 92.92%, exact 66.67% on the same 240 rows.
- This means the reinforced training run helped, but did not make the model robust to the plain default reasoning prompt.
- The learned behavior is still strongly prompt-conditional: the model can emit useful executable actions when the prompt enforces the contract, but falls back to prose-style outputs under the default prompt.
- A few reinforced-prompt outputs still skip the opening `<think>` while producing the right final action, so structure compliance improved a lot without becoming perfect.

## 2026-03-21 Qwen3.5-2B action-only ablation launch

- Verified the correct 2B identifier on HF / config load is `Qwen/Qwen3.5-2B`; it resolves to the same `Qwen3_5ForConditionalGeneration` family and uses the corrected conditional-generation eval path.
- Measured a fresh action-only baseline for Qwen3.5-2B on 240 validation rows.
- Baseline result: parseable 100.00%, exact 58.33%.
- This is a large jump over the Qwen3.5-0.8B action-only baseline and makes 2B the right next capacity test.
- Launched a fresh action-only Unsloth fine-tune for Qwen3.5-2B.
- Process: `proc_f88e7d47f6b9`
- Output dir: `outputs/qwen35-2b-browser-action-unsloth`
- Next planned step after this run is to evaluate before/after and then decide whether to proceed to a 2B reinforced reasoning-action ablation.

## 2026-03-21 Qwen3.5-2B action-only evaluation update

- Confirmed the Qwen3.5-2B action-only Unsloth run (`proc_f88e7d47f6b9`) finished successfully.
- Baseline before fine-tuning on 240 validation rows: parseable 100.00%, exact 58.33%.
- Post-train eval with the corrected conditional-generation loader on the same split: parseable 100.00%, exact 87.50%.
- This is a strong result and materially better than the Qwen3.5-0.8B action-only baseline and its post-train score.
- Qwen3.5-2B appears to be the better next foundation for follow-up reasoning-action experiments.

## 2026-03-21 Qwen3.5-2B reinforced reasoning-action launch

- Measured a fresh Qwen3.5-2B baseline on the original reasoning-action validation split with the default prompt and corrected conditional-generation loader.
- Baseline result at 1536 generation tokens: parseable 96.67%, exact 53.33% on 240 validation rows.
- This is a major improvement over the 0.8B reasoning baseline and strongly suggests capacity was a real limiting factor.
- Launched a fresh reinforced reasoning-action Unsloth fine-tune for Qwen3.5-2B using the v3 reinforced dataset.
- Process: `proc_26665d1b1e2d`
- Dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Post-train plan remains the same: evaluate under both the default reasoning prompt and the reinforced prompt to determine robustness versus prompt-conditionality.

## 2026-03-21 Qwen3.5-2B reinforced reasoning-action relaunch

- Investigated the failed first 2B reinforced reasoning run (`proc_26665d1b1e2d`).
- Root event occurred at step 100/407 during the first eval/save boundary: CUDA driver error during evaluation, followed by open-file exhaustion (`Errno 24`) during cleanup.
- Confirmed the shell soft open-file limit was only 1024.
- Patched `scripts/train_unsloth_sft_generic.py` so training can disable mid-training eval and mid-training checkpoint saves.
- Relaunched the 2B reinforced reasoning run with `ulimit -n 65535`, `--disable-eval`, and `--disable-mid-save`.
- New process: `proc_fbc9350ab32c`
- Output dir remains `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`.
- Plan: let training finish without intermediate eval/save interruptions, then run the default-prompt and reinforced-prompt evals afterward.

## 2026-03-22 Qwen3.5-2B reinforced reasoning-action evaluation update

- Confirmed the relaunched Qwen3.5-2B reinforced reasoning-action run completed successfully and final adapter artifacts are present under `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`.
- Ran the planned post-train evaluation on both prompt families at max_new_tokens=1536 using the corrected conditional-generation loader.
- On the plain default reasoning prompt, the finished adapter scored parseable 41.67% and exact 29.17% on 240 validation rows.
- On the reinforced reasoning prompt that mirrors the train-time format scaffolding, the same adapter scored parseable 95.42% and exact 84.17% on the same 240 rows.
- Relative to the pre-train default-prompt baseline (96.67% parseable, 53.33% exact), this run substantially improved reinforced-prompt behavior but materially worsened robustness under the plain default reasoning prompt.
- Inspected the default-prompt failures and found the dominant issue is still output-format collapse rather than obvious loss of task knowledge.
- Of the 170 non-matching rows, 140 were unparseable; the biggest buckets were prose-only action descriptions (81 rows) and other natural-language completions with no valid final BrowserGym action line (59 rows).
- A smaller residual bucket used near-correct but non-canonical syntax such as `click(bid='18')` (14 rows), plus a few genuine action-selection mistakes on checkbox-heavy tasks.
- Worst default-prompt task families by exact-match included `multi-layouts`, `click-tab`, `click-test-2`, `focus-text-2`, and `read-table`, all of which fell to 0 exact on their sampled validation rows.
- Current read: at 2B scale, reinforced reasoning training can preserve strong behavior when the inference prompt enforces the contract, but it still does not internalize a stable default-prompt action format.

## 2026-03-22 local BrowserGym GRPO integration

- Inspected the Liquid4All browser-control GRPO example and adapted the core idea into a local-only path that fits this repo's existing one-step BrowserGym workflow.
- Chose not to depend on Modal or Python-side `openenv` / `browsergym` packages for the first smoke path; instead, the new trainer reuses the existing OpenEnv websocket server protocol already used by `scripts/collect_rollouts.py`.
- Added `scripts/train_browsergym_grpo.py`, which builds static one-step prompt rows from real BrowserGym reset observations and trains a GRPO policy with environment-backed reward functions.
- The reward path resets the same task+seed, executes the model's predicted BrowserGym action, and scores parseability plus environment reward / success bonus / action-error penalty.
- Added `configs/grpo_smoke_qwen25_1p5b_click-test.yaml` for a tiny local smoke run and `docs/grpo-local.md` documenting setup and usage.
- Updated `pyproject.toml` with optional RL dependency groups so the RL stack can live in a dedicated environment instead of contaminating the lighter default workflow.
- Created a dedicated RL venv at `/home/saisamarth/venvs/browser-agent-rl` with pinned `trl==0.25.1`, `transformers==4.57.3`, CUDA `torch==2.5.1+cu124`, and the supporting TRL dependencies required for GRPO imports.
- Started a local `browsergym-env:latest` container pinned to MiniWoB `click-test` on port 8000 and verified reward-function dry-run sanity.
- Ran the full smoke training command successfully with `Qwen/Qwen2.5-1.5B-Instruct` + LoRA; GRPO completed 2 optimizer steps and saved outputs to `outputs/qwen25-1.5b-browser-action-grpo-smoke`.
- Observed caveat: `click-test` is too easy for a meaningful RL signal here, so sampled completions all got the same reward and `reward_std` collapsed to 0.0. The pipeline is functioning, but the next real RL run should use a harder task family or broader prompt slice to get non-zero advantage signal.

## 2026-03-22 multi-turn RL Phase A start

- Reviewed the earlier one-step GRPO results and concluded that the broad 30-task run was not a reliable policy-improvement experiment because too many prompts collapsed to parseability-only reward with zero reward variance.
- Decided to keep the Qwen2.5-1.5B action-only SFT adapter as the RL warm start, but move RL itself from one-step reward to multi-turn rollout reward.
- Added `scripts/train_browsergym_grpo_multiturn.py` as a separate trainer so the original one-step GRPO script remains available for comparison.
- The new trainer performs browser rollouts up to 10 steps, stops early on success, keeps short trajectory history in the prompt, and scores the full rollout with success reward plus small penalties for invalid actions, action errors, and wasted steps.
- Chose a narrow Phase A curriculum of `click-button`, `click-option`, and `enter-text-2` so reward behavior can be inspected before scaling back to the full phase-1 task set.
- Added config `configs/grpo_multiturn_phase_a_qwen25_action_adapter.yaml` for the first multi-turn RL validation run.
- Confirmed the local multitask BrowserGym server is healthy and the Phase A dry-run prompt construction succeeded on 12 prompt rows.
- Launched the first multi-turn Phase A run in the background using the Qwen2.5-1.5B action-only SFT adapter.

## 2026-03-22 multi-turn RL Phase A.2 analysis

- Patched `scripts/train_browsergym_grpo_multiturn.py` so follow-up rollout steps sample instead of decoding greedily.
- Phase A.2 used the Qwen2.5-1.5B action-only SFT adapter with tasks `click-option`, `enter-text-2`, and `enter-password`.
- This improved rollout diversity immediately: the first logged GRPO batch showed `reward_std≈2.49`, much better than the earlier deterministic follow-up behavior.
- However, reward variance still collapsed on many later batches, so sampling fixed one bottleneck without fully solving curriculum saturation.
- Task-level replay analysis over the Phase A.2 seed slice showed:
  - `click-option`: 3/4 successes, avg reward 2.32, avg steps 1.75
  - `enter-text-2`: 3/4 successes, avg reward 2.315, avg steps 2.25
  - `enter-password`: 1/4 successes, avg reward 0.85, avg steps 2.5
- Interpretation: `click-option` and `enter-text-2` are already close to saturation under the current setup, while `enter-password` remains hard enough to provide meaningful RL headroom.
- Concrete failure mode for `enter-password`: the model often fills only the second field and clicks submit, or rewrites the second field, instead of reliably filling both password boxes before submit.
- Current recommendation is not to scale back to all 30 tasks yet. The next RL curriculum should stay narrow and harder, centered on `enter-password` plus one or two medium-difficulty tasks.

## 2026-03-22 multi-turn RL Phase B launch

- After inspecting Phase A.2, kept the recommendation to stay narrow but widened the curriculum moderately instead of jumping back to all 30 tasks.
- Added a Phase B config with 10 seed rollouts per task so we get more outcome diversity and more chances to observe positive trajectories.
- Phase B task mix is `enter-password`, `click-option`, `enter-text-2`, `click-test-2`, and `click-checkboxes-transfer`.
- Rationale: keep `enter-password` as the hardest and most informative task, retain two medium tasks that already showed useful signal, and add two additional tasks to broaden the rollout distribution without going fully broad again.
- Increased later-step rollout sampling slightly to `temperature=0.9` while keeping `top_p=0.95`.
- Added config `configs/grpo_multiturn_phase_b_qwen25_action_adapter.yaml` and confirmed the dry-run prompt construction on 50 prompt rows succeeded under the multitask BrowserGym server.
- Launched the Phase B multi-turn RL run in the background with process `proc_0cec13e7c8f4`.

## 2026-03-23 Qwen3.5-2B RL rotary mismatch fix

- Reproduced the Qwen3.5-2B GRPO crash directly on iom4090 with `configs/grpo_smoke_qwen35_2b_action_adapter_click_button.yaml`.
- Exact failing stack ended inside `transformers/models/qwen3_5/modeling_qwen3_5.py` at `apply_rotary_pos_emb`, with `RuntimeError: Sizes of tensors must match except in dimension 3. Expected size 0 but got size 1 for tensor number 1 in the list.`
- Instrumented the Qwen3.5 forward path and confirmed the real failure mode: during GRPO generation -> logprob scoring, stale `rope_deltas` state inside `Qwen3_5Model` produced malformed `position_ids` with batch dimension 0, which collapsed RoPE cos/sin tensors to shape `(0, seq_len, 64)`.
- Added a guarded patch in `scripts/train_browsergym_grpo.py` that clears or batch-aligns stale `rope_deltas` before `compute_3d_position_ids` reuses them.
- Verified the patched one-step GRPO smoke now runs end-to-end for the Qwen3.5-2B action adapter `outputs/qwen35-2b-browser-action-unsloth`.
- Added the same Qwen3.5 guard and conditional-loader support to `scripts/train_browsergym_grpo_multiturn.py`.
- Verified a multi-turn smoke now runs end-to-end for:
  - `outputs/qwen35-2b-browser-action-unsloth`
  - `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Smoke configs used for validation:
  - `configs/grpo_smoke_qwen35_2b_action_adapter_click_button.yaml`
  - `configs/grpo_smoke_qwen35_2b_reasoning_reinforced_click_button.yaml`
  - `configs/grpo_multiturn_smoke_qwen35_2b_action.yaml`
  - `configs/grpo_multiturn_smoke_qwen35_2b_reasoning.yaml`
- Important caveat: the `click-button` smoke is too easy, so both Qwen3.5-2B smoke runs still showed `reward_std = 0` despite completing cleanly. These runs validate stack correctness, not useful RL signal.
- Recommendation from this point: use the fixed Qwen3.5-2B path on a harder narrow multi-turn curriculum rather than broad full-task RL immediately.

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

