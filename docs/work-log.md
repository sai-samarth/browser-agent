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
