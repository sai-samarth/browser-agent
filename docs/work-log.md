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
