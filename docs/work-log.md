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
