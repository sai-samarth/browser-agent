# Current Status

Last updated: 2026-03-20T18:25:00Z

## Active monitored run
- No active training run right now.

## Most recent completed result
### Qwen3.5-0.8B on action-only dataset
- Process: `proc_8d6ef4acc03b`
- Output dir: `outputs/qwen35-0.8b-browser-action-unsloth`
- Status: training finished successfully
- Train runtime: ~5343s (~89.1 min)
- Final eval loss: 0.05195
- Final train loss: 0.2513

### Current eval state
- Baseline val: parseable 100.00%, exact 17.92%
- Saved post-train eval via old shared eval path: parseable 100.00%, exact 17.92% (invalid for this model family).
- Root cause: the old eval loaded Qwen3.5-0.8B through a causal-LM path that did not correctly express the trained adapter; PEFT emitted missing-adapter-key warnings under that path.
- Corrected post-train eval via conditional-generation path: parseable 100.00%, exact 80.83% on 240 validation rows.
- Interpretation: Qwen3.5-0.8B fine-tuning did work; the bug was in evaluation, not training.

## Prior completed result
### Qwen2.5-1.5B on reasoning-action dataset
- Output dir: `outputs/qwen25-1.5b-browser-reasoning-unsloth`
- Saved low-budget eval: parseable 83.75%, exact 69.58% (undercounted due to truncation)
- Corrected high-budget eval: parseable 100.00%, exact 81.67%
- Canonicalized local equivalence: 83.75%
- Main remaining weakness: checkbox-planning tasks, especially `click-checkboxes-large`

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Current focus is exclusively Qwen3.5-0.8B debugging; do not branch into other model investigations until this eval mismatch is understood.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
