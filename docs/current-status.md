# Current Status

Last updated: 2026-03-21T16:00:52Z

## Active monitored run
- Qwen3.5-2B reinforced reasoning-action Unsloth is running now.
- Process: `proc_26665d1b1e2d`
- Dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`

## Why this run exists
- Qwen3.5-2B action-only already showed a much stronger baseline and post-train result than 0.8B.
- Qwen3.5-2B on the default reasoning-action validation split also starts far stronger than 0.8B.
- This run tests whether the reinforced reasoning-action prompt format can work much better once the model has enough capacity.

## New baseline
### Qwen3.5-2B on reasoning-action dataset
- Model: `Qwen/Qwen3.5-2B`
- Default reasoning prompt baseline at 1536 tokens with corrected conditional-generation loader: parseable 96.67%, exact 53.33% on 240 validation rows.
- This is dramatically stronger than the Qwen3.5-0.8B reasoning baseline.

## Prior completed result
### Qwen3.5-2B on action-only dataset
- Process: `proc_f88e7d47f6b9`
- Dataset: `data/exports/phase1_sft_v2/action_only/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-action-unsloth`
- Baseline val: parseable 100.00%, exact 58.33%.
- Post-train eval: parseable 100.00%, exact 87.50%.

## Prior completed result
### Qwen3.5-0.8B on reinforced reasoning-action dataset
- Default reasoning prompt eval at 1536 tokens: parseable 10.00%, exact 3.75%.
- Reinforced evaluation prompt at 1536 tokens: parseable 92.92%, exact 66.67%.

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
