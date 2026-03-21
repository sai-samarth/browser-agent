# Current Status

Last updated: 2026-03-21T07:06:00Z

## Active monitored run
- Qwen3.5-0.8B reasoning-action reinforced-prompt Unsloth is running now.
- Process: `proc_b4a7b2545e93`
- Dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Output dir: `outputs/qwen35-0.8b-browser-reasoning-reinforced-unsloth`
- This is a clean ablation built from the original reasoning-action dataset with stricter output instructions, a one-shot worked example, and a user-side format reminder.

## Why this run exists
- The previous reasoning-action adapter often explained the correct next action in prose but omitted the final BrowserGym action line.
- A 40-row inference-only prompt-enforcement ablation improved that finished adapter from 7.50% parseable / 7.50% exact-match to 100.00% parseable / 67.50% exact-match under a reinforced prompt.
- That strongly suggests prompt-sensitive formatting failure rather than total loss of task knowledge.
- This new run tests whether training on the reinforced prompt can internalize that structure.

## Prior completed result
### Qwen3.5-0.8B on reasoning-action dataset
- Process: `proc_d740bb1d5398`
- Output dir: `outputs/qwen35-0.8b-browser-reasoning-unsloth`
- Status: training finished successfully
- Baseline val with corrected conditional-generation loader: parseable 100.00%, exact 10.83% on 240 validation rows.
- Post-train strict parser eval at 512 tokens: parseable 11.25%, exact 8.75% on 240 validation rows.
- Post-train strict parser eval at 1536 tokens: parseable 11.25%, exact 8.75% on 240 validation rows.
- Full local semantic judging with the local `Qwen3.5-9B-judge` server on all 240 rows estimated about 60.00% action-equivalent outputs.
- Interpretation: the previous reasoning-action run learned partial task semantics but failed the executable output contract.

## Prior completed result
### Qwen3.5-0.8B on action-only dataset
- Process: `proc_8d6ef4acc03b`
- Output dir: `outputs/qwen35-0.8b-browser-action-unsloth`
- Status: training finished successfully
- Corrected post-train eval via conditional-generation path: parseable 100.00%, exact 80.83% on 240 validation rows.

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
