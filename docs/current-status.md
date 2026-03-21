# Current Status

Last updated: 2026-03-21T13:34:28Z

## Active monitored run
- Qwen3.5-2B action-only Unsloth is running now.
- Process: `proc_f88e7d47f6b9`
- Dataset: `data/exports/phase1_sft_v2/action_only/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-action-unsloth`

## Why this run exists
- Qwen3.5-0.8B action-only fine-tuning worked, but the 0.8B reasoning-action runs remained too prompt-fragile.
- Qwen3.5-2B is the next clean capacity ablation in the same family.
- The plan is to establish the 2B action-only baseline first, then compare it to a later 2B reinforced reasoning-action run.

## New baseline
### Qwen3.5-2B on action-only dataset
- Model: `Qwen/Qwen3.5-2B`
- Baseline val using corrected conditional-generation loader: parseable 100.00%, exact 58.33% on 240 validation rows.
- This is already much stronger than the Qwen3.5-0.8B action-only baseline of 17.92%.

## Prior completed result
### Qwen3.5-0.8B on reinforced reasoning-action dataset
- Process: `proc_b4a7b2545e93`
- Dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Output dir: `outputs/qwen35-0.8b-browser-reasoning-reinforced-unsloth`
- Default reasoning prompt eval at 1536 tokens: parseable 10.00%, exact 3.75% on 240 validation rows.
- Reinforced evaluation prompt at 1536 tokens: parseable 92.92%, exact 66.67% on 240 validation rows.
- Interpretation: the 0.8B reasoning-action line improved under scaffolding but stayed highly prompt-conditional.

## Prior completed result
### Qwen3.5-0.8B on action-only dataset
- Process: `proc_8d6ef4acc03b`
- Output dir: `outputs/qwen35-0.8b-browser-action-unsloth`
- Corrected post-train eval via conditional-generation path: parseable 100.00%, exact 80.83% on 240 validation rows.

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
