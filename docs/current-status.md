# Current Status

Last updated: 2026-03-21T17:24:37Z

## Active monitored run
- Qwen3.5-2B reinforced reasoning-action Unsloth has been relaunched and is running now.
- Process: `proc_fbc9350ab32c`
- Dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Mid-training eval and mid-training checkpoint saves are disabled for this relaunch.

## Why it was relaunched
- The first 2B reinforced reasoning run failed at step 100/407 during the first eval/checkpoint boundary.
- The failure stack showed a CUDA driver error during eval followed by open-file exhaustion (`OSError: [Errno 24] Too many open files`).
- The shell open-file soft limit was only 1024.
- The relaunch now raises `ulimit -n` and avoids all mid-training eval/save boundaries to remove that failure path.

## New baseline
### Qwen3.5-2B on reasoning-action dataset
- Default reasoning prompt baseline at 1536 tokens with corrected conditional-generation loader: parseable 96.67%, exact 53.33% on 240 validation rows.

## Prior completed result
### Qwen3.5-2B on action-only dataset
- Process: `proc_f88e7d47f6b9`
- Baseline val: parseable 100.00%, exact 58.33%.
- Post-train eval: parseable 100.00%, exact 87.50%.

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
