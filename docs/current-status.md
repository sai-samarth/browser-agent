# Current Status

Last updated: 2026-03-21T15:19:33Z

## Active monitored run
- No Qwen3.5-2B fine-tune is currently running.
- The latest completed run is the Qwen3.5-2B action-only Unsloth experiment on `outputs/qwen35-2b-browser-action-unsloth`.

## Most recent completed result
### Qwen3.5-2B on action-only dataset
- Process: `proc_f88e7d47f6b9`
- Dataset: `data/exports/phase1_sft_v2/action_only/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-action-unsloth`
- Status: training finished successfully
- Final train loss: 0.2382
- Final eval loss: 0.04981
- Baseline val with corrected conditional-generation loader: parseable 100.00%, exact 58.33% on 240 validation rows.
- Post-train eval with the same corrected loader: parseable 100.00%, exact 87.50% on 240 validation rows.
- Interpretation: Qwen3.5-2B is a much stronger base than 0.8B and fine-tunes cleanly on action-only.

## Prior completed result
### Qwen3.5-0.8B on reinforced reasoning-action dataset
- Default reasoning prompt eval at 1536 tokens: parseable 10.00%, exact 3.75% on 240 validation rows.
- Reinforced evaluation prompt at 1536 tokens: parseable 92.92%, exact 66.67% on 240 validation rows.

## Prior completed result
### Qwen3.5-0.8B on action-only dataset
- Corrected post-train eval via conditional-generation path: parseable 100.00%, exact 80.83% on 240 validation rows.

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
