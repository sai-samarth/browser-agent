# Current Status

Last updated: 2026-03-22T04:05:30Z

## Latest completed result
### Qwen3.5-2B reinforced reasoning-action
- Training dataset: `data/exports/phase1_sft_v3/reasoning_action_reinforced/hf_dataset`
- Output dir: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Default reasoning prompt eval at 1536 tokens: parseable 41.67%, exact 29.17% on 240 validation rows.
- Reinforced reasoning prompt eval at 1536 tokens: parseable 95.42%, exact 84.17% on 240 validation rows.
- Relative to the default-prompt baseline (96.67% parseable, 53.33% exact), reinforced training improved reinforced-prompt behavior strongly but made plain default-prompt robustness materially worse.

## Failure read on the default prompt
- Most of the default-prompt gap is still output-format collapse rather than missing task knowledge.
- Of the 170 non-matching rows, 140 were unparseable under the strict BrowserGym parser.
- The largest buckets were prose-only action descriptions (81 rows) and other unparseable natural-language completions (59 rows).
- A smaller residual bucket used near-correct but non-canonical syntax such as `click(bid='18')` (14 rows).

## Prior completed result
### Qwen3.5-2B on action-only dataset
- Process: `proc_f88e7d47f6b9`
- Baseline val: parseable 100.00%, exact 58.33%.
- Post-train eval: parseable 100.00%, exact 87.50%.

## Operating rule
- Only use unpaid local models or free models unless explicit approval is given for paid model/API use.
- Update `docs/current-status.md`, `docs/work-log.md`, and `docs/experiment-results.md` whenever runs finish or metrics materially change.
