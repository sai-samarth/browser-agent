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

## RL infrastructure status
- Added a local GRPO training path at `scripts/train_browsergym_grpo.py` for one-step BrowserGym action models.
- Added smoke config `configs/grpo_smoke_qwen25_1p5b_click-test.yaml` and runbook `docs/grpo-local.md`.
- Verified end-to-end on local BrowserGym websocket env using a dedicated RL venv at `/home/saisamarth/venvs/browser-agent-rl`.
- Smoke run used `Qwen/Qwen2.5-1.5B-Instruct` with LoRA on `click-test` and completed successfully, saving outputs under `outputs/qwen25-1.5b-browser-action-grpo-smoke`.
- Because `click-test` is trivial, both sampled generations solved the task and reward variance collapsed to zero, so this should be treated as an infrastructure smoke test rather than evidence of useful RL learning.

## RL next phase
- One-step GRPO was validated as infrastructure but the broad 30-task phase-1 run mostly collapsed to parseability-only reward.
- Multi-turn Phase A.2 showed that sampling later rollout steps improves rollout diversity, but the 3-task curriculum still partially saturated because `click-option` and `enter-text-2` were already close to solved.
- The current active curriculum is Phase B multi-turn RL with 5 tasks and 10 seed rollouts per task.
- Phase B task mix: `enter-password`, `click-option`, `enter-text-2`, `click-test-2`, and `click-checkboxes-transfer`.
- The curriculum is intentionally centered on `enter-password` while adding several medium-difficulty tasks to raise the chance of positive-but-nontrivial outcomes.
- Active script: `scripts/train_browsergym_grpo_multiturn.py`
- Active config: `configs/grpo_multiturn_phase_b_qwen25_action_adapter.yaml`
- Active run: background process `proc_0cec13e7c8f4`
