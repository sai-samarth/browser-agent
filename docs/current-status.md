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

## Qwen3.5-2B RL status update
- Last updated: 2026-03-23T05:04:18Z
- The earlier Qwen3.5-2B GRPO blocker was a real stack bug, now reproduced and patched.
- Root cause: stale `rope_deltas` in the Qwen3.5 multimodal wrapper caused zero-batch `position_ids` during GRPO logprob scoring after generation.
- Patched scripts:
  - `scripts/train_browsergym_grpo.py`
  - `scripts/train_browsergym_grpo_multiturn.py`
- Validated end-to-end smoke after patching:
  - action adapter: `outputs/qwen35-2b-browser-action-unsloth`
  - reasoning adapter: `outputs/qwen35-2b-browser-reasoning-reinforced-unsloth`
- Current read: Qwen3.5-2B is no longer blocked by the rotary/tensor crash, but harder-task multi-turn runs are still needed before judging whether it beats the Qwen2.5 RL path.

## Qwen3.5-2B action-only RL small-run update
- Last updated: 2026-03-23T05:26:30Z
- Short multi-turn RL check on `enter-password`, `click-option`, and `enter-text-2` finished cleanly.
- Sampling was already raised to `temperature=0.9`, `top_p=0.95`.
- Reward stayed around 3.08 to 3.12, but `reward_std` stayed 0.0 throughout.
- Current read: the fixed 2B action-only path is stable, but this curriculum is too saturated to generate useful GRPO preference signal.

## Qwen3.5-2B action-only hard-run update
- Last updated: 2026-03-23T05:58:16Z
- Harder action-only subset run on `click-checkboxes-large`, `click-checkboxes-transfer`, and `find-word` finished cleanly.
- This run did produce real non-zero GRPO signal, with `reward_std` peaks around `1.994` and `2.135`.
- Some later batches still collapsed back toward tied rewards, so the signal is intermittent rather than fully stable.
- Current read: this harder subset is the first action-only Qwen3.5-2B RL setup that looks experimentally worth continuing.

## Qwen3.5-2B reasoning-vs-action RL comparison update
- Last updated: 2026-03-23T06:06:19Z
- The matched reinforced-reasoning hard-subset RL run finished cleanly and did produce non-zero reward variance, with peaks around `2.305` and `2.15`.
- But compared with the matched action-only run, it was slower, produced much longer completions, hit clipping more often, and showed multiple negative-reward batches.
- Current read: action-only remains the better RL warm start for the next continuation step on this stack.

## Qwen3.5-2B action-only scaled-run update
- Last updated: 2026-03-23T06:14:48Z
- The scaled hard-subset action-only run finished cleanly.
- Strong nonzero variance still appeared late in the run (`reward_std≈2.121`), so the useful signal survived scaling.
- But many batches still collapsed to tied rewards, including both high-reward and low-reward collapsed phases.
- Current read: action-only on the hard subset is still the best RL path, but the signal remains intermittent rather than consistently dense.

## Qwen3.5-2B task-signal analysis update
- Last updated: 2026-03-23T06:28:43Z
- Proper task-level replay analysis now suggests the best current RL signal carriers are `enter-text-2` and `enter-password`.
- `click-checkboxes-large` looks useful as a third task, but weaker than those two.
- `click-option` is saturated and should be dropped.
- `click-checkboxes-transfer` is weaker than `click-checkboxes-large` and should be deprioritized.
- `find-word` has variance but low average reward, so it is not the best immediate curriculum choice.
- Recommended next action-only curriculum: `enter-text-2`, `enter-password`, `click-checkboxes-large`.

## Qwen3.5-2B refined-curriculum RL update
- Last updated: 2026-03-23T06:37:52Z
- The refined action-only curriculum (`enter-text-2`, `enter-password`, `click-checkboxes-large`) finished cleanly but mostly collapsed to tied rewards in live GRPO training.
- This means the earlier task-signal replay analysis did not transfer cleanly to full training dynamics.
- Current read: the best live RL signal still comes from the harder checkbox-heavy curriculum rather than the refined mostly-text-entry curriculum.

## Qwen3.5-2B exact-slice RL update
- Last updated: 2026-03-23T06:47:34Z
- The exact live-signal slice run did not reproduce the earlier strong variance batches.
- This means the strongest earlier batches were not just explained by prompt identity; broader rollout dynamics matter more.
- Current read: curriculum narrowing alone is no longer the best next knob. Reward / sampling design is now the more likely bottleneck.

## Qwen3.5-2B num_generations=4 update
- Last updated: 2026-03-23T06:58:22Z
- Increasing `num_generations` to 4 did help somewhat: the run produced a meaningful non-zero variance batch (`reward_std≈1.657`).
- But many batches still collapsed to tied rewards.
- Current read: generation diversity helps, but not enough on its own. Reward design is now the main bottleneck.

## Reward-design analysis update
- Last updated: 2026-03-23T07:16:49Z
- Current read: reward shaping is now the main bottleneck, not curriculum-only tuning.
- Recommended next implementation is task-specific progress shaping for `enter-text-2`, `enter-password`, and `click-checkboxes-large` layered on top of the existing base reward.

## Reward-shaping validation update
- Last updated: 2026-03-23T08:32:38Z
- The first task-specific shaping layer materially improved reward variance on the refined action-only curriculum.
- Strong non-zero variance batches now appear (`reward_std≈2.31` and `≈1.994`), where the unshaped refined run was mostly tied.
- Current read: reward shaping is working and should be retained for the next RL iteration.

## Shaped hard-curriculum update
- Last updated: 2026-03-23T09:41:14Z
- The shaped hard action-only run produced a real strong non-zero variance batch (`reward_std≈2.044`) but still had many tied batches.
- Current read: shaping helps, but the tied-reward problem is reduced rather than solved.

## Qwen3.5-0.8B weak-task focus update
- Last updated: 2026-03-23T10:04:39Z
- Current 0.8B weak-task focus is `click-checkboxes-large`, `find-word`, and `enter-text-2`.
- `click-checkboxes-large` is the highest-leverage weak task and appears to be a real state-tracking/set-progress failure rather than a formatting problem.
- `find-word` and `enter-text-2` mostly need content-level shaping rather than action-format shaping.
- Recommended next step is targeted SFT/data improvement plus task-specific reward shaping on those tasks before broader GRPO.

## Qwen3.5-0.8B shaped weak-task update
- Last updated: 2026-03-23T10:24:13Z
- The shaped 0.8B weak-task validation produced multiple strong non-zero variance batches, including `reward_std≈2.758` and `≈2.313`.
- Current read: this is the cleanest reward-signal path we have so far for demonstrating targeted improvement.

## Qwen3.5-0.8B GRPO + eval update
- Last updated: 2026-03-23T10:39:22Z
- The first weak-task shaped GRPO continuation finished cleanly and improved full-val exact-match from 80.83% to 81.67%.
- Current read: the path is viable but the gain is still modest, so the next big improvements likely require stronger shaping and/or more targeted SFT/data work on the same weak tasks.

## 0.8B sweep update
- Last updated: 2026-03-23T11:07:18Z
- The shaped weak-task gen4 run produced strong reward variance (`reward_std≈2.758` and `≈1.495`).
- Current read: this parameter setting belongs in the short-list of better GRPO knob combinations for 0.8B.

