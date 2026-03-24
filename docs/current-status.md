# Current Status

Last updated: 2026-03-24T12:59:13Z

## Latest completed result
### Qwen3.5-0.8B mixed weak-task continuation SFT
- Warm start: `outputs/qwen35-0.8b-browser-action-unsloth`
- Output dir: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
- Full-val eval at 256 tokens with corrected conditional-generation loader: parseable 100.00%, exact 83.33% on 240 validation rows.
- This is currently the best continuation-SFT variant on the 0.8B action-only line.
- Carry-forward recommendation: use this checkpoint for the next continuation-stage comparison or downstream post-SFT follow-up work.

## Prior completed result
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

## 0.8B GRPO sweep summary update
- Last updated: 2026-03-23T11:19:40Z
- Best post-GRPO exact-match in the weak-task sweep is 81.67% (vs 80.83% post-SFT baseline).
- `num_generations=4` improved reward variance but not final eval.
- Stronger shaping improved reward variance substantially, but its first post-eval tied the current best rather than beating it.

## 2026-03-24 Qwen3.5-0.8B weak-task 1000-row continuation SFT launch

- Last updated: 2026-03-24T04:41:26Z
- Launched a continuation SFT run from the existing 0.8B action-only adapter.
- Warm start adapter: `outputs/qwen35-0.8b-browser-action-unsloth`
- Base model: `Qwen/Qwen3.5-0.8B`
- Training dataset: `data/exports/phase1_sft_v2_action_weak3_exact1000/hf_dataset`
- Output dir: `outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft`
- Log file: `logs/qwen35_08b_weak3_exact1000_cont_sft.log`
- Weak-task train subset is restricted to:
  - `click-checkboxes-large`
  - `find-word`
  - `enter-text-2`
- Exact train-size construction:
  - 930 unique weak-task rows available in the source train split
  - added 70 duplicated rows to reach exactly 1000 train rows
  - resulting train counts:
    - `click-checkboxes-large`: 667
    - `enter-text-2`: 191
    - `find-word`: 142
- Validation split was kept as the full action-only validation set to preserve end-of-training full-val evaluation compatibility.
- Continuation SFT hyperparameters:
  - `max_length=2048`
  - `num_train_epochs=2.0`
  - `per_device_train_batch_size=4`
  - `gradient_accumulation_steps=4`
  - `learning_rate=1e-4`
  - `seed=3407`
- Initial launch check passed: model load, adapter attach, dataset map/tokenization, and trainer step-loop start all completed cleanly.
- Expected training length at launch: about 126 optimizer steps.

## 2026-03-24 Qwen3.5-0.8B weak-task 1000-row continuation SFT completion

- Last updated: 2026-03-24T05:42:57Z
- The weak-task 1000-row continuation SFT finished cleanly.
- Warm start adapter: `outputs/qwen35-0.8b-browser-action-unsloth`
- Output dir: `outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft`
- Log file: `logs/qwen35_08b_weak3_exact1000_cont_sft.log`
- Final adapter artifacts are present, including `adapter_model.safetensors`, tokenizer files, processor config, and `run_summary.json`.
- Final training stats from the trainer log:
  - `train_runtime`: 3626s
  - `train_samples_per_second`: 0.552
  - `train_steps_per_second`: 0.035
  - `train_loss`: 0.04394
  - `epoch`: 2.0
- Logged loss trend over the run improved from about `0.08149` early to about `0.02019` late.
- No end-of-training evaluation has been run yet.
- Next required step is a full end-of-training evaluation on the standard action-only validation split, then compare against the prior post-SFT baseline of 80.83% exact-match.

## 2026-03-24 Qwen3.5-0.8B weak-task continuation SFT post-train eval

- Last updated: 2026-03-24T05:50:30Z
- Ran the required end-of-training evaluation on the full standard action-only validation split using the corrected conditional-generation eval path.
- Evaluated adapter: `outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft`
- Eval file: `outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft/eval_after_conditional_256.json`
- Eval loader: `conditional_generation`
- `max_new_tokens=256`

### Full validation result
- Parseable rate: `100.00%`
- Exact-match: `81.25%`
- Prior 0.8B post-SFT baseline on the same eval path: `80.83%`
- Net full-val gain from this continuation SFT: about `+0.42` exact-match points.

### Targeted weak-task result
- `click-checkboxes-large`: `38.60% -> 45.61%` (`+7.02` points)
- `find-word`: `75.00% -> 87.50%` (`+12.50` points)
- `enter-text-2`: `78.57% -> 85.71%` (`+7.14` points)
- All three targeted tasks improved.

### Weak-task generation read
- `find-word` improved in the intended content-selection way.
  - Example improvement: `fill('14', 'non') -> fill('14', 'interdum')` on a row whose target was `fill('14', 'interdum')`.
- `enter-text-2` improved in the intended transformation-fidelity way.
  - Example improvement: `fill('14', 'kase') -> fill('14', 'kasie')` on a row whose target was `fill('14', 'kasie')`.
- `click-checkboxes-large` improved materially but remains unstable.
  - Several rows that previously guessed the wrong checkbox ID now emit the exact target ID.
  - But some previously correct rows regressed to nearby plausible checkbox IDs, so the continuation improved target-ID selection overall without fully solving state-tracking / set-progress errors.

### Collateral regressions
- Non-target tasks as a group dropped from `96.27%` to `93.17%` exact-match.
- Largest observed regressions:
  - `click-checkboxes-transfer`: `86.96% -> 73.91%` (`-13.04` points)
  - `read-table`: `92.86% -> 85.71%` (`-7.14` points)
  - `click-option`: `100.00% -> 95.00%` (`-5.00` points)
- This means the continuation achieved the intended weak-task improvement, but it over-specialized enough to hurt a few neighboring/general tasks, especially `click-checkboxes-transfer`.

### Current read
- The 1000-row weak-task continuation SFT is a real but modest win if judged only by full-val exact-match (`81.25% > 80.83%`).
- More importantly, it validates that targeted SFT can move the intended weak tasks in the right direction.
- However, the collateral damage means this exact continuation recipe should not be treated as the final best path.
- Best next SFT direction is likely a mixed dataset that still boosts the weak tasks, but preserves more broad-task coverage than this weak-task-only 1000-row continuation.

## 2026-03-24 Qwen3.5-0.8B mixed weak-task continuation SFT launch

- Last updated: 2026-03-24T06:02:11Z
- Launched a fresh continuation SFT run from the original 0.8B action-only adapter (`80.83%` baseline), not from the prior weak-task-only continuation.
- Warm start adapter: `outputs/qwen35-0.8b-browser-action-unsloth`
- Base model: `Qwen/Qwen3.5-0.8B`
- Training dataset: `data/exports/phase1_sft_v2_action_weak3_mixed50_1000/hf_dataset`
- Output dir: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
- Log file: `logs/qwen35_08b_weak3_mixed50_1000_cont_sft.log`

### Dataset construction
- Total train size: `1000`
- Weak-task rows: `500`
- Non-weak rows: `500`
- No duplication was required on either side.
- Weak-task set:
  - `click-checkboxes-large`
  - `find-word`
  - `enter-text-2`
- Sampled weak-task counts in the final train split:
  - `click-checkboxes-large`: `333`
  - `enter-text-2`: `91`
  - `find-word`: `76`
- Validation split was kept as the full standard action-only validation set for end-of-training comparison against the same 240-row benchmark.

### Training hyperparameters
- `max_length=2048`
- `num_train_epochs=2.0`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4`
- `learning_rate=1e-4`
- `seed=3407`

### Initial launch check
- Model load passed.
- Adapter attach passed.
- Dataset rendering and tokenization passed.
- Trainer entered the step loop cleanly.
- Expected training length at launch: about `126` optimizer steps.

### Why this run exists
- The earlier weak-task-only 1000-row continuation improved the intended weak tasks but caused collateral regressions on other tasks, especially `click-checkboxes-transfer`.
- This mixed 50/50 continuation tests whether we can keep most of the targeted weak-task gains while preserving broader action-only behavior.

## 2026-03-24 Qwen3.5-0.8B mixed weak-task continuation SFT completion

- Last updated: 2026-03-24T07:38:57Z
- The mixed 50/50 weak-task continuation SFT finished cleanly.
- Warm start adapter: `outputs/qwen35-0.8b-browser-action-unsloth`
- Output dir: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
- Log file: `logs/qwen35_08b_weak3_mixed50_1000_cont_sft.log`
- Final adapter artifacts are present, including `adapter_model.safetensors`, tokenizer files, processor config, and `run_summary.json`.
- Final training stats from the trainer log:
  - `train_runtime`: 4214s
  - `train_samples_per_second`: 0.475
  - `train_steps_per_second`: 0.03
  - `train_loss`: 0.0499
  - `epoch`: 2.0
- Logged loss trend improved from about `0.0652` early to about `0.04005` late.

## 2026-03-24 Qwen3.5-0.8B mixed weak-task continuation SFT post-train eval

- Last updated: 2026-03-24T07:38:57Z
- Ran the required end-of-training evaluation on the full standard action-only validation split using the corrected conditional-generation eval path.
- Evaluated adapter: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
- Eval file: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft/eval_after_conditional_256.json`
- Eval loader: `conditional_generation`
- `max_new_tokens=256`

### Full validation result
- Parseable rate: `100.00%`
- Exact-match: `83.33%`
- Prior 0.8B post-SFT baseline on the same eval path: `80.83%`
- Prior weak-task-only continuation result: `81.25%`
- Net gain vs baseline: about `+2.50` exact-match points.
- Net gain vs weak-task-only continuation: about `+2.08` exact-match points.

### Targeted weak-task result vs baseline
- `click-checkboxes-large`: `38.60% -> 49.12%` (`+10.53` points)
- `find-word`: `75.00% -> 75.00%` (`0.00` points)
- `enter-text-2`: `78.57% -> 85.71%` (`+7.14` points)

### Tradeoff read vs baseline
- The biggest targeted gain came from `click-checkboxes-large`, which improved more than either prior continuation variant.
- `enter-text-2` retained its earlier improvement.
- `find-word` gave back the weak-task-only gain and returned to baseline level.
- The key win of the mixed run is that it preserved broader behavior much better than the weak-task-only continuation.
  - `click-checkboxes-transfer` recovered from `73.91%` back to `86.96%`.
  - `click-option` recovered from `95.00%` back to `100.00%`.
  - `read-table` stayed at `85.71%`, still below the original `92.86%` baseline.

### Current read
- This mixed 50/50 continuation is currently the best continuation-SFT variant tested on the 0.8B action-only line.
- It dominates the weak-task-only continuation on full-val exact-match because it keeps most of the intended weak-task gains while avoiding the major collateral regressions on neighboring tasks.
- If choosing a continuation-SFT checkpoint to carry forward, `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft` is now the strongest candidate.

## 2026-03-24 Qwen3.5-0.8B mixed-best checkpoint weak-task analysis and GRPO launch

- Last updated: 2026-03-24T08:15:41Z
- Started from the new best continuation-SFT checkpoint:
  - `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
- Fresh weak-task read on the full 240-row validation split gave:
  - `click-checkboxes-large`: `49.12%`
  - `find-word`: `75.00%`
  - `enter-text-2`: `85.71%`
  - `read-table`: `85.71%`
  - `click-checkboxes-transfer`: `86.96%`
- Chosen GRPO task subset:
  - `click-checkboxes-large`
  - `find-word`
  - `enter-text-2`
  - `click-checkboxes-transfer`
- `read-table` was left out for now even though it is similarly weak, because the current GRPO shaping already supports:
  - checkbox tasks (`click-checkboxes-large`, `click-checkboxes-transfer`)
  - `find-word`
  - `enter-text-2`
  and therefore provides the best immediate high-signal continuation without first adding new reward logic for `read-table`.

### GRPO config
- Config file: `configs/grpo_multiturn_qwen35_08b_action_mixedbest_weak4_shaped.yaml`
- Warm start adapter: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
- Output dir: `outputs/qwen35-0.8b-browser-action-grpo-multiturn-mixedbest-weak4-shaped`
- Log file: `logs/qwen35_08b_mixedbest_weak4_grpo.log`
- Key settings:
  - `samples_per_task=8`
  - `max_steps=36`
  - `num_generations=2`
  - `generation_batch_size=2`
  - `rollout_temperature=0.9`
  - `rollout_top_p=0.95`
  - `progress_shaping_scale=1.5`
  - `premature_submit_penalty=0.2`

### Launch status
- The GRPO run launched cleanly.
- Process is live and output scaffolding was created successfully.
- Initial artifacts present:
  - `grpo_run_config.json`
  - `prompt_dataset_preview.jsonl`
- Hermes tracking session: `proc_d838336513bd`

### Current read
- This is the first GRPO continuation launched from the new best 83.33% mixed continuation-SFT checkpoint.
- The selected 4-task subset preserves the strongest currently-supported weak-task families while avoiding unsupported reward-shaping work before launch.

## 2026-03-24 Qwen3.5 RL environment repair and GRPO relaunch

- Last updated: 2026-03-24T10:13:38Z
- The first GRPO launch from the mixed-best 0.8B checkpoint failed before training during environment import/model-load, not because of reward logic or task config.

### Root cause
- The dedicated RL env at `/home/saisamarth/venvs/browser-agent-rl` had `transformers==4.57.3`, but that install did **not** include the `qwen3_5` module.
- Direct verification in the RL env showed both of these failed before repair:
  - `AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)`
  - `AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)`
- Cross-check against the working SFT env showed the key mismatch:
  - working env `/home/saisamarth/venvs/ft-qwen25`: `transformers==5.3.0`, `qwen3_5` present
  - broken env `/home/saisamarth/venvs/browser-agent-rl`: `transformers==4.57.3`, `qwen3_5` absent
- After upgrading the RL env to `transformers==5.3.0`, the next import failure surfaced:
  - `No module named 'weave'`
- Installing `weave==0.52.35` resolved the TRL import path required by `GRPOTrainer` in this env.

### Verified repair
- In the repaired RL env:
  - `transformers==5.3.0`
  - `trl==0.25.1`
  - `qwen3_5` module present
  - `AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)` works
  - `AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)` works

### Relaunch result
- Relaunched the same GRPO config unchanged after the env repair:
  - `configs/grpo_multiturn_qwen35_08b_action_mixedbest_weak4_shaped.yaml`
- Current live run:
  - output dir: `outputs/qwen35-0.8b-browser-action-grpo-multiturn-mixedbest-weak4-shaped`
  - log file: `logs/qwen35_08b_mixedbest_weak4_grpo.log`
  - Hermes tracking session: `proc_cce4120d3dc3`
- The repaired launch passed the previous failure stages and entered the trainer step loop.
- Current read: the Qwen3.5 GRPO blocker on this path was an RL-environment package mismatch, not a checkpoint problem.

## 2026-03-24 Qwen3.5-0.8B mixed-best GRPO post-train eval

- Last updated: 2026-03-24T11:45:23Z
- Ran end-of-training evaluation on the completed GRPO continuation adapter:
  - `outputs/qwen35-0.8b-browser-action-grpo-multiturn-mixedbest-weak4-shaped`
- Eval file:
  - `outputs/qwen35-0.8b-browser-action-grpo-multiturn-mixedbest-weak4-shaped/eval_after_grpo_256.json`
- Eval loader: `conditional_generation`
- `max_new_tokens=256`

### Full validation result
- Parseable rate: `100.00%`
- Exact-match: `82.92%`
- Starting mixed-SFT checkpoint exact-match: `83.33%`
- Net GRPO delta on top of the mixed SFT checkpoint: about `-0.42` exact-match points.
- Relative to the older pre-continuation 0.8B post-SFT baseline (`80.83%`), the GRPO result is still `+2.08` points higher, but it does not improve over the best mixed continuation SFT checkpoint.

### Task-level read vs mixed SFT checkpoint
- `click-checkboxes-large`: `49.12% -> 45.61%` (`-3.51` points)
- `find-word`: `75.00% -> 75.00%` (`0.00` points)
- `enter-text-2`: `85.71% -> 85.71%` (`0.00` points)
- `click-checkboxes-transfer`: `86.96% -> 86.96%` (`0.00` points)
- `read-table`: `85.71% -> 92.86%` (`+7.14` points)
- All other evaluated tasks were unchanged.

### Interpretation
- This GRPO continuation did not improve on the best mixed continuation-SFT checkpoint overall.
- The only clear positive movement in the 240-row eval was `read-table`, even though that task was not in the shaped GRPO subset.
- The main negative movement was regression on `click-checkboxes-large`, which was the most important weak checkbox task we were trying to improve.
- Current read: for this checkpoint/subset/reward configuration, GRPO did not add value on top of the mixed SFT warm start.
- The mixed 50/50 continuation SFT checkpoint (`83.33%`) remains the best current action-only result to carry forward.

## 2026-03-24 Qwen3.5-9B 4-bit teacher baseline on action-only validation

- Last updated: 2026-03-24T12:59:13Z
- Ran the standard action-only validation eval with bare `Qwen/Qwen3.5-9B` loaded locally in 4-bit through the shared eval path.
- Eval file: `outputs/qwen35-9b-4bit-action-baseline/eval_action_only_240_256.json`
- Eval loader: `conditional_generation`
- `max_new_tokens=256`
- No adapter was applied; this is the raw 9B teacher-style baseline under the same strict parser/exact-match protocol used for the smaller models.

### Result
- Parseable rate: `72.08%`
- Exact-match: `58.33%`

### Comparison
- This is far below the best current 0.8B mixed continuation SFT checkpoint:
  - `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft`
  - parseable `100.00%`, exact-match `83.33%`
- It is also below the original 0.8B action-only post-SFT checkpoint (`80.83%`).

### Interpretation
- Under the default strict action-only eval prompt, raw Qwen3.5-9B 4-bit is much less reliable at emitting valid canonical BrowserGym action lines than the tuned 0.8B action-only checkpoints.
- This does not mean the 9B model lacks task knowledge; it means that under this exact one-step strict-action protocol, the smaller tuned action-only models are substantially better calibrated to the output contract.
- Current consolidation read: the strongest validated action-only result in this project remains the mixed-data 0.8B continuation SFT checkpoint at `83.33%` exact-match.

