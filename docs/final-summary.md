# Final Project Summary

## Scope

This project asked a simple question:

Can small open models learn to produce executable BrowserGym actions from real browser traces well enough to be useful, and what data / training format actually moves the metric?

The answer is yes, but the details mattered a lot.

## Main conclusions

### 1. Action-only SFT beat reasoning+action for reliable one-step execution

The cleanest general pattern in the repo is that action-only supervision was much more robust than reasoning+action when the evaluation required a strict executable BrowserGym action.

Reasoning+action often preserved meaningful task knowledge, but too often drifted into prose, partial formatting, or prompt-sensitive behavior.

### 2. Qwen3.5 required a model-specific evaluation path

A major project lesson was that Qwen3.5 adapters could be silently mis-evaluated if loaded through the wrong architecture path.

The critical fix was to evaluate Qwen3.5 through the conditional-generation / processor-backed loader rather than the old causal-LM path. Before that fix, the 0.8B action-only run looked broken. After the fix, it became one of the strongest results in the repo.

### 3. Weak-task targeting worked, but only when mixed with broad coverage

A pure 1000-row weak-task continuation moved the intended weak tasks, but it also caused collateral regressions.

The mixed continuation solved that tradeoff better:
- 500 weak-task rows
- 500 broad rows
- full validation exact-match improved from 80.83% to 83.33%

That made the mixed continuation the best 0.8B SFT checkpoint in the project.

### 4. RL / GRPO infrastructure became real, but not yet the winning performance path

The repo now contains working one-step and multi-turn GRPO training paths for BrowserGym-style tasks.

Reward shaping improved reward variance materially, especially on weak tasks like:
- `click-checkboxes-large`
- `find-word`
- `enter-text-2`

But the best RL runs still did not beat the best SFT checkpoint on full validation exact-match. For this project, RL was valuable as infrastructure and diagnosis, not as the final best model path.

## Final metrics worth citing

### Best absolute score
- `Qwen/Qwen3.5-2B`
- action-only SFT
- 87.50% exact-match on 240 validation rows

### Best small-model / final release checkpoint
- `Qwen/Qwen3.5-0.8B`
- action-only baseline SFT: 80.83%
- mixed weak-task continuation SFT: 83.33%
- baseline before fine-tuning: 17.92%

### Other strong reference points
- `Qwen/Qwen2.5-1.5B-Instruct` action-only: 79.58%
- `Qwen/Qwen2.5-1.5B-Instruct` reasoning+action: 81.67% strict / 83.75% canonicalized

## Recommended interpretation

If the goal is highest score regardless of size, the 2B action-only checkpoint is the strongest result.

If the goal is a compact model with a clean story, a large improvement margin, and a reproducible final checkpoint, the 0.8B mixed continuation is the best project endpoint.

That is why this repo is consolidated around the 0.8B mixed continuation as the final small-model release path.

## Release recommendation

If you are reproducing or building on this repo:

1. start with the phase1 action-only export
2. train the 0.8B action-only baseline
3. build the 500/500 mixed weak-task continuation subset
4. run continuation SFT
5. evaluate with the corrected Qwen3.5 conditional loader
