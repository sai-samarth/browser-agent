# browser-agent

A reproducible BrowserGym + MiniWoB research repo for collecting browser-use traces, exporting SFT datasets, fine-tuning small multimodal models, and stress-testing GRPO on browser tasks.

This repo is the final consolidated snapshot of the project.

## What this project showed

### Headline results

| Model | Training setup | Validation exact-match |
|---|---|---:|
| Qwen2.5-1.5B-Instruct | action-only SFT | 79.58% |
| Qwen2.5-1.5B-Instruct | reasoning+action SFT | 81.67% strict / 83.75% canonicalized |
| Qwen3.5-0.8B | action-only SFT | 80.83% |
| Qwen3.5-0.8B | mixed weak-task continuation SFT | 83.33% |
| Qwen3.5-2B | action-only SFT | 87.50% |
| Qwen3.5-2B | reinforced reasoning+action SFT | 84.17% with reinforced prompt, 29.17% on default prompt |

### Final takeaways

- Action-only SFT was the most reliable format for one-step BrowserGym execution.
- Reasoning+action often preserved task knowledge but made executable formatting brittle under default prompts.
- A Qwen3.5-specific evaluation bug initially hid real gains. Correct conditional-generation loading was required for valid Qwen3.5 evaluation.
- Targeted weak-task continuation helped, but a pure weak-task continuation over-specialized.
- The best 0.8B checkpoint came from a mixed continuation: 500 weak-task rows + 500 broad rows, reaching 83.33% exact-match.
- GRPO infrastructure was validated and reward shaping improved signal, but no GRPO run beat the best SFT checkpoint.

## Which checkpoint to use

Two answers are true at once:

- Best absolute score in this repo: `outputs/qwen35-2b-browser-action-unsloth` at 87.50% exact-match.
- Best small-model / final release checkpoint: `outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft` at 83.33% exact-match.

If you want the smallest strong checkpoint to build on, start with the 0.8B mixed continuation.

## Repository contents

- `scripts/collect_rollouts.py` — OpenEnv / BrowserGym rollout collection
- `scripts/export_sft_dataset.py` — export rollout traces into action-only and reasoning+action chat datasets
- `scripts/train_unsloth_sft_generic.py` — baseline Unsloth SFT
- `scripts/train_qwen35_continuation_sft.py` — continuation SFT from an existing Qwen3.5 adapter
- `scripts/eval_action_model.py` — strict exact-match evaluation with the corrected Qwen3.5 loader path
- `scripts/train_browsergym_grpo.py` and `scripts/train_browsergym_grpo_multiturn.py` — local RL / GRPO experiments
- `configs/` — tracked configs used for the main experiments
- `docs/` — consolidated notes, methodology, reproducibility, and final writeups

## Reproducibility paths

### Fastest path: reproduce the final 0.8B result

1. Prepare the exported base dataset locally, either by:
   - downloading the released dataset from Hugging Face, or
   - rebuilding it from local rollout traces.
2. Train the Qwen3.5-0.8B action-only baseline adapter.
3. Build the mixed weak-task continuation subset.
4. Run continuation SFT.
5. Evaluate with the corrected conditional-generation path.

See:
- `docs/reproducibility.md`
- `scripts/download_hf_dataset.py`
- `scripts/train_qwen35_08b_action_baseline.sh`
- `scripts/reproduce_qwen35_08b_mixed_best.sh`

## Important docs

- `docs/final-summary.md` — concise project conclusions
- `docs/reproducibility.md` — exact commands and environment notes
- `docs/current-status.md` — final consolidated status snapshot
- `docs/experiment-results.md` — detailed experiment log with metrics
- `docs/work-log.md` — chronological project notes

## Environment notes

This repo has two practical dependency tiers:

- Lightweight analysis / export: `uv sync`
- GPU fine-tuning / RL: a dedicated CUDA environment with `torch`, `transformers`, `peft`, `datasets`, and `unsloth`

Important RL note for Qwen3.5:

- the old RL env with `transformers==4.57.3` was not sufficient for the later Qwen3.5 work
- the fixed path used `transformers==5.3.0` and `weave==0.52.35`

## What is intentionally not committed

To keep the repo reproducible without becoming enormous, raw rollout corpora and full adapter weights are not the main source of truth in git.

The repo commits:
- manifests
- configs
- scripts
- compact evaluation artifacts
- experiment notes

The repo does not rely on chat memory for project state. The final record lives in `docs/`.
