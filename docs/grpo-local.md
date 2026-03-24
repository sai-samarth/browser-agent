# Local BrowserGym GRPO Runbook

## What this adds

This repo now has a local GRPO training path for one-step BrowserGym action models.

It is intentionally minimal:
- no Modal
- no Python `browsergym` / `openenv` dependency for the first smoke path
- reuses the existing OpenEnv BrowserGym websocket server already used by rollout collection
- trains against static one-step prompts built from real initial BrowserGym observations
- computes rewards by resetting the same task+seed and executing the model's predicted action

This fits the current project objective better than a full online multi-step RL loop because the existing SFT/eval stack is already one-step next-action supervision.

## Optional Python dependencies

Use a dedicated RL environment and install the repo with the `rl` extra:

```bash
pip install -e '.[rl]'
```

If you later want vLLM-backed generation for GRPO, also install:

```bash
pip install -e '.[rl-vllm]'
```

## Start a local BrowserGym server for smoke testing

```bash
docker rm -f browsergym-click-test 2>/dev/null || true
docker run -d \
  --name browsergym-click-test \
  -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK=miniwob \
  -e BROWSERGYM_TASK_NAME=click-test \
  browsergym-env:latest
```

## Dry-run reward sanity check

```bash
python scripts/train_browsergym_grpo.py \
  --config configs/grpo_smoke_qwen25_1p5b_click-test.yaml \
  --dry-run-reward
```

This will:
- reset the BrowserGym env across a small seed slice
- build and save a prompt preview dataset
- run the reward functions on a trivial `noop()` completion

## Smoke training run

```bash
python scripts/train_browsergym_grpo.py \
  --config configs/grpo_smoke_qwen25_1p5b_click-test.yaml
```

The default smoke config is intentionally tiny:
- Qwen2.5-1.5B-Instruct
- LoRA enabled
- 8 prompt rows
- 2 GRPO optimizer steps
- 2 generations per prompt
- no vLLM

## Current scope and caveats

- This is a local one-step GRPO path, not yet a full multi-step online RL loop.
- Reward is based on the initial BrowserGym state only.
- For the first integration, causal text models are the safest target.
- Qwen3.5 conditional-generation models may require extra work for stable GRPO support.
- Once this local path is validated, the next extension should be multi-step episode rewards over the same websocket server.


## Multi-turn Phase A

The next RL step after one-step smoke validation is a narrow multi-turn GRPO curriculum.

Current Phase A design:
- warm start: `outputs/qwen25-1.5b-browser-action-lora`
- trainer: `scripts/train_browsergym_grpo_multiturn.py`
- config: `configs/grpo_multiturn_phase_a_qwen25_action_adapter.yaml`
- tasks: `click-button`, `click-option`, `enter-text-2`
- rollout length: up to 10 steps
- stop early on success
- reward: rollout-level success reward plus small penalties for invalid actions, action errors, and extra steps

Run it with:

```bash
source /home/saisamarth/venvs/ft-qwen25/bin/activate
python scripts/train_browsergym_grpo_multiturn.py   --config configs/grpo_multiturn_phase_a_qwen25_action_adapter.yaml
```

Why Phase A exists:
- the earlier full 30-task one-step GRPO run completed, but too many prompts got only parseability reward and zero reward variance
- multi-turn RL is intended to credit useful early actions that only pay off several steps later
- the narrow task subset keeps rollout behavior inspectable before scaling up


## Phase A.2 findings

Phase A.2 changed multi-turn rollout generation so later steps sample instead of decoding greedily.

Observed effect:
- rollout diversity improved immediately
- the first logged batch showed strong reward variance (`reward_std≈2.49`)
- later batches still often collapsed to tied rewards, meaning the curriculum is still partially too easy

Task-level replay on the Phase A.2 seed slice:
- `click-option`: 3/4 successes, avg reward 2.32
- `enter-text-2`: 3/4 successes, avg reward 2.315
- `enter-password`: 1/4 successes, avg reward 0.85

Current recommendation:
- do not scale back to the full 30-task phase-1 set yet
- use a harder narrow curriculum next
- prioritize `enter-password`, which is currently the clearest remaining source of useful RL signal


## Phase B

Phase B widens the multi-turn curriculum while staying far narrower than the full 30-task phase-1 set.

Config:
- `configs/grpo_multiturn_phase_b_qwen25_action_adapter.yaml`

Task mix:
- `enter-password`
- `click-option`
- `enter-text-2`
- `click-test-2`
- `click-checkboxes-transfer`

Key changes versus Phase A.2:
- 10 seed rollouts per task instead of 4
- 5 tasks instead of 3
- slightly higher later-step rollout sampling temperature (`0.9`)

Why this exists:
- Phase A.2 showed that sampled later rollout steps improve reward variance
- but two of the three tasks were already close to saturation
- Phase B keeps the curriculum manageable while increasing both task diversity and the chance of positive-but-nontrivial outcomes
