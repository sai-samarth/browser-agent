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
