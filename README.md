# Browser Agent

BrowserGym + OpenEnv trace-generation workspace for collecting browser-use trajectories, analyzing rollout quality, and preparing datasets for small-model fine-tuning.

Current focus:
- generate high-quality MiniWoB traces
- convert traces into SFT-ready formats
- later extend the same stack toward RL methods such as GRPO

## Current capabilities

The repo already contains working collection and analysis primitives:

- `scripts/collect_rollouts.py`
  - connects to OpenEnv over WebSocket
  - supports scripted and teacher-driven action generation
  - writes `trajectory_steps.jsonl` and `episode_summaries.jsonl`
- `scripts/run_parallel_miniwob.py`
  - shards MiniWoB tasks across workers
  - launches per-worker BrowserGym containers and collectors
- `scripts/summarize_parallel_run.py`
  - aggregates worker outputs for one parallel run

## Repository layout

- `configs/` — rollout and teacher configs
- `scripts/` — collection, orchestration, summarization, and analysis scripts
- `docs/` — runbooks, schema notes, and project documentation
- `data/` — local generated artifacts and task lists
- `reports/` — committed analysis snapshots and derived summaries

## Data model

Raw collection outputs land in timestamped run directories under `data/rollouts/`:

- `trajectory_steps.jsonl` — one JSON object per environment step
- `episode_summaries.jsonl` — one JSON object per episode
- `resolved_config.yaml` — exact config used for the run

Parallel runs write orchestration metadata under `data/parallel_runs/`:

- per-worker configs
- per-worker logs
- references to the rollout directories created by each worker

See `docs/data-schema.md` for the current schema and intended downstream exports.

## Quick start

### 1. Sync dependencies

```bash
uv sync
```

### 2. Run a single collector

```bash
uv run python scripts/collect_rollouts.py --config configs/rollout_config.yaml
```

### 3. Run a parallel MiniWoB collection job

```bash
uv run python scripts/run_parallel_miniwob.py   --config-template configs/rollout_config.yaml   --workers 4   --run-name-prefix miniwob_train
```

### 4. Summarize one parallel run

```bash
uv run python scripts/summarize_parallel_run.py   --parallel-run-root data/parallel_runs/<run_name>
```

### 5. Build a corpus baseline report across existing runs

```bash
uv run python scripts/analyze_rollout_corpus.py
```

Generated reports are written under `reports/baselines/` by default.

## Phase 1 plan

Phase 1 is about producing enough high-quality traces to fine-tune a small browser-use model.

Execution order:
1. make the repo proper and reproducible
2. baseline the existing runs
3. diagnose accuracy bottlenecks by task family and failure mode
4. stabilize one teacher backend
5. improve trace quality on a fixed evaluation slice
6. define SFT eligibility rules
7. export training-ready datasets
8. scale generation

A detailed plan is kept in `.hermes/plans/` for local project planning.

## Current status

- collection infrastructure works
- multiple real rollout corpora already exist locally under `data/`
- best historical runs were produced on 2026-02-21
- later 2026-02-24 attempts appear to have failed mostly from teacher backend auth/connectivity issues

## Notes

- This repo intentionally does not track raw rollout corpora in git.
- Large generated artifacts under `data/rollouts/` and `data/parallel_runs/` stay local.
- Committed reports should be small, derived summaries rather than raw traces.
