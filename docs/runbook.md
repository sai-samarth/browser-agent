# Browser Agent Runbook

## Environment assumptions

This project expects:
- Python 3.10+
- `uv` for dependency management
- a working OpenEnv BrowserGym server, usually exposed on `http://localhost:8000`
- optional teacher endpoint compatible with the OpenAI chat-completions API

## Core workflows

### Single-run collection

```bash
uv run python scripts/collect_rollouts.py --config configs/rollout_config.yaml
```

Outputs:
- `data/rollouts/<run_name>_<timestamp>/trajectory_steps.jsonl`
- `data/rollouts/<run_name>_<timestamp>/episode_summaries.jsonl`
- `data/rollouts/<run_name>_<timestamp>/resolved_config.yaml`

### Parallel collection

```bash
uv run python scripts/run_parallel_miniwob.py   --config-template configs/rollout_config.yaml   --workers 4   --run-name-prefix miniwob_train
```

This writes worker configs and logs under `data/parallel_runs/<run_name>_<timestamp>/`.

### Summarize one parallel run

```bash
uv run python scripts/summarize_parallel_run.py   --parallel-run-root data/parallel_runs/<run_name>
```

### Build a baseline report over all discovered runs

```bash
uv run python scripts/analyze_rollout_corpus.py
```

Default outputs:
- `reports/baselines/latest.md`
- `reports/baselines/latest.json`

## Recommended operating pattern

1. Validate a teacher/backend on a small fixed eval slice.
2. Run a larger parallel job only after connectivity is stable.
3. Summarize the run immediately.
4. Classify the result as:
   - usable for SFT
   - usable only for debugging
   - broken / incomplete
5. Only then merge those traces into a curated export.

## Known historical issues

- task switching used to be inconsistent until the server reset path was fixed
- screenshot-heavy observations previously hurt latency badly
- later Z.ai runs failed due missing key or connection errors

## Logging expectations

Good production runs should have:
- worker log files for every worker
- `episode_summaries.jsonl` for every completed rollout dir
- low fallback rates
- low action error counts
- non-zero task coverage

## Git hygiene

This repo tracks code, configs, docs, and small derived reports.
Raw generated corpora remain local and should stay out of git.
