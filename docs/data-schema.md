# Data Schema Notes

## Raw rollout structure

Each rollout run lives under:

```text
data/rollouts/<run_name>_<timestamp>/
```

Expected files:
- `trajectory_steps.jsonl`
- `episode_summaries.jsonl`
- `resolved_config.yaml`

## `trajectory_steps.jsonl`

One JSON object per environment step.

Observed fields include:
- `timestamp`
- `run_id`
- `benchmark`
- `task_name`
- `episode_id`
- `episode_idx`
- `step_idx`
- `seed`
- `policy_mode`
- `pre_observation`
- `post_observation`
- `action_str`
- `reward`
- `done`
- `last_action_error`
- `latency_ms`
- `teacher_model`
- `teacher_response_answer`
- `teacher_response_reasoning`
- `teacher_latency_ms`
- `teacher_usage`
- `teacher_used_fallback`
- `observation_actionable_node_count`
- `observation_text_line_count`
- `observation_is_root_only`
- `observation_is_sparse`
- `same_action_as_previous`
- `consecutive_same_action_count`
- `no_progress_streak`
- `repeated_action_loop`

The exact set can vary across older runs.

## `episode_summaries.jsonl`

One JSON object per episode.

Observed summary-level fields include:
- `timestamp`
- `task_name`
- `episode_idx`
- `seed`
- `success`
- `cum_reward`
- `num_steps`
- `final_done`
- `action_error_count`
- optional `teacher_fallback_count`
- `root_only_observation_count`
- `sparse_observation_count`
- `repeated_action_loop_count`
- `max_consecutive_same_action_count`
- `max_no_progress_streak`

## Intended downstream exports

The dataset-prep stage should support at least these exports:

1. Step-level action supervision
   - input: observation plus short history
   - target: action only

2. Step-level reasoning plus action supervision
   - input: observation plus short history
   - target: reasoning plus action

3. Episode-expanded export
   - one record per step
   - deterministic schema suitable for train/val splitting

## Data quality concepts

We will likely classify traces into:
- Tier A — successful and clean
- Tier B — partially useful but imperfect
- Tier C — debugging only

This document is descriptive for now. Filtering rules will live separately once formalized.
