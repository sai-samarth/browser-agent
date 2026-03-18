# Phase 1 local collection campaign

This campaign script launches repeated local-production collection runs with distinct seed offsets.

Defaults:
- runs: 8
- workers: 4
- episodes per task: 10
- seed stride: 1,000,000
- task file: `data/task_lists/phase1_production_tasks.txt`
- config: `configs/rollout_config_phase1_local_qwen35_core.yaml`

Purpose:
- accumulate multiple non-overlapping runs toward the first 10k-step corpus milestone
- preserve reproducibility through deterministic seed offsets
- keep the teacher fixed to the local Qwen3.5-9B endpoint
