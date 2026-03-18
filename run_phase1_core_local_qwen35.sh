#!/usr/bin/env bash
set -euo pipefail
cd /home/saisamarth/projects/browser-agent
.venv/bin/python scripts/run_parallel_miniwob.py   --config-template configs/rollout_config_phase1_local_qwen35_core.yaml   --task-list-file data/task_lists/phase1_core.txt   --workers 4   --episodes-per-task 3   --run-name-prefix miniwob_phase1_core_local_qwen35
