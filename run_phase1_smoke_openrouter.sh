#!/usr/bin/env bash
set -euo pipefail
cd /home/saisamarth/projects/browser-agent
.venv/bin/python scripts/run_parallel_miniwob.py   --config-template configs/rollout_config_phase1_openrouter.yaml   --task-list-file data/task_lists/phase1_smoke.txt   --workers 2   --episodes-per-task 1   --run-name-prefix miniwob_phase1_smoke_or
