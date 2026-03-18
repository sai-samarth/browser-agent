#!/usr/bin/env bash
set -euo pipefail
cd /home/saisamarth/projects/browser-agent

RUNS=${RUNS:-8}
SEED_STRIDE=${SEED_STRIDE:-1000000}
WORKERS=${WORKERS:-4}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-10}
TASK_FILE=${TASK_FILE:-data/task_lists/phase1_production_tasks.txt}
CONFIG=${CONFIG:-configs/rollout_config_phase1_local_qwen35_core.yaml}
CAMPAIGN=${CAMPAIGN:-batch1}

for ((i=0; i<RUNS; i++)); do
  OFFSET=$((i * SEED_STRIDE))
  RUN_NAME=$(printf 'miniwob_phase1_prod_local_qwen35_%s_r%02d' "$CAMPAIGN" "$i")
  echo "[campaign] starting run=$RUN_NAME seed_offset_base=$OFFSET"
  .venv/bin/python scripts/run_parallel_miniwob.py     --config-template "$CONFIG"     --task-list-file "$TASK_FILE"     --workers "$WORKERS"     --episodes-per-task "$EPISODES_PER_TASK"     --seed-offset-base "$OFFSET"     --run-name-prefix "$RUN_NAME"
  echo "[campaign] completed run=$RUN_NAME"
  echo
  sleep 5
done
