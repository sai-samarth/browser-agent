# Phase 1 Core MiniWoB Task Set

This file defines the initial MiniWoB task subset for Phase 1 trace generation.

Goal:
- produce a clean SFT-oriented browser-use corpus for a small model
- bias toward tasks the current collector and teacher interface already handle well
- avoid wasting generation budget on task families that are currently bottlenecked by environment mismatch or poor grounding

## Selection rule

This initial set was chosen from tasks that looked reliably solvable across the two strongest historical runs:

- `miniwob_train_20260221_164638` — best coverage baseline
- `miniwob_train_zai_20260221_180613` — best accuracy baseline

Heuristic used:
- prioritize tasks with consistently high success across both runs
- prefer tasks with straightforward browser-control semantics
- de-prioritize geometry, canvas-heavy, and clearly action-mismatch-sensitive tasks for now

## Included task families

The current Phase 1 core covers:
- direct clicking and simple selection
- form and text entry
- low-horizon navigation
- structured lookup and extraction
- a small amount of multi-step interaction that already appears stable

Representative examples:
- `click-test`
- `click-checkboxes`
- `click-dialog`
- `enter-text`
- `focus-text`
- `phone-book`
- `read-table`
- `simple-arithmetic`
- `useful low-horizon tasks with clear accessible-tree grounding`

## Excluded for now

These are not permanently excluded. They are excluded from the first large SFT-oriented generation wave because they currently look like poor uses of collection budget.

### Action-mismatch heavy
- `ascending-numbers`
- `choose-date`
- related widget families where the teacher seems right but the action binding fails

### Observation-quality heavy
- `click-color`
- any tasks that frequently expose root-only or near-empty observations

### Long-horizon / hidden-state heavy
- `click-pie`
- other tasks with repeated-action loops and low reward progress

### Geometry / canvas heavy
- `bisect-angle`
- `circle-center`
- drag/draw style tasks that likely need richer grounding or action support

## How to use it

For future parallel runs, point `run_parallel_miniwob.py` at this task list via `--task-list-file`.

Example:

```bash
.venv/bin/python scripts/run_parallel_miniwob.py   --config-template configs/rollout_config.yaml   --task-list-file data/task_lists/phase1_core.txt   --workers 4   --run-name-prefix miniwob_phase1_core
```

## Why this matters

Phase 1 is about getting enough *useful* traces for fine-tuning, not maximizing raw benchmark coverage immediately.

A narrower but cleaner task slice should improve:
- corpus quality
- teacher efficiency
- debugging speed
- signal-to-noise ratio in the first student fine-tune

Later phases can expand coverage once action compatibility and observation diagnostics improve.
