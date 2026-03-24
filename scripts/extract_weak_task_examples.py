#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

TASKS = ["click-checkboxes-large", "find-word", "enter-text-2"]
OLD = Path("outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json")
NEW = Path("outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft/eval_after_conditional_256.json")
old = json.loads(OLD.read_text())['rows']
new = json.loads(NEW.read_text())['rows']

for task in TASKS:
    print(f"=== TASK: {task} ===")
    improved = []
    regressed = []
    remaining = []
    for i, (o, n) in enumerate(zip(old, new)):
        if n['task_name'] != task:
            continue
        rec = {
            'index': i,
            'target': n['target'],
            'old_prediction': o['prediction'],
            'new_prediction': n['prediction'],
            'old_match': o['match'],
            'new_match': n['match'],
            'old_raw_generation': o.get('raw_generation'),
            'new_raw_generation': n.get('raw_generation'),
        }
        if (not o['match']) and n['match']:
            improved.append(rec)
        elif o['match'] and (not n['match']):
            regressed.append(rec)
        elif (not o['match']) and (not n['match']) and (o['prediction'] != n['prediction']):
            remaining.append(rec)
    print('-- improved --')
    print(json.dumps(improved[:4], indent=2))
    print('-- regressed --')
    print(json.dumps(regressed[:4], indent=2))
    print('-- changed but still wrong --')
    print(json.dumps(remaining[:4], indent=2))
