#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OLD = Path("outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json")
NEW = Path("outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft/eval_after_conditional_256.json")
old = json.loads(OLD.read_text())
new = json.loads(NEW.read_text())
old_rows = old['rows']
new_rows = new['rows']

by_task = defaultdict(lambda: {'count':0,'before':0,'after':0,'improved':0,'regressed':0})
for o,n in zip(old_rows,new_rows):
    t = n['task_name']
    d = by_task[t]
    d['count'] += 1
    d['before'] += int(o['match'])
    d['after'] += int(n['match'])
    d['improved'] += int((not o['match']) and n['match'])
    d['regressed'] += int(o['match'] and (not n['match']))

rows=[]
for t,d in by_task.items():
    rows.append({
        'task_name': t,
        'count': d['count'],
        'before_exact': d['before']/d['count'],
        'after_exact': d['after']/d['count'],
        'delta': (d['after']-d['before'])/d['count'],
        'improved_rows': d['improved'],
        'regressed_rows': d['regressed'],
    })
rows.sort(key=lambda x: x['delta'])
print(json.dumps(rows, indent=2))
