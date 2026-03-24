#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BASE = Path('outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json')
WEAK = Path('outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft/eval_after_conditional_256.json')
MIXED = Path('outputs/qwen35-0.8b-browser-action-weak3-mixed50-1000-cont-sft/eval_after_conditional_256.json')
TASKS = {'click-checkboxes-large', 'find-word', 'enter-text-2'}

base = json.loads(BASE.read_text())
weak = json.loads(WEAK.read_text())
mixed = json.loads(MIXED.read_text())

assert len(base['rows']) == len(weak['rows']) == len(mixed['rows'])

def summarize(a_rows, b_rows):
    by_task = defaultdict(lambda: {'count':0,'a':0,'b':0,'improved':0,'regressed':0})
    for a,b in zip(a_rows,b_rows):
        t = a['task_name']
        d = by_task[t]
        d['count'] += 1
        d['a'] += int(a['match'])
        d['b'] += int(b['match'])
        d['improved'] += int((not a['match']) and b['match'])
        d['regressed'] += int(a['match'] and (not b['match']))
    rows = []
    for t,d in sorted(by_task.items()):
        rows.append({
            'task_name': t,
            'count': d['count'],
            'before_exact': d['a']/d['count'],
            'after_exact': d['b']/d['count'],
            'delta': (d['b']-d['a'])/d['count'],
            'improved_rows': d['improved'],
            'regressed_rows': d['regressed'],
        })
    return rows

base_to_mixed = summarize(base['rows'], mixed['rows'])
weak_to_mixed = summarize(weak['rows'], mixed['rows'])

report = {
    'summary': {
        'baseline_exact': base['summary']['exact_match'],
        'weak_only_exact': weak['summary']['exact_match'],
        'mixed_exact': mixed['summary']['exact_match'],
        'mixed_minus_baseline': mixed['summary']['exact_match'] - base['summary']['exact_match'],
        'mixed_minus_weak_only': mixed['summary']['exact_match'] - weak['summary']['exact_match'],
        'baseline_parseable': base['summary']['parseable_rate'],
        'weak_only_parseable': weak['summary']['parseable_rate'],
        'mixed_parseable': mixed['summary']['parseable_rate'],
    },
    'baseline_to_mixed': base_to_mixed,
    'weak_only_to_mixed': weak_to_mixed,
    'target_tasks_baseline_to_mixed': [r for r in base_to_mixed if r['task_name'] in TASKS],
    'target_tasks_weak_only_to_mixed': [r for r in weak_to_mixed if r['task_name'] in TASKS],
}
print(json.dumps(report, indent=2))
