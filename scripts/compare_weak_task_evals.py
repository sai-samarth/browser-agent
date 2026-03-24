#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ["click-checkboxes-large", "find-word", "enter-text-2"]
OLD = Path("outputs/qwen35-0.8b-browser-action-unsloth/eval_after_conditional_256.json")
NEW = Path("outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft/eval_after_conditional_256.json")
OUT = Path("outputs/qwen35-0.8b-browser-action-weak3-exact1000-cont-sft/weak3_comparison_report.json")

old = json.loads(OLD.read_text())
new = json.loads(NEW.read_text())
old_rows = old["rows"]
new_rows = new["rows"]
assert len(old_rows) == len(new_rows)

per_task = {}
examples = defaultdict(list)
transition_counts = Counter()

for task in TASKS:
    idxs = [i for i, r in enumerate(new_rows) if r["task_name"] == task]
    old_task = [old_rows[i] for i in idxs]
    new_task = [new_rows[i] for i in idxs]
    before = sum(1 for r in old_task if r["match"])
    after = sum(1 for r in new_task if r["match"])
    per_task[task] = {
        "count": len(idxs),
        "before_exact": before / len(idxs) if idxs else 0.0,
        "after_exact": after / len(idxs) if idxs else 0.0,
        "delta": (after - before) / len(idxs) if idxs else 0.0,
    }
    for i in idxs:
        o = old_rows[i]
        n = new_rows[i]
        trans = f"{int(o['match'])}->{int(n['match'])}"
        transition_counts[(task, trans)] += 1
        if o["prediction"] != n["prediction"] or o["match"] != n["match"]:
            examples[task].append({
                "index": i,
                "target": n["target"],
                "old_prediction": o["prediction"],
                "new_prediction": n["prediction"],
                "old_match": o["match"],
                "new_match": n["match"],
                "old_raw_generation": o.get("raw_generation"),
                "new_raw_generation": n.get("raw_generation"),
            })

other_idxs = [i for i, r in enumerate(new_rows) if r["task_name"] not in TASKS]
other_old = [old_rows[i] for i in other_idxs]
other_new = [new_rows[i] for i in other_idxs]
summary = {
    "overall_before_exact": old["summary"]["exact_match"],
    "overall_after_exact": new["summary"]["exact_match"],
    "overall_delta": new["summary"]["exact_match"] - old["summary"]["exact_match"],
    "overall_before_parseable": old["summary"]["parseable_rate"],
    "overall_after_parseable": new["summary"]["parseable_rate"],
    "weak_tasks": per_task,
    "other_tasks": {
        "count": len(other_idxs),
        "before_exact": sum(1 for r in other_old if r["match"]) / len(other_idxs) if other_idxs else 0.0,
        "after_exact": sum(1 for r in other_new if r["match"]) / len(other_idxs) if other_idxs else 0.0,
    },
    "transition_counts": {
        task: {
            trans: transition_counts[(task, trans)]
            for trans in ["0->0", "0->1", "1->0", "1->1"]
        }
        for task in TASKS
    },
    "changed_examples": {task: examples[task][:12] for task in TASKS},
}

OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
