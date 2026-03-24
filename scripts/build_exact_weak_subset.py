#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dataset-dir', required=True)
    ap.add_argument('--output-dataset-dir', required=True)
    ap.add_argument('--tasks', nargs='+', required=True)
    ap.add_argument('--train-size', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--val-mode', choices=['full', 'filtered'], default='full')
    args = ap.parse_args()

    source = load_from_disk(args.source_dataset_dir)
    task_set = set(args.tasks)
    rng = random.Random(args.seed)

    weak_train = [dict(x) for x in source['train'] if x.get('metadata', {}).get('task_name') in task_set]
    if not weak_train:
        raise SystemExit('No matching train rows found for requested tasks')

    weak_counts = Counter(x.get('metadata', {}).get('task_name') for x in weak_train)
    train_rows = list(weak_train)
    if len(train_rows) > args.train_size:
        rng.shuffle(train_rows)
        train_rows = train_rows[: args.train_size]
    elif len(train_rows) < args.train_size:
        deficit = args.train_size - len(train_rows)
        sampled = [dict(rng.choice(weak_train)) for _ in range(deficit)]
        train_rows.extend(sampled)
    rng.shuffle(train_rows)

    if args.val_mode == 'filtered':
        val_rows = [dict(x) for x in source['val'] if x.get('metadata', {}).get('task_name') in task_set]
    else:
        val_rows = [dict(x) for x in source['val']]

    out = DatasetDict({
        'train': Dataset.from_list(train_rows),
        'val': Dataset.from_list(val_rows),
    })
    out_path = Path(args.output_dataset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_path))

    train_counts = Counter(x.get('metadata', {}).get('task_name') for x in train_rows)
    val_counts = Counter(x.get('metadata', {}).get('task_name') for x in val_rows)
    manifest = {
        'source_dataset_dir': args.source_dataset_dir,
        'output_dataset_dir': str(out_path),
        'tasks': sorted(task_set),
        'seed': args.seed,
        'target_train_size': args.train_size,
        'unique_weak_train_rows': len(weak_train),
        'duplicated_rows_added': max(0, args.train_size - len(weak_train)),
        'val_mode': args.val_mode,
        'source_weak_train_counts': dict(weak_counts),
        'train_counts': dict(train_counts),
        'val_counts': dict(val_counts),
    }
    (out_path / 'subset_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
