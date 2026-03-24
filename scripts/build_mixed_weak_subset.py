#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def sample_with_replacement(rows, n, rng):
    return [dict(rng.choice(rows)) for _ in range(n)]


def sample_without_replacement(rows, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return [dict(x) for x in rows[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dataset-dir', required=True)
    ap.add_argument('--output-dataset-dir', required=True)
    ap.add_argument('--weak-tasks', nargs='+', required=True)
    ap.add_argument('--weak-train-size', type=int, default=500)
    ap.add_argument('--other-train-size', type=int, default=500)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--val-mode', choices=['full', 'filtered'], default='full')
    args = ap.parse_args()

    source = load_from_disk(args.source_dataset_dir)
    weak_set = set(args.weak_tasks)
    rng = random.Random(args.seed)

    all_train = [dict(x) for x in source['train']]
    weak_train = [x for x in all_train if x.get('metadata', {}).get('task_name') in weak_set]
    other_train = [x for x in all_train if x.get('metadata', {}).get('task_name') not in weak_set]
    if not weak_train:
        raise SystemExit('No weak-task train rows found')
    if not other_train:
        raise SystemExit('No non-weak train rows found')

    if len(weak_train) >= args.weak_train_size:
        sampled_weak = sample_without_replacement(weak_train, args.weak_train_size, rng)
        weak_dup = 0
    else:
        sampled_weak = [dict(x) for x in weak_train] + sample_with_replacement(weak_train, args.weak_train_size - len(weak_train), rng)
        weak_dup = args.weak_train_size - len(weak_train)

    if len(other_train) >= args.other_train_size:
        sampled_other = sample_without_replacement(other_train, args.other_train_size, rng)
        other_dup = 0
    else:
        sampled_other = [dict(x) for x in other_train] + sample_with_replacement(other_train, args.other_train_size - len(other_train), rng)
        other_dup = args.other_train_size - len(other_train)

    train_rows = sampled_weak + sampled_other
    rng.shuffle(train_rows)

    if args.val_mode == 'filtered':
        val_rows = [dict(x) for x in source['val'] if x.get('metadata', {}).get('task_name') in weak_set]
    else:
        val_rows = [dict(x) for x in source['val']]

    out = DatasetDict({
        'train': Dataset.from_list(train_rows),
        'val': Dataset.from_list(val_rows),
    })
    out_path = Path(args.output_dataset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_path))

    manifest = {
        'source_dataset_dir': args.source_dataset_dir,
        'output_dataset_dir': str(out_path),
        'weak_tasks': sorted(weak_set),
        'seed': args.seed,
        'weak_train_size': args.weak_train_size,
        'other_train_size': args.other_train_size,
        'total_train_size': len(train_rows),
        'available_weak_rows': len(weak_train),
        'available_other_rows': len(other_train),
        'weak_duplicates_added': weak_dup,
        'other_duplicates_added': other_dup,
        'val_mode': args.val_mode,
        'source_weak_counts': dict(Counter(x.get('metadata', {}).get('task_name') for x in weak_train)),
        'source_other_counts': dict(Counter(x.get('metadata', {}).get('task_name') for x in other_train)),
        'train_counts': dict(Counter(x.get('metadata', {}).get('task_name') for x in train_rows)),
        'val_counts': dict(Counter(x.get('metadata', {}).get('task_name') for x in val_rows)),
    }
    (out_path / 'subset_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
