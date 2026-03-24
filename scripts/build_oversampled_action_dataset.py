#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dataset-dir', required=True)
    ap.add_argument('--output-dataset-dir', required=True)
    ap.add_argument('--weak-tasks', nargs='+', required=True)
    ap.add_argument('--weak-multiplier', type=int, default=4, help='Total copies for weak-task train rows')
    args = ap.parse_args()

    source = load_from_disk(args.source_dataset_dir)
    train_rows = [dict(x) for x in source['train']]
    val_rows = [dict(x) for x in source['val']]
    weak = set(args.weak_tasks)
    boosted_train = []
    for row in train_rows:
        task = row.get('metadata', {}).get('task_name')
        copies = args.weak_multiplier if task in weak else 1
        for _ in range(copies):
            boosted_train.append(dict(row))

    out = DatasetDict({
        'train': Dataset.from_list(boosted_train),
        'val': Dataset.from_list(val_rows),
    })
    out_path = Path(args.output_dataset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_path))

    src_counts = Counter(r.get('metadata', {}).get('task_name') for r in train_rows)
    out_counts = Counter(r.get('metadata', {}).get('task_name') for r in boosted_train)
    manifest = {
        'source_dataset_dir': args.source_dataset_dir,
        'output_dataset_dir': str(out_path),
        'weak_tasks': sorted(weak),
        'weak_multiplier': args.weak_multiplier,
        'source_train_rows': len(train_rows),
        'boosted_train_rows': len(boosted_train),
        'val_rows': len(val_rows),
        'source_counts': dict(src_counts),
        'boosted_counts': dict(out_counts),
    }
    (out_path / 'oversample_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
