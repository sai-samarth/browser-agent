#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-json', required=True)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()

    obj = json.loads(Path(args.eval_json).read_text())
    rows = obj['rows']
    misses = [r for r in rows if not r.get('match')][:args.limit]
    print(json.dumps(obj['summary'], indent=2))
    print(f"misses={len([r for r in rows if not r.get('match')])}")
    for m in misses:
        print('---')
        print('task:', m['task_name'])
        print('target:', m['target'])
        print('prediction:', m['prediction'])
        print((m['raw_generation'] or '')[:1200].replace('\n', '\\n'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
