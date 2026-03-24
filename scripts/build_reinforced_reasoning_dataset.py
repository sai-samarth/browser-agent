#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

STRICT_SUFFIX = (
    "\n\nStrict output rule:\n"
    "1. Always produce exactly one <think>...</think> block.\n"
    "2. After </think>, the final non-empty line must be exactly one valid BrowserGym action.\n"
    "3. Never end with prose only.\n"
    "4. Do not omit the final action line.\n"
    "5. Follow this exact shape:\n"
    "<think>\nbrief reasoning\n</think>\naction_here"
)

EXAMPLE_USER = """Example Task: click-button
Goal: Click the button labeled submit.
Current URL: http://example.invalid
Last action had error: False
Observation diagnostics: actionable_nodes=1, text_lines=2, sparse=False, root_only=False

Recent history:
(none)

Current observation:
RootWebArea 'Example'
	[18] button 'submit'"""

EXAMPLE_ASSISTANT = """<think>
There is a single submit button with bid 18. I should click it.
</think>
click('18')"""

USER_REMINDER = (
    "\n\nReminder: answer in exactly this form:\n"
    "<think>\nreasoning\n</think>\n"
    "one final BrowserGym action line"
)


def _rewrite_system(system_text: str) -> str:
    needle = "Think step by step before outputting the single next BrowserGym action. Put reasoning inside exactly one <think>...</think> block, then output only the action on the next line."
    if needle in system_text:
        return system_text.replace(needle, needle + STRICT_SUFFIX)
    return system_text.rstrip() + STRICT_SUFFIX


def _transform_messages(messages: list[dict]) -> list[dict]:
    if len(messages) != 3:
        raise ValueError(f"Expected 3 messages, got {len(messages)}")
    system, user, assistant = messages
    return [
        {"role": "system", "content": _rewrite_system(system["content"])},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT},
        {"role": "user", "content": user["content"] + USER_REMINDER},
        {"role": "assistant", "content": assistant["content"]},
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dataset-dir', default='data/exports/phase1_sft_v2/reasoning_action/hf_dataset')
    ap.add_argument('--output-root', default='data/exports/phase1_sft_v3')
    ap.add_argument('--variant-name', default='reasoning_action_reinforced')
    args = ap.parse_args()

    source = load_from_disk(args.source_dataset_dir)
    out_root = Path(args.output_root)
    variant_dir = out_root / args.variant_name
    hf_dir = variant_dir / 'hf_dataset'

    transformed = {}
    counts = {}
    for split, ds in source.items():
        rows = []
        for ex in ds:
            meta = dict(ex.get('metadata', {}))
            meta['variant'] = args.variant_name
            meta['source_variant'] = ex.get('metadata', {}).get('variant', 'reasoning_action')
            rows.append({
                'messages': _transform_messages(ex['messages']),
                'metadata': meta,
            })
        transformed[split] = rows
        counts[split] = len(rows)
        _write_jsonl(variant_dir / f'{split}.jsonl', rows)

    dset = DatasetDict({split: Dataset.from_list(rows) for split, rows in transformed.items()})
    dset.save_to_disk(str(hf_dir))

    manifest = {
        'variant_name': args.variant_name,
        'source_dataset_dir': args.source_dataset_dir,
        'output_hf_dataset_dir': str(hf_dir),
        'counts': counts,
        'format_changes': {
            'strict_suffix_added': True,
            'one_shot_example_added': True,
            'user_reminder_added': True,
        },
    }
    (variant_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
