#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JUDGE_SYSTEM = """You are judging one-step BrowserGym action equivalence.
Decide whether the model output's intended FINAL next browser action is semantically equivalent to the gold answer.
Treat these as equivalent:
- single vs double quotes
- click('15') vs click('15', button='left')
- extra prose or <think> text before the final action
If the output is truncated before a final action is clear, answer NOT_EQUIVALENT.
If it clearly chooses a different next action, answer NOT_EQUIVALENT.
Return exactly one token sequence on the first line: EQUIVALENT or NOT_EQUIVALENT.
Optionally add one short reason on the second line.
"""


def tail_lines(text: str, max_lines: int = 8, max_chars: int = 700) -> str:
    lines = (text or '').splitlines()
    tail = '\n'.join(lines[-max_lines:]) if lines else ''
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail.strip()


def parse_label(text: str) -> bool:
    s = (text or '').strip().upper()
    first = s.splitlines()[0].strip() if s.splitlines() else s
    if 'NOT_EQUIVALENT' in first:
        return False
    if 'EQUIVALENT' in first:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-json', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--model-name', default='unsloth/Qwen3-1.7B-unsloth-bnb-4bit')
    ap.add_argument('--limit', type=int, default=240)
    ap.add_argument('--max-new-tokens', type=int, default=48)
    args = ap.parse_args()

    data = json.loads(Path(args.eval_json).read_text())
    rows = data['rows'][:args.limit] if args.limit else data['rows']

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = 'left'

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    judged = []
    eq = 0
    for i, row in enumerate(rows, start=1):
        user = {
            'task_name': row.get('task_name', ''),
            'gold_answer': row.get('target', ''),
            'model_output_tail': tail_lines(row.get('raw_generation', '')),
        }
        messages = [
            {'role': 'system', 'content': JUDGE_SYSTEM},
            {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        equivalent = parse_label(text)
        eq += int(equivalent)
        judged.append({
            'task_name': row.get('task_name'),
            'target': row.get('target'),
            'prediction': row.get('prediction'),
            'parser_match': row.get('match'),
            'raw_generation_tail': tail_lines(row.get('raw_generation', '')),
            'judge_equivalent': equivalent,
            'judge_output': text.strip(),
        })
        if i % 20 == 0:
            print(f'judged {i}/{len(rows)}')

    out = {
        'summary': {
            'num_examples': len(rows),
            'parser_exact_match': data['summary'].get('exact_match'),
            'judge_equivalent_rate': eq / len(rows) if rows else 0.0,
            'judge_model': args.model_name,
        },
        'rows': judged,
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out['summary'], indent=2))
    disagreements = [r for r in judged if bool(r['parser_match']) != bool(r['judge_equivalent'])]
    print('disagreements', len(disagreements))
    for row in disagreements[:12]:
        print(json.dumps(row, ensure_ascii=False)[:2000])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
