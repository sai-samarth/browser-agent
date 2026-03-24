#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


def _normalize_browsergym_action(action: str) -> str:
    action = action.strip()
    if not action:
        return action
    bid_like_actions = {'click','dblclick','hover','focus','fill','clear','select_option'}
    m = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>[\s\S]*)\)$", action)
    if not m:
        return action
    name = m.group('name')
    args = m.group('args')
    if name not in bid_like_actions:
        return action
    first_arg_match = re.match(r"\s*(?P<bid>\d+)\b(?P<rest>[\s\S]*)$", args)
    if first_arg_match:
        bid = first_arg_match.group('bid')
        rest = first_arg_match.group('rest')
        args = f"'{bid}'{rest}"
    if name == 'fill':
        bare_fill_match = re.match(r"\s*(?P<bid>'[^']+'|\"[^\"]+\")\s*,\s*(?P<text>[A-Za-z_][A-Za-z0-9_ ./:-]*)\s*$", args)
        if bare_fill_match:
            bid = bare_fill_match.group('bid')
            text_arg = bare_fill_match.group('text').strip()
            if text_arg and not text_arg.startswith(("'", '"')):
                args = f"{bid}, '{text_arg}'"
    return f"{name}({args})"


def _extract_action_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    allowed_actions = {'noop','click','dblclick','hover','focus','fill','clear','select_option','drag_and_drop','scroll','goto','send_keys','press'}
    code_blocks = re.findall(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)```", cleaned)
    candidates = code_blocks if code_blocks else [cleaned]

    def _extract_calls(s: str) -> list[str]:
        out=[]; n=len(s); i=0
        while i<n:
            ch=s[i]
            if not (ch.isalpha() or ch=='_'):
                i+=1; continue
            start=i; i+=1
            while i<n and (s[i].isalnum() or s[i]=='_'):
                i+=1
            name=s[start:i]
            if i>=n or s[i] != '(' or name not in allowed_actions:
                continue
            depth=0; j=i; in_quote=None; escaped=False
            while j<n:
                c=s[j]
                if in_quote is not None:
                    if escaped: escaped=False
                    elif c=='\\': escaped=True
                    elif c==in_quote: in_quote=None
                    j+=1; continue
                if c in ("'", '"'):
                    in_quote=c; j+=1; continue
                if c=='(': depth+=1
                elif c==')':
                    depth-=1
                    if depth==0:
                        out.append(s[start:j+1]); break
                j+=1
            i=j+1
        return out

    standalone_pattern = re.compile(r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?`?\s*([A-Za-z_][A-Za-z0-9_]*\(.*\))\s*`?\s*;?\s*$")
    scored=[]
    for cand_idx, candidate in enumerate(candidates):
        candidate = candidate.strip()
        if not candidate:
            continue
        for line_idx, raw_line in enumerate(candidate.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            m = standalone_pattern.match(line)
            if not m:
                continue
            line_calls = _extract_calls(m.group(1))
            if line_calls:
                action = line_calls[-1].rstrip(';').strip()
                if action:
                    scored.append((1000 + cand_idx*100 + line_idx, len(scored), action))
        inline_calls = _extract_calls(candidate)
        for call_idx, action in enumerate(inline_calls):
            action = action.rstrip(';').strip()
            if action:
                scored.append((100 + cand_idx*100 + call_idx, len(scored), action))
    if not scored:
        return None
    scored.sort(key=lambda x:(x[0],x[1]))
    return _normalize_browsergym_action(scored[-1][2])


def _looks_conditional(model_ref: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
        archs = cfg.architectures or []
        model_type = (getattr(cfg, 'model_type', '') or '').lower()
        if any('Qwen3_5ForConditionalGeneration' in a for a in archs):
            return True
        if 'qwen3_5' in model_type:
            return True
    except Exception:
        pass
    s = model_ref.lower()
    return 'qwen3.5' in s or 'qwen3_5' in s


def _should_use_conditional_loader(model_name: str, adapter_dir: Optional[str]) -> bool:
    if _looks_conditional(model_name):
        return True
    if adapter_dir and _looks_conditional(adapter_dir):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--adapter-dir', default=None)
    parser.add_argument('--split', default='val')
    parser.add_argument('--limit', type=int, default=240)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_dir)[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    use_conditional = _should_use_conditional_loader(args.model_name, args.adapter_dir)
    model_ref = args.adapter_dir or args.model_name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if use_conditional:
        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
        tokenizer = getattr(processor, 'tokenizer', processor)
        if getattr(tokenizer, 'pad_token', None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = 'left'
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir) if args.adapter_dir else base_model
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = 'left'
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir) if args.adapter_dir else base_model

    model.eval()

    exact=0; parseable=0; rows=[]
    for i, ex in enumerate(ds, start=1):
        messages = ex['messages']
        prompt_messages = messages[:-1]
        target = messages[-1]['content'].strip()
        if processor is not None:
            prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        pred = _extract_action_from_text(gen)
        target_action = _extract_action_from_text(target) or target
        if pred is not None:
            parseable += 1
        if pred == target_action:
            exact += 1
        rows.append({'task_name': ex['metadata']['task_name'], 'target': target_action, 'prediction': pred, 'raw_generation': gen, 'match': pred == target_action})
        if i % 20 == 0:
            print(f'processed {i}/{len(ds)}')

    result = {
        'split': args.split,
        'num_examples': len(rows),
        'parseable_rate': parseable / len(rows) if rows else 0.0,
        'exact_match': exact / len(rows) if rows else 0.0,
        'adapter_dir': args.adapter_dir,
        'model_name': args.model_name,
        'max_new_tokens': args.max_new_tokens,
        'loader': 'conditional_generation' if use_conditional else 'causal_lm',
    }
    out = {'summary': result, 'rows': rows}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
