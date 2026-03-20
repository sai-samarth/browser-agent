#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """You are evaluating one-step BrowserGym action equivalence.

You will receive:
- the task name
- the gold next-step answer
- the tail of the model output

Your job is to decide whether the model output's intended FINAL next browser action is semantically equivalent to the gold answer.

Important rules:
1. Judge the intended final next action, not whether the reasoning is good.
2. Treat superficial formatting differences as equivalent:
   - single vs double quotes
   - click('15') vs click('15', button='left')
   - extra prose or <think> blocks before the final action
3. If the model output is truncated before a final action is clear, mark not equivalent.
4. If the model clearly chooses a different next action, mark not equivalent.
5. If the model output contains multiple candidate actions, judge the final intended action.
6. Be conservative. Only mark equivalent when the intended action is clearly the same.

Return JSON only:
{"equivalent": true|false, "confidence": 0.0-1.0, "reason": "short explanation"}
"""


def tail_lines(text: str, max_lines: int = 8, max_chars: int = 900) -> str:
    lines = (text or "").splitlines()
    tail = "\n".join(lines[-max_lines:]) if lines else ""
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail.strip()


def make_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def judge_one(client: OpenAI, model: str, task_name: str, gold: str, raw_generation: str, max_lines: int) -> dict[str, Any]:
    user_prompt = {
        "task_name": task_name,
        "gold_answer": gold,
        "model_output_tail": tail_lines(raw_generation, max_lines=max_lines),
    }
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    )
    text = resp.choices[0].message.content
    obj = json.loads(text)
    return {
        "equivalent": bool(obj.get("equivalent", False)),
        "confidence": float(obj.get("confidence", 0.0)),
        "reason": str(obj.get("reason", "")),
        "input": user_prompt,
        "raw_judge": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tail-lines", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    data = json.loads(Path(args.eval_json).read_text())
    rows = data["rows"]
    if args.limit:
        rows = rows[: args.limit]

    client = make_client()

    judged_rows = []
    equiv = 0
    for idx, row in enumerate(rows, start=1):
        result = judge_one(
            client=client,
            model=args.judge_model,
            task_name=row.get("task_name", ""),
            gold=row.get("target", ""),
            raw_generation=row.get("raw_generation", ""),
            max_lines=args.tail_lines,
        )
        if result["equivalent"]:
            equiv += 1
        judged_rows.append({
            "task_name": row.get("task_name"),
            "target": row.get("target"),
            "prediction": row.get("prediction"),
            "parser_match": row.get("match"),
            "raw_generation_tail": tail_lines(row.get("raw_generation", ""), max_lines=args.tail_lines),
            "judge_equivalent": result["equivalent"],
            "judge_confidence": result["confidence"],
            "judge_reason": result["reason"],
        })
        if idx % 20 == 0:
            print(f"judged {idx}/{len(rows)}")
        if args.sleep:
            time.sleep(args.sleep)

    summary = {
        "num_examples": len(rows),
        "judge_model": args.judge_model,
        "judge_equivalent_rate": equiv / len(rows) if rows else 0.0,
        "parser_exact_match": data["summary"].get("exact_match"),
        "tail_lines": args.tail_lines,
    }
    out = {"summary": summary, "rows": judged_rows}
    Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    disagreements = [r for r in judged_rows if r["judge_equivalent"] != bool(r.get("parser_match"))]
    print(f"parser_judge_disagreements={len(disagreements)}")
    for row in disagreements[:10]:
        print(json.dumps(row, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
