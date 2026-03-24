#!/usr/bin/env python3
"""Export browser-agent rollouts into SFT-ready datasets.

Outputs:
- action-only chat JSONL (train/val)
- reasoning+action chat JSONL (train/val)
- manifest JSON with counts and filters
- optional Hugging Face datasets save_to_disk directories
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

FALLBACK_SYSTEM_PROMPT = """You control a web browser through BrowserGym actions.
You must complete the given web task by interacting with the page.

Available actions (this server accepts these forms):
- noop() - Do nothing
- click(bid, button='left', modifiers=None) - Click by BrowserGym ID. Example modifier click: click('18', modifiers=['Control'])
- dblclick(bid) - Double-click an element
- hover(bid) - Move mouse over an element
- focus(bid) - Focus an element
- fill(bid, text) - Fill a text field
- clear(bid) - Clear a text field
- select_option(bid, option_text_or_list) - Select option(s) by visible text (not by bid)
- drag_and_drop(source_bid, target_bid) - Drag source element to target element
- scroll(delta_x, delta_y) - Scroll by pixel delta. Example: scroll(0, 300) for down, scroll(0, -300) for up
- goto(url) - Navigate to a URL

The page structure shows elements as: [bid] element_type 'element_text'
For example: [13] button 'Click Me!' means bid='13'
""".strip()


def _sha_bucket(text: str) -> float:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return (value % 1_000_000) / 1_000_000.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_generation_system_prompt(rollout_dir: Path) -> str:
    cfg_path = rollout_dir / "resolved_config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            if isinstance(cfg, dict):
                teacher = cfg.get("teacher_api") or {}
                if isinstance(teacher, dict):
                    prompt = teacher.get("system_prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        return prompt.strip()
        except Exception:
            pass
    return FALLBACK_SYSTEM_PROMPT


def _strip_generation_format_instructions(base_prompt: str) -> str:
    markers = [
        "\nReply using EXACTLY this format:",
        "\nReply with exactly ONE action on a single line",
        "\nExample:",
        "\nFinal constraints:",
    ]
    cut_points = [base_prompt.find(marker) for marker in markers if base_prompt.find(marker) != -1]
    if cut_points:
        return base_prompt[: min(cut_points)].rstrip()
    return base_prompt.rstrip()


def _action_only_system_prompt(base_prompt: str) -> str:
    base = _strip_generation_format_instructions(base_prompt)
    return base + "\n\nOutput only the single next BrowserGym action on one line."


def _reasoning_action_system_prompt(base_prompt: str) -> str:
    base = _strip_generation_format_instructions(base_prompt)
    return base + "\n\nThink step by step before outputting the single next BrowserGym action. Put reasoning inside exactly one <think>...</think> block, then output only the action on the next line."


def _render_history(history_rows: list[dict[str, Any]], max_chars: int = 1200) -> str:
    if not history_rows:
        return "(none)"
    parts: list[str] = []
    for row in history_rows:
        action = row.get("action_str", "")
        reward = row.get("reward", 0.0)
        done = row.get("done", False)
        err = row.get("last_action_error", False)
        obs = row.get("post_observation", {}) if isinstance(row.get("post_observation"), dict) else {}
        obs_text = str(obs.get("text", "")).replace("\n", " ").strip()
        if len(obs_text) > 220:
            obs_text = obs_text[:220] + "..."
        parts.append(
            f"- step {row.get('step_idx')}: action={action}; reward={reward}; done={done}; action_error={err}; post_obs={obs_text}"
        )
    text = "\n".join(parts)
    return text[:max_chars]


def _render_user_message(step: dict[str, Any], history_rows: list[dict[str, Any]]) -> str:
    pre = step.get("pre_observation", {}) if isinstance(step.get("pre_observation"), dict) else {}
    diag = pre.get("diagnostics", {}) if isinstance(pre.get("diagnostics"), dict) else {}
    obs_text = str(pre.get("text", ""))
    return (
        f"Task: {step.get('task_name', '')}\n"
        f"Goal: {pre.get('goal', '')}\n"
        f"Current URL: {pre.get('url', '')}\n"
        f"Last action had error: {bool(pre.get('last_action_error', False))}\n"
        f"Observation diagnostics: actionable_nodes={diag.get('actionable_node_count', 0)}, "
        f"text_lines={diag.get('text_line_count', 0)}, sparse={bool(diag.get('is_sparse', False))}, "
        f"root_only={bool(diag.get('is_root_only', False))}\n\n"
        f"Recent history:\n{_render_history(history_rows)}\n\n"
        f"Current observation:\n{obs_text}"
    )


def _make_action_only_sample(step: dict[str, Any], history_rows: list[dict[str, Any]], base_system_prompt: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": _action_only_system_prompt(base_system_prompt)},
            {"role": "user", "content": _render_user_message(step, history_rows)},
            {"role": "assistant", "content": str(step.get("action_str", "")).strip()},
        ],
        "metadata": {
            "variant": "action_only",
            "task_name": step.get("task_name"),
            "episode_id": step.get("episode_id"),
            "run_id": step.get("run_id"),
            "seed": step.get("seed"),
            "step_idx": step.get("step_idx"),
            "teacher_model": step.get("teacher_model"),
            "teacher_used_fallback": bool(step.get("teacher_used_fallback", False)),
        },
    }


def _make_reasoning_action_sample(step: dict[str, Any], history_rows: list[dict[str, Any]], base_system_prompt: str) -> dict[str, Any] | None:
    reasoning = str(step.get("teacher_response_reasoning", "")).strip()
    action = str(step.get("action_str", "")).strip()
    if not reasoning:
        return None
    assistant = f"<think>\n{reasoning}\n</think>\n{action}"
    return {
        "messages": [
            {"role": "system", "content": _reasoning_action_system_prompt(base_system_prompt)},
            {"role": "user", "content": _render_user_message(step, history_rows)},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "variant": "reasoning_action",
            "task_name": step.get("task_name"),
            "episode_id": step.get("episode_id"),
            "run_id": step.get("run_id"),
            "seed": step.get("seed"),
            "step_idx": step.get("step_idx"),
            "teacher_model": step.get("teacher_model"),
            "teacher_used_fallback": bool(step.get("teacher_used_fallback", False)),
        },
    }


def _episode_passes_filters(ep: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.successful_only and not bool(ep.get("success", False)):
        return False
    if int(ep.get("action_error_count", 0) or 0) > args.max_action_errors:
        return False
    if int(ep.get("repeated_action_loop_count", 0) or 0) > args.max_repeated_loops:
        return False
    if int(ep.get("sparse_observation_count", 0) or 0) > args.max_sparse_observations:
        return False
    if int(ep.get("root_only_observation_count", 0) or 0) > args.max_root_only_observations:
        return False
    if int(ep.get("teacher_fallback_count", 0) or 0) > args.max_fallback_count:
        return False
    return True


def _split_name(step: dict[str, Any], args: argparse.Namespace) -> str:
    key = str(step.get("run_id" if args.split_by == "run" else "episode_id", ""))
    return "val" if _sha_bucket(key) < args.val_ratio else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_build_hf_dataset(out_dir: Path, variant: str, rows_by_split: dict[str, list[dict[str, Any]]]) -> str | None:
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return None
    ds_dict = {split: Dataset.from_list(rows) for split, rows in rows_by_split.items()}
    dset = DatasetDict(ds_dict)
    target = out_dir / variant / "hf_dataset"
    dset.save_to_disk(str(target))
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export browser-agent rollouts into SFT-ready datasets.")
    parser.add_argument("--rollout-glob", default="miniwob_phase1_prod_local_qwen35_batch*_r*_*")
    parser.add_argument("--rollouts-root", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/exports/phase1_sft_v2"))
    parser.add_argument("--history-steps", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-by", choices=["run", "episode"], default="run")
    parser.add_argument("--successful-only", action="store_true", default=True)
    parser.add_argument("--max-action-errors", type=int, default=0)
    parser.add_argument("--max-repeated-loops", type=int, default=0)
    parser.add_argument("--max-sparse-observations", type=int, default=2)
    parser.add_argument("--max-root-only-observations", type=int, default=0)
    parser.add_argument("--max-fallback-count", type=int, default=0)
    args = parser.parse_args()

    rollout_dirs = sorted(args.rollouts_root.glob(args.rollout_glob))
    if not rollout_dirs:
        raise SystemExit(f"No rollout dirs matched: {args.rollouts_root}/{args.rollout_glob}")

    action_rows = {"train": [], "val": []}
    reasoning_rows = {"train": [], "val": []}

    manifest: dict[str, Any] = {
        "matched_rollout_dirs": [str(p) for p in rollout_dirs],
        "filters": {
            "successful_only": args.successful_only,
            "max_action_errors": args.max_action_errors,
            "max_repeated_loops": args.max_repeated_loops,
            "max_sparse_observations": args.max_sparse_observations,
            "max_root_only_observations": args.max_root_only_observations,
            "max_fallback_count": args.max_fallback_count,
        },
        "split_by": args.split_by,
        "val_ratio": args.val_ratio,
        "history_steps": args.history_steps,
        "episodes_seen": 0,
        "episodes_kept": 0,
        "step_rows_seen": 0,
        "action_only": {"train": 0, "val": 0},
        "reasoning_action": {"train": 0, "val": 0},
    }

    for rollout_dir in rollout_dirs:
        base_system_prompt = _load_generation_system_prompt(rollout_dir)
        ep_rows = _load_jsonl(rollout_dir / "episode_summaries.jsonl")
        step_rows = _load_jsonl(rollout_dir / "trajectory_steps.jsonl")
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for row in step_rows:
            by_episode.setdefault(str(row.get("episode_id", "")), []).append(row)
        for rows in by_episode.values():
            rows.sort(key=lambda r: int(r.get("step_idx", 0) or 0))

        manifest["episodes_seen"] += len(ep_rows)
        manifest["step_rows_seen"] += len(step_rows)

        for ep in ep_rows:
            ep_id = str(ep.get("episode_id", ""))
            if not _episode_passes_filters(ep, args):
                continue
            steps = by_episode.get(ep_id, [])
            if not steps:
                continue
            manifest["episodes_kept"] += 1
            for idx, step in enumerate(steps):
                history = steps[max(0, idx - args.history_steps):idx]
                split = _split_name(step, args)
                action_sample = _make_action_only_sample(step, history, base_system_prompt)
                action_rows[split].append(action_sample)
                manifest["action_only"][split] += 1
                reasoning_sample = _make_reasoning_action_sample(step, history, base_system_prompt)
                if reasoning_sample is not None:
                    reasoning_rows[split].append(reasoning_sample)
                    manifest["reasoning_action"][split] += 1

    _write_jsonl(args.output_dir / "action_only" / "train.jsonl", action_rows["train"])
    _write_jsonl(args.output_dir / "action_only" / "val.jsonl", action_rows["val"])
    _write_jsonl(args.output_dir / "reasoning_action" / "train.jsonl", reasoning_rows["train"])
    _write_jsonl(args.output_dir / "reasoning_action" / "val.jsonl", reasoning_rows["val"])

    action_hf = _maybe_build_hf_dataset(args.output_dir, "action_only", action_rows)
    reasoning_hf = _maybe_build_hf_dataset(args.output_dir, "reasoning_action", reasoning_rows)
    manifest["hf_dataset_dirs"] = {
        "action_only": action_hf,
        "reasoning_action": reasoning_hf,
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    push_script = args.output_dir / "push_to_hub.py"
    push_script.write_text(
        "from datasets import load_from_disk\nimport argparse\n\nparser = argparse.ArgumentParser()\nparser.add_argument('--dataset-dir', required=True)\nparser.add_argument('--repo-id', required=True)\nparser.add_argument('--private', action='store_true')\nargs = parser.parse_args()\n\nds = load_from_disk(args.dataset_dir)\nds.push_to_hub(args.repo_id, private=args.private)\nprint(f'pushed {args.dataset_dir} -> {args.repo_id}')\n",
        encoding="utf-8",
    )

    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps({
        "episodes_seen": manifest["episodes_seen"],
        "episodes_kept": manifest["episodes_kept"],
        "action_only": manifest["action_only"],
        "reasoning_action": manifest["reasoning_action"],
        "hf_dataset_dirs": manifest["hf_dataset_dirs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
