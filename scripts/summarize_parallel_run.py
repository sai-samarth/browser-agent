#!/usr/bin/env python3
"""Summarize rollout quality metrics for a parallel MiniWoB run."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


RUN_DIR_RE = re.compile(r"^\[info\] Run directory:\s*(.+)\s*$")


@dataclass
class WorkerInput:
    worker_id: str
    config_path: Path
    log_path: Path
    run_dir: Path | None
    tasks: list[str]
    episodes_per_task: int | None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _parse_run_dir_from_log(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = RUN_DIR_RE.match(line.rstrip("\n"))
            if m:
                return Path(m.group(1).strip())
    return None


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _safe_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _safe_mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _discover_workers(parallel_run_root: Path) -> list[WorkerInput]:
    cfg_dir = parallel_run_root / "configs"
    log_dir = parallel_run_root / "logs"
    if not cfg_dir.exists():
        raise FileNotFoundError(f"Missing configs dir: {cfg_dir}")
    if not log_dir.exists():
        raise FileNotFoundError(f"Missing logs dir: {log_dir}")

    workers: list[WorkerInput] = []
    for config_path in sorted(cfg_dir.glob("worker_*.yaml")):
        worker_id = config_path.stem.split("_")[-1]
        log_path = log_dir / f"worker_{worker_id}.log"
        cfg = _load_yaml(config_path)
        collection = cfg.get("collection", {}) if isinstance(cfg.get("collection"), dict) else {}
        tasks_raw = collection.get("tasks", [])
        tasks = [str(x) for x in tasks_raw] if isinstance(tasks_raw, list) else []
        episodes_per_task = collection.get("episodes_per_task")
        episodes_per_task_int = int(episodes_per_task) if isinstance(episodes_per_task, int) else None
        workers.append(
            WorkerInput(
                worker_id=worker_id,
                config_path=config_path,
                log_path=log_path,
                run_dir=_parse_run_dir_from_log(log_path),
                tasks=tasks,
                episodes_per_task=episodes_per_task_int,
            )
        )
    return workers


def _summarize(parallel_run_root: Path) -> dict[str, Any]:
    workers = _discover_workers(parallel_run_root)
    if not workers:
        raise RuntimeError(f"No worker configs found in {parallel_run_root / 'configs'}")

    expected_tasks = sorted({t for w in workers for t in w.tasks})
    expected_episodes = 0
    for w in workers:
        if w.episodes_per_task is not None:
            expected_episodes += len(w.tasks) * w.episodes_per_task

    all_rows: list[dict[str, Any]] = []
    worker_stats: list[dict[str, Any]] = []
    run_dirs_found = 0
    rows_with_teacher_fallback = False
    for w in workers:
        run_dir = w.run_dir
        episode_path = run_dir / "episode_summaries.jsonl" if run_dir is not None else None
        rows = _parse_jsonl(episode_path) if episode_path is not None else []
        all_rows.extend(rows)
        if run_dir is not None:
            run_dirs_found += 1

        successes = sum(1 for r in rows if bool(r.get("success", False)))
        worker_stats.append(
            {
                "worker_id": w.worker_id,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "tasks_configured": len(w.tasks),
                "episodes_expected": (len(w.tasks) * w.episodes_per_task) if w.episodes_per_task is not None else None,
                "episodes_observed": len(rows),
                "successes": successes,
                "success_rate": (successes / len(rows)) if rows else 0.0,
                "missing_episode_file": (episode_path is None or not episode_path.exists()),
            }
        )
        if rows and "teacher_fallback_count" in rows[0]:
            rows_with_teacher_fallback = True

    total_episodes = len(all_rows)
    total_success = sum(1 for r in all_rows if bool(r.get("success", False)))
    observed_tasks = sorted({str(r.get("task_name", "")) for r in all_rows if r.get("task_name")})

    rewards = [float(r.get("cum_reward", 0.0) or 0.0) for r in all_rows]
    steps = [float(r.get("num_steps", 0.0) or 0.0) for r in all_rows]
    action_errs = [float(r.get("action_error_count", 0.0) or 0.0) for r in all_rows]
    done_flags = [bool(r.get("final_done", False)) for r in all_rows]
    fallback_counts = [
        float(r.get("teacher_fallback_count", 0.0) or 0.0)
        for r in all_rows
        if "teacher_fallback_count" in r
    ]

    per_task: dict[str, dict[str, float]] = defaultdict(
        lambda: {"episodes": 0.0, "successes": 0.0, "reward_sum": 0.0, "steps_sum": 0.0}
    )
    timestamps: list[datetime] = []
    for r in all_rows:
        task = str(r.get("task_name", ""))
        if not task:
            continue
        bucket = per_task[task]
        bucket["episodes"] += 1
        bucket["successes"] += 1 if bool(r.get("success", False)) else 0
        bucket["reward_sum"] += float(r.get("cum_reward", 0.0) or 0.0)
        bucket["steps_sum"] += float(r.get("num_steps", 0.0) or 0.0)
        dt = _safe_dt(r.get("timestamp"))
        if dt is not None:
            timestamps.append(dt)

    per_task_rows: list[dict[str, Any]] = []
    for task, s in per_task.items():
        episodes = int(s["episodes"])
        successes = int(s["successes"])
        per_task_rows.append(
            {
                "task_name": task,
                "episodes": episodes,
                "successes": successes,
                "success_rate": (successes / episodes) if episodes else 0.0,
                "avg_cum_reward": (s["reward_sum"] / episodes) if episodes else 0.0,
                "avg_steps": (s["steps_sum"] / episodes) if episodes else 0.0,
            }
        )
    per_task_rows.sort(key=lambda x: (x["success_rate"], x["task_name"]))

    macro_accuracy = _safe_mean([float(x["success_rate"]) for x in per_task_rows])

    started_at = min(timestamps).isoformat() if timestamps else None
    ended_at = max(timestamps).isoformat() if timestamps else None
    duration_s = (
        (max(timestamps) - min(timestamps)).total_seconds()
        if len(timestamps) >= 2
        else None
    )

    summary: dict[str, Any] = {
        "parallel_run_root": str(parallel_run_root),
        "worker_count": len(workers),
        "run_dirs_found": run_dirs_found,
        "expected_task_count": len(expected_tasks),
        "observed_task_count": len(observed_tasks),
        "missing_tasks": sorted(set(expected_tasks) - set(observed_tasks)),
        "total_episodes_expected": expected_episodes if expected_episodes > 0 else None,
        "total_episodes_observed": total_episodes,
        "total_successes": total_success,
        "micro_accuracy": (total_success / total_episodes) if total_episodes else 0.0,
        "macro_task_accuracy": macro_accuracy,
        "avg_cum_reward": _safe_mean(rewards),
        "avg_num_steps": _safe_mean(steps),
        "avg_action_error_count": _safe_mean(action_errs),
        "final_done_rate": (_safe_mean([1.0 if x else 0.0 for x in done_flags]) if done_flags else 0.0),
        "teacher_fallback_episode_rate": (
            _safe_mean([1.0 if x > 0 else 0.0 for x in fallback_counts]) if fallback_counts else None
        ),
        "avg_teacher_fallback_count": (_safe_mean(fallback_counts) if fallback_counts else None),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_s,
        "rows_have_teacher_fallback_count": rows_with_teacher_fallback,
        "workers": worker_stats,
        "per_task": per_task_rows,
    }
    return summary


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _print_report(summary: dict[str, Any], top_k: int) -> None:
    print(f"Parallel run: {summary['parallel_run_root']}")
    print(
        "Coverage:"
        f" workers={summary['worker_count']} run_dirs={summary['run_dirs_found']} "
        f"tasks={summary['observed_task_count']}/{summary['expected_task_count']} "
        f"episodes={summary['total_episodes_observed']}"
        + (
            f"/{summary['total_episodes_expected']}"
            if summary.get("total_episodes_expected") is not None
            else ""
        )
    )
    if summary["missing_tasks"]:
        print(f"Missing tasks ({len(summary['missing_tasks'])}): {', '.join(summary['missing_tasks'])}")

    print(
        "Overall:"
        f" micro_acc={_fmt_pct(summary['micro_accuracy'])}"
        f" macro_task_acc={_fmt_pct(summary['macro_task_accuracy'])}"
        f" avg_reward={summary['avg_cum_reward']:.3f}"
        f" avg_steps={summary['avg_num_steps']:.2f}"
        f" avg_action_errors={summary['avg_action_error_count']:.2f}"
        f" final_done_rate={_fmt_pct(summary['final_done_rate'])}"
    )
    if summary.get("teacher_fallback_episode_rate") is not None:
        print(
            "Teacher:"
            f" fallback_episode_rate={_fmt_pct(summary['teacher_fallback_episode_rate'])}"
            f" avg_fallback_count={summary['avg_teacher_fallback_count']:.3f}"
        )
    if summary.get("duration_seconds") is not None:
        print(
            f"Timing: start={summary['started_at']} end={summary['ended_at']} "
            f"duration_s={summary['duration_seconds']:.1f}"
        )

    workers = summary["workers"]
    failed_workers = [w for w in workers if w["missing_episode_file"]]
    if failed_workers:
        ids = ", ".join(w["worker_id"] for w in failed_workers)
        print(f"Workers missing episode_summaries.jsonl: {ids}")

    per_task = summary["per_task"]
    if not per_task:
        return
    top_k = max(1, top_k)
    worst = per_task[:top_k]
    best = sorted(per_task, key=lambda x: (x["success_rate"], x["task_name"]), reverse=True)[:top_k]

    print(f"Worst {len(worst)} tasks by success rate:")
    for row in worst:
        print(
            f"  {row['task_name']}: acc={_fmt_pct(row['success_rate'])} "
            f"({row['successes']}/{row['episodes']}), "
            f"avg_reward={row['avg_cum_reward']:.3f}, avg_steps={row['avg_steps']:.2f}"
        )
    print(f"Best {len(best)} tasks by success rate:")
    for row in best:
        print(
            f"  {row['task_name']}: acc={_fmt_pct(row['success_rate'])} "
            f"({row['successes']}/{row['episodes']}), "
            f"avg_reward={row['avg_cum_reward']:.3f}, avg_steps={row['avg_steps']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parallel MiniWoB rollout metrics.")
    parser.add_argument(
        "--parallel-run-root",
        type=Path,
        required=True,
        help="Path like data/parallel_runs/miniwob_train_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best/worst tasks to print.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = _summarize(args.parallel_run_root)
    _print_report(summary, args.top_k)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        print(f"Saved JSON summary: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
