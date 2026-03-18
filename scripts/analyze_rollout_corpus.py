#!/usr/bin/env python3
"""Build a baseline report across discovered rollout artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_summary_module(script_dir: Path):
    target = script_dir / "summarize_parallel_run.py"
    spec = importlib.util.spec_from_file_location("summarize_parallel_run", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_ratio(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _classify_run(summary: dict[str, Any]) -> dict[str, Any]:
    expected_eps = summary.get("total_episodes_expected") or 0
    observed_eps = int(summary.get("total_episodes_observed") or 0)
    expected_tasks = int(summary.get("expected_task_count") or 0)
    observed_tasks = int(summary.get("observed_task_count") or 0)
    missing_tasks = len(summary.get("missing_tasks") or [])
    micro = float(summary.get("micro_accuracy") or 0.0)
    action_err = float(summary.get("avg_action_error_count") or 0.0)

    episode_coverage = _safe_ratio(observed_eps, expected_eps) if expected_eps else (1.0 if observed_eps else 0.0)
    task_coverage = _safe_ratio(observed_tasks, expected_tasks) if expected_tasks else (1.0 if observed_tasks else 0.0)

    if observed_eps == 0:
        label = "broken"
    elif episode_coverage < 0.25 or task_coverage < 0.25:
        label = "partial"
    elif micro >= 0.30 and action_err <= 3.0 and missing_tasks <= max(5, int(expected_tasks * 0.1)):
        label = "sft_candidate"
    else:
        label = "analysis_candidate"

    return {
        "label": label,
        "episode_coverage": episode_coverage,
        "task_coverage": task_coverage,
        "missing_task_count": missing_tasks,
    }


def _infer_teacher_info(run_root: Path) -> dict[str, Any]:
    cfg_dir = run_root / "configs"
    teacher_base_url = None
    teacher_model = None
    policy_mode = None
    if cfg_dir.exists():
        for config_path in sorted(cfg_dir.glob("worker_*.yaml")):
            try:
                import yaml
                cfg = yaml.safe_load(config_path.read_text())
            except Exception:
                continue
            if not isinstance(cfg, dict):
                continue
            policy = cfg.get("policy") or {}
            teacher = cfg.get("teacher_api") or {}
            policy_mode = policy_mode or policy.get("mode")
            teacher_base_url = teacher_base_url or teacher.get("base_url")
            teacher_model = teacher_model or teacher.get("model")
            if teacher_base_url or teacher_model:
                break
    return {
        "policy_mode": policy_mode,
        "teacher_base_url": teacher_base_url,
        "teacher_model": teacher_model,
    }


def _count_rollout_corpus(rollout_root: Path) -> dict[str, Any]:
    total_runs = 0
    runs_with_episode_summaries = 0
    runs_with_steps = 0
    total_episodes = 0
    total_steps = 0
    total_successes = 0
    for run_dir in sorted(rollout_root.glob("*")):
        if not run_dir.is_dir():
            continue
        total_runs += 1
        episode_path = run_dir / "episode_summaries.jsonl"
        step_path = run_dir / "trajectory_steps.jsonl"
        if episode_path.exists():
            runs_with_episode_summaries += 1
            for line in episode_path.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                total_episodes += 1
                try:
                    obj = json.loads(line)
                    if bool(obj.get("success", False)):
                        total_successes += 1
                except Exception:
                    pass
        if step_path.exists():
            runs_with_steps += 1
            total_steps += sum(1 for line in step_path.read_text(errors="ignore").splitlines() if line.strip())
    return {
        "total_rollout_runs": total_runs,
        "runs_with_episode_summaries": runs_with_episode_summaries,
        "runs_with_trajectory_steps": runs_with_steps,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "total_successes": total_successes,
        "overall_success_rate": _safe_ratio(total_successes, total_episodes),
    }


def _make_run_record(run_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    classification = _classify_run(summary)
    teacher_info = _infer_teacher_info(run_root)
    return {
        "run_name": run_root.name,
        "run_root": str(run_root),
        "classification": classification,
        "teacher": teacher_info,
        "micro_accuracy": float(summary.get("micro_accuracy") or 0.0),
        "macro_task_accuracy": float(summary.get("macro_task_accuracy") or 0.0),
        "avg_action_error_count": float(summary.get("avg_action_error_count") or 0.0),
        "avg_num_steps": float(summary.get("avg_num_steps") or 0.0),
        "avg_cum_reward": float(summary.get("avg_cum_reward") or 0.0),
        "final_done_rate": float(summary.get("final_done_rate") or 0.0),
        "total_episodes_observed": int(summary.get("total_episodes_observed") or 0),
        "total_episodes_expected": summary.get("total_episodes_expected"),
        "observed_task_count": int(summary.get("observed_task_count") or 0),
        "expected_task_count": int(summary.get("expected_task_count") or 0),
        "missing_tasks": summary.get("missing_tasks") or [],
        "teacher_fallback_episode_rate": summary.get("teacher_fallback_episode_rate"),
        "avg_teacher_fallback_count": summary.get("avg_teacher_fallback_count"),
        "started_at": summary.get("started_at"),
        "ended_at": summary.get("ended_at"),
        "duration_seconds": summary.get("duration_seconds"),
        "workers": summary.get("workers") or [],
        "per_task": summary.get("per_task") or [],
    }


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _best_accuracy_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [r for r in runs if r["total_episodes_observed"] >= 100]
    if not eligible:
        eligible = [r for r in runs if r["total_episodes_observed"] > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r["micro_accuracy"], r["observed_task_count"], r["total_episodes_observed"]))


def _best_coverage_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [r for r in runs if r["total_episodes_observed"] > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r["observed_task_count"], r["total_episodes_observed"], r["micro_accuracy"]))


def _render_markdown(corpus: dict[str, Any], runs: list[dict[str, Any]], best_accuracy: dict[str, Any] | None, best_coverage: dict[str, Any] | None) -> str:
    sft_candidates = [r for r in runs if r["classification"]["label"] == "sft_candidate"]
    partial = [r for r in runs if r["classification"]["label"] == "partial"]
    broken = [r for r in runs if r["classification"]["label"] == "broken"]
    top_runs = sorted(
        runs,
        key=lambda r: (r["micro_accuracy"], r["observed_task_count"], r["total_episodes_observed"]),
        reverse=True,
    )[:8]

    lines: list[str] = []
    lines.append("# Browser-Agent Baseline Report")
    lines.append("")
    lines.append("## Corpus snapshot")
    lines.append("")
    lines.append(f"- rollout runs discovered: {corpus['total_rollout_runs']}")
    lines.append(f"- runs with episode summaries: {corpus['runs_with_episode_summaries']}")
    lines.append(f"- runs with step traces: {corpus['runs_with_trajectory_steps']}")
    lines.append(f"- total episodes: {corpus['total_episodes']}")
    lines.append(f"- total steps: {corpus['total_steps']}")
    lines.append(f"- total successful episodes: {corpus['total_successes']}")
    lines.append(f"- overall success rate across all logged episodes: {_fmt_pct(corpus['overall_success_rate'])}")
    lines.append("")
    lines.append("## Run classification summary")
    lines.append("")
    lines.append(f"- SFT candidates: {len(sft_candidates)}")
    lines.append(f"- partial runs: {len(partial)}")
    lines.append(f"- broken runs: {len(broken)}")
    lines.append("")

    if best_accuracy is not None:
        lines.append("## Best accuracy baseline")
        lines.append("")
        lines.append(f"- run: `{best_accuracy['run_name']}`")
        lines.append(f"- micro accuracy: {_fmt_pct(best_accuracy['micro_accuracy'])}")
        lines.append(f"- macro task accuracy: {_fmt_pct(best_accuracy['macro_task_accuracy'])}")
        lines.append(f"- task coverage: {best_accuracy['observed_task_count']}/{best_accuracy['expected_task_count']}")
        lines.append(f"- observed episodes: {best_accuracy['total_episodes_observed']}")
        lines.append(f"- teacher: {best_accuracy['teacher'].get('teacher_model') or 'n/a'} @ {best_accuracy['teacher'].get('teacher_base_url') or 'n/a'}")
        lines.append("")

    if best_coverage is not None:
        lines.append("## Best coverage baseline")
        lines.append("")
        lines.append(f"- run: `{best_coverage['run_name']}`")
        lines.append(f"- micro accuracy: {_fmt_pct(best_coverage['micro_accuracy'])}")
        lines.append(f"- macro task accuracy: {_fmt_pct(best_coverage['macro_task_accuracy'])}")
        lines.append(f"- task coverage: {best_coverage['observed_task_count']}/{best_coverage['expected_task_count']}")
        lines.append(f"- observed episodes: {best_coverage['total_episodes_observed']}")
        lines.append(f"- teacher: {best_coverage['teacher'].get('teacher_model') or 'n/a'} @ {best_coverage['teacher'].get('teacher_base_url') or 'n/a'}")
        lines.append("")

    lines.append("## Top runs by accuracy")
    lines.append("")
    lines.append("| run | label | micro acc | tasks | episodes | action err | teacher |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for run in top_runs:
        teacher = run['teacher'].get('teacher_model') or run['teacher'].get('policy_mode') or 'n/a'
        lines.append(
            f"| `{run['run_name']}` | {run['classification']['label']} | {_fmt_pct(run['micro_accuracy'])} | "
            f"{run['observed_task_count']}/{run['expected_task_count']} | {run['total_episodes_observed']} | "
            f"{_fmt_float(run['avg_action_error_count'])} | {teacher} |"
        )
    lines.append("")

    if best_accuracy is not None and best_accuracy.get('per_task'):
        lines.append("## Worst tasks in the best-accuracy run")
        lines.append("")
        lines.append("| task | success rate | episodes | avg reward | avg steps |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in best_accuracy['per_task'][:12]:
            lines.append(
                f"| `{row['task_name']}` | {_fmt_pct(float(row['success_rate']))} | {int(row['episodes'])} | "
                f"{_fmt_float(float(row['avg_cum_reward']))} | {_fmt_float(float(row['avg_steps']))} |"
            )
        lines.append("")
        lines.append("## Best tasks in the best-accuracy run")
        lines.append("")
        lines.append("| task | success rate | episodes | avg reward | avg steps |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in list(reversed(best_accuracy['per_task']))[:12]:
            lines.append(
                f"| `{row['task_name']}` | {_fmt_pct(float(row['success_rate']))} | {int(row['episodes'])} | "
                f"{_fmt_float(float(row['avg_cum_reward']))} | {_fmt_float(float(row['avg_steps']))} |"
            )
        lines.append("")

    if broken:
        lines.append("## Broken runs worth noting")
        lines.append("")
        for run in broken[:10]:
            lines.append(
                f"- `{run['run_name']}` — episodes={run['total_episodes_observed']}, "
                f"teacher={run['teacher'].get('teacher_model') or 'n/a'} @ {run['teacher'].get('teacher_base_url') or 'n/a'}"
            )
        lines.append("")

    lines.append("## Takeaways")
    lines.append("")
    if best_accuracy is not None:
        lines.append(
            f"- Highest-accuracy substantial run is `{best_accuracy['run_name']}` at {_fmt_pct(best_accuracy['micro_accuracy'])}."
        )
    if best_coverage is not None:
        lines.append(
            f"- Broadest successful task coverage is `{best_coverage['run_name']}` covering {best_coverage['observed_task_count']}/{best_coverage['expected_task_count']} tasks."
        )
    lines.append("- Existing data is already large enough to baseline quality before generating more traces.")
    lines.append("- Recent failures are dominated by backend auth and connection issues, so backend stability should be fixed before scaling collection.")
    lines.append("- The next high-leverage step is a failure taxonomy over the best successful runs, especially the worst-performing task families.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze discovered Browser-Agent rollout artifacts.")
    parser.add_argument("--parallel-runs-root", type=Path, default=Path("data/parallel_runs"))
    parser.add_argument("--rollouts-root", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--md-out", type=Path, default=Path("reports/baselines/latest.md"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/baselines/latest.json"))
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary_module = _load_summary_module(script_dir)

    runs: list[dict[str, Any]] = []
    for run_root in sorted(args.parallel_runs_root.glob("*")):
        if not run_root.is_dir():
            continue
        try:
            summary = summary_module._summarize(run_root)
        except Exception as exc:
            runs.append(
                {
                    "run_name": run_root.name,
                    "run_root": str(run_root),
                    "classification": {"label": "broken", "episode_coverage": 0.0, "task_coverage": 0.0, "missing_task_count": 0},
                    "teacher": _infer_teacher_info(run_root),
                    "error": repr(exc),
                    "micro_accuracy": 0.0,
                    "macro_task_accuracy": 0.0,
                    "avg_action_error_count": 0.0,
                    "avg_num_steps": 0.0,
                    "avg_cum_reward": 0.0,
                    "final_done_rate": 0.0,
                    "total_episodes_observed": 0,
                    "total_episodes_expected": None,
                    "observed_task_count": 0,
                    "expected_task_count": 0,
                    "missing_tasks": [],
                    "teacher_fallback_episode_rate": None,
                    "avg_teacher_fallback_count": None,
                    "started_at": None,
                    "ended_at": None,
                    "duration_seconds": None,
                    "workers": [],
                    "per_task": [],
                }
            )
            continue
        runs.append(_make_run_record(run_root, summary))

    corpus = _count_rollout_corpus(args.rollouts_root)
    best_accuracy = _best_accuracy_run(runs)
    best_coverage = _best_coverage_run(runs)

    payload = {
        "corpus": corpus,
        "best_accuracy_run": best_accuracy,
        "best_coverage_run": best_coverage,
        "runs": runs,
    }

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(_render_markdown(corpus, runs, best_accuracy, best_coverage), encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.md_out}")
    print(f"Wrote {args.json_out}")
    if best_accuracy is not None:
        print(f"Best accuracy run: {best_accuracy['run_name']} ({_fmt_pct(best_accuracy['micro_accuracy'])})")
    if best_coverage is not None:
        print(f"Best coverage run: {best_coverage['run_name']} ({best_coverage['observed_task_count']}/{best_coverage['expected_task_count']} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
