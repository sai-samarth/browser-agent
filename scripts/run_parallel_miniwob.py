#!/usr/bin/env python3
"""Launch parallel MiniWoB env containers and rollout collectors."""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorkerSpec:
    worker_idx: int
    port: int
    container_name: str
    tasks: list[str]
    config_path: Path
    log_path: Path


def _run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a YAML mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _discover_registered_tasks(image: str, benchmark: str) -> list[str]:
    script = rf"""
set -euo pipefail
python - <<'PY'
import gymnasium as gym
import importlib

benchmark = {benchmark!r}
try:
    importlib.import_module(f"browsergym.{{benchmark}}")
except Exception:
    # Continue; registry may still be populated in some images.
    pass

prefix = f"browsergym/{{benchmark}}."
tasks = sorted(
    str(env_id).split(prefix, 1)[1]
    for env_id in gym.envs.registry.keys()
    if str(env_id).startswith(prefix)
)
for task in tasks:
    print(task)
PY
"""
    cmd = ["docker", "run", "--rm", "--entrypoint", "bash", image, "-lc", script]
    proc = _run_cmd(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to discover registered BrowserGym tasks.\n"
            f"Command: {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stderr: {proc.stderr.strip()}"
        )

    tasks = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    tasks = sorted(set(tasks))
    if not tasks:
        raise RuntimeError(
            f"No registered tasks found for benchmark '{benchmark}'. "
            "Check image benchmark packages."
        )
    return tasks


def _discover_miniwob_html_tasks(image: str) -> list[str]:
    script = r"""
set -euo pipefail
for root in \
  /app/miniwob-plusplus/miniwob/html/miniwob \
  /miniwob-plusplus/miniwob/html/miniwob
do
  if [ -d "$root" ]; then
    for f in "$root"/*.html; do
      t="$(basename "$f" .html)"
      [ "$t" = "index" ] && continue
      printf "%s\n" "$t"
    done
    exit 0
  fi
done
echo "MiniWoB HTML directory not found in image." >&2
exit 2
"""
    cmd = ["docker", "run", "--rm", "--entrypoint", "bash", image, "-lc", script]
    proc = _run_cmd(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to discover MiniWoB tasks.\n"
            f"Command: {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stderr: {proc.stderr.strip()}"
        )

    tasks = sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})
    if not tasks:
        raise RuntimeError("Task discovery returned an empty task list.")
    return tasks


def _load_tasks(task_list_file: Path) -> list[str]:
    tasks: list[str] = []
    with task_list_file.open("r", encoding="utf-8") as f:
        for raw in f:
            task = raw.strip()
            if not task or task.startswith("#"):
                continue
            if task.startswith("- "):
                task = task[2:].strip().strip('"').strip("'")
            tasks.append(task)
    tasks = sorted(set(tasks))
    if not tasks:
        raise ValueError(f"No tasks found in {task_list_file}")
    return tasks


def _validate_tasks_or_fail(tasks: list[str], valid_tasks: set[str], source: str) -> None:
    invalid = sorted({task for task in tasks if task not in valid_tasks})
    if not invalid:
        return
    preview = ", ".join(invalid[:10])
    suffix = "" if len(invalid) <= 10 else f" ... (+{len(invalid) - 10} more)"
    raise ValueError(
        f"Invalid tasks from {source}: {preview}{suffix}. "
        "These are not registered BrowserGym env IDs for this benchmark/image."
    )


def _round_robin_shard(tasks: list[str], workers: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(workers)]
    for idx, task in enumerate(tasks):
        shards[idx % workers].append(task)
    return shards


def _is_host_port_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _allocate_ports(base_port: int, count: int) -> list[int]:
    if base_port < 1 or base_port > 65535:
        raise ValueError("--base-port must be in range [1, 65535].")
    if count < 1:
        return []

    ports: list[int] = []
    port = base_port
    while len(ports) < count:
        if port > 65535:
            raise RuntimeError("Ran out of ports while allocating worker ports.")
        if _is_host_port_free(port):
            ports.append(port)
        port += 1
    return ports


def _container_status(name: str) -> str:
    cmd = [
        "docker",
        "inspect",
        "--format",
        "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
        name,
    ]
    proc = _run_cmd(cmd, check=False)
    if proc.returncode != 0:
        return "missing"
    return proc.stdout.strip() or "unknown"


def _wait_for_container_ready(name: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_status = "unknown"
    while time.monotonic() < deadline:
        status = _container_status(name)
        last_status = status
        if status in {"healthy", "running"}:
            return
        if status in {"exited", "dead", "unhealthy", "missing"}:
            break
        time.sleep(1.0)
    raise RuntimeError(f"Container {name} did not become ready. Last status={last_status}")


def _remove_container_if_exists(name: str) -> None:
    _run_cmd(["docker", "rm", "-f", name], check=False)


def _launch_container(
    image: str,
    benchmark: str,
    name: str,
    port: int,
    task_name_env: str | None,
    startup_timeout_s: int,
    dry_run: bool,
) -> None:
    _remove_container_if_exists(name)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{port}:8000",
        "-e",
        f"BROWSERGYM_BENCHMARK={benchmark}",
    ]
    if task_name_env is not None:
        cmd.extend(["-e", f"BROWSERGYM_TASK_NAME={task_name_env}"])
    cmd.append(image)
    print(f"[container] launch {name} on :{port}")
    if dry_run:
        print(f"[dry-run] {' '.join(shlex.quote(x) for x in cmd)}")
        return

    proc = _run_cmd(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to start container {name}.\n"
            f"stderr: {proc.stderr.strip()}"
        )
    _wait_for_container_ready(name, startup_timeout_s)


def _build_worker_config(
    template: dict[str, Any],
    run_name_prefix: str,
    worker_idx: int,
    port: int,
    tasks: list[str],
    episodes_per_task: int | None,
    max_steps_per_episode: int | None,
    seed_offset_base: int,
    seed_stride: int,
    teacher_base_url: str | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(template)
    cfg.setdefault("run", {})
    cfg.setdefault("server", {})
    cfg.setdefault("collection", {})

    cfg["run"]["run_name"] = f"{run_name_prefix}_w{worker_idx:02d}"
    cfg["server"]["base_url"] = f"http://localhost:{port}"
    cfg["collection"]["tasks"] = tasks
    cfg["collection"]["episode_start_index"] = 0
    cfg["collection"]["seed_offset"] = seed_offset_base + (worker_idx * seed_stride)

    if episodes_per_task is not None:
        cfg["collection"]["episodes_per_task"] = int(episodes_per_task)
    if max_steps_per_episode is not None:
        cfg["collection"]["max_steps_per_episode"] = int(max_steps_per_episode)

    if teacher_base_url is not None:
        cfg.setdefault("teacher_api", {})
        cfg["teacher_api"]["base_url"] = teacher_base_url

    return cfg


def _launch_collectors(
    workers: list[WorkerSpec],
    collector_python: str,
    collector_script: Path,
    dry_run: bool,
) -> dict[int, int]:
    procs: dict[int, subprocess.Popen[str]] = {}
    exit_codes: dict[int, int] = {}

    try:
        for spec in workers:
            cmd = [collector_python, str(collector_script), "--config", str(spec.config_path)]
            print(f"[collector] worker={spec.worker_idx} tasks={len(spec.tasks)} log={spec.log_path}")
            if dry_run:
                print(f"[dry-run] {' '.join(shlex.quote(x) for x in cmd)}")
                continue
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_f = spec.log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            log_f.close()
            procs[spec.worker_idx] = proc

        while procs:
            done: list[int] = []
            for worker_idx, proc in procs.items():
                code = proc.poll()
                if code is None:
                    continue
                exit_codes[worker_idx] = code
                done.append(worker_idx)
                print(f"[collector] worker={worker_idx} finished exit_code={code}")
            for worker_idx in done:
                procs.pop(worker_idx, None)
            if procs:
                time.sleep(1.0)
    finally:
        for worker_idx, proc in procs.items():
            print(f"[collector] terminating worker={worker_idx}")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for worker_idx, proc in procs.items():
            exit_codes[worker_idx] = proc.returncode if proc.returncode is not None else -9

    return exit_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shard MiniWoB tasks and run parallel BrowserGym rollout collectors."
    )
    parser.add_argument("--config-template", type=Path, default=Path("configs/rollout_config.yaml"))
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel env+collector workers.")
    parser.add_argument("--base-port", type=int, default=8000, help="Worker 0 port; worker i uses base+i.")
    parser.add_argument("--image", type=str, default="browsergym-env:latest")
    parser.add_argument("--benchmark", type=str, default="miniwob")
    parser.add_argument(
        "--task-name-env",
        type=str,
        default=None,
        help="Optional BROWSERGYM_TASK_NAME env value for all workers.",
    )
    parser.add_argument("--run-name-prefix", type=str, default="miniwob_parallel")
    parser.add_argument("--output-dir", type=Path, default=Path("data/parallel_runs"))
    parser.add_argument("--task-list-file", type=Path, default=None, help="Optional custom task list file.")
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--seed-offset-base", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=100000)
    parser.add_argument("--teacher-base-url", type=str, default=None, help="Optional override, e.g. http://localhost:7999/v1")
    parser.add_argument("--collector-python", type=str, default=sys.executable)
    parser.add_argument("--collector-script", type=Path, default=Path("scripts/collect_rollouts.py"))
    parser.add_argument("--startup-timeout-s", type=int, default=120)
    parser.add_argument("--keep-containers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    template = _load_yaml(args.config_template)

    valid_tasks = set(_discover_registered_tasks(args.image, args.benchmark))
    if args.task_list_file is not None:
        tasks = _load_tasks(args.task_list_file)
        _validate_tasks_or_fail(tasks, valid_tasks, str(args.task_list_file))
    else:
        tasks = sorted(valid_tasks)

    shards = _round_robin_shard(tasks, args.workers)
    active_workers = [idx for idx, shard in enumerate(shards) if shard]
    if not active_workers:
        raise RuntimeError("No tasks assigned to workers.")
    allocated_ports = _allocate_ports(args.base_port, len(active_workers))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir / f"{args.run_name_prefix}_{timestamp}"
    configs_dir = run_root / "configs"
    logs_dir = run_root / "logs"
    configs_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)

    worker_specs: list[WorkerSpec] = []
    for slot_idx, worker_idx in enumerate(active_workers):
        port = allocated_ports[slot_idx]
        name = f"bg-{args.run_name_prefix}-{worker_idx:02d}-{timestamp}"
        worker_cfg = _build_worker_config(
            template=template,
            run_name_prefix=args.run_name_prefix,
            worker_idx=worker_idx,
            port=port,
            tasks=shards[worker_idx],
            episodes_per_task=args.episodes_per_task,
            max_steps_per_episode=args.max_steps_per_episode,
            seed_offset_base=args.seed_offset_base,
            seed_stride=args.seed_stride,
            teacher_base_url=args.teacher_base_url,
        )
        config_path = configs_dir / f"worker_{worker_idx:02d}.yaml"
        log_path = logs_dir / f"worker_{worker_idx:02d}.log"
        _write_yaml(config_path, worker_cfg)
        worker_specs.append(
            WorkerSpec(
                worker_idx=worker_idx,
                port=port,
                container_name=name,
                tasks=shards[worker_idx],
                config_path=config_path,
                log_path=log_path,
            )
        )

    print(f"[info] run_root={run_root}")
    print(f"[info] total_tasks={len(tasks)} workers={len(worker_specs)}")
    for spec in worker_specs:
        print(
            f"[plan] worker={spec.worker_idx} port={spec.port} "
            f"tasks={len(spec.tasks)} container={spec.container_name}"
        )

    started_containers: list[str] = []
    exit_code = 0
    try:
        for spec in worker_specs:
            _launch_container(
                image=args.image,
                benchmark=args.benchmark,
                name=spec.container_name,
                port=spec.port,
                task_name_env=args.task_name_env,
                startup_timeout_s=args.startup_timeout_s,
                dry_run=args.dry_run,
            )
            started_containers.append(spec.container_name)

        collector_results = _launch_collectors(
            workers=worker_specs,
            collector_python=args.collector_python,
            collector_script=args.collector_script,
            dry_run=args.dry_run,
        )
        failed = {idx: code for idx, code in collector_results.items() if code != 0}
        if failed:
            exit_code = 1
            print(f"[error] worker failures: {failed}")
        else:
            print("[done] all workers completed successfully")
    except KeyboardInterrupt:
        exit_code = 130
        print("[warn] interrupted by user")
    finally:
        if args.keep_containers:
            print("[info] keep_containers=true, leaving containers running")
        else:
            for name in started_containers:
                print(f"[container] cleanup {name}")
                _remove_container_if_exists(name)

    print(f"[info] configs: {configs_dir}")
    print(f"[info] logs: {logs_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
