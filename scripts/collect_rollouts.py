#!/usr/bin/env python3
"""Collect MiniWoB rollouts via OpenEnv WebSocket API.

Current scope:
- Config-driven data collection with YAML
- Scripted and teacher-model action policies
- JSONL trajectory logging + episode summaries
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import yaml
from websockets.asyncio.client import connect as ws_connect

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


@dataclass
class RunPaths:
    run_dir: Path
    steps_jsonl: Path
    episodes_jsonl: Path
    resolved_config_yaml: Path


@dataclass
class TeacherRuntime:
    client: Any
    model: str
    base_url: str
    timeout_s: int
    fallback_action: str
    max_history_turns: int
    max_observation_chars: int
    system_prompt: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    extra_body: Optional[Dict[str, Any]]
    max_retries: int
    retry_backoff_s: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text.strip().lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "run"


def _to_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme in ("ws", "wss"):
        ws_scheme = parsed.scheme
    elif parsed.scheme == "https":
        ws_scheme = "wss"
    else:
        ws_scheme = "ws"
    ws_path = parsed.path.rstrip("/") + "/ws"
    return urlunparse((ws_scheme, parsed.netloc, ws_path, "", "", ""))


def _infer_miniwob_task_from_url(url: str) -> Optional[str]:
    """Extract MiniWoB task name from URL like /miniwob/click-test.html."""
    if not url:
        return None
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if not path:
        return None
    filename = path.rsplit("/", 1)[-1]
    if not filename.endswith(".html"):
        return None
    return filename[: -len(".html")] or None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a YAML mapping/object.")
    return data


def _prepare_paths(config: Dict[str, Any]) -> RunPaths:
    run_cfg = config.get("run", {})
    run_name = str(run_cfg.get("run_name", "rollout_run"))
    output_root = Path(str(run_cfg.get("output_root", "data/rollouts")))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{_slugify(run_name)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        steps_jsonl=run_dir / "trajectory_steps.jsonl",
        episodes_jsonl=run_dir / "episode_summaries.jsonl",
        resolved_config_yaml=run_dir / "resolved_config.yaml",
    )


def _get_seed(collection_cfg: Dict[str, Any], episode_idx: int) -> Optional[int]:
    mode = str(collection_cfg.get("seed_mode", "episode_index")).lower()
    seed_offset = int(collection_cfg.get("seed_offset", 0))
    if mode == "episode_index":
        return episode_idx + seed_offset
    if mode == "fixed":
        return int(collection_cfg.get("fixed_seed", 0)) + seed_offset
    if mode == "none":
        return None
    raise ValueError(f"Unsupported seed_mode: {mode}")


def _get_scripted_action(config: Dict[str, Any], task_name: str, step_idx: int) -> str:
    policy = config.get("policy", {})
    if str(policy.get("mode", "")).lower() != "scripted":
        raise ValueError("Step 1 supports only policy.mode=scripted.")
    scripted_actions = policy.get("scripted_actions", {})
    if not isinstance(scripted_actions, dict):
        raise ValueError("policy.scripted_actions must be a mapping.")

    actions = scripted_actions.get(task_name, scripted_actions.get("default"))
    if not actions:
        raise ValueError(f"No scripted actions found for task '{task_name}' and no default actions.")
    if not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise ValueError(f"Actions for task '{task_name}' must be a list of strings.")

    return actions[min(step_idx, len(actions) - 1)]


def _get_policy_mode(config: Dict[str, Any]) -> str:
    policy = config.get("policy", {})
    mode = str(policy.get("mode", "scripted")).strip().lower()
    if mode not in {"scripted", "teacher"}:
        raise ValueError("policy.mode must be either 'scripted' or 'teacher'.")
    return mode


def _extract_action_from_teacher_text(text: str) -> Optional[str]:
    """Extract one BrowserGym action string from teacher output."""
    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    allowed_actions = {
        "noop",
        "click",
        "dblclick",
        "hover",
        "focus",
        "fill",
        "clear",
        "select_option",
        "drag_and_drop",
        "scroll",
        "goto",
        "send_keys",
        "press",
    }

    # Prefer fenced code blocks if present.
    code_blocks = re.findall(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)```", cleaned)
    candidates = code_blocks if code_blocks else [cleaned]

    def _extract_calls(s: str) -> list[str]:
        out: list[str] = []
        n = len(s)
        i = 0
        while i < n:
            ch = s[i]
            if not (ch.isalpha() or ch == "_"):
                i += 1
                continue

            start = i
            i += 1
            while i < n and (s[i].isalnum() or s[i] == "_"):
                i += 1
            name = s[start:i]
            if i >= n or s[i] != "(":
                continue
            if name not in allowed_actions:
                continue

            depth = 0
            j = i
            in_quote: Optional[str] = None
            escaped = False
            while j < n:
                c = s[j]
                if in_quote is not None:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == in_quote:
                        in_quote = None
                    j += 1
                    continue

                if c in ("'", '"'):
                    in_quote = c
                    j += 1
                    continue
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        out.append(s[start : j + 1])
                        break
                j += 1
            i = j + 1
        return out

    standalone_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?`?\s*([A-Za-z_][A-Za-z0-9_]*\(.*\))\s*`?\s*;?\s*$"
    )
    scored: list[tuple[int, int, str]] = []

    for cand_idx, candidate in enumerate(candidates):
        candidate = candidate.strip()
        if not candidate:
            continue

        # Strong preference: a line that's just one action call.
        for line_idx, raw_line in enumerate(candidate.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            m = standalone_pattern.match(line)
            if not m:
                continue
            line_calls = _extract_calls(m.group(1))
            if not line_calls:
                continue
            action = line_calls[-1].rstrip(";").strip()
            if action:
                score = 1000 + (cand_idx * 100) + line_idx
                scored.append((score, len(scored), action))

        # Fallback: action calls embedded in prose.
        inline_calls = _extract_calls(candidate)
        for call_idx, action in enumerate(inline_calls):
            action = action.rstrip(";").strip()
            if not action:
                continue
            score = 100 + (cand_idx * 100) + call_idx
            scored.append((score, len(scored), action))

    if not scored:
        return None

    # Highest score wins; tie-break by last appended.
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[-1][2]


def _collect_text_fragments(value: Any, out: list[str]) -> None:
    """Collect text recursively from provider-specific message payload shapes."""
    if value is None:
        return

    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_text_fragments(item, out)
        return

    if isinstance(value, dict):
        # Prefer known text-bearing keys first.
        preferred_keys = (
            "content",
            "text",
            "reasoning_content",
            "reasoning",
            "output_text",
            "value",
            "arguments",
        )
        found_preferred = False
        for key in preferred_keys:
            if key in value:
                found_preferred = True
                _collect_text_fragments(value.get(key), out)

        if not found_preferred:
            for item in value.values():
                _collect_text_fragments(item, out)
        return

    # Pydantic/openai objects.
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            dumped = None
        if dumped is not None:
            _collect_text_fragments(dumped, out)
            return

    # Generic object fallback for provider SDK response classes.
    for attr in ("content", "text", "reasoning_content", "reasoning", "output_text", "value"):
        if hasattr(value, attr):
            try:
                _collect_text_fragments(getattr(value, attr), out)
            except Exception:
                continue


def _dedupe_nonempty_text(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _split_think_blocks(text: str) -> tuple[str, str]:
    """Split `<think>...</think>` blocks into (reasoning, answer)."""
    if not text:
        return "", ""
    cleaned = text.strip()
    if not cleaned:
        return "", ""

    think_re = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
    reasoning_parts = [m.strip() for m in think_re.findall(cleaned) if m and m.strip()]
    answer = think_re.sub("", cleaned).strip()
    reasoning = "\n\n".join(reasoning_parts).strip()
    return reasoning, answer


def _extract_teacher_message_parts(message: Any) -> Dict[str, Any]:
    """Extract structured answer/reasoning text from a provider message object."""
    answer_fragments: list[str] = []
    reasoning_fragments: list[str] = []

    if hasattr(message, "content"):
        try:
            _collect_text_fragments(getattr(message, "content"), answer_fragments)
        except Exception:
            pass
    for attr in ("reasoning_content", "reasoning"):
        if hasattr(message, attr):
            try:
                _collect_text_fragments(getattr(message, attr), reasoning_fragments)
            except Exception:
                continue

    message_dump = None
    if hasattr(message, "model_dump"):
        try:
            message_dump = message.model_dump()
        except Exception:
            message_dump = None

    if isinstance(message_dump, dict):
        for key in ("content", "text", "output_text"):
            if key in message_dump:
                _collect_text_fragments(message_dump.get(key), answer_fragments)
        for key in ("reasoning_content", "reasoning"):
            if key in message_dump:
                _collect_text_fragments(message_dump.get(key), reasoning_fragments)

    # Some models emit reasoning + answer inline in content with think tags.
    processed_answers: list[str] = []
    for fragment in answer_fragments:
        reasoning_from_content, answer_from_content = _split_think_blocks(fragment)
        if reasoning_from_content:
            reasoning_fragments.append(reasoning_from_content)
        if answer_from_content:
            processed_answers.append(answer_from_content)

    answer_candidates = _dedupe_nonempty_text(processed_answers)
    reasoning_candidates = _dedupe_nonempty_text(reasoning_fragments)
    extraction_candidates = answer_candidates + [
        candidate for candidate in reasoning_candidates if candidate not in answer_candidates
    ]

    # Last fallback: collect any text from dump if provider uses an unknown shape.
    if not extraction_candidates and message_dump is not None:
        fallback_fragments: list[str] = []
        _collect_text_fragments(message_dump, fallback_fragments)
        extraction_candidates = _dedupe_nonempty_text(fallback_fragments)

    answer_text = "\n\n".join(answer_candidates).strip()
    reasoning_text = "\n\n".join(reasoning_candidates).strip()
    raw_response = "\n\n".join(extraction_candidates).strip()

    return {
        "answer_candidates": answer_candidates,
        "reasoning_candidates": reasoning_candidates,
        "extraction_candidates": extraction_candidates,
        "answer_text": answer_text,
        "reasoning_text": reasoning_text,
        "raw_response": raw_response,
        "message_dump": message_dump,
    }


def _sum_usage(usages: list[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    total_prompt = 0
    total_completion = 0
    total_all = 0
    has_any = False

    for usage in usages:
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        all_tokens = usage.get("total_tokens")
        if isinstance(prompt, (int, float)):
            total_prompt += int(prompt)
            has_any = True
        if isinstance(completion, (int, float)):
            total_completion += int(completion)
            has_any = True
        if isinstance(all_tokens, (int, float)):
            total_all += int(all_tokens)
            has_any = True

    if not has_any:
        return None

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_all,
    }


def _build_teacher_runtime(config: Dict[str, Any]) -> TeacherRuntime:
    teacher_cfg = config.get("teacher_api", {})
    if not isinstance(teacher_cfg, dict):
        raise ValueError("teacher_api must be a mapping.")
    if not bool(teacher_cfg.get("enabled", False)):
        raise ValueError("teacher policy requires teacher_api.enabled=true.")

    if OpenAI is None:
        raise ValueError(
            "OpenAI SDK is not installed. Install with: pip install --upgrade 'openai>=1.0'"
        )

    api_key = teacher_cfg.get("api_key")
    if api_key is None:
        api_key_env_var = str(teacher_cfg.get("api_key_env_var", "ZAI_API_KEY"))
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Missing teacher API key. Set env var {api_key_env_var} or teacher_api.api_key."
            )

    model = str(teacher_cfg.get("model", "glm-5"))
    base_url = str(teacher_cfg.get("base_url", "https://api.z.ai/api/paas/v4/"))
    timeout_s = int(teacher_cfg.get("request_timeout_s", 60))
    fallback_action = str(teacher_cfg.get("fallback_action", "noop()"))
    max_history_turns = int(teacher_cfg.get("max_history_turns", 6))
    max_observation_chars = int(teacher_cfg.get("max_observation_chars", 4000))
    max_retries = int(teacher_cfg.get("max_retries", 2))
    retry_backoff_s = float(teacher_cfg.get("retry_backoff_s", 1.5))
    temperature = (
        float(teacher_cfg["temperature"])
        if teacher_cfg.get("temperature") is not None
        else None
    )
    top_p = float(teacher_cfg["top_p"]) if teacher_cfg.get("top_p") is not None else None
    max_tokens = (
        int(teacher_cfg["max_tokens"])
        if teacher_cfg.get("max_tokens") is not None
        else None
    )
    extra_body = teacher_cfg.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("teacher_api.extra_body must be a mapping/object.")

    system_prompt = str(
        teacher_cfg.get(
            "system_prompt",
            (
                "You control a web browser through BrowserGym actions.\n"
                "You must complete the given web task by interacting with the page.\n\n"
                "Available actions:\n"
                "- noop() - Do nothing\n"
                "- click(bid, modifiers=None) - Click element by BrowserGym ID. "
                "Use modifiers=['Control'] for multi-select clicks.\n"
                "- fill(bid, text) - Fill input field with text\n"
                "- send_keys(text) - Send keyboard input\n"
                "- scroll(direction) - Scroll up/down\n\n"
                "The page structure shows elements as: [bid] element_type 'element_text'\n"
                "For example: [13] button 'Click Me!' means bid='13'\n\n"
                "Reply with exactly ONE action on a single line, e.g.:\n"
                "click('13')\n"
                "click('18', modifiers=['Control'])\n"
                "fill('42', 'hello world')\n"
                "noop()\n\n"
                "Do not include explanations or multiple actions."
            ),
        )
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    return TeacherRuntime(
        client=client,
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        fallback_action=fallback_action,
        max_history_turns=max_history_turns,
        max_observation_chars=max_observation_chars,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )


def _build_teacher_user_prompt(
    task_name: str,
    step_idx: int,
    obs: Dict[str, Any],
    max_observation_chars: int,
) -> str:
    obs_text = str(obs.get("text", ""))
    if max_observation_chars > 0 and len(obs_text) > max_observation_chars:
        obs_text = obs_text[:max_observation_chars]

    multi_select_hint = ""
    if "multiselectable=True" in obs_text:
        multi_select_hint = (
            "Important: If the observation shows a multiselectable listbox and you need to "
            "select multiple options, keep existing selections by using "
            "click('<bid>', modifiers=['Control']) for additional option clicks.\n\n"
        )

    return (
        f"Goal: {obs.get('goal', '')}\n"
        "Observation:\n"
        f"{obs_text}\n\n"
        f"{multi_select_hint}"
        "Return ONLY one BrowserGym action string."
    )


def _teacher_action_sync(
    runtime: TeacherRuntime,
    task_name: str,
    step_idx: int,
    obs: Dict[str, Any],
    conversation_history: list[Dict[str, str]],
) -> Dict[str, Any]:
    user_prompt = _build_teacher_user_prompt(
        task_name=task_name,
        step_idx=step_idx,
        obs=obs,
        max_observation_chars=runtime.max_observation_chars,
    )
    max_history_messages = max(runtime.max_history_turns, 0) * 2
    history = (
        conversation_history[-max_history_messages:]
        if max_history_messages > 0
        else []
    )
    messages = [{"role": "system", "content": runtime.system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    def _call_teacher(messages_payload: list[Dict[str, str]]) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": runtime.model,
            "messages": messages_payload,
            "timeout": runtime.timeout_s,
        }
        if runtime.temperature is not None:
            request_kwargs["temperature"] = runtime.temperature
        if runtime.top_p is not None:
            request_kwargs["top_p"] = runtime.top_p
        if runtime.max_tokens is not None:
            request_kwargs["max_tokens"] = runtime.max_tokens
        if runtime.extra_body is not None:
            request_kwargs["extra_body"] = runtime.extra_body

        response = None
        total_attempts = max(runtime.max_retries, 0) + 1
        for attempt_idx in range(total_attempts):
            try:
                response = runtime.client.chat.completions.create(**request_kwargs)
                break
            except Exception as exc:
                attempt_no = attempt_idx + 1
                if attempt_no >= total_attempts:
                    raise RuntimeError(
                        "Teacher API request failed "
                        f"(base_url={runtime.base_url}, model={runtime.model}, "
                        f"attempts={total_attempts}, timeout_s={runtime.timeout_s}): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                sleep_s = runtime.retry_backoff_s * (2 ** attempt_idx)
                print(
                    "[warn] Teacher API request failed "
                    f"attempt={attempt_no}/{total_attempts} "
                    f"(base_url={runtime.base_url}, model={runtime.model}) "
                    f"retry_in={sleep_s:.1f}s: {type(exc).__name__}: {exc}"
                )
                time.sleep(sleep_s)
        if response is None:
            raise RuntimeError("Teacher API request failed unexpectedly with no response.")

        message = response.choices[0].message
        message_parts = _extract_teacher_message_parts(message)

        extracted_action = None
        for candidate_text in message_parts["extraction_candidates"]:
            extracted_action = _extract_action_from_teacher_text(candidate_text)
            if extracted_action is not None:
                break

        usage_data = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_data = usage.model_dump()
            else:
                usage_data = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

        finish_reason = None
        try:
            finish_reason = response.choices[0].finish_reason
        except Exception:
            pass

        return {
            "response": response,
            "message_parts": message_parts,
            "extracted_action": extracted_action,
            "usage_data": usage_data,
            "finish_reason": finish_reason,
        }

    attempt_results: list[Dict[str, Any]] = []
    first_attempt = _call_teacher(messages)
    attempt_results.append(first_attempt)

    # Some local vLLM models can emit an empty first reply. Retry once with
    # a synthetic noop history turn to coax a non-empty action response.
    first_parts = first_attempt["message_parts"]
    first_extracted = first_attempt["extracted_action"]
    is_empty_first_reply = (
        first_extracted is None
        and not str(first_parts.get("answer_text", "")).strip()
        and not str(first_parts.get("reasoning_text", "")).strip()
    )
    if is_empty_first_reply:
        retry_messages = list(messages)
        retry_messages.append({"role": "assistant", "content": "noop()"})
        retry_messages.append({"role": "user", "content": user_prompt})
        attempt_results.append(_call_teacher(retry_messages))

    selected = attempt_results[0]
    for attempt in attempt_results:
        if attempt.get("extracted_action") is not None:
            selected = attempt
            break
    else:
        if len(attempt_results) > 1:
            selected = attempt_results[-1]

    selected_parts = selected["message_parts"]
    extracted = selected["extracted_action"]
    used_fallback = extracted is None
    action_str = extracted or runtime.fallback_action
    raw_response = str(selected_parts.get("raw_response", "")).strip()
    answer_text = str(selected_parts.get("answer_text", "")).strip()
    reasoning_text = str(selected_parts.get("reasoning_text", "")).strip()
    # Keep logged answer aligned with the parsed action actually sent to env.
    if extracted is not None:
        answer_text = extracted
    usage_data = _sum_usage([attempt.get("usage_data") for attempt in attempt_results])

    return {
        "action_str": action_str,
        "used_fallback": used_fallback,
        "raw_response": raw_response,
        "answer_text": answer_text,
        "reasoning_text": reasoning_text,
        "attempt_count": len(attempt_results),
        "used_empty_retry": len(attempt_results) > 1,
        "finish_reason": selected.get("finish_reason"),
        "user_prompt": user_prompt,
        # Keep assistant history clean: only the final action string.
        "assistant_message": action_str,
        "messages_sent": len(messages),
        "model": runtime.model,
        "usage": usage_data,
        "raw_message": selected_parts.get("message_dump"),
    }


def _observation_diagnostics(obs: Dict[str, Any]) -> Dict[str, Any]:
    text = str(obs.get("text") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    actionable_node_count = len(re.findall(r"\[[^\]]+\]", text))
    non_root_lines = [line for line in lines if not line.startswith("RootWebArea")]
    is_root_only = bool(lines) and not non_root_lines
    is_sparse = actionable_node_count <= 1 and len(non_root_lines) <= 1
    return {
        "text_line_count": len(lines),
        "actionable_node_count": actionable_node_count,
        "is_root_only": is_root_only,
        "is_sparse": is_sparse,
    }


def _step_diagnostics(
    *,
    action_str: str,
    previous_action_str: Optional[str],
    previous_consecutive_same_action_count: int,
    previous_no_progress_streak: int,
    step_reward: float,
    done: bool,
) -> Dict[str, Any]:
    same_action_as_previous = previous_action_str == action_str if previous_action_str is not None else False
    consecutive_same_action_count = (
        previous_consecutive_same_action_count + 1 if same_action_as_previous else 1
    )
    made_progress = bool(done or step_reward > 0)
    no_progress_streak = 0 if made_progress else (previous_no_progress_streak + 1)
    repeated_action_loop = bool(same_action_as_previous and not made_progress)
    return {
        "same_action_as_previous": same_action_as_previous,
        "consecutive_same_action_count": consecutive_same_action_count,
        "no_progress_streak": no_progress_streak,
        "repeated_action_loop": repeated_action_loop,
    }


def _extract_obs_text(obs: Dict[str, Any], logging_cfg: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = _observation_diagnostics(obs)
    out: Dict[str, Any] = {
        "goal": obs.get("goal", ""),
        "url": obs.get("url", ""),
        "error": obs.get("error", ""),
        "last_action_error": bool(obs.get("last_action_error", False)),
        "done": bool(obs.get("done", False)),
        "reward": obs.get("reward"),
        "diagnostics": diagnostics,
    }
    if bool(logging_cfg.get("save_observation_text", True)):
        out["text"] = obs.get("text", "")
    if bool(logging_cfg.get("save_axtree_txt", False)):
        out["axtree_txt"] = obs.get("axtree_txt", "")
    if bool(logging_cfg.get("save_pruned_html", False)):
        out["pruned_html"] = obs.get("pruned_html", "")
    return out


def _normalize_ws_observation(message: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize OpenEnv WebSocket observation payloads to a flat observation dict.

    OpenEnv `/ws` currently returns:
    {
      "type": "observation",
      "data": {
        "observation": {...},
        "reward": ...,
        "done": ...
      }
    }

    Older/custom servers may return observation fields directly under `data`.
    This helper supports both formats and returns a single dict containing
    observation fields plus `reward` and `done`.
    """
    data = message.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid observation payload: data is {type(data).__name__}")

    nested_obs = data.get("observation")
    if isinstance(nested_obs, dict):
        normalized = dict(nested_obs)
        # Reward/done live at the wrapper level in OpenEnv's websocket format.
        normalized["reward"] = data.get("reward")
        normalized["done"] = data.get("done", False)
        return normalized

    # Backward compatibility with servers that put observation fields directly in `data`.
    return data


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


async def _ws_send_recv(ws, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    await ws.send(json.dumps(payload))
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    msg = json.loads(raw)
    if msg.get("type") == "error":
        raise RuntimeError(f"Server error: {msg.get('data', {})}")
    return msg


async def _collect_async(config: Dict[str, Any], paths: RunPaths) -> None:
    server_cfg = config.get("server", {})
    collection_cfg = config.get("collection", {})
    logging_cfg = config.get("logging", {})
    policy_mode = _get_policy_mode(config)
    teacher_runtime: Optional[TeacherRuntime] = None
    if policy_mode == "teacher":
        teacher_runtime = _build_teacher_runtime(config)

    ws_url = _to_ws_url(str(server_cfg.get("base_url", "http://localhost:8000")))
    timeout_s = int(server_cfg.get("request_timeout_s", 60))
    max_message_size_mb = float(server_cfg.get("max_message_size_mb", 100))
    tasks = collection_cfg.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("collection.tasks must be a non-empty list.")

    episodes_per_task = int(collection_cfg.get("episodes_per_task", 1))
    episode_start_index = int(collection_cfg.get("episode_start_index", 0))
    max_steps = int(collection_cfg.get("max_steps_per_episode", 10))
    pass_task_name = bool(collection_cfg.get("pass_task_name_in_reset", True))
    enforce_task_match = bool(collection_cfg.get("enforce_task_match", True))
    benchmark = str(collection_cfg.get("benchmark", "miniwob"))
    run_id = paths.run_dir.name

    print(f"[info] Connecting to {ws_url}")
    print(f"[info] Policy mode: {policy_mode}")
    async with ws_connect(
        ws_url,
        open_timeout=timeout_s,
        max_size=int(max_message_size_mb * 1024 * 1024),
    ) as ws:
        for task_name in tasks:
            for local_episode_idx in range(episodes_per_task):
                episode_idx = episode_start_index + local_episode_idx
                seed = _get_seed(collection_cfg, episode_idx)
                reset_data: Dict[str, Any] = {}
                if seed is not None:
                    reset_data["seed"] = seed
                if pass_task_name:
                    reset_data["task_name"] = task_name

                reset_msg = await _ws_send_recv(
                    ws, {"type": "reset", "data": reset_data}, timeout_s=timeout_s
                )
                if reset_msg.get("type") != "observation":
                    raise RuntimeError(f"Unexpected reset response: {reset_msg}")

                current_obs = _normalize_ws_observation(reset_msg)
                if (
                    enforce_task_match
                    and benchmark == "miniwob"
                    and pass_task_name
                ):
                    observed_task = _infer_miniwob_task_from_url(str(current_obs.get("url", "")))
                    if observed_task is not None and observed_task != task_name:
                        raise RuntimeError(
                            "Task mismatch after reset: "
                            f"requested task='{task_name}' but server loaded '{observed_task}' "
                            f"(url={current_obs.get('url', '')}). "
                            "This BrowserGym server appears pinned to one task "
                            "(often via BROWSERGYM_TASK_NAME) and does not switch tasks per reset."
                        )
                done = bool(current_obs.get("done", False))
                cum_reward = float(current_obs.get("reward") or 0.0)
                action_errors = 0
                steps_taken = 0
                teacher_fallback_count = 0
                root_only_observation_count = 0
                sparse_observation_count = 0
                repeated_action_loop_count = 0
                max_consecutive_same_action_count = 0
                max_no_progress_streak = 0
                previous_action_str: Optional[str] = None
                consecutive_same_action_count = 0
                no_progress_streak = 0
                teacher_conversation: list[Dict[str, str]] = []
                episode_id = f"{run_id}:{task_name}:ep{episode_idx}"

                print(f"[episode] task={task_name} ep={episode_idx} seed={seed}")

                for step_idx in range(max_steps):
                    if done:
                        break

                    teacher_meta: Optional[Dict[str, Any]] = None
                    teacher_latency_ms: Optional[float] = None
                    if policy_mode == "scripted":
                        action_str = _get_scripted_action(config, task_name, step_idx)
                    else:
                        if teacher_runtime is None:
                            raise RuntimeError("Teacher runtime is not initialized.")
                        teacher_started = time.perf_counter()
                        teacher_meta = await asyncio.to_thread(
                            _teacher_action_sync,
                            teacher_runtime,
                            task_name,
                            step_idx,
                            current_obs,
                            teacher_conversation,
                        )
                        teacher_ended = time.perf_counter()
                        teacher_latency_ms = (teacher_ended - teacher_started) * 1000.0
                        action_str = str(teacher_meta["action_str"])
                        if bool(teacher_meta.get("used_fallback", False)):
                            teacher_fallback_count += 1

                    pre_obs = _extract_obs_text(current_obs, logging_cfg)
                    pre_obs_diag = dict(pre_obs.get("diagnostics", {}))
                    if bool(pre_obs_diag.get("is_root_only", False)):
                        root_only_observation_count += 1
                    if bool(pre_obs_diag.get("is_sparse", False)):
                        sparse_observation_count += 1
                    env_step_started = time.perf_counter()
                    step_msg = await _ws_send_recv(
                        ws,
                        {
                            "type": "step",
                            "data": {
                                "action_str": action_str,
                                "metadata": {"task_name": task_name, "episode_idx": episode_idx},
                            },
                        },
                        timeout_s=timeout_s,
                    )
                    step_ended = time.perf_counter()

                    if step_msg.get("type") != "observation":
                        raise RuntimeError(f"Unexpected step response: {step_msg}")

                    next_obs = _normalize_ws_observation(step_msg)
                    step_reward = float(next_obs.get("reward") or 0.0)
                    done = bool(next_obs.get("done", False))
                    last_action_error = bool(next_obs.get("last_action_error", False))
                    if last_action_error:
                        action_errors += 1
                    cum_reward += step_reward
                    steps_taken += 1

                    step_diag = _step_diagnostics(
                        action_str=action_str,
                        previous_action_str=previous_action_str,
                        previous_consecutive_same_action_count=consecutive_same_action_count,
                        previous_no_progress_streak=no_progress_streak,
                        step_reward=step_reward,
                        done=done,
                    )
                    consecutive_same_action_count = int(step_diag["consecutive_same_action_count"])
                    no_progress_streak = int(step_diag["no_progress_streak"])
                    previous_action_str = action_str
                    if bool(step_diag["repeated_action_loop"]):
                        repeated_action_loop_count += 1
                    max_consecutive_same_action_count = max(
                        max_consecutive_same_action_count,
                        consecutive_same_action_count,
                    )
                    max_no_progress_streak = max(max_no_progress_streak, no_progress_streak)

                    row = {
                        "run_id": run_id,
                        "timestamp": _utc_now_iso(),
                        "benchmark": benchmark,
                        "task_name": task_name,
                        "episode_id": episode_id,
                        "episode_idx": episode_idx,
                        "seed": seed,
                        "step_idx": step_idx,
                        "policy_mode": policy_mode,
                        "action_str": action_str,
                        "pre_observation": pre_obs,
                        "post_observation": _extract_obs_text(next_obs, logging_cfg),
                        "reward": step_reward,
                        "done": done,
                        "last_action_error": last_action_error,
                        "latency_ms": (step_ended - env_step_started) * 1000.0,
                        "observation_actionable_node_count": int(pre_obs_diag.get("actionable_node_count", 0)),
                        "observation_text_line_count": int(pre_obs_diag.get("text_line_count", 0)),
                        "observation_is_root_only": bool(pre_obs_diag.get("is_root_only", False)),
                        "observation_is_sparse": bool(pre_obs_diag.get("is_sparse", False)),
                        "same_action_as_previous": bool(step_diag["same_action_as_previous"]),
                        "consecutive_same_action_count": int(step_diag["consecutive_same_action_count"]),
                        "no_progress_streak": int(step_diag["no_progress_streak"]),
                        "repeated_action_loop": bool(step_diag["repeated_action_loop"]),
                    }
                    if teacher_meta is not None:
                        row["teacher_model"] = teacher_meta.get("model")
                        row["teacher_used_fallback"] = bool(
                            teacher_meta.get("used_fallback", False)
                        )
                        if teacher_meta.get("usage") is not None:
                            row["teacher_usage"] = teacher_meta.get("usage")
                        row["teacher_response_answer"] = teacher_meta.get("answer_text", "")
                        row["teacher_response_reasoning"] = teacher_meta.get(
                            "reasoning_text", ""
                        )
                        if teacher_latency_ms is not None:
                            row["teacher_latency_ms"] = teacher_latency_ms
                    _append_jsonl(paths.steps_jsonl, row)

                    if teacher_meta is not None:
                        user_prompt = str(teacher_meta.get("user_prompt", "")).strip()
                        assistant_msg = str(
                            teacher_meta.get("assistant_message", action_str)
                        ).strip()
                        if user_prompt:
                            teacher_conversation.append(
                                {"role": "user", "content": user_prompt}
                            )
                        teacher_conversation.append(
                            {"role": "assistant", "content": assistant_msg or action_str}
                        )

                        if teacher_runtime is not None and teacher_runtime.max_history_turns > 0:
                            max_history_messages = teacher_runtime.max_history_turns * 2
                            if len(teacher_conversation) > max_history_messages:
                                teacher_conversation = teacher_conversation[-max_history_messages:]
                    current_obs = next_obs

                summary = {
                    "run_id": run_id,
                    "timestamp": _utc_now_iso(),
                    "benchmark": benchmark,
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "episode_idx": episode_idx,
                    "seed": seed,
                    "num_steps": steps_taken,
                    "cum_reward": cum_reward,
                    "success": bool(done and cum_reward > 0),
                    "final_done": done,
                    "action_error_count": action_errors,
                    "root_only_observation_count": root_only_observation_count,
                    "sparse_observation_count": sparse_observation_count,
                    "repeated_action_loop_count": repeated_action_loop_count,
                    "max_consecutive_same_action_count": max_consecutive_same_action_count,
                    "max_no_progress_streak": max_no_progress_streak,
                }
                if policy_mode == "teacher":
                    summary["teacher_fallback_count"] = teacher_fallback_count
                _append_jsonl(paths.episodes_jsonl, summary)

        await ws.send(json.dumps({"type": "close"}))


def _write_resolved_config(config: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect BrowserGym rollouts into JSONL.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rollout_config.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_yaml(args.config)
    paths = _prepare_paths(config)
    _write_resolved_config(config, paths.resolved_config_yaml)

    print(f"[info] Run directory: {paths.run_dir}")
    try:
        asyncio.run(_collect_async(config, paths))
    except KeyboardInterrupt:
        print("[warn] Interrupted by user.")
        return 1
    except Exception as exc:
        print(f"[error] Collection failed: {type(exc).__name__}: {exc}")
        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            print(f"[error] Root cause: {type(cause).__name__}: {cause}")
        return 1

    print("[done] Collection complete.")
    print(f"[done] Step logs: {paths.steps_jsonl}")
    print(f"[done] Episode summaries: {paths.episodes_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
