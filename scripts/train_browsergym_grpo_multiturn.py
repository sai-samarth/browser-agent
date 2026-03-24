#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
import transformers.utils.hub as transformers_hub
from datasets import Dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
    transformers_hub.TRANSFORMERS_CACHE = getattr(transformers_hub, "default_cache_path", None) or ""

from trl import GRPOConfig, GRPOTrainer
from websockets.asyncio.client import connect as ws_connect

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_rollouts import (  # noqa: E402
    _extract_action_from_teacher_text,
    _normalize_browsergym_action,
    _normalize_ws_observation,
    _observation_diagnostics,
    _to_ws_url,
    _ws_send_recv,
)
from scripts.export_sft_dataset import (  # noqa: E402
    FALLBACK_SYSTEM_PROMPT,
    _action_only_system_prompt,
    _render_history,
)


@dataclass
class MultiTurnGRPOConfig:
    model_name: str
    browsergym_base_url: str
    output_dir: str
    adapter_dir: str | None = None
    task_name: str | None = None
    task_names: list[str] | None = None
    task_names_file: str | None = None
    task_seed_pairs_file: str | None = None
    dataset_size: int = 12
    samples_per_task: int | None = None
    seed_offset: int = 980000
    system_prompt: str | None = None
    max_prompt_length: int = 1536
    max_completion_length: int = 64
    learning_rate: float = 5e-6
    warmup_steps: int = 0
    logging_steps: int = 1
    save_steps: int = 10
    max_steps: int = 20
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 2
    generation_batch_size: int = 2
    bf16: bool = True
    report_to: str = "none"
    use_vllm: bool = False
    rollout_max_steps: int = 10
    rollout_generation_max_new_tokens: int = 48
    rollout_do_sample: bool = True
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95
    final_success_reward: float = 2.0
    env_reward_scale: float = 1.0
    parse_valid_reward: float = 0.05
    parse_invalid_penalty: float = 0.05
    action_error_penalty: float = 0.1
    per_step_penalty: float = 0.01
    max_consecutive_invalid_actions: int = 3
    progress_shaping_scale: float = 1.0
    premature_submit_penalty: float = 0.1

    @classmethod
    def from_yaml(cls, path: Path) -> "MultiTurnGRPOConfig":
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Config must be a YAML mapping")
        return cls(**data)


class BrowserGymWSClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.ws_url = _to_ws_url(base_url)

    async def reset(self, *, task_name: str, seed: int | None) -> dict[str, Any]:
        async with ws_connect(self.ws_url, open_timeout=30, max_size=100 * 1024 * 1024) as ws:
            payload = {"type": "reset", "data": {"task_name": task_name}}
            if seed is not None:
                payload["data"]["seed"] = seed
            msg = await _ws_send_recv(ws, payload, timeout_s=60)
            if msg.get("type") != "observation":
                raise RuntimeError(f"Unexpected reset response: {msg}")
            return _normalize_ws_observation(msg)

    async def rollout(
        self,
        *,
        task_name: str,
        seed: int,
        first_completion_text: str,
        model,
        tokenizer,
        system_prompt: str,
        config: MultiTurnGRPOConfig,
    ) -> dict[str, Any]:
        history_rows: list[dict[str, Any]] = []
        total_reward = 0.0
        consecutive_invalid = 0
        done = False
        success = False
        action_error_count = 0
        parseable_action_count = 0
        raw_completions: list[str] = []

        async with ws_connect(self.ws_url, open_timeout=30, max_size=100 * 1024 * 1024) as ws:
            reset_payload = {"type": "reset", "data": {"task_name": task_name, "seed": seed}}
            reset_msg = await _ws_send_recv(ws, reset_payload, timeout_s=60)
            if reset_msg.get("type") != "observation":
                raise RuntimeError(f"Unexpected reset response: {reset_msg}")
            current_obs = _normalize_ws_observation(reset_msg)

            for step_idx in range(config.rollout_max_steps):
                prev_progress = _task_progress(
                    task_name,
                    str(current_obs.get("goal") or ""),
                    str(current_obs.get("text") or ""),
                    history_rows,
                )
                total_reward -= config.per_step_penalty
                if step_idx == 0:
                    completion_text = first_completion_text
                else:
                    completion_text = generate_completion_text(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=system_prompt,
                        task_name=task_name,
                        current_obs=current_obs,
                        history_rows=history_rows,
                        max_new_tokens=config.rollout_generation_max_new_tokens,
                        do_sample=config.rollout_do_sample,
                        temperature=config.rollout_temperature,
                        top_p=config.rollout_top_p,
                    )
                raw_completions.append(completion_text)
                parsed_action = parse_action(completion_text)
                if parsed_action is None:
                    total_reward -= config.parse_invalid_penalty
                    consecutive_invalid += 1
                    if consecutive_invalid >= config.max_consecutive_invalid_actions:
                        break
                    parsed_action = "noop()"
                else:
                    total_reward += config.parse_valid_reward
                    consecutive_invalid = 0
                    parseable_action_count += 1
                    total_reward -= _submit_penalty(parsed_action, str(current_obs.get("text") or ""), prev_progress, config)

                step_msg = await _ws_send_recv(
                    ws,
                    {
                        "type": "step",
                        "data": {
                            "action_str": parsed_action,
                            "metadata": {"task_name": task_name, "seed": seed, "step_idx": step_idx},
                        },
                    },
                    timeout_s=60,
                )
                if step_msg.get("type") != "observation":
                    raise RuntimeError(f"Unexpected step response: {step_msg}")
                next_obs = _normalize_ws_observation(step_msg)
                step_env_reward = float(next_obs.get("reward") or 0.0)
                total_reward += config.env_reward_scale * step_env_reward
                step_done = bool(next_obs.get("done", False))
                last_action_error = bool(next_obs.get("last_action_error", False))
                if last_action_error:
                    action_error_count += 1
                    total_reward -= config.action_error_penalty

                history_rows.append(
                    {
                        "step_idx": step_idx,
                        "action_str": parsed_action,
                        "reward": step_env_reward,
                        "done": step_done,
                        "last_action_error": last_action_error,
                        "post_observation": {"text": str(next_obs.get("text") or "")},
                    }
                )
                next_progress = _task_progress(
                    task_name,
                    str(next_obs.get("goal") or ""),
                    str(next_obs.get("text") or ""),
                    history_rows,
                )
                if prev_progress is not None and next_progress is not None:
                    total_reward += config.progress_shaping_scale * (next_progress - prev_progress)
                current_obs = next_obs
                done = step_done
                if done and step_env_reward > 0:
                    success = True
                    total_reward += config.final_success_reward
                    break
                if done:
                    break

        return {
            "reward": total_reward,
            "success": success,
            "done": done,
            "steps_taken": len(history_rows),
            "action_error_count": action_error_count,
            "parseable_action_count": parseable_action_count,
            "raw_completions": raw_completions,
        }


def _load_task_names(config: MultiTurnGRPOConfig) -> list[str]:
    names: list[str] = []
    if config.task_name:
        names.append(config.task_name)
    if config.task_names:
        names.extend(str(x) for x in config.task_names)
    if config.task_names_file:
        for raw in Path(config.task_names_file).read_text().splitlines():
            task = raw.strip()
            if not task or task.startswith("#"):
                continue
            names.append(task)
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        raise ValueError("Provide task_name, task_names, or task_names_file")
    return deduped


def _build_task_seed_pairs(config: MultiTurnGRPOConfig) -> list[tuple[str, int]]:
    if config.task_seed_pairs_file:
        pairs: list[tuple[str, int]] = []
        for raw in Path(config.task_seed_pairs_file).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '\t' in line:
                task, seed_text = line.split('\t', 1)
            elif ',' in line:
                task, seed_text = line.split(',', 1)
            else:
                raise ValueError(f"Invalid task-seed pair line: {raw!r}")
            pairs.append((task.strip(), int(seed_text.strip())))
        if not pairs:
            raise ValueError('task_seed_pairs_file provided but no valid pairs found')
        return pairs

    tasks = _load_task_names(config)
    pairs: list[tuple[str, int]] = []
    seed = config.seed_offset
    if config.samples_per_task is not None:
        for task in tasks:
            for _ in range(config.samples_per_task):
                pairs.append((task, seed))
                seed += 1
        return pairs
    for idx in range(config.dataset_size):
        pairs.append((tasks[idx % len(tasks)], seed))
        seed += 1
    return pairs


def build_pre_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": obs.get("goal", ""),
        "url": obs.get("url", ""),
        "last_action_error": bool(obs.get("last_action_error", False)),
        "text": str(obs.get("text", "")),
        "diagnostics": _observation_diagnostics(obs),
    }


def render_user_message(task_name: str, current_obs: dict[str, Any], history_rows: list[dict[str, Any]]) -> str:
    pre = build_pre_observation(current_obs)
    diag = pre.get("diagnostics", {})
    return (
        f"Task: {task_name}\n"
        f"Goal: {pre.get('goal', '')}\n"
        f"Current URL: {pre.get('url', '')}\n"
        f"Last action had error: {bool(pre.get('last_action_error', False))}\n"
        f"Observation diagnostics: actionable_nodes={diag.get('actionable_node_count', 0)}, "
        f"text_lines={diag.get('text_line_count', 0)}, sparse={bool(diag.get('is_sparse', False))}, "
        f"root_only={bool(diag.get('is_root_only', False))}\n\n"
        f"Recent history:\n{_render_history(history_rows)}\n\n"
        f"Current observation:\n{pre.get('text', '')}"
    )


def build_prompt_dataset(config: MultiTurnGRPOConfig) -> list[dict[str, Any]]:
    client = BrowserGymWSClient(config.browsergym_base_url)
    system_prompt = config.system_prompt or _action_only_system_prompt(FALLBACK_SYSTEM_PROMPT)
    rows: list[dict[str, Any]] = []
    for task_name, seed in _build_task_seed_pairs(config):
        obs = asyncio.run(client.reset(task_name=task_name, seed=seed))
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": render_user_message(task_name, obs, [])},
                ],
                "task_name": task_name,
                "seed": seed,
                "goal": str(obs.get("goal") or ""),
                "initial_url": str(obs.get("url") or ""),
            }
        )
    return rows


def completion_texts(completions: list[Any]) -> list[str]:
    texts: list[str] = []
    for completion in completions:
        text = ""
        if isinstance(completion, list) and completion:
            first = completion[0]
            if isinstance(first, dict):
                text = str(first.get("content") or first.get("text") or "")
            else:
                text = str(first)
        elif isinstance(completion, dict):
            text = str(completion.get("content") or completion.get("text") or "")
        else:
            text = str(completion or "")
        texts.append(text)
    return texts


def parse_action(response_text: str) -> str | None:
    extracted = _extract_action_from_teacher_text(response_text)
    if extracted is None:
        return None
    return _normalize_browsergym_action(extracted)

def _extract_goal_quoted_text(goal: str) -> str | None:
    m = re.search(r'"([^"]+)"', goal or '')
    return m.group(1) if m else None


def _target_text_from_goal(goal: str) -> str | None:
    target = _extract_goal_quoted_text(goal or '')
    if not target:
        return None
    goal_l = (goal or '').lower()
    if 'all upper case' in goal_l:
        return target.upper()
    if 'all lower case' in goal_l:
        return target.lower()
    return target


def _extract_textbox_values(obs_text: str) -> list[tuple[str, str]]:
    return re.findall(r"\[(\d+)\] textbox '([^']*)'", obs_text or '')


def _extract_button_labels(obs_text: str) -> dict[str, str]:
    return {bid: label for bid, label in re.findall(r"\[(\d+)\] button '([^']*)'", obs_text or '')}


def _extract_checkbox_states(obs_text: str) -> dict[str, bool]:
    states = {}
    pattern = re.compile(r"\[(\d+)\] checkbox '([^']*)', checked='(true|false)'")
    for _bid, label, checked in pattern.findall(obs_text or ''):
        states[label] = checked == 'true'
    return states


def _parse_fill_action(action: str) -> tuple[str, str] | None:
    m = re.match(r"fill\('([^']+)',\s*'([^']*)'\)", action or '')
    if not m:
        return None
    return m.group(1), m.group(2)


def _parse_click_bid(action: str) -> str | None:
    m = re.match(r"click\('([^']+)'(?:,.*)?\)", action or '')
    if not m:
        return None
    return m.group(1)


def _latest_fill_values(history_rows: list[dict[str, Any]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in history_rows:
        parsed = _parse_fill_action(str(row.get('action_str') or ''))
        if parsed is not None:
            bid, value = parsed
            values[bid] = value
    return values


def _checkbox_target_set(goal: str) -> set[str] | None:
    g = (goal or '').strip()
    if not g.lower().startswith('select '):
        return None
    body = g[len('Select '):]
    body = re.sub(r' and click submit\.?$', '', body, flags=re.I).strip()
    if body.lower() == 'nothing':
        return set()
    return {p.strip() for p in body.split(',') if p.strip()}


def _enter_text_progress(goal: str, obs_text: str, history_rows: list[dict[str, Any]]) -> float | None:
    target = _target_text_from_goal(goal)
    if not target:
        return None
    textbox_values = _extract_textbox_values(obs_text)
    current = textbox_values[0][1] if textbox_values else ''
    if not current and textbox_values:
        fills = _latest_fill_values(history_rows)
        current = fills.get(textbox_values[0][0], '')
    return 1.0 if target == '' else SequenceMatcher(None, current, target).ratio()


def _enter_password_progress(goal: str, obs_text: str, history_rows: list[dict[str, Any]]) -> float | None:
    target = _extract_goal_quoted_text(goal or '')
    if not target:
        return None
    textbox_bids = [bid for bid, _value in _extract_textbox_values(obs_text)]
    if len(textbox_bids) < 2:
        return None
    fills = _latest_fill_values(history_rows)
    correct = sum(1 for bid in textbox_bids[:2] if fills.get(bid) == target)
    return correct / 2.0


def _extract_find_word_target(goal: str, obs_text: str) -> str | None:
    m = re.search(r'Find the (\d+)(?:st|nd|rd|th) word', goal or '', flags=re.I)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    para_match = re.search(r"StaticText '([^']+)'", obs_text or '')
    if not para_match:
        return None
    words = re.findall(r"[A-Za-z0-9'-]+", para_match.group(1))
    if idx < 0 or idx >= len(words):
        return None
    return words[idx]


def _find_word_progress(goal: str, obs_text: str, history_rows: list[dict[str, Any]]) -> float | None:
    target = _extract_find_word_target(goal, obs_text)
    if not target:
        return None
    textbox_values = _extract_textbox_values(obs_text)
    current = textbox_values[0][1] if textbox_values else ''
    if not current and textbox_values:
        fills = _latest_fill_values(history_rows)
        current = fills.get(textbox_values[0][0], '')
    return SequenceMatcher(None, current, target).ratio()


def _checkbox_progress(goal: str, obs_text: str) -> float | None:
    target = _checkbox_target_set(goal)
    if target is None:
        return None
    states = _extract_checkbox_states(obs_text)
    checked = {label for label, is_checked in states.items() if is_checked}
    if not target:
        return 1.0 if not checked else max(-1.0, -0.25 * len(checked))
    correct = len(checked & target)
    incorrect = len(checked - target)
    score = (correct - incorrect) / max(1, len(target))
    return max(-1.0, min(1.0, score))


def _task_progress(task_name: str, goal: str, obs_text: str, history_rows: list[dict[str, Any]]) -> float | None:
    if task_name == 'enter-text-2':
        return _enter_text_progress(goal, obs_text, history_rows)
    if task_name == 'enter-password':
        return _enter_password_progress(goal, obs_text, history_rows)
    if task_name == 'find-word':
        return _find_word_progress(goal, obs_text, history_rows)
    if task_name.startswith('click-checkboxes'):
        return _checkbox_progress(goal, obs_text)
    return None


def _submit_penalty(action: str, obs_text: str, progress: float | None, config: MultiTurnGRPOConfig) -> float:
    if progress is None or progress >= 0.999:
        return 0.0
    bid = _parse_click_bid(action)
    if bid is None:
        return 0.0
    label = (_extract_button_labels(obs_text).get(bid) or '').strip().lower()
    return config.premature_submit_penalty if label == 'submit' else 0.0



def _looks_conditional(model_ref: str | None) -> bool:
    if not model_ref:
        return False
    try:
        cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
        archs = cfg.architectures or []
        model_type = (getattr(cfg, "model_type", "") or "").lower()
        if any("Qwen3_5ForConditionalGeneration" in a for a in archs):
            return True
        if "qwen3_5" in model_type:
            return True
    except Exception:
        pass
    s = model_ref.lower()
    return "qwen3.5" in s or "qwen3_5" in s


def _should_use_conditional_loader(model_name: str, adapter_dir: str | None) -> bool:
    return _looks_conditional(model_name) or _looks_conditional(adapter_dir)


def _apply_qwen35_rope_delta_guard() -> None:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception:
        return

    qwen_model_cls = getattr(modeling_qwen3_5, "Qwen3_5Model", None)
    if qwen_model_cls is None or not hasattr(qwen_model_cls, "compute_3d_position_ids"):
        return
    if getattr(qwen_model_cls, "_rope_delta_guard_patched", False):
        return

    original_compute = qwen_model_cls.compute_3d_position_ids

    def _patched_compute_3d_position_ids(
        self,
        input_ids,
        inputs_embeds,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
        mm_token_type_ids=None,
    ):
        rope_deltas = getattr(self, "rope_deltas", None)
        if rope_deltas is not None and inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            delta_rows = rope_deltas.shape[0] if getattr(rope_deltas, "ndim", 0) > 0 else 0
            if delta_rows == 0:
                self.rope_deltas = None
            elif delta_rows != batch_size:
                if delta_rows > batch_size:
                    self.rope_deltas = rope_deltas[:batch_size]
                else:
                    repeats = math.ceil(batch_size / delta_rows)
                    self.rope_deltas = rope_deltas.repeat_interleave(repeats, dim=0)[:batch_size]
        return original_compute(
            self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    qwen_model_cls.compute_3d_position_ids = _patched_compute_3d_position_ids
    qwen_model_cls._rope_delta_guard_patched = True


def load_model_and_tokenizer(config: MultiTurnGRPOConfig):
    model_ref = config.adapter_dir or config.model_name
    use_conditional = _should_use_conditional_loader(config.model_name, config.adapter_dir)

    if use_conditional:
        _apply_qwen35_rope_delta_guard()
        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"
        base_model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    base_model.warnings_issued = {}
    if config.adapter_dir:
        model = PeftModel.from_pretrained(base_model, config.adapter_dir, is_trainable=True)
        model.warnings_issued = {}
    else:
        model = base_model
    return model, tokenizer


def generate_completion_text(*, model, tokenizer, system_prompt: str, task_name: str, current_obs: dict[str, Any], history_rows: list[dict[str, Any]], max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    user_message = render_user_message(task_name, current_obs, history_rows)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    if was_training:
        model.train()
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def build_multiturn_reward(config: MultiTurnGRPOConfig, runtime: dict[str, Any]):
    client = BrowserGymWSClient(config.browsergym_base_url)
    system_prompt = config.system_prompt or _action_only_system_prompt(FALLBACK_SYSTEM_PROMPT)

    def reward(prompts, completions, task_name, seed, **kwargs) -> list[float]:
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]
        rewards: list[float] = []
        for text, task, row_seed in zip(completion_texts(completions), task_name, seed):
            result = asyncio.run(
                client.rollout(
                    task_name=str(task),
                    seed=int(row_seed),
                    first_completion_text=text,
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    config=config,
                )
            )
            rewards.append(float(result["reward"]))
        return rewards

    reward.__name__ = "browsergym_multiturn_reward"
    return reward


def write_preview(rows: list[dict[str, Any]], output_dir: Path) -> None:
    preview_path = output_dir / "prompt_dataset_preview.jsonl"
    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dry_run_preview(config: MultiTurnGRPOConfig, rows: list[dict[str, Any]]) -> None:
    print("Prepared prompt rows:", len(rows))
    print("First prompt user message:\n", rows[0]["prompt"][1]["content"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Local multi-turn BrowserGym GRPO training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run-reward", action="store_true")
    args = parser.parse_args()

    config = MultiTurnGRPOConfig.from_yaml(Path(args.config))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grpo_run_config.json").write_text(json.dumps(asdict(config), indent=2))

    rows = build_prompt_dataset(config)
    write_preview(rows, output_dir)
    dataset = Dataset.from_list(rows)
    if args.dry_run_reward:
        dry_run_preview(config, rows)
        return 0

    model, tokenizer = load_model_and_tokenizer(config)
    runtime = {"model": model, "tokenizer": tokenizer}
    print(json.dumps({
        "model_name": config.model_name,
        "adapter_dir": config.adapter_dir,
        "loader": "causal_lm",
        "rollout_max_steps": config.rollout_max_steps,
        "task_count": len({task for task, _seed in _build_task_seed_pairs(config)}),
    }, indent=2))

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        bf16=(config.bf16 and torch.cuda.is_available()),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=config.report_to,
        use_vllm=config.use_vllm,
    )

    reward_func = build_multiturn_reward(config, runtime)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=None,
    )
    runtime["model"] = trainer.model
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
