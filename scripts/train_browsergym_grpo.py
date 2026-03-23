#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
import transformers.utils.hub as transformers_hub
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

if not hasattr(transformers_hub, 'TRANSFORMERS_CACHE'):
    transformers_hub.TRANSFORMERS_CACHE = getattr(transformers_hub, 'default_cache_path', None) or ''

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
    _render_user_message,
)


@dataclass
class GRPOBrowserConfig:
    model_name: str
    browsergym_base_url: str
    task_name: str | None = None
    task_names: list[str] | None = None
    task_names_file: str | None = None
    output_dir: str = ''
    adapter_dir: str | None = None
    dataset_size: int = 8
    samples_per_task: int | None = None
    seed_offset: int = 910000
    system_prompt: str | None = None
    max_prompt_length: int = 1536
    max_completion_length: int = 64
    learning_rate: float = 5e-6
    warmup_steps: int = 0
    logging_steps: int = 1
    save_steps: int = 10
    max_steps: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 2
    generation_batch_size: int = 2
    bf16: bool = True
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    report_to: str = "none"
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.15
    reward_parse_valid: float = 0.1
    reward_parse_invalid: float = -0.2
    reward_done_bonus: float = 1.0
    reward_action_error_penalty: float = 0.2
    reward_env_scale: float = 1.0

    @classmethod
    def from_yaml(cls, path: Path) -> "GRPOBrowserConfig":
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

    async def step_once(self, *, task_name: str, seed: int | None, action_str: str) -> dict[str, Any]:
        async with ws_connect(self.ws_url, open_timeout=30, max_size=100 * 1024 * 1024) as ws:
            reset_payload = {"type": "reset", "data": {"task_name": task_name}}
            if seed is not None:
                reset_payload["data"]["seed"] = seed
            reset_msg = await _ws_send_recv(ws, reset_payload, timeout_s=60)
            if reset_msg.get("type") != "observation":
                raise RuntimeError(f"Unexpected reset response: {reset_msg}")
            step_msg = await _ws_send_recv(
                ws,
                {
                    "type": "step",
                    "data": {
                        "action_str": action_str,
                        "metadata": {"task_name": task_name, "seed": seed},
                    },
                },
                timeout_s=60,
            )
            if step_msg.get("type") != "observation":
                raise RuntimeError(f"Unexpected step response: {step_msg}")
            return _normalize_ws_observation(step_msg)


def _load_task_names(config: GRPOBrowserConfig) -> list[str]:
    names: list[str] = []
    if config.task_name:
        names.append(config.task_name)
    if config.task_names:
        names.extend(str(x) for x in config.task_names)
    if config.task_names_file:
        path = Path(config.task_names_file)
        for raw in path.read_text().splitlines():
            task = raw.strip()
            if not task or task.startswith('#'):
                continue
            names.append(task)
    names = [n for i,n in enumerate(names) if n and n not in names[:i]]
    if not names:
        raise ValueError('Provide task_name, task_names, or task_names_file')
    return names


def _build_task_seed_pairs(config: GRPOBrowserConfig) -> list[tuple[str, int]]:
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
        task = tasks[idx % len(tasks)]
        pairs.append((task, seed))
        seed += 1
    return pairs


def _build_prompt_row(system_prompt: str, task_name: str, seed: int, obs: dict[str, Any]) -> dict[str, Any]:
    pre_observation = {
        "goal": obs.get("goal", ""),
        "url": obs.get("url", ""),
        "last_action_error": bool(obs.get("last_action_error", False)),
        "text": obs.get("text", ""),
        "diagnostics": _observation_diagnostics(obs),
    }
    synthetic_step = {
        "task_name": task_name,
        "seed": seed,
        "step_idx": 0,
        "pre_observation": pre_observation,
    }
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _render_user_message(synthetic_step, [])},
        ],
        "task_name": task_name,
        "seed": seed,
        "goal": str(pre_observation["goal"]),
        "initial_url": str(pre_observation["url"]),
    }


def build_prompt_dataset(config: GRPOBrowserConfig) -> list[dict[str, Any]]:
    client = BrowserGymWSClient(config.browsergym_base_url)
    system_prompt = config.system_prompt or _action_only_system_prompt(FALLBACK_SYSTEM_PROMPT)
    rows: list[dict[str, Any]] = []
    for task_name, seed in _build_task_seed_pairs(config):
        obs = asyncio.run(client.reset(task_name=task_name, seed=seed))
        rows.append(_build_prompt_row(system_prompt, task_name, seed, obs))
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



def _looks_conditional(model_ref: str | None) -> bool:
    if not model_ref:
        return False
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


def _load_model_and_processing(config: GRPOBrowserConfig):
    use_conditional = _should_use_conditional_loader(config.model_name, config.adapter_dir)
    model_ref = config.adapter_dir or config.model_name

    if use_conditional:
        _apply_qwen35_rope_delta_guard()
        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
        tokenizer = getattr(processor, 'tokenizer', processor)
        if getattr(tokenizer, 'pad_token', None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = 'left'
        base_model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='auto',
            trust_remote_code=True,
        )
        base_model.warnings_issued = {}
        if config.adapter_dir:
            model = PeftModel.from_pretrained(base_model, config.adapter_dir, is_trainable=True)
            model.warnings_issued = {}
            peft_config = None
        else:
            model = base_model
            peft_config = build_peft_config(config)
        return model, tokenizer, peft_config, use_conditional

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = 'left'
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        trust_remote_code=True,
    )
    base_model.warnings_issued = {}
    if config.adapter_dir:
        model = PeftModel.from_pretrained(base_model, config.adapter_dir, is_trainable=True)
        model.warnings_issued = {}
        peft_config = None
    else:
        model = base_model
        peft_config = build_peft_config(config)
    return model, tokenizer, peft_config, use_conditional

def build_parse_reward(config: GRPOBrowserConfig):
    def reward(completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        for text in completion_texts(completions):
            rewards.append(config.reward_parse_valid if parse_action(text) else config.reward_parse_invalid)
        return rewards

    reward.__name__ = "parseability_reward"
    return reward


def build_env_reward(config: GRPOBrowserConfig):
    client = BrowserGymWSClient(config.browsergym_base_url)

    def reward(prompts, completions, task_name, seed, **kwargs) -> list[float]:
        rewards: list[float] = []
        for text, task, row_seed in zip(completion_texts(completions), task_name, seed):
            action = parse_action(text)
            if action is None:
                rewards.append(0.0)
                continue
            obs = asyncio.run(client.step_once(task_name=str(task), seed=int(row_seed), action_str=action))
            score = float(obs.get("reward") or 0.0) * config.reward_env_scale
            if bool(obs.get("done", False)) and float(obs.get("reward") or 0.0) > 0:
                score += config.reward_done_bonus
            if bool(obs.get("last_action_error", False)):
                score -= config.reward_action_error_penalty
            rewards.append(score)
        return rewards

    reward.__name__ = "browsergym_step_reward"
    return reward


def build_peft_config(config: GRPOBrowserConfig) -> LoraConfig | None:
    if not config.use_peft:
        return None
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def write_prompt_preview(rows: list[dict[str, Any]], output_dir: Path) -> None:
    preview_path = output_dir / "prompt_dataset_preview.jsonl"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def smoke_reward_preview(config: GRPOBrowserConfig, rows: list[dict[str, Any]]) -> None:
    parse_reward = build_parse_reward(config)
    env_reward = build_env_reward(config)
    fake_good = [[[{"content": "click('13')"}][0]]]
    print("Reward function preview on a synthetic completion is not meaningful without task-specific bids.")
    print("Prepared prompt rows:", len(rows))
    print("First prompt user message:\n", rows[0]["prompt"][1]["content"])
    print("Parse reward on noop():", parse_reward([[{"content": "noop()"}]])[0])
    first = rows[0]
    print(
        "Env reward on noop() for first row:",
        env_reward(
            prompts=[first["prompt"]],
            completions=[[{"content": "noop()"}]],
            task_name=[first["task_name"]],
            seed=[first["seed"]],
        )[0],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Local BrowserGym GRPO training over one-step prompts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run-reward", action="store_true", help="Prepare prompt dataset and exercise reward functions without training.")
    args = parser.parse_args()

    config = GRPOBrowserConfig.from_yaml(Path(args.config))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grpo_run_config.json").write_text(json.dumps(asdict(config), indent=2))

    rows = build_prompt_dataset(config)
    write_prompt_preview(rows, output_dir)
    dataset = Dataset.from_list(rows)

    if args.dry_run_reward:
        smoke_reward_preview(config, rows)
        return 0

    model, tokenizer, peft_config, use_conditional = _load_model_and_processing(config)
    print(json.dumps({
        "model_name": config.model_name,
        "adapter_dir": config.adapter_dir,
        "loader": "conditional_generation" if use_conditional else "causal_lm",
        "use_peft_init": peft_config is not None,
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
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[build_parse_reward(config), build_env_reward(config)],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
