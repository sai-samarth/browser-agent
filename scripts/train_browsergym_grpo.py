#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    task_name: str
    output_dir: str
    dataset_size: int = 8
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
    for idx in range(config.dataset_size):
        seed = config.seed_offset + idx
        obs = asyncio.run(client.reset(task_name=config.task_name, seed=seed))
        rows.append(_build_prompt_row(system_prompt, config.task_name, seed, obs))
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

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
        peft_config=build_peft_config(config),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
