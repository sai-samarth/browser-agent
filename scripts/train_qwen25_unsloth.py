#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='data/exports/phase1_sft_v2/action_only/hf_dataset')
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--output-dir', default='outputs/qwen25-1.5b-browser-action-lora-unsloth')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--num-train-epochs', type=float, default=1.0)
    parser.add_argument('--per-device-train-batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--logging-steps', type=int, default=10)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    ds = load_from_disk(args.dataset_dir)

    def render(example):
        text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
        return {'text': text}

    rendered = ds.map(render, remove_columns=ds['train'].column_names)

    def tokenize(example):
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length',
        )
        tokens['labels'] = list(tokens['input_ids'])
        return tokens

    tokenized = rendered.map(tokenize, batched=False, remove_columns=['text'])
    tokenized.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    def collate(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch]),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['val'],
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        'framework': 'unsloth',
        'model_name': args.model_name,
        'dataset_dir': args.dataset_dir,
        'output_dir': args.output_dir,
        'train_rows': len(ds['train']),
        'val_rows': len(ds['val']),
        'max_length': args.max_length,
        'epochs': args.num_train_epochs,
        'load_in_4bit': True,
        'quantization': 'bitsandbytes 4bit NF4 + bf16 compute',
    }
    Path(args.output-dir if False else args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
