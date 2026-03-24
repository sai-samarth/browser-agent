#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--adapter-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--num-train-epochs', type=float, default=2.0)
    ap.add_argument('--per-device-train-batch-size', type=int, default=4)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = 'left'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=True)
    model.print_trainable_parameters()

    ds = load_from_disk(args.dataset_dir)

    def render(example):
        text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
        return {'text': text}

    rendered = ds.map(render, remove_columns=ds['train'].column_names)

    def tokenize_fn(example):
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length',
        )
        tokens['labels'] = list(tokens['input_ids'])
        return tokens

    tokenized = rendered.map(tokenize_fn, batched=False, remove_columns=['text'])
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

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
        eval_strategy='no',
        save_strategy='no',
        report_to='none',
        gradient_checkpointing=True,
        seed=args.seed,
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
    if hasattr(processor, 'save_pretrained'):
        processor.save_pretrained(args.output_dir)

    summary = {
        'base_model': args.base_model,
        'adapter_init': args.adapter_dir,
        'dataset_dir': args.dataset-dir if False else args.dataset_dir,
        'output_dir': args.output_dir,
        'train_rows': len(ds['train']),
        'val_rows': len(ds['val']),
        'max_length': args.max_length,
        'epochs': args.num_train_epochs,
        'learning_rate': args.learning_rate,
        'continuation': True,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
