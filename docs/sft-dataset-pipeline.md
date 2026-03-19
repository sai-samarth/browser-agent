# SFT dataset export pipeline

This project exports browser-agent rollouts into step-level chat SFT datasets.

## Export variants

### 1. action_only
Primary training format.

Each row contains:
- `messages[system]` — browser-agent action instruction prompt
- `messages[user]` — task goal, URL, compact history, observation diagnostics, and current observation text
- `messages[assistant]` — the next BrowserGym action only
- `metadata` — run/task/episode/step/teacher info

### 2. reasoning_action
Experimental training format.

Same structure, but the assistant target is:

```text
<think>
...
</think>
action(...)
```

Rows without teacher reasoning are skipped from this export.

## Filtering defaults

The default export is intentionally conservative:
- successful episodes only
- max action errors: 0
- max repeated loops: 0
- max sparse observations: 2
- max root-only observations: 0
- max fallback count: 0

This approximates a Tier A dataset.

## Splitting

Default split strategy:
- split by `run_id`
- `val_ratio = 0.1`

This reduces near-duplicate leakage better than splitting step-by-step.

## Outputs

Example output directory:

```text
data/exports/phase1_sft_v1/
  manifest.json
  push_to_hub.py
  action_only/
    train.jsonl
    val.jsonl
    hf_dataset/
  reasoning_action/
    train.jsonl
    val.jsonl
    hf_dataset/
```

## Build command

```bash
.venv/bin/python scripts/export_sft_dataset.py   --rollout-glob 'miniwob_phase1_prod_local_qwen35_batch*_r*_ *'
```

Use the real glob without the space before `*`.

## Hugging Face dataset usage

The exporter attempts to save Hugging Face `datasets` objects to disk if the `datasets` package is installed.

To push later:

```bash
huggingface-cli login
.venv/bin/python data/exports/phase1_sft_v1/push_to_hub.py   --dataset-dir data/exports/phase1_sft_v1/action_only/hf_dataset   --repo-id <username>/<repo-name>
```

Do the same for the reasoning-action variant if you want both on the hub.
