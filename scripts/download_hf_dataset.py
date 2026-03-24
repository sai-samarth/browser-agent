#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a dataset from the Hugging Face Hub and save it locally with save_to_disk().")
    ap.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. saital/browser-agent-phase1-sft-action-only")
    ap.add_argument("--output-dir", required=True, help="Local save_to_disk target directory")
    ap.add_argument("--config-name", default=None, help="Optional dataset config name")
    args = ap.parse_args()

    ds = load_dataset(args.repo_id, args.config_name)
    out = Path(args.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"saved {args.repo_id} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
