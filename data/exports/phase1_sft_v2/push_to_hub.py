from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', required=True)
parser.add_argument('--repo-id', required=True)
parser.add_argument('--private', action='store_true')
args = parser.parse_args()

ds = load_from_disk(args.dataset_dir)
ds.push_to_hub(args.repo_id, private=args.private)
print(f'pushed {args.dataset_dir} -> {args.repo_id}')
