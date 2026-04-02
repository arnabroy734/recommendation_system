"""
deploy/promote.py
-----------------
Promotes a trained MF or SASRec MLflow run to production
by writing its run_id and metadata to prod_config.json.

Usage:
    python deploy/promote.py --model mf --run_id 3f2a1bc
    python deploy/promote.py --model sasrec --run_id 9e1d4fa
    python deploy/promote.py --model mf --run_id 3f2a1bc --model sasrec --run_id 9e1d4fa
    python deploy/promote.py --show
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mlflow

PROD_CONFIG_PATH = Path("prod_config.json")
VALID_MODELS     = {"mf", "sasrec"}


def load_prod_config() -> dict:
    if PROD_CONFIG_PATH.exists():
        with open(PROD_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_prod_config(config: dict):
    with open(PROD_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"prod_config.json updated → {PROD_CONFIG_PATH.resolve()}")


def fetch_run_metadata(run_id: str) -> dict:
    try:
        client   = mlflow.tracking.MlflowClient()
        run      = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", "unknown")
        params   = run.data.params
        train_end = params.get("end", None)
        if train_end is None:
            print(f"  WARNING: 'end' param not found in run {run_id}.")
        return {
            "run_id":      run_id,
            "run_name":    run_name,
            "train_end":   train_end,
            "params":      params,
            "promoted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        print(f"ERROR: Could not fetch run {run_id} from MLflow: {e}")
        sys.exit(1)


def promote(model: str, run_id: str, config: dict) -> dict:
    print(f"\nPromoting {model.upper()} run: {run_id}")
    metadata = fetch_run_metadata(run_id)
    config[model] = metadata
    print(f"  run_name  : {metadata['run_name']}")
    print(f"  train_end : {metadata['train_end']}")
    print(f"  promoted  : {metadata['promoted_at']}")
    return config


def show_config(config: dict):
    if not config:
        print("prod_config.json is empty or does not exist.")
        return
    print("\n=== Current Production Config ===")
    for model, meta in config.items():
        print(f"\n  [{model.upper()}]")
        print(f"    run_id     : {meta.get('run_id')}")
        print(f"    run_name   : {meta.get('run_name')}")
        print(f"    train_end  : {meta.get('train_end')}")
        print(f"    promoted   : {meta.get('promoted_at')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  type=str, choices=list(VALID_MODELS),
                   action="append", dest="models",  metavar="MODEL")
    p.add_argument("--run_id", type=str,
                   action="append", dest="run_ids", metavar="RUN_ID")
    p.add_argument("--show",   action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_prod_config()

    if args.show:
        show_config(config)
        return

    if not args.models or not args.run_ids:
        print("ERROR: Provide at least one --model and --run_id pair.")
        sys.exit(1)

    if len(args.models) != len(args.run_ids):
        print("ERROR: Number of --model and --run_id arguments must match.")
        sys.exit(1)

    for model, run_id in zip(args.models, args.run_ids):
        config = promote(model, run_id, config)

    save_prod_config(config)
    print("\nDone. Run `python deploy/promote.py --show` to verify.")


if __name__ == "__main__":
    main()