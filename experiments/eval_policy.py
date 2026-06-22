"""Apply a saved policy JSON to a dataset's held-out test split.

This is the canonical eval for the new pipeline. It uses the same
DatasetSpec balanced loader as the pipeline so train/test splits match.

    python experiments/eval_policy.py --policy experiments/policies/cic/cic-anthropic-haiku-seed42-20260525-203602.json
    python experiments/eval_policy.py --latest cic       # newest policy for CIC
    python experiments/eval_policy.py --all              # all latest policies per (dataset, provider)
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.eval import classification_metrics
from lib.policy_pipeline.io import load_policy
from lib.policy_pipeline.voting import predict


def _latest_policy(dataset_key: str, provider: str | None = None) -> Path | None:
    pat = f"experiments/policies/{dataset_key}/*.json"
    files = sorted(Path(REPO).glob(pat))
    if provider:
        files = [f for f in files if provider in f.name]
    return files[-1] if files else None


def eval_one(policy_path: Path, seed: int | None = None) -> dict:
    doc, policy = load_policy(str(policy_path))
    cfg = doc["config"]
    use_seed = seed if seed is not None else cfg["seed"]

    split = load_dataset(cfg["dataset_key"], seed=use_seed)
    X_te = pd.concat([split.normal_test, split.attack_test], axis=0, ignore_index=True)
    y_te = np.concatenate([
        np.full(len(split.normal_test), "normal"),
        np.full(len(split.attack_test), "attack"),
    ])
    pred = predict(policy, X_te)
    m = classification_metrics(y_te, pred)
    tags = sorted({r.phenomenon_tag for r in policy.rules})
    return {
        "run_id": doc["run_id"],
        "dataset": split.spec.display_name,
        "provider": cfg["provider"],
        "model_id": cfg["model_id"],
        "seed": cfg["seed"],
        "k": cfg["k"],
        "voting_mode": policy.voting_mode,
        "n_rules": len(policy.rules),
        "n_distinct_tags": len(tags),
        "tags": tags,
        **m,
        "rules": [r.to_dict() for r in policy.rules],
    }


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_report(r: dict) -> None:
    print(f"\n=== {r['dataset']} | {r['provider']}:{r['model_id']} | seed={r['seed']} ===")
    print(f"  run: {r['run_id']}")
    print(f"  policy: {r['n_rules']} rules ({r['n_distinct_tags']} distinct tags: {','.join(r['tags'])})")
    print(f"  voting: {r['voting_mode']}")
    for rule in r["rules"]:
        print(f"    [{rule['phenomenon_tag']:9}] {rule['feature']} {rule['op']} {rule['value']}")
    print(f"  macro_f1={fmt(r['macro_f1'])}  attack_f1={fmt(r['attack_f1'])}")
    print(f"  attack_precision={fmt(r['attack_precision'])}  attack_recall={fmt(r['attack_recall'])}")
    print(f"  normal_f1={fmt(r['normal_f1'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="Path to policy JSON")
    ap.add_argument("--latest", help="Dataset key; evaluate newest policy")
    ap.add_argument("--provider", help="Filter by provider when using --latest or --all")
    ap.add_argument("--all", action="store_true", help="Evaluate latest per (dataset, provider)")
    ap.add_argument("--seed", type=int, default=None, help="Override seed for data split")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.policy:
        targets.append(Path(args.policy))
    elif args.latest:
        p = _latest_policy(args.latest, args.provider)
        if p is None:
            print(f"No policies found for {args.latest}")
            return
        targets.append(p)
    elif args.all:
        for dataset in ["cic", "wustl", "ton", "bot", "unsw"]:
            for prov in ["anthropic", "google"]:
                p = _latest_policy(dataset, prov)
                if p:
                    targets.append(p)
    else:
        ap.error("Need --policy PATH, --latest DATASET, or --all")

    results = [eval_one(p, args.seed) for p in targets]

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for r in results:
            print_report(r)


if __name__ == "__main__":
    main()
