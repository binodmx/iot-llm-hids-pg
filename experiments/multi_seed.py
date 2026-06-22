"""Multi-seed campaign with a fixed config per (dataset, model).

The WINNERS dict below is a hardcoded snapshot of the config that won the
hyperparameter sweep (sweep_policy.py / build_sweep_summary.py, archived —
not part of the published, fixed-hyperparameter pipeline). Runs each
(dataset, provider) at seeds {42, 123, 456} and writes results to
experiments/sweeps/multi_seed-<ts>.csv.

    python experiments/multi_seed.py --dataset all --provider both \
        --seeds 42,123,456
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.policy_pipeline import RunConfig, run_pipeline
from lib.policy_pipeline.datasets import load_dataset
from lib.policy_pipeline.eval import classification_metrics
from lib.policy_pipeline.io import save_policy
from lib.policy_pipeline.voting import predict


# Winners from iot-llm-hids-pg-archive/experiments/sweeps/winners-macro_f1.csv
# (picked by sweep + post-fix manual entry for ton/google after the
# diversity.py quantile fix).
WINNERS: dict[tuple[str, str], dict] = {
    ("bot",   "anthropic"): {"voting_mode": "weighted", "selection_metric": "macro_f1",  "composite_alpha": 0.6, "k": 5},
    ("bot",   "google"):    {"voting_mode": "weighted", "selection_metric": "macro_f1",  "composite_alpha": 0.6, "k": 5},
    ("cic",   "anthropic"): {"voting_mode": "weighted", "selection_metric": "macro_f1",  "composite_alpha": 0.6, "k": 5},
    ("cic",   "google"):    {"voting_mode": "weighted", "selection_metric": "attack_f1", "composite_alpha": 0.6, "k": 5},
    ("ton",   "anthropic"): {"voting_mode": "weighted", "selection_metric": "attack_f1", "composite_alpha": 0.6, "k": 5},
    ("ton",   "google"):    {"voting_mode": "weighted", "selection_metric": "macro_f1",  "composite_alpha": 0.6, "k": 5},
    ("unsw",  "anthropic"): {"voting_mode": "weighted", "selection_metric": "attack_f1", "composite_alpha": 0.6, "k": 5},
    ("unsw",  "google"):    {"voting_mode": "weighted", "selection_metric": "macro_f1",  "composite_alpha": 0.6, "k": 5},
    ("wustl", "anthropic"): {"voting_mode": "weighted", "selection_metric": "attack_f1", "composite_alpha": 0.6, "k": 5},
    ("wustl", "google"):    {"voting_mode": "weighted", "selection_metric": "attack_f1", "composite_alpha": 0.6, "k": 5},
}


def _config(dataset: str, provider: str, seed: int) -> RunConfig:
    base = dict(
        provider=provider,
        model_id=("claude-haiku-4-5-20251001" if provider == "anthropic"
                  else "gemini-2.5-flash"),
        seed=seed, k=5, max_rounds=4, early_stop_patience=2,
        voting_mode="weighted", selection_metric="macro_f1",
        temperature=0.1,
    )
    base.update(WINNERS.get((dataset, provider), {}))
    base["dataset_key"] = dataset
    return RunConfig(**base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Comma-separated keys or 'all'")
    ap.add_argument("--provider", default="both",
                    help="anthropic|google|both")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--out-dir", default="experiments/sweeps")
    args = ap.parse_args()

    if args.dataset == "all":
        datasets = ["cic", "wustl", "ton", "bot", "unsw"]
    else:
        datasets = args.dataset.split(",")
    providers = (["anthropic", "google"] if args.provider == "both"
                 else [args.provider])
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"multi_seed-{ts}.csv"

    rows = []
    for ds in datasets:
        for prov in providers:
            for sd in seeds:
                cfg = _config(ds, prov, sd)
                print(f"\n--- {ds} | {prov} | seed={sd} ---")
                split = load_dataset(ds, seed=sd)
                t0 = time.time()
                try:
                    result = run_pipeline(
                        cfg, split.normal_train, split.attack_train,
                        split.attack_class_labels_train, verbose=False,
                    )
                    X_te = pd.concat([split.normal_test, split.attack_test], axis=0, ignore_index=True)
                    y_te = np.concatenate([
                        np.full(len(split.normal_test), "normal"),
                        np.full(len(split.attack_test), "attack"),
                    ])
                    m = classification_metrics(y_te, predict(result.best_policy, X_te))
                    n_tags = len({r.phenomenon_tag for r in result.best_policy.rules})
                    path = save_policy(cfg, result)
                    wall = time.time() - t0
                    row = {
                        "dataset": ds, "provider": prov, "seed": sd,
                        "best_round": result.best_round_index + 1,
                        "n_distinct_tags": n_tags,
                        "tokens_in": result.total_tokens_in,
                        "tokens_out": result.total_tokens_out,
                        "wall_seconds": round(wall, 1),
                        "policy_path": path,
                        **m,
                    }
                    rows.append(row)
                    print(f"   macro_f1={m['macro_f1']:.4f} attack_f1={m['attack_f1']:.4f} "
                          f"tags={n_tags} {wall:.0f}s")
                except Exception as e:
                    print(f"   ERROR: {e}")
                    rows.append({"dataset": ds, "provider": prov, "seed": sd, "error": str(e)})

    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
