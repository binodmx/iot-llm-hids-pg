"""Unified driver: run the policy pipeline for one (dataset, provider) combo.

    python experiments/run_pipeline.py --dataset cic --provider anthropic
    python experiments/run_pipeline.py --dataset bot --provider google --seed 123

Output: writes a policy JSON to experiments/policies/{dataset}/{run_id}.json
and prints the held-out-test metrics to stdout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.policy_pipeline import RunConfig, run_pipeline
from lib.policy_pipeline.datasets import load_dataset, describe_split
from lib.policy_pipeline.eval import classification_metrics
from lib.policy_pipeline.voting import predict
from lib.policy_pipeline.io import save_policy
from lib.policy_pipeline.models import KNOWN_MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["cic", "wustl", "ton", "bot", "unsw"])
    ap.add_argument("--provider", required=True, choices=["anthropic", "google"])
    ap.add_argument("--model-id", default=None,
                    help="Override default model. Defaults: haiku-4-5 for anthropic, gemini-2.5-flash for google.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--voting", default="weighted",
                    choices=["weighted", "majority", "or_gate"])
    ap.add_argument("--selection-metric", default="macro_f1",
                    choices=["macro_f1", "attack_f1", "attack_precision_at_recall_0.9"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.model_id is None:
        args.model_id = KNOWN_MODELS["haiku"][1] if args.provider == "anthropic" else KNOWN_MODELS["gemini"][1]

    cfg = RunConfig(
        dataset_key=args.dataset, provider=args.provider, model_id=args.model_id,
        seed=args.seed, k=args.k, max_rounds=args.max_rounds,
        early_stop_patience=args.patience,
        voting_mode=args.voting,
        selection_metric=args.selection_metric,
    )

    split = load_dataset(args.dataset, seed=args.seed)
    print(f"\n=== {cfg.dataset_key} / {cfg.provider}:{args.model_id.split('-')[-1] if '-' in args.model_id else args.model_id} ===")
    print(f"Split: {describe_split(split)}")

    t0 = time.time()
    result = run_pipeline(
        cfg, split.normal_train, split.attack_train,
        split.attack_class_labels_train,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    # Held-out test
    X_te = pd.concat([split.normal_test, split.attack_test], axis=0, ignore_index=True)
    y_te = np.concatenate([
        np.full(len(split.normal_test), "normal"),
        np.full(len(split.attack_test), "attack"),
    ])
    pred_te = predict(result.best_policy, X_te)
    m = classification_metrics(y_te, pred_te)
    n_tags = len({r.phenomenon_tag for r in result.best_policy.rules})

    print(f"\n--- {split.spec.display_name} TEST (best round {result.best_round_index + 1}) ---")
    for r, w in zip(result.best_policy.rules, result.best_policy.weights):
        print(f"  [{r.phenomenon_tag:9}] (w={w:.3f}) {r.feature} {r.op} {r.value}")
    print(f"  voting={result.best_policy.voting_mode}, tau={result.best_policy.tau:.4f}")
    print(f"  macro_f1={m['macro_f1']:.4f}  attack_f1={m['attack_f1']:.4f}")
    print(f"  attack_precision={m['attack_precision']:.4f}  attack_recall={m['attack_recall']:.4f}")
    print(f"  normal_f1={m['normal_f1']:.4f}")
    print(f"  distinct_phenomenon_tags={n_tags}")
    print(f"  tokens={result.total_tokens_in}/{result.total_tokens_out}  wall={elapsed:.1f}s")

    path = save_policy(cfg, result)
    print(f"Saved policy: {path}")

    return m, n_tags, path


if __name__ == "__main__":
    main()
