"""Policy JSON contract between the pipeline and canonical eval.

A policy JSON is the single artefact handed off to
experiments/standardize-results.py (after that script is refactored
to consume JSONs instead of its hardcoded DATASETS dict).
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

from .config import Policy, RunConfig, RunResult


def policy_to_dict(policy: Policy) -> dict[str, Any]:
    return {
        "voting": {
            "mode":    policy.voting_mode,
            "tau":     float(policy.tau),
            "weights": [float(w) for w in policy.weights],
        },
        "rules": [r.to_dict() for r in policy.rules],
    }


def policy_from_dict(d: dict) -> Policy:
    from .config import Rule
    rules = [
        Rule(
            feature=r["feature"], op=r["op"], value=r["value"],
            phenomenon_tag=r.get("phenomenon_tag", "VOLUME"),
            rationale=r.get("rationale", ""),
        )
        for r in d["rules"]
    ]
    v = d.get("voting", {})
    return Policy(
        rules=rules,
        voting_mode=v.get("mode", "majority"),
        tau=float(v.get("tau", 0.5)),
        weights=[float(w) for w in v.get("weights", [])],
    )


def build_run_id(cfg: RunConfig, when: datetime.datetime | None = None) -> str:
    when = when or datetime.datetime.now()
    from .models import short_model_name
    return (
        f"{cfg.dataset_key}-{cfg.provider}-{short_model_name(cfg.provider, cfg.model_id)}"
        f"-seed{cfg.seed}-{when:%Y%m%d-%H%M%S}"
    )


def save_policy(
    cfg: RunConfig, result: RunResult,
    out_dir: str = None,
) -> str:
    """Write `<out_dir>/<run_id>.json` and return the path.

    Default out_dir is `<repo>/experiments/policies/<dataset_key>/`.
    """
    if out_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(repo, "experiments", "policies", cfg.dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    run_id = build_run_id(cfg)
    path = os.path.join(out_dir, f"{run_id}.json")
    doc = {
        "run_id":       run_id,
        "config":       cfg.as_dict(),
        "best_round":   result.best_round_index,
        "best_metrics": result.best_metrics,
        "tokens": {
            "input":  result.total_tokens_in,
            "output": result.total_tokens_out,
        },
        "policy":       policy_to_dict(result.best_policy),
        "round_history": [
            {
                "round_index":  r.round_index,
                "metrics":      r.policy_metrics,
                "accepted":     [a.to_dict() for a in r.accepted],
                "tokens_in":    r.tokens_in,
                "tokens_out":   r.tokens_out,
                "rejected":     r.rejected_reasons,
            }
            for r in result.history
        ],
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


def load_policy(path: str) -> tuple[dict, Policy]:
    """Return (metadata_dict, Policy)."""
    with open(path) as f:
        doc = json.load(f)
    return doc, policy_from_dict(doc["policy"])
