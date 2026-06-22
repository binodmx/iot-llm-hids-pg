"""Vote-combination over rule outputs.

The voting layer is where rule outputs are turned into per-row predictions.
Three modes:

- "majority": each rule votes attack/normal, mode wins (notebook 4 default).
- "or_gate":  any rule firing -> attack. Higher recall, lower precision.
              Useful for zero-day / anomaly framing.
- "weighted": score = sum(w_i * fired_i); attack iff score >= tau.
              w_i is per-rule precision on a validation slice; tau is
              auto-calibrated on the same slice to maximise the chosen
              selection_metric. Stored deterministically so canonical
              eval can replay.
"""

from __future__ import annotations

import operator
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Policy, Rule

_OPS = {
    "<": operator.lt, ">": operator.gt,
    "<=": operator.le, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
}


def _to_scalar(v):
    if isinstance(v, str):
        s = v.strip().strip("'\"")
        try:
            return float(s)
        except ValueError:
            return s
    return v


def rule_fires(rule: Rule, X: pd.DataFrame) -> np.ndarray:
    """Return a boolean array: True where rule fires (predicts attack) on each row of X."""
    op = _OPS.get(rule.op)
    if op is None:
        return np.zeros(len(X), dtype=bool)
    if rule.feature not in X.columns:
        return np.zeros(len(X), dtype=bool)
    col = X[rule.feature]
    val = _to_scalar(rule.value)
    # Equality on float columns is fragile but allowed; the prompt steers
    # the LLM away from it for continuous features.
    try:
        if isinstance(val, str):
            return op(col.astype(str), val).to_numpy()
        return op(col, val).to_numpy()
    except Exception:
        return np.zeros(len(X), dtype=bool)


def rules_firing_matrix(rules: Iterable[Rule], X: pd.DataFrame) -> np.ndarray:
    """Shape (n_rows, n_rules) boolean array of fire decisions."""
    cols = [rule_fires(r, X) for r in rules]
    if not cols:
        return np.zeros((len(X), 0), dtype=bool)
    return np.stack(cols, axis=1)


def predict(policy: Policy, X: pd.DataFrame) -> np.ndarray:
    """Apply the policy to X. Returns array of {"attack","normal"} strings.

    05-anonymization-ablation-*.ipynb / standardize-results.py use string labels; we match.
    """
    fires = rules_firing_matrix(policy.rules, X)
    n_rows, n_rules = fires.shape
    if n_rules == 0:
        return np.array(["normal"] * n_rows)

    if policy.voting_mode == "majority":
        n_attack_votes = fires.sum(axis=1)
        # Strict majority over k rules; tie -> normal (matches notebook 4 mode())
        attack = n_attack_votes > (n_rules / 2)
    elif policy.voting_mode == "or_gate":
        attack = fires.any(axis=1)
    elif policy.voting_mode == "weighted":
        weights = np.array(policy.weights, dtype=float)
        if len(weights) != n_rules:
            weights = np.ones(n_rules, dtype=float)
        scores = fires.astype(float) @ weights
        attack = scores >= policy.tau
    else:
        raise ValueError(f"Unknown voting_mode: {policy.voting_mode}")

    return np.where(attack, "attack", "normal")


def calibrate_tau(
    fires_val: np.ndarray, y_val: np.ndarray, weights: np.ndarray,
    metric: str = "macro_f1",
) -> float:
    """Sweep tau over possible score thresholds and return the best by `metric`.

    `y_val` should be an array of {"attack","normal"}.
    """
    from .eval import classification_metrics

    if fires_val.size == 0:
        return 0.5
    scores = fires_val.astype(float) @ weights
    # Candidate thresholds: 0, then between each pair of distinct scores, then a value above max.
    uniq = np.unique(scores)
    if len(uniq) == 0:
        return 0.5
    cands = list(uniq) + [uniq.max() + 1.0]
    best_tau, best_score = float(uniq[0]), -1.0
    for t in cands:
        pred = np.where(scores >= t, "attack", "normal")
        m = classification_metrics(y_val, pred)
        s = m.get(metric, m["macro_f1"])
        if s > best_score:
            best_score, best_tau = s, float(t)
    return best_tau
