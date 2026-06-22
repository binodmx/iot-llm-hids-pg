"""Diversity scoring and greedy diverse-top-k selection.

Replaces the broken 5%-disagreement filter in notebook 7's complex pipeline.

Four-axis score for a candidate rule against an already-accepted set:

    div(r, accepted) = w_tag * tag_novelty
                     + w_disagreement * decision_disagreement_imbalance_aware
                     + w_feature * feature_novelty
                     + w_threshold * threshold_separation

Final selection: composite = alpha * f1 + (1 - alpha) * div, greedy fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Rule, RunConfig
from .voting import rule_fires


@dataclass
class CandidateScore:
    rule: Rule
    val_f1: float
    div_score: float
    composite: float
    rejection_reason: str = ""


def _imbalance_aware_disagreement(
    new_fires_normal: np.ndarray, new_fires_attack: np.ndarray,
    accepted_fires_normal: np.ndarray, accepted_fires_attack: np.ndarray,
) -> float:
    """1 - (J_normal + J_attack) / 2; J = Jaccard similarity of fire-sets within a class.

    Handles class imbalance: a rule that disagrees only on the tiny normal
    class still gets credit. Returns 1.0 when accepted set is empty.
    """
    if accepted_fires_normal.size == 0 or accepted_fires_normal.shape[1] == 0:
        return 1.0

    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        # a, b: boolean 1-D arrays
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    union_acc_normal = accepted_fires_normal.any(axis=1)
    union_acc_attack = accepted_fires_attack.any(axis=1)
    j_n = jaccard(new_fires_normal, union_acc_normal)
    j_a = jaccard(new_fires_attack, union_acc_attack)
    return 1.0 - (j_n + j_a) / 2.0


def _tag_novelty(candidate: Rule, accepted: list[Rule], k: int) -> float:
    same = sum(1 for a in accepted if a.phenomenon_tag == candidate.phenomenon_tag)
    return max(0.0, 1.0 - same / k)


def _feature_novelty(candidate: Rule, accepted: list[Rule]) -> float:
    return 0.0 if any(a.feature == candidate.feature for a in accepted) else 1.0


def _threshold_separation(
    candidate: Rule, accepted: list[Rule], normal_df: pd.DataFrame,
) -> float:
    same_feat = [a for a in accepted if a.feature == candidate.feature]
    if not same_feat:
        return 1.0
    if candidate.feature not in normal_df.columns:
        return 0.0
    # Coerce defensively — categorical columns may pass through here for == / != rules
    col_num = pd.to_numeric(normal_df[candidate.feature], errors="coerce").dropna()
    if len(col_num) == 0:
        return 0.0
    try:
        iqr = float(col_num.quantile(0.75) - col_num.quantile(0.25)) or 1.0
    except Exception:
        return 0.0
    try:
        cval = float(candidate.value)
    except (TypeError, ValueError):
        return 0.0
    seps = []
    for a in same_feat:
        try:
            aval = float(a.value)
        except (TypeError, ValueError):
            continue
        seps.append(min(abs(cval - aval) / iqr, 1.0))
    return min(seps) if seps else 0.0


def score_rule_diversity(
    candidate: Rule,
    accepted: list[Rule],
    normal_train: pd.DataFrame,
    attack_train: pd.DataFrame,
    cfg: RunConfig,
) -> float:
    """Multi-axis diversity score in [0, 1]."""
    cand_n = rule_fires(candidate, normal_train)
    cand_a = rule_fires(candidate, attack_train)
    if accepted:
        acc_n = np.stack([rule_fires(a, normal_train) for a in accepted], axis=1)
        acc_a = np.stack([rule_fires(a, attack_train) for a in accepted], axis=1)
    else:
        acc_n = np.empty((len(normal_train), 0), dtype=bool)
        acc_a = np.empty((len(attack_train), 0), dtype=bool)

    d_disagree = _imbalance_aware_disagreement(cand_n, cand_a, acc_n, acc_a)
    d_tag      = _tag_novelty(candidate, accepted, cfg.k)
    d_feature  = _feature_novelty(candidate, accepted)
    d_thresh   = _threshold_separation(candidate, accepted, normal_train)

    return (
        cfg.div_w_tag         * d_tag
        + cfg.div_w_disagreement * d_disagree
        + cfg.div_w_feature      * d_feature
        + cfg.div_w_threshold    * d_thresh
    )


def dominant_rejection_reason(
    candidate: Rule, accepted: list[Rule],
    normal_train: pd.DataFrame, attack_train: pd.DataFrame, cfg: RunConfig,
) -> str:
    """Return a short string for the next-round prompt explaining why a candidate lost."""
    if not accepted:
        return ""
    if _feature_novelty(candidate, accepted) == 0.0:
        return f"duplicate_feature:{candidate.feature}"
    if _tag_novelty(candidate, accepted, cfg.k) == 0.0:
        return f"saturated_tag:{candidate.phenomenon_tag}"
    cand_n = rule_fires(candidate, normal_train)
    cand_a = rule_fires(candidate, attack_train)
    acc_n = np.stack([rule_fires(a, normal_train) for a in accepted], axis=1)
    acc_a = np.stack([rule_fires(a, attack_train) for a in accepted], axis=1)
    d = _imbalance_aware_disagreement(cand_n, cand_a, acc_n, acc_a)
    if d < 0.1:
        return f"low_disagreement:{d:.2f}"
    return "lower_composite_than_alternatives"


def select_accepted(
    proposals: list[Rule],
    val_f1s: list[float],
    normal_train: pd.DataFrame,
    attack_train: pd.DataFrame,
    cfg: RunConfig,
) -> tuple[list[Rule], list[str]]:
    """Greedy diverse top-k selection.

    Returns (accepted_rules, rejection_reasons_for_unaccepted_proposals).
    """
    remaining: list[tuple[Rule, float]] = list(zip(proposals, val_f1s))
    accepted: list[Rule] = []
    while remaining and len(accepted) < cfg.k:
        # Score every remaining candidate against the current accepted set
        scored = []
        for rule, f1 in remaining:
            div = score_rule_diversity(rule, accepted, normal_train, attack_train, cfg)
            comp = cfg.composite_alpha * f1 + (1 - cfg.composite_alpha) * div
            scored.append((rule, f1, div, comp))
        scored.sort(key=lambda x: -x[3])  # highest composite first
        winner = scored[0][0]
        accepted.append(winner)
        remaining = [(r, f1) for r, f1 in remaining if r is not winner]

    rejection_reasons = [
        f"{r.feature} {r.op} {r.value} [{r.phenomenon_tag}]: "
        + dominant_rejection_reason(r, accepted, normal_train, attack_train, cfg)
        for r, _ in remaining
    ]
    return accepted, rejection_reasons
