"""Build a statistics package that the LLM consumes as context.

The package summarises normal vs attack distributions plus per-attack-class
divergence on the most discriminative features. This is the contextual
information that lets the LLM propose rules that generalise across attack
classes (zero-day-friendly) rather than fitting one class at a time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_NUMERIC_KINDS = "iufc"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.kind in _NUMERIC_KINDS]


def _summary(s: pd.Series) -> dict:
    # Coerce to numeric defensively — pandas may carry mixed-type 'object'
    # columns that pass the dtype.kind check but contain strings.
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if len(s_num) == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0}
    try:
        return {
            "mean":   float(s_num.mean()),
            "std":    float(s_num.std()),
            "p10":    float(s_num.quantile(0.10)),
            "median": float(s_num.median()),
            "p90":    float(s_num.quantile(0.90)),
        }
    except Exception:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0}


def build_stats_package(
    normal_df: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_class_labels: Optional[pd.Series] = None,
    top_n_divergent: int = 15,
) -> dict:
    """Return a dict that's safe to JSON-dump for the LLM prompt.

    - aggregate: normal vs attack summary stats for top-N divergent features.
    - per_class: when attack_class_labels is provided, mean per attack class
                 for the top-N features (this is the zero-day signal).
    - categoricals: value-count distributions for non-numeric features.
    """
    feats = _numeric_cols(normal_df)
    feats = [f for f in feats if f in attack_df.columns]

    # Rank features by standardised mean difference (proxy for separability).
    divs = []
    for f in feats:
        try:
            n = pd.to_numeric(normal_df[f], errors="coerce").dropna()
            a = pd.to_numeric(attack_df[f], errors="coerce").dropna()
            if len(n) == 0 or len(a) == 0:
                continue
            pooled = ((n.std() + a.std()) / 2.0) or 1.0
            divs.append((f, abs(a.mean() - n.mean()) / pooled))
        except Exception:
            continue
    divs.sort(key=lambda x: -x[1])
    top_feats = [f for f, _ in divs[:top_n_divergent]]

    aggregate = {}
    for f in top_feats:
        aggregate[f] = {
            "normal": _summary(normal_df[f]),
            "attack": _summary(attack_df[f]),
        }

    per_class = {}
    if attack_class_labels is not None and len(attack_class_labels) == len(attack_df):
        unique_classes = list(pd.Series(attack_class_labels).unique())
        for f in top_feats:
            row = {}
            col_num = pd.to_numeric(attack_df[f], errors="coerce")
            for cls in unique_classes:
                mask = (attack_class_labels.values == cls)
                if mask.sum() == 0:
                    continue
                try:
                    vals = col_num.values[mask]
                    finite = vals[~pd.isna(vals)]
                    if len(finite) == 0:
                        continue
                    row[str(cls)] = float(finite.mean())
                except Exception:
                    continue
            per_class[f] = row

    # Categorical / discrete features: value-count overview for non-numeric cols
    cat_feats = [c for c in normal_df.columns if c not in feats]
    categoricals = {}
    for c in cat_feats[:10]:
        if c not in attack_df.columns:
            continue
        try:
            n_top = normal_df[c].astype(str).value_counts().head(5).to_dict()
            a_top = attack_df[c].astype(str).value_counts().head(5).to_dict()
            categoricals[c] = {"normal_top5": n_top, "attack_top5": a_top}
        except Exception:
            continue

    return {
        "top_features": top_feats,
        "aggregate": aggregate,
        "per_class_attack_means": per_class,
        "categorical_overview": categoricals,
        "n_normal": int(len(normal_df)),
        "n_attack": int(len(attack_df)),
        "attack_class_count": (
            int(len(set(attack_class_labels))) if attack_class_labels is not None else None
        ),
    }
