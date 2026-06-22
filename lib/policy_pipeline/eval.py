"""Classification-report wrappers.

We keep label strings {"attack","normal"} for compatibility with the
existing canonical eval script (standardize-results.py).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a flat dict of the metrics the pipeline + sweep care about."""
    rpt = classification_report(
        y_true, y_pred, labels=["normal", "attack"],
        output_dict=True, zero_division=0,
    )
    out = {
        "macro_f1":         rpt["macro avg"]["f1-score"],
        "accuracy":         rpt["accuracy"],
        "attack_precision": rpt["attack"]["precision"],
        "attack_recall":    rpt["attack"]["recall"],
        "attack_f1":        rpt["attack"]["f1-score"],
        "normal_precision": rpt["normal"]["precision"],
        "normal_recall":    rpt["normal"]["recall"],
        "normal_f1":        rpt["normal"]["f1-score"],
    }
    # Useful tunable target: attack precision when recall is at least 0.9
    try:
        y_score = (y_pred == "attack").astype(float)
        y_bin = (y_true == "attack").astype(int)
        p, r, _ = precision_recall_curve(y_bin, y_score)
        mask = r >= 0.9
        out["attack_precision_at_recall_0.9"] = float(p[mask].max()) if mask.any() else 0.0
    except Exception:
        out["attack_precision_at_recall_0.9"] = 0.0
    return out


def per_rule_metrics(fires: np.ndarray, y_true: np.ndarray) -> dict:
    """Per-rule precision/recall/F1 when treating that rule as the sole classifier."""
    y_bin = (y_true == "attack").astype(int)
    n_rules = fires.shape[1]
    rows = []
    for j in range(n_rules):
        pred = fires[:, j].astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum())
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"precision": prec, "recall": rec, "f1": f1})
    return {"per_rule": rows}
