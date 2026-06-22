"""Rule identity helpers.

Replaces notebook 7's `feature TEXT PRIMARY KEY` SQLite schema, which
forced one-rule-per-feature collapse. Here multiple rules per feature
are allowed; uniqueness is on the full (feature, op, rounded value)
triplet.

The pipeline is in-memory; we don't actually persist to SQLite. The
final policy is serialised to JSON via io.save_policy().
"""

from __future__ import annotations

import hashlib

from .config import Rule


def rule_id(rule: Rule, sig: int = 3) -> str:
    """Deterministic id for a rule. Numeric values are rounded to `sig` decimals
    so two rules with thresholds 0.001 and 0.0011 hash identically."""
    try:
        val_repr = f"{float(rule.value):.{sig}f}"
    except (TypeError, ValueError):
        val_repr = str(rule.value)
    key = f"{rule.feature}|{rule.op}|{val_repr}".encode()
    return hashlib.sha1(key).hexdigest()[:12]


def deduplicate(rules: list[Rule]) -> list[Rule]:
    """Drop later duplicates by rule_id, preserving order."""
    seen: set[str] = set()
    out: list[Rule] = []
    for r in rules:
        rid = rule_id(r)
        if rid not in seen:
            seen.add(rid)
            out.append(r)
    return out
