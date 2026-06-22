"""Prompt construction.

System message: priorities + operator constraints + rule-shape constraint
                + phenomenon-tag vocabulary.

Human message: stats package + history package (accepted rules with
                per-rule precision/recall + rejected candidates + which
                tags remain uncovered).

Tool schema: propose_rule(feature, op, value, phenomenon_tag, rationale)
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import PHENOMENON_TAGS, RoundLog, Rule


# ─────────────────────────────────────────────────────────────────────
# Tool schema (pydantic so both providers' tool-binding work uniformly)
# ─────────────────────────────────────────────────────────────────────

_TAG_DOC = ", ".join(PHENOMENON_TAGS)


class propose_rule(BaseModel):
    """Propose one ATTACK-flagging threshold rule.

    Each rule is a single condition (feature op value). When TRUE, the
    flow is flagged as ATTACK by this rule. The voting layer combines
    multiple rules later.
    """

    feature: str = Field(description="Exact feature name as listed in the input.")
    op: str = Field(description="One of: >, <, >=, <=, ==, !=")
    value: str = Field(description=(
        "Threshold value. For numeric features pass a number (as string is OK). "
        "For categorical/discrete features pass the literal value."
    ))
    phenomenon_tag: str = Field(description=(
        f"One of the controlled tags: {_TAG_DOC}. Indicates the protocol-level "
        "phenomenon this rule captures."
    ))
    rationale: str = Field(description=(
        "≤25 words: what phenomenon this captures and which attack classes "
        "should satisfy it."
    ))


# ─────────────────────────────────────────────────────────────────────
# System message
# ─────────────────────────────────────────────────────────────────────

def build_system_message(k: int, n_known_classes: Optional[int] = None) -> str:
    n_classes_clause = (
        f"the {n_known_classes} known attack classes shown" if n_known_classes
        else "the known attack classes shown"
    )
    tag_lines = (
        "    HANDSHAKE   incomplete TCP handshake (flag counts)\n"
        "    ASYMMETRY   asymmetric byte/packet ratios (src vs dst)\n"
        "    TIMING      inter-arrival jitter, micro-durations\n"
        "    HEADER      header length / TTL outliers\n"
        "    SERVICE     unexpected dst_port for the proto\n"
        "    STATE       connection-state pathology (OTH, S0, REJ)\n"
        "    VOLUME      raw byte/packet outliers\n"
        "    ENTROPY     payload / port-distribution irregularity"
    )
    return (
        f"You are an IoT intrusion-detection analyst. You will propose {k} simple\n"
        f"threshold rules that EACH, on its own, flags a network flow as ATTACK\n"
        f"when true. The rules are combined later by a voting layer — DO NOT try\n"
        f"to construct a single classifier in your head.\n\n"
        f"REQUIREMENTS, in priority order:\n\n"
        f"1. GENERALISATION. Rules must fire on PREVIOUSLY UNSEEN attack families,\n"
        f"   not only {n_classes_clause}. Prefer features where the attack\n"
        f"   distribution differs from normal CONSISTENTLY across every attack\n"
        f"   class in the per-class table — not features where one class\n"
        f"   dominates the average.\n\n"
        f"2. SEMANTIC DIVERSITY. The {k} rules must each capture a DIFFERENT\n"
        f"   protocol-level phenomenon. Phenomenon tags:\n"
        f"{tag_lines}\n"
        f"   Do NOT propose two rules over the same feature; do NOT propose two\n"
        f"   rules that encode the same phenomenon under different features.\n\n"
        f"3. NON-TRIVIAL REASONING. A single threshold on `flow_duration` or\n"
        f"   `Header_Length` is the kind of split a decision tree already finds.\n"
        f"   Prefer features whose discriminative power requires PROTOCOL\n"
        f"   knowledge a tree cannot articulate (handshake completion, control-\n"
        f"   plane vs data-plane ratios, service-specific port semantics).\n\n"
        f"4. ROBUSTNESS. Avoid thresholds inside the normal p10–p90 band where\n"
        f"   possible. Avoid equality on continuous floats.\n\n"
        f"OPERATOR CONSTRAINTS:\n"
        f"- Use >, <, >=, <= ONLY for continuous numeric features.\n"
        f"- Use ==, != ONLY for categorical or discrete features (string values,\n"
        f"  port numbers, protocol types, binary flags).\n"
        f"- Do NOT use ==/!= on float-valued continuous features.\n\n"
        f"RULE SHAPE: Each rule is a SINGLE condition (feature op value). Do NOT\n"
        f"emit compound AND/OR expressions — composition happens in the voting\n"
        f"layer. The prior compound-rule experiment produced near-clone\n"
        f"expressions and worse F1, so single conditions stay.\n\n"
        f"OUTPUT: exactly {k} tool calls to `propose_rule`. Each MUST include\n"
        f"rationale (≤25 words) and a phenomenon_tag from the controlled\n"
        f"vocabulary above."
    )


# ─────────────────────────────────────────────────────────────────────
# Human message
# ─────────────────────────────────────────────────────────────────────

def _format_history(history: list[RoundLog], cfg_k: int) -> dict[str, Any]:
    if not history:
        return {"is_first_round": True}
    last = history[-1]
    covered_tags = sorted({a.phenomenon_tag for a in last.accepted})
    uncovered_tags = [t for t in PHENOMENON_TAGS if t not in covered_tags]
    return {
        "is_first_round": False,
        "round_number": last.round_index + 2,  # human-friendly 1-indexed next round
        "previous_round_metrics": last.policy_metrics,
        "previously_accepted": [
            {
                "feature": a.feature, "op": a.op, "value": a.value,
                "phenomenon_tag": a.phenomenon_tag,
                "rationale": a.rationale,
            }
            for a in last.accepted
        ],
        "covered_tags": covered_tags,
        "uncovered_tags": uncovered_tags,
        "rejected_last_round": last.rejected_reasons,
    }


def build_human_message(
    stats_pkg: dict, history: list[RoundLog], k: int,
) -> str:
    history_pkg = _format_history(history, k)

    if history_pkg["is_first_round"]:
        framing = (
            f"This is round 1. Propose your first {k} rules covering the most\n"
            f"semantically distinct phenomena you can identify from the stats."
        )
    else:
        covered = history_pkg["covered_tags"]
        uncovered = history_pkg["uncovered_tags"]
        framing = (
            f"This is round {history_pkg['round_number']}.\n\n"
            f"PREVIOUSLY ACCEPTED RULES (these are the current policy):\n"
            f"{json.dumps(history_pkg['previously_accepted'], indent=2)}\n\n"
            f"PREVIOUS ROUND METRICS:\n"
            f"{json.dumps(history_pkg['previous_round_metrics'], indent=2)}\n\n"
            f"PHENOMENON TAGS ALREADY COVERED: {covered or 'none'} — avoid these.\n"
            f"PHENOMENON TAGS STILL UNCOVERED: {uncovered or 'none'} — prioritise these.\n\n"
            f"REJECTED CANDIDATES FROM LAST ROUND (do not repeat these mistakes):\n"
            f"{chr(10).join(history_pkg['rejected_last_round']) or '(none)'}\n\n"
            f"Propose {k} NEW rules that:\n"
            f"- Cover at least {max(1, k - len(covered))} uncovered phenomenon tag(s).\n"
            f"- Use FEATURES NOT already in the current policy where possible.\n"
            f"- Avoid the rejection patterns listed above.\n"
            f"- Aim to improve attack recall on rows where current policy mispredicts."
        )

    return (
        f"DATASET STATISTICS\n"
        f"==================\n"
        f"n_normal: {stats_pkg['n_normal']}, n_attack: {stats_pkg['n_attack']}, "
        f"attack_class_count: {stats_pkg.get('attack_class_count')}\n\n"
        f"TOP DIVERGENT FEATURES (ranked by standardised mean diff):\n"
        f"{json.dumps(stats_pkg['top_features'])}\n\n"
        f"PER-FEATURE AGGREGATE (normal vs attack: mean/std/p10/median/p90):\n"
        f"{json.dumps(stats_pkg['aggregate'], indent=2)}\n\n"
        f"PER-CLASS ATTACK MEANS (use this for GENERALISATION — features where\n"
        f"the value is high/low across MOST classes are better than features\n"
        f"that one class dominates):\n"
        f"{json.dumps(stats_pkg['per_class_attack_means'], indent=2)}\n\n"
        f"CATEGORICAL OVERVIEW (top-5 values, normal vs attack):\n"
        f"{json.dumps(stats_pkg['categorical_overview'], indent=2)}\n\n"
        f"{framing}\n\n"
        f"Emit exactly {k} tool calls to `propose_rule`."
    )


def build_human_message_hybrid(
    stats_pkg: dict,
    history: list[RoundLog],
    k: int,
    normal_examples: list[dict],
    attack_examples: list[dict],
) -> str:
    """Build human message with stats AND concrete flow examples.

    Args:
        stats_pkg: statistical aggregates (mean/std/percentiles)
        history: prior rounds' accepted rules and metrics
        k: number of rules to propose
        normal_examples: list of example normal flows as dicts
        attack_examples: list of example attack flows as dicts

    Returns:
        Formatted prompt string with examples + stats.
    """

    # Format examples as compact JSON
    normal_examples_json = json.dumps(normal_examples[:5], indent=2)[:800]
    attack_examples_json = json.dumps(attack_examples[:5], indent=2)[:800]

    examples_section = (
        f"CONCRETE EXAMPLES FROM TRAINING DATA:\n"
        f"(Top-5 normal + attack flows by centroid similarity; showing first 5 features per flow)\n\n"
        f"Normal Flow Examples:\n{normal_examples_json}\n\n"
        f"Attack Flow Examples:\n{attack_examples_json}\n\n"
    )

    # Get base message (includes stats, history, framing)
    base_message = build_human_message(stats_pkg, history, k)

    return examples_section + base_message
