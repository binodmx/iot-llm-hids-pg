"""Hybrid policy generation pipeline: stats RAG + vector examples.

Standalone copy of pipeline.py with hybrid prompt injection.
Injects concrete examples alongside statistical guidance for improved
generalization on imbalanced datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from .config import (
    PHENOMENON_TAGS, Policy, ProposalScore, RoundLog, Rule, RunConfig, RunResult,
)
from .diversity import score_rule_diversity, select_accepted
from .eval import classification_metrics
from .models import extract_token_usage, extract_tool_calls, make_llm
from .persistence import deduplicate
from .prompts import (
    build_human_message_hybrid, build_system_message, propose_rule
)
from .stats import build_stats_package
from .voting import calibrate_tau, predict, rule_fires, rules_firing_matrix


def _stratified_carve(
    normal_train: pd.DataFrame, attack_train: pd.DataFrame, val_frac: float, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve a stratified `val_frac` slice off each class."""
    n_val = normal_train.sample(frac=val_frac, random_state=seed)
    n_tr  = normal_train.drop(n_val.index)
    a_val = attack_train.sample(frac=val_frac, random_state=seed)
    a_tr  = attack_train.drop(a_val.index)
    return n_tr, n_val, a_tr, a_val


def _y_for(normal: pd.DataFrame, attack: pd.DataFrame) -> np.ndarray:
    return np.concatenate([
        np.full(len(normal), "normal"),
        np.full(len(attack), "attack"),
    ])


def _X_for(normal: pd.DataFrame, attack: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([normal, attack], axis=0, ignore_index=True)


def _score_proposal(
    rule: Rule,
    train_normal: pd.DataFrame, train_attack: pd.DataFrame,
    val_normal: pd.DataFrame, val_attack: pd.DataFrame,
) -> ProposalScore:
    X_tr, y_tr = _X_for(train_normal, train_attack), _y_for(train_normal, train_attack)
    X_va, y_va = _X_for(val_normal,   val_attack),   _y_for(val_normal,   val_attack)
    fires_tr = rule_fires(rule, X_tr)
    fires_va = rule_fires(rule, X_va)
    pred_tr  = np.where(fires_tr, "attack", "normal")
    pred_va  = np.where(fires_va, "attack", "normal")
    m_tr = classification_metrics(y_tr, pred_tr)
    m_va = classification_metrics(y_va, pred_va)
    return ProposalScore(
        rule=rule,
        train_f1=m_tr["macro_f1"], train_precision=m_tr["attack_precision"],
        train_recall=m_tr["attack_recall"],
        val_f1=m_va["macro_f1"], val_precision=m_va["attack_precision"],
        val_recall=m_va["attack_recall"],
    )


def _build_policy(
    accepted: list[Rule],
    val_normal: pd.DataFrame, val_attack: pd.DataFrame,
    cfg: RunConfig,
) -> Policy:
    X_va, y_va = _X_for(val_normal, val_attack), _y_for(val_normal, val_attack)
    fires_va = rules_firing_matrix(accepted, X_va)
    if cfg.voting_mode == "weighted":
        weights = []
        for j in range(fires_va.shape[1]):
            pred = np.where(fires_va[:, j], "attack", "normal")
            m = classification_metrics(y_va, pred)
            if cfg.weight_fn == "precision":
                weights.append(m["attack_precision"])
            elif cfg.weight_fn == "f1":
                weights.append(m["attack_f1"])
            else:
                weights.append(1.0)
        weights_arr = np.array(weights, dtype=float)
        if weights_arr.sum() == 0:
            weights_arr = np.ones(len(weights), dtype=float)
        if cfg.tau is None:
            tau = calibrate_tau(fires_va, y_va, weights_arr, metric=cfg.selection_metric)
        else:
            tau = float(cfg.tau)
        return Policy(rules=accepted, voting_mode="weighted",
                       tau=tau, weights=weights_arr.tolist())
    elif cfg.voting_mode == "majority":
        return Policy(rules=accepted, voting_mode="majority",
                       tau=len(accepted) / 2.0 + 1e-9,
                       weights=[1.0] * len(accepted))
    elif cfg.voting_mode == "or_gate":
        return Policy(rules=accepted, voting_mode="or_gate",
                       tau=0.5, weights=[1.0] * len(accepted))
    else:
        raise ValueError(f"Unknown voting_mode: {cfg.voting_mode}")


def _safe_tag(tag) -> str:
    t = (str(tag) or "VOLUME").upper().strip()
    return t if t in PHENOMENON_TAGS else "VOLUME"


def _parse_proposals(ai_msg) -> list[Rule]:
    rules: list[Rule] = []
    for tc in extract_tool_calls(ai_msg):
        if tc.get("name") != "propose_rule":
            continue
        a = tc.get("args", {}) or {}
        try:
            rules.append(Rule(
                feature=str(a.get("feature", "")).strip(),
                op=str(a.get("op", "")).strip(),
                value=a.get("value"),
                phenomenon_tag=_safe_tag(a.get("phenomenon_tag")),
                rationale=str(a.get("rationale", ""))[:200],
            ))
        except Exception:
            continue
    return deduplicate([r for r in rules if r.feature and r.op])


def _run_llm_round_hybrid(
    llm_bound, stats_pkg: dict, history: list[RoundLog], cfg: RunConfig,
    normal_examples: list[dict], attack_examples: list[dict],
) -> tuple[list[Rule], int, int]:
    """LLM round with hybrid prompts (stats + examples)."""
    sys = SystemMessage(build_system_message(cfg.k, stats_pkg.get("attack_class_count")))
    hum = HumanMessage(build_human_message_hybrid(
        stats_pkg, history, cfg.k,
        normal_examples, attack_examples
    ))
    ai = llm_bound.invoke([sys, hum])
    proposals = _parse_proposals(ai)
    tin, tout = extract_token_usage(ai)
    return proposals, tin, tout


def run_pipeline_hybrid(
    cfg: RunConfig,
    normal_train: pd.DataFrame, attack_train: pd.DataFrame,
    normal_examples: list[dict], attack_examples: list[dict],
    attack_class_labels: pd.Series | None = None,
    verbose: bool = True,
) -> RunResult:
    """Run policy generation with hybrid RAG (stats + concrete examples).

    Args:
        cfg: Configuration (model, voting, k, rounds, etc.)
        normal_train: Normal (benign) training flows
        attack_train: Attack training flows
        normal_examples: Example normal flows for prompt (from hybrid_retrieve)
        attack_examples: Example attack flows for prompt (from hybrid_retrieve)
        attack_class_labels: Optional per-class labels for attacks
        verbose: Print progress

    Returns:
        RunResult with best policy, metrics, and round history.
    """

    rng_seed = cfg.seed
    np.random.seed(rng_seed)

    n_tr, n_val, a_tr, a_val = _stratified_carve(
        normal_train, attack_train, cfg.val_slice_frac, rng_seed,
    )
    if attack_class_labels is not None:
        a_labels_tr = attack_class_labels.loc[a_tr.index] if hasattr(attack_class_labels, "loc") else None
    else:
        a_labels_tr = None

    stats_pkg = build_stats_package(n_tr, a_tr, a_labels_tr, top_n_divergent=cfg.top_n_divergent)

    llm = make_llm(cfg.provider, cfg.model_id, temperature=cfg.temperature)
    llm_bound = llm.bind_tools([propose_rule])

    history: list[RoundLog] = []
    best_round_idx = -1
    best_policy: Policy | None = None
    best_metric_value: float = -1.0
    best_metrics_dict: dict = {}
    total_in = total_out = 0
    rounds_without_improvement = 0

    for round_idx in range(cfg.max_rounds):
        if verbose:
            print(f"\n=== Round {round_idx + 1}/{cfg.max_rounds} ===")
        proposals, tin, tout = _run_llm_round_hybrid(
            llm_bound, stats_pkg, history, cfg,
            normal_examples, attack_examples
        )
        total_in += tin; total_out += tout
        if verbose:
            print(f"  LLM emitted {len(proposals)} proposals (tokens in/out: {tin}/{tout})")
            for p in proposals:
                print(f"    [{p.phenomenon_tag:9}] {p.feature} {p.op} {p.value}")

        scored: list[ProposalScore] = []
        for p in proposals:
            try:
                scored.append(_score_proposal(p, n_tr, a_tr, n_val, a_val))
            except Exception as e:
                if verbose:
                    print(f"    score-error for {p.feature}: {e}")

        accepted, reject_reasons = select_accepted(
            [s.rule for s in scored],
            [s.val_f1 for s in scored],
            n_tr, a_tr, cfg,
        )

        if not accepted:
            if verbose: print("  No proposals accepted; stopping.")
            break

        policy = _build_policy(accepted, n_val, a_val, cfg)
        X_va, y_va = _X_for(n_val, a_val), _y_for(n_val, a_val)
        pred_va = predict(policy, X_va)
        metrics = classification_metrics(y_va, pred_va)
        metrics["n_distinct_tags"] = len({r.phenomenon_tag for r in accepted})
        metric_value = metrics.get(cfg.selection_metric, metrics["macro_f1"])

        if verbose:
            print(f"  Accepted {len(accepted)} rules, {metrics['n_distinct_tags']} distinct tags")
            print(f"  Val {cfg.selection_metric}: {metric_value:.4f}  "
                  f"(macro_f1={metrics['macro_f1']:.4f}, "
                  f"attack_f1={metrics['attack_f1']:.4f}, "
                  f"prec={metrics['attack_precision']:.4f}, "
                  f"rec={metrics['attack_recall']:.4f})")

        history.append(RoundLog(
            round_index=round_idx, proposals=proposals, scored=scored,
            accepted=accepted, rejected_reasons=reject_reasons,
            policy_metrics=metrics, tokens_in=tin, tokens_out=tout,
        ))

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_round_idx = round_idx
            best_policy = policy
            best_metrics_dict = metrics
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= cfg.early_stop_patience:
                if verbose:
                    print(f"  Early-stop: no improvement for {rounds_without_improvement} rounds")
                break

    if best_policy is None:
        best_policy = Policy(rules=[], voting_mode=cfg.voting_mode, tau=0.5, weights=[])
        best_metrics_dict = {"macro_f1": 0.0}

    return RunResult(
        config=cfg, best_round_index=best_round_idx,
        best_policy=best_policy, best_metrics=best_metrics_dict,
        history=history,
        total_tokens_in=total_in, total_tokens_out=total_out,
    )
