"""Configuration and result dataclasses.

All tunables live on RunConfig so a hyperparameter sweep is a simple
`for cfg in grid: run_pipeline(cfg)` loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# Controlled vocabulary of phenomenon tags. The next-round prompt uses
# this set to tell the LLM which tags are already covered vs uncovered.
PHENOMENON_TAGS = (
    "HANDSHAKE",   # incomplete TCP handshake (flag counts)
    "ASYMMETRY",   # asymmetric byte/packet ratios (src vs dst)
    "TIMING",      # inter-arrival jitter, micro-durations
    "HEADER",      # header length / TTL outliers
    "SERVICE",     # unexpected dst_port for the proto
    "STATE",       # connection-state pathology (OTH, S0, REJ)
    "VOLUME",      # raw byte/packet outliers
    "ENTROPY",     # payload / port-distribution irregularity
)


@dataclass(frozen=True)
class RunConfig:
    # identity
    dataset_key: str                  # "cic" | "wustl" | "ton" | "bot" | "unsw"
    provider: str                     # "anthropic" | "google"
    model_id: str                     # provider-specific model id
    seed: int = 42

    # pipeline shape (tunable)
    k: int = 5
    max_rounds: int = 8
    early_stop_patience: int = 2
    val_slice_frac: float = 0.20
    top_n_divergent: int = 15
    temperature: float = 0.1

    # voting (tunable)
    voting_mode: Literal["majority", "or_gate", "weighted"] = "weighted"
    tau: Optional[float] = None       # if None, auto-calibrate on val slice
    weight_fn: Literal["precision", "f1", "uniform"] = "precision"

    # diversity weights (tunable)
    div_w_tag: float = 0.35
    div_w_disagreement: float = 0.30
    div_w_feature: float = 0.20
    div_w_threshold: float = 0.15
    composite_alpha: float = 0.60     # weight on F1 in composite = a*F1 + (1-a)*div

    # target metric (tunable)
    selection_metric: Literal[
        "macro_f1",
        "attack_f1",
        "attack_precision_at_recall_0.9",
    ] = "macro_f1"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    feature: str
    op: str
    value: float | int | str
    phenomenon_tag: str
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "op": self.op,
            "value": self.value,
            "phenomenon_tag": self.phenomenon_tag,
            "rationale": self.rationale,
        }


@dataclass
class Policy:
    rules: list[Rule]
    voting_mode: str
    tau: float = 0.5
    weights: list[float] = field(default_factory=list)


@dataclass
class ProposalScore:
    rule: Rule
    train_f1: float
    train_precision: float
    train_recall: float
    val_f1: float
    val_precision: float
    val_recall: float


@dataclass
class RoundLog:
    round_index: int
    proposals: list[Rule]
    scored: list[ProposalScore]
    accepted: list[Rule]
    rejected_reasons: list[str]
    policy_metrics: dict
    tokens_in: int
    tokens_out: int


@dataclass
class RunResult:
    config: RunConfig
    best_round_index: int
    best_policy: Policy
    best_metrics: dict
    history: list[RoundLog]
    total_tokens_in: int
    total_tokens_out: int
