"""Per-dataset loaders with balanced-corpus support.

Each dataset uses a different file path, label column, and label
encoding. Bot-IoT especially needs balanced sampling because only 370
normal flows exist in the 2.93M-row population (1:7687 imbalance).

Entry point: `load_dataset(key, balanced=True)` -> DatasetSplit.

A DatasetSplit holds the train/test split plus an optional attack-class
label series for the stats package's per-class table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
HOME_DATA = os.path.expanduser("~/Documents/Projects/RAG Paper/data")


# ─────────────────────────────────────────────────────────────────────
# Dataset specifications
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    key: str
    display_name: str
    paths: list[str]                # one path = single file; two = (train, test) pre-split
    label_col: str
    benign_value: object             # value indicating "normal"; everything else = attack
    drop_cols: list[str]             # columns to drop (including label_col and any auxiliaries)
    attack_class_col: Optional[str]  # optional: column for per-class attack labels
    exclude_classes: list[str] = field(default_factory=list)
    cap_per_class: Optional[int] = 50_000  # max samples per class for balanced corpus
    presplit: bool = False           # True for UNSW-NB15 (pre-split train/test files)

    def resolved_paths(self) -> list[str]:
        out = []
        for p in self.paths:
            if p.startswith("~"):
                out.append(os.path.expanduser(p))
            elif p.startswith("/"):
                out.append(p)
            else:
                out.append(os.path.join(REPO_ROOT, p))
        return out


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cic": DatasetSpec(
        key="cic",
        display_name="CIC-IoT2023",
        paths=["iot-llm-hids-pg/1-cic-iot/data/sample-100000-2.csv"],
        label_col="label",
        benign_value="BenignTraffic",
        drop_cols=["label"],
        attack_class_col="label",
        cap_per_class=None,           # already pre-sampled
    ),
    "wustl": DatasetSpec(
        key="wustl",
        display_name="WUSTL-IIoT",
        paths=["iot-llm-hids-pg/2-wustl-iiot/data/population.csv"],
        label_col="Target",
        benign_value=0,
        drop_cols=["Target", "Traffic"],
        attack_class_col="Traffic",
        cap_per_class=50_000,         # opposite-imbalance (mostly normal) — cap both
    ),
    "ton": DatasetSpec(
        key="ton",
        display_name="TON_IoT",
        paths=[f"{HOME_DATA}/ton-iot/ton-iot-population.csv"],
        label_col="label",
        benign_value=0,
        drop_cols=["label", "type"],
        attack_class_col="type",
        cap_per_class=42_000,         # cap to normal-class size (already 42K/148K)
    ),
    "bot": DatasetSpec(
        key="bot",
        display_name="Bot-IoT",
        paths=[f"{HOME_DATA}/bot-iot/bot-iot-population.csv"],
        label_col="attack",
        benign_value=0,
        drop_cols=["attack", "category", "subcategory"],
        attack_class_col="subcategory",
        cap_per_class=370,            # only 370 normal exist — match it
    ),
    "unsw": DatasetSpec(
        key="unsw",
        display_name="UNSW-NB15",
        paths=[
            f"{HOME_DATA}/unsw-nb15/UNSW_NB15_training-set.csv",
            f"{HOME_DATA}/unsw-nb15/UNSW_NB15_testing-set.csv",
        ],
        label_col="label",
        benign_value=0,
        drop_cols=["label", "attack_cat", "id"],
        attack_class_col="attack_cat",
        exclude_classes=["Worms"],
        presplit=True,
        cap_per_class=50_000,
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSplit:
    spec: DatasetSpec
    normal_train: pd.DataFrame
    normal_test: pd.DataFrame
    attack_train: pd.DataFrame
    attack_test: pd.DataFrame
    attack_class_labels_train: Optional[pd.Series]   # aligned with attack_train.index


def _read_one(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def _drop_string_cols(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Keep only numeric features after split; LLM rules can't compare on strings
    except where the spec explicitly supports it (categorical via ==/!=).

    We keep object columns whose unique cardinality is <= 50 (likely categorical
    discriminators like proto/service/conn_state); drop high-cardinality strings
    (IPs, timestamps)."""
    keep = []
    for c in df.columns:
        if c == label_col:
            keep.append(c)
            continue
        if df[c].dtype.kind in "iufcb":
            keep.append(c)
        else:
            nun = df[c].nunique(dropna=True)
            if nun > 0 and nun <= 50:
                keep.append(c)
    return df[keep]


def load_dataset(key: str, seed: int = 42) -> DatasetSplit:
    spec = DATASET_SPECS[key]
    paths = spec.resolved_paths()

    if spec.presplit:
        df_train_raw = _read_one(paths[0])
        df_test_raw  = _read_one(paths[1])
        # Strip whitespace in attack_cat (UNSW-NB15 has leading spaces)
        if spec.attack_class_col and spec.attack_class_col in df_train_raw.columns:
            df_train_raw[spec.attack_class_col] = df_train_raw[spec.attack_class_col].astype(str).str.strip()
            df_test_raw[spec.attack_class_col]  = df_test_raw[spec.attack_class_col].astype(str).str.strip()
        # Apply exclude_classes (on attack_cat if available)
        if spec.exclude_classes and spec.attack_class_col in df_train_raw.columns:
            df_train_raw = df_train_raw[~df_train_raw[spec.attack_class_col].isin(spec.exclude_classes)]
            df_test_raw  = df_test_raw[~df_test_raw[spec.attack_class_col].isin(spec.exclude_classes)]
        df_train_raw = _drop_string_cols(df_train_raw, spec.label_col)
        df_test_raw  = _drop_string_cols(df_test_raw, spec.label_col)
        # Realign drop_cols to surviving columns
        drop_tr = [c for c in spec.drop_cols if c in df_train_raw.columns]
        drop_te = [c for c in spec.drop_cols if c in df_test_raw.columns]

        n_tr = df_train_raw[df_train_raw[spec.label_col] == spec.benign_value]
        a_tr = df_train_raw[df_train_raw[spec.label_col] != spec.benign_value]
        n_te = df_test_raw[df_test_raw[spec.label_col] == spec.benign_value]
        a_te = df_test_raw[df_test_raw[spec.label_col] != spec.benign_value]

        labels_train = (
            a_tr[spec.attack_class_col].copy()
            if spec.attack_class_col and spec.attack_class_col in a_tr.columns
            else None
        )
        n_tr = n_tr.drop(columns=drop_tr); a_tr = a_tr.drop(columns=drop_tr)
        n_te = n_te.drop(columns=drop_te); a_te = a_te.drop(columns=drop_te)

        # Cap per class if the corpus is huge
        if spec.cap_per_class:
            cap = spec.cap_per_class
            if len(n_tr) > cap:
                n_tr = n_tr.sample(n=cap, random_state=seed)
            if len(a_tr) > cap:
                idx = a_tr.sample(n=cap, random_state=seed).index
                a_tr = a_tr.loc[idx]
                if labels_train is not None:
                    labels_train = labels_train.loc[idx]
            # Test cap is half of train cap, mirrors the 80/20 ratio
            test_cap = max(1, cap // 4)
            if len(n_te) > test_cap:
                n_te = n_te.sample(n=test_cap, random_state=seed)
            if len(a_te) > test_cap:
                a_te = a_te.sample(n=test_cap, random_state=seed)

        return DatasetSplit(
            spec=spec, normal_train=n_tr, normal_test=n_te,
            attack_train=a_tr, attack_test=a_te,
            attack_class_labels_train=labels_train,
        )

    # ───── single-file datasets ─────
    df = _read_one(paths[0])
    df = _drop_string_cols(df, spec.label_col)
    drop = [c for c in spec.drop_cols if c in df.columns]
    if spec.exclude_classes and spec.attack_class_col in df.columns:
        df = df[~df[spec.attack_class_col].astype(str).isin(spec.exclude_classes)]
    normal_df = df[df[spec.label_col] == spec.benign_value]
    attack_df = df[df[spec.label_col] != spec.benign_value]
    # Capture per-row attack subcategory before dropping
    if spec.attack_class_col and spec.attack_class_col in attack_df.columns:
        attack_class_series = attack_df[spec.attack_class_col].copy()
    else:
        attack_class_series = None
    normal_df = normal_df.drop(columns=drop)
    attack_df = attack_df.drop(columns=drop)

    # ───── Balanced cap: undersample to min(normal_count, attack_count, cap) ─────
    n_normal = len(normal_df)
    n_attack = len(attack_df)
    if spec.cap_per_class:
        cap = min(spec.cap_per_class, n_normal, n_attack)
    else:
        cap = min(n_normal, n_attack)

    # If cap < either class size, undersample that class
    if n_normal > cap:
        normal_df = normal_df.sample(n=cap, random_state=seed)
    if n_attack > cap:
        # Sample attack indices, then realign labels
        idx = attack_df.sample(n=cap, random_state=seed).index
        attack_df = attack_df.loc[idx]
        if attack_class_series is not None:
            attack_class_series = attack_class_series.loc[idx]

    # ───── 80/20 stratified split ─────
    n_train = normal_df.sample(frac=0.8, random_state=seed)
    n_test  = normal_df.drop(n_train.index)
    a_train = attack_df.sample(frac=0.8, random_state=seed)
    a_test  = attack_df.drop(a_train.index)
    labels_train = (
        attack_class_series.loc[a_train.index]
        if attack_class_series is not None else None
    )

    return DatasetSplit(
        spec=spec,
        normal_train=n_train, normal_test=n_test,
        attack_train=a_train, attack_test=a_test,
        attack_class_labels_train=labels_train,
    )


def describe_split(split: DatasetSplit) -> dict:
    """Quick summary printable from a notebook driver."""
    s = split
    return {
        "dataset": s.spec.display_name,
        "normal_train": len(s.normal_train), "normal_test": len(s.normal_test),
        "attack_train": len(s.attack_train), "attack_test": len(s.attack_test),
        "n_features": s.normal_train.shape[1],
        "attack_classes": (
            int(s.attack_class_labels_train.nunique())
            if s.attack_class_labels_train is not None else None
        ),
    }
