"""Standardize attack-precision reports and regenerate ML feature-importance exports.

This script:
1. Generates standardized binary classification reports for all 5 datasets with model/seed headers
2. Regenerates ML feature-importance (permutation, DT, RF) for WUSTL, TON, Bot, UNSW
   (CIC already has these files)
3. Aggregates all standardized outputs into results/policy_refinement_cost_feature_summary.{csv,json}

Inlines logic from 08-edge-benchmark-matched-inference.py to avoid module import issues.
"""

from __future__ import annotations

import argparse
import json
import operator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "data"
# Per-dataset results/ folders live outside the repo, in a sibling directory.
EXTERNAL_RESULTS_ROOT = REPO_ROOT.parent / "iot-llm-hids-pg-results"

OPS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

MODEL_NAME = "claude-haiku-4-5-20251001"
SEED = 42
BEST_ROUNDS = {
    "cic": 4,
    "wustl": 1,
    "ton": 4,
    "bot": 2,
    "unsw": 3,
}


@dataclass(frozen=True)
class Rule:
    feature: str
    op: str
    value: Any


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    dataset_dir: str
    sample_size: int
    one_file_candidates: tuple[str, ...]
    train_file_candidates: tuple[str, ...]
    test_file_candidates: tuple[str, ...]
    one_file_sample_counts: tuple[tuple[Any, int, bool], ...]
    train_file_sample_counts: tuple[tuple[Any, int, bool], ...]
    test_file_sample_counts: tuple[tuple[Any, int, bool], ...]
    label_col: str
    normal_value: Any
    attack_value: Any
    drop_cols: tuple[str, ...]
    split_by_class: bool
    encode_categorical: bool
    notebook_style_test_encoding: bool
    rules: tuple[Rule, ...]


@dataclass
class PreparedDataset:
    config: DatasetConfig
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    categorical_cols: list[str]


DATASETS: dict[str, DatasetConfig] = {
    "cic": DatasetConfig(
        key="cic",
        display_name="CIC-IoT2023",
        dataset_dir="1-cic-iot",
        sample_size=100000,
        one_file_candidates=("data/sample-100000-2.csv",),
        train_file_candidates=(),
        test_file_candidates=(),
        one_file_sample_counts=(),
        train_file_sample_counts=(),
        test_file_sample_counts=(),
        label_col="label",
        normal_value="normal",
        attack_value="attack",
        drop_cols=("label",),
        split_by_class=True,
        encode_categorical=False,
        notebook_style_test_encoding=False,
        rules=(
            Rule("flow_duration", "<", 1),
            Rule("Header_Length", "<", 1000),
            Rule("Duration", "<=", 70),
            Rule("Srate", ">", 50),
            Rule("ack_flag_number", "==", 0),
        ),
    ),
    "wustl": DatasetConfig(
        key="wustl",
        display_name="WUSTL-IIoT",
        dataset_dir="2-wustl-iiot",
        sample_size=100000,
        one_file_candidates=(
            "data/sample-100000-2.csv",
            "data/sample-10000-2.csv",
            str(DATA_ROOT / "wustl-iiot" / "wustl-iiot-population.csv"),
        ),
        train_file_candidates=(),
        test_file_candidates=(),
        one_file_sample_counts=((0, 50000, False), (1, 50000, False)),
        train_file_sample_counts=(),
        test_file_sample_counts=(),
        label_col="Target",
        normal_value=0,
        attack_value=1,
        drop_cols=("Target", "Traffic"),
        split_by_class=True,
        encode_categorical=True,
        notebook_style_test_encoding=False,
        rules=(
            Rule("SrcAddr", "!=", "192.168.0.20"),
            Rule("TotPkts", "==", 2),
            Rule("DstRate", "==", 0),
            Rule("DstPkts", "==", 0),
            Rule("DstLoad", "==", 0),
        ),
    ),
    "ton": DatasetConfig(
        key="ton",
        display_name="TON_IoT",
        dataset_dir="3-ton-iot",
        sample_size=100000,
        one_file_candidates=(
            "data/sample-100000-2.csv",
            "data/sample-10000-2.csv",
            str(DATA_ROOT / "ton-iot" / "ton-iot-population.csv"),
        ),
        train_file_candidates=(),
        test_file_candidates=(),
        one_file_sample_counts=((0, 42040, False), (1, 57960, False)),
        train_file_sample_counts=(),
        test_file_sample_counts=(),
        label_col="label",
        normal_value=0,
        attack_value=1,
        drop_cols=("label", "type"),
        split_by_class=True,
        encode_categorical=True,
        notebook_style_test_encoding=False,
        rules=(
            Rule("dst_port", "==", 4444),
            Rule("conn_state", "==", "OTH"),
            Rule("duration", "==", 0.0),
            Rule("weird_name", "==", "-"),
        ),
    ),
    "bot": DatasetConfig(
        key="bot",
        display_name="Bot-IoT",
        dataset_dir="4-bot-iot",
        sample_size=100000,
        one_file_candidates=(str(DATA_ROOT / "bot-iot" / "bot-iot-population.csv"),),
        train_file_candidates=("data/sample-100000-2_train.csv", "data/sample-10000-2_train.csv"),
        test_file_candidates=("data/sample-100000-2_test.csv", "data/sample-10000-2_test.csv"),
        one_file_sample_counts=(),
        train_file_sample_counts=((0, 40000, True), (1, 40000, False)),
        test_file_sample_counts=((0, 10000, True), (1, 10000, False)),
        label_col="attack",
        normal_value=0,
        attack_value=1,
        drop_cols=("attack", "category", "subcategory"),
        split_by_class=False,
        encode_categorical=True,
        notebook_style_test_encoding=True,
        rules=(
            Rule("saddr", "==", "192.168.100.150"),
            Rule("dport", "==", 80),
            Rule("stddev", ">", 0),
            Rule("N_IN_Conn_P_SrcIP", ">", 20),
            Rule("state_number", "!=", 2),
        ),
    ),
    "unsw": DatasetConfig(
        key="unsw",
        display_name="UNSW-NB15",
        dataset_dir="5-unsw-nb15",
        sample_size=100000,
        one_file_candidates=(),
        train_file_candidates=(
            "data/sample-100000-2_train.csv",
            "data/sample-10000-2_train.csv",
            str(DATA_ROOT / "unsw-nb15" / "UNSW_NB15_training-set.csv"),
        ),
        test_file_candidates=(
            "data/sample-100000-2_test.csv",
            "data/sample-10000-2_test.csv",
            str(DATA_ROOT / "unsw-nb15" / "UNSW_NB15_testing-set.csv"),
        ),
        one_file_sample_counts=(),
        train_file_sample_counts=((0, 40000, False), (1, 40000, False)),
        test_file_sample_counts=((0, 10000, False), (1, 10000, False)),
        label_col="label",
        normal_value=0,
        attack_value=1,
        drop_cols=("id", "attack_cat", "label"),
        split_by_class=False,
        encode_categorical=True,
        notebook_style_test_encoding=True,
        rules=(
            Rule("proto", "!=", "tcp"),
            Rule("dur", "<", 0.001),
            Rule("state", "==", "INT"),
            Rule("dpkts", "==", 0),
            Rule("dttl", "==", 0),
        ),
    ),
}


def candidate_path(base: Path, candidate: str) -> Path:
    expanded = Path(candidate).expanduser()
    if expanded.is_absolute():
        return expanded
    return base / expanded


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT.parent))
        except ValueError:
            return str(path)


def find_first_existing(base: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = candidate_path(base, candidate)
        if path.exists():
            return path
    return None


def sample_by_counts(
    df: pd.DataFrame,
    label_col: str,
    sample_counts: tuple[tuple[Any, int, bool], ...],
) -> pd.DataFrame:
    if not sample_counts:
        return df
    sampled_parts = []
    for label_value, count, replace in sample_counts:
        part = df[df[label_col] == label_value]
        if part.empty:
            raise ValueError(f"No rows found for {label_col} == {label_value!r}")
        if len(part) < count and not replace:
            raise ValueError(
                f"Need {count} rows for {label_col} == {label_value!r}, "
                f"but only {len(part)} are available"
            )
        sampled_parts.append(part.sample(n=count, random_state=42, replace=replace))
    return pd.concat(sampled_parts, ignore_index=True)


def sample_train_test_from_population(
    df: pd.DataFrame,
    label_col: str,
    train_counts: tuple[tuple[Any, int, bool], ...],
    test_counts: tuple[tuple[Any, int, bool], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for label_value, train_count, train_replace in train_counts:
        matching_tests = [item for item in test_counts if item[0] == label_value]
        if not matching_tests:
            raise ValueError(f"Missing test sample count for {label_col} == {label_value!r}")
        _, test_count, test_replace = matching_tests[0]
        part = df[df[label_col] == label_value]
        if part.empty:
            raise ValueError(f"No rows found for {label_col} == {label_value!r}")
        if len(part) < train_count and not train_replace:
            raise ValueError(
                f"Need {train_count} train rows for {label_col} == {label_value!r}, "
                f"but only {len(part)} are available"
            )
        train_part = part.sample(n=train_count, random_state=42, replace=train_replace)
        if train_replace or test_replace:
            test_pool = part
        else:
            test_pool = part.drop(train_part.index)
        if len(test_pool) < test_count and not test_replace:
            raise ValueError(
                f"Need {test_count} test rows for {label_col} == {label_value!r}, "
                f"but only {len(test_pool)} are available after train sampling"
            )
        test_part = test_pool.sample(n=test_count, random_state=42, replace=test_replace)
        train_parts.append(train_part)
        test_parts.append(test_part)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def load_raw_frames(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    dataset_base = REPO_ROOT / config.dataset_dir
    train_path = find_first_existing(dataset_base, config.train_file_candidates)
    test_path = find_first_existing(dataset_base, config.test_file_candidates)
    if train_path and test_path:
        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)
        train_df = sample_by_counts(train_df, config.label_col, config.train_file_sample_counts)
        test_df = sample_by_counts(test_df, config.label_col, config.test_file_sample_counts)
        return train_df, test_df, f"{display_path(train_path)} + {display_path(test_path)}"
    one_file_path = find_first_existing(dataset_base, config.one_file_candidates)
    if one_file_path:
        df = pd.read_csv(one_file_path, low_memory=False)
        if config.train_file_sample_counts and config.test_file_sample_counts:
            train_df, test_df = sample_train_test_from_population(
                df,
                config.label_col,
                config.train_file_sample_counts,
                config.test_file_sample_counts,
            )
            return train_df, test_df, display_path(one_file_path)
        df = sample_by_counts(df, config.label_col, config.one_file_sample_counts)
        return df, None, display_path(one_file_path)
    checked = [
        display_path(candidate_path(dataset_base, candidate))
        for candidate in (
            config.one_file_candidates
            + config.train_file_candidates
            + config.test_file_candidates
        )
    ]
    raise FileNotFoundError(
        f"No local CSV found for {config.display_name}. Checked: {', '.join(checked)}"
    )


def relabel_cic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[:, "label"] = np.where(out["label"] == "BenignTraffic", "normal", "attack")
    return out


def split_one_file(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.key == "cic":
        df = relabel_cic(df)
    if not config.split_by_class:
        raise ValueError(f"{config.key} is configured for train/test files, not one-file split")
    normal_df = df[df[config.label_col] == config.normal_value]
    attack_df = df[df[config.label_col] == config.attack_value]
    if normal_df.empty or attack_df.empty:
        raise ValueError(
            f"{config.display_name} split failed: normal={len(normal_df)}, attack={len(attack_df)}"
        )
    normal_train = normal_df.sample(frac=0.8, random_state=42)
    normal_test = normal_df.drop(normal_train.index)
    attack_train = attack_df.sample(frac=0.8, random_state=42)
    attack_test = attack_df.drop(attack_train.index)
    train_df = pd.concat([normal_train, attack_train])
    test_df = pd.concat([normal_test, attack_test])
    return train_df, test_df


def split_train_test_files(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_normal = train_df[train_df[config.label_col] == config.normal_value]
    train_attack = train_df[train_df[config.label_col] == config.attack_value]
    test_normal = test_df[test_df[config.label_col] == config.normal_value]
    test_attack = test_df[test_df[config.label_col] == config.attack_value]
    return pd.concat([train_normal, train_attack]), pd.concat([test_normal, test_attack])


def encode_notebook_style(
    train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_cols: list[str], independent_test: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in categorical_cols:
        train_encoder = LabelEncoder()
        train_out[col] = train_encoder.fit_transform(train_out[col].astype(str))
        if independent_test:
            test_encoder = LabelEncoder()
            test_out[col] = test_encoder.fit_transform(test_out[col].astype(str))
        else:
            values = pd.concat([train_df[col], test_df[col]], ignore_index=True).astype(str)
            encoder = LabelEncoder().fit(values)
            train_out[col] = encoder.transform(train_df[col].astype(str))
            test_out[col] = encoder.transform(test_df[col].astype(str))
    return train_out, test_out


def prepare_dataset(config: DatasetConfig) -> PreparedDataset:
    raw_a, raw_b, _source = load_raw_frames(config)
    if raw_b is None:
        raw_train, raw_test = split_one_file(raw_a, config)
    else:
        raw_train, raw_test = split_train_test_files(raw_a, raw_b, config)
    categorical_cols = []
    train_for_ml = raw_train.copy()
    test_for_ml = raw_test.copy()
    if config.encode_categorical:
        categorical_cols = train_for_ml.select_dtypes(include=["object"]).columns.tolist()
        train_for_ml, test_for_ml = encode_notebook_style(
            train_for_ml,
            test_for_ml,
            categorical_cols,
            independent_test=config.notebook_style_test_encoding,
        )
    missing_drops = [col for col in config.drop_cols if col not in train_for_ml.columns]
    if missing_drops:
        raise KeyError(f"{config.display_name} missing expected drop columns: {missing_drops}")
    x_train = train_for_ml.drop(columns=list(config.drop_cols))
    y_train = train_for_ml[config.label_col]
    x_test = test_for_ml.drop(columns=list(config.drop_cols))
    y_test = test_for_ml[config.label_col]
    return PreparedDataset(
        config=config,
        raw_train=raw_train,
        raw_test=raw_test,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        categorical_cols=categorical_cols,
    )


def rule_to_series(rule: Rule, rows: pd.DataFrame) -> pd.Series:
    if rule.feature not in rows.columns:
        raise KeyError(f"Rule feature {rule.feature!r} not present in test rows")
    series = rows[rule.feature]
    value = rule.value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        series = pd.to_numeric(series, errors="coerce")
    elif isinstance(value, str):
        series = series.astype(str)
    return OPS[rule.op](series, value)


def llm_rule_predict(rows: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    votes = [rule_to_series(rule, rows).to_numpy(dtype=bool) for rule in config.rules]
    if config.key == "ton":
        votes.append(((rows["dst_bytes"] == 0) & (rows["src_bytes"] == 0)).to_numpy(dtype=bool))
    attack_votes = np.vstack(votes).sum(axis=0)
    return np.where(attack_votes > (len(votes) / 2), config.attack_value, config.normal_value)


def save_attack_precision_report(
    config: DatasetConfig,
    y_test: pd.Series,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Save binary classification report with metadata header."""
    report = classification_report(y_test, y_pred, digits=4, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    header = f"""dataset: {config.display_name}
sample_size: {config.sample_size}
seed: {SEED}
model_name: {MODEL_NAME}
rule_file: ../results/llm-rule-features-{config.sample_size}-{MODEL_NAME}.json
date: {datetime.now().strftime('%Y-%m-%d')}
---
"""
    with open(out_path, "w") as f:
        f.write(header)
        f.write("Classification Report\n")
        f.write(report)
        f.write("\n\nConfusion Matrix\n")
        f.write(str(cm))


def rule_to_record(rule: Rule) -> dict[str, Any]:
    return {
        "feature_name": rule.feature,
        "op": rule.op,
        "value": rule.value,
        "rule": f"{rule.feature} {rule.op} {rule.value}",
    }


def export_llm_rule_features(configs: list[DatasetConfig]) -> Path:
    """Persist the exact LLM rule features used by standardized binary reports."""
    sample_sizes = {config.sample_size for config in configs}
    if len(sample_sizes) != 1:
        raise ValueError(f"Expected one sample size, got {sorted(sample_sizes)}")
    sample_size = sample_sizes.pop()
    out_path = REPO_ROOT / "results" / f"llm-rule-features-{sample_size}-{MODEL_NAME}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in configs:
        rows.append(
            {
                "dataset": config.display_name,
                "seed": SEED,
                "round": BEST_ROUNDS[config.key],
                "rule_id": (
                    f"standardize-results:{config.key}:seed{SEED}:"
                    f"round{BEST_ROUNDS[config.key]}:{MODEL_NAME}"
                ),
                "feature_names": [rule.feature for rule in config.rules],
                "rules": [rule_to_record(rule) for rule in config.rules],
            }
        )

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    return out_path


def compute_gini_importance(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_runs: int = 100,
) -> dict[str, float]:
    """Compute averaged gini-based feature importance over n_runs."""
    feature_importances = {}
    for i in range(n_runs):
        model = model_class(random_state=42 + i)
        model.fit(X_train, y_train)
        sorted_features = sorted(
            zip(model.feature_importances_, model.feature_names_in_), reverse=True
        )
        for importance, name in sorted_features:
            if name in feature_importances:
                feature_importances[name].append(importance)
            else:
                feature_importances[name] = [importance]

    average_importances = {}
    for name, importances in feature_importances.items():
        average_importances[name] = sum(importances) / len(importances)
    return average_importances


def compute_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute permutation-based feature importance as in notebooks."""
    models = {
        "LR": LogisticRegression(random_state=42, max_iter=1000),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
    }

    importances_df = pd.DataFrame(index=feature_names)

    for model_name, model in models.items():
        print(f"  Computing {model_name} permutation importance...")
        model.fit(X_train, y_train)
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances_df[model_name] = result.importances_mean

        min_val = importances_df[model_name].abs().min()
        max_val = importances_df[model_name].abs().max()
        if max_val - min_val == 0:
            importances_df[model_name + "_minmax_normalized"] = importances_df[model_name]
        else:
            importances_df[model_name + "_minmax_normalized"] = (
                importances_df[model_name].abs() - min_val
            ) / (max_val - min_val)

    normalized_columns = [col for col in importances_df.columns if "_minmax_normalized" in col]
    importances_df["Mean"] = importances_df[normalized_columns].mean(axis=1)
    importances_sorted_df = importances_df.sort_values(by="Mean", ascending=False)
    return importances_sorted_df


def parse_attack_precision_report(path: Path, attack_label: str = "attack") -> dict[str, Any]:
    """Parse metadata header and attack/macro-avg rows from a standardized report file.

    attack_label must match the first token of the attack class row in the report.
    CIC uses the string "attack"; other datasets use "1" (numeric label).
    """
    lines = path.read_text().splitlines()
    meta: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
        if ": " in line:
            key, val = line.split(": ", 1)
            meta[key.strip()] = val.strip()

    attack_prec = attack_recall = attack_f1 = test_macro_f1 = None
    for line in lines[body_start:]:
        stripped = line.strip()
        parts = stripped.split()
        if not parts:
            continue
        if parts[0] == attack_label and len(parts) >= 4:
            attack_prec, attack_recall, attack_f1 = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        elif stripped.startswith("macro avg") and len(parts) >= 5:
            test_macro_f1 = float(parts[4])

    return {
        "dataset": meta.get("dataset"),
        "sample_size": int(meta.get("sample_size", 0)),
        "seed": int(meta.get("seed", 0)),
        "model_name": meta.get("model_name"),
        "final_rule_file": meta.get("rule_file"),
        "test_macro_f1": test_macro_f1,
        "attack_precision": attack_prec,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
    }


def parse_policy_refinement_summary(path: Path, best_round: int) -> dict[str, Any]:
    """Extract train F1 and cumulative token counts from a policy-refinement-summary JSON.

    Returns None for each field when the file is absent or all token counts are zero
    (which indicates the data was not recorded, not that zero tokens were used).
    """
    if not path.exists():
        return {
            "train_macro_f1": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    data = json.loads(path.read_text())
    rounds = data.get("rounds", [])
    round_idx = best_round - 1
    train_macro_f1 = rounds[round_idx].get("macro_f1") if round_idx < len(rounds) else None
    prompt = sum(r.get("round_prompt_tokens", 0) for r in rounds)
    completion = sum(r.get("round_completion_tokens", 0) for r in rounds)
    total = sum(r.get("round_total_tokens", 0) for r in rounds)
    if prompt == 0 and completion == 0 and total == 0:
        prompt = completion = total = None
    return {
        "train_macro_f1": train_macro_f1,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def parse_permutation_top5(path: Path, col: str) -> list[str]:
    """Return top-5 feature names from a permutation-importance CSV sorted by col descending."""
    if not path.exists():
        return []
    df = pd.read_csv(path, index_col=0)
    if col not in df.columns:
        return []
    return df[col].sort_values(ascending=False).head(5).index.tolist()


def generate_summary_artifact(configs: list[DatasetConfig]) -> None:
    """Aggregate all standardized outputs into a single per-dataset summary CSV + JSON.

    Reads from whatever result files currently exist on disk; missing files produce
    None values rather than raising errors. Output goes to
    results/policy_refinement_cost_feature_summary.{csv,json}.
    """
    sample_sizes = {c.sample_size for c in configs}
    sample_size = next(iter(sample_sizes)) if len(sample_sizes) == 1 else 100000
    llm_features_path = REPO_ROOT / "results" / f"llm-rule-features-{sample_size}-{MODEL_NAME}.json"
    llm_features_map: dict[str, list[str]] = {}
    if llm_features_path.exists():
        for entry in json.loads(llm_features_path.read_text()):
            llm_features_map[entry["dataset"]] = entry["feature_names"]

    rows: list[dict[str, Any]] = []
    for config in configs:
        results_dir = EXTERNAL_RESULTS_ROOT / config.dataset_dir
        report_path = (
            results_dir
            / f"attack-precision-report-{config.sample_size}-seed42-claude-haiku-4-5.txt"
        )
        if not report_path.exists():
            print(f"  WARN: no attack report for {config.display_name} — row omitted")
            continue

        report = parse_attack_precision_report(report_path, attack_label=str(config.attack_value))
        refinement = parse_policy_refinement_summary(
            results_dir
            / "llm"
            / f"policy-refinement-summary-{config.sample_size}-{MODEL_NAME}.json",
            BEST_ROUNDS[config.key],
        )
        top_llm = llm_features_map.get(config.display_name, [])
        perm_path = results_dir / f"feature-importance-{config.sample_size}-permutation.csv"
        top_rf = parse_permutation_top5(perm_path, "RF_minmax_normalized")
        top_dt = parse_permutation_top5(perm_path, "DT_minmax_normalized")
        llm_set = set(top_llm)

        rows.append(
            {
                "dataset": report["dataset"],
                "sample_size": report["sample_size"],
                "seed": report["seed"],
                "model_name": report["model_name"],
                "round": BEST_ROUNDS[config.key],
                "train_macro_f1": refinement["train_macro_f1"],
                "test_macro_f1": report["test_macro_f1"],
                "attack_precision": report["attack_precision"],
                "attack_recall": report["attack_recall"],
                "attack_f1": report["attack_f1"],
                "prompt_tokens": refinement["prompt_tokens"],
                "completion_tokens": refinement["completion_tokens"],
                "total_tokens": refinement["total_tokens"],
                "final_rule_file": report["final_rule_file"],
                "top_llm_features": "; ".join(top_llm),
                "top_rf_features": "; ".join(top_rf),
                "top_dt_features": "; ".join(top_dt),
                "rf_overlap": len(llm_set & set(top_rf)),
                "dt_overlap": len(llm_set & set(top_dt)),
            }
        )

    if not rows:
        print("No result files found; summary artifact not written.")
        return

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "policy_refinement_cost_feature_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved summary CSV  → {csv_path.relative_to(REPO_ROOT)}")

    json_path = out_dir / "policy_refinement_cost_feature_summary.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Saved summary JSON → {json_path.relative_to(REPO_ROOT)}")


def process_dataset(config: DatasetConfig) -> None:
    """Process one dataset: save attack report and (optionally) feature importance."""
    dataset_key = config.key
    results_dir = EXTERNAL_RESULTS_ROOT / config.dataset_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing {config.display_name}")
    print(f"{'='*70}")

    try:
        prepared = prepare_dataset(config)
        print(f"Loaded: {len(prepared.x_train)} train, {len(prepared.x_test)} test rows")
    except FileNotFoundError as exc:
        print(f"SKIPPED: {exc}")
        return

    # Task 1: Save attack precision report
    print(f"Computing LLM rule predictions...")
    y_pred = llm_rule_predict(prepared.raw_test, config)
    attack_report_path = results_dir / f"attack-precision-report-{config.sample_size}-seed42-claude-haiku-4-5.txt"
    save_attack_precision_report(config, prepared.y_test, y_pred, attack_report_path)
    print(f"Saved attack report to {attack_report_path.relative_to(REPO_ROOT)}")

    # Task 2: Feature importance (skip CIC, it already has these files)
    if dataset_key == "cic":
        print(f"Skipping feature importance (already exists for CIC)")
        return

    print(f"Computing feature importance (DT, RF gini + permutation)...")

    # DT gini importance
    dt_importance = compute_gini_importance(DecisionTreeClassifier, prepared.x_train, prepared.y_train)
    dt_top5 = sorted(dt_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    dt_path = results_dir / f"feature-importance-{config.sample_size}-dt.txt"
    with open(dt_path, "w") as f:
        f.write("\n".join([str(feature) for feature in dt_top5]))
    print(f"Saved DT top-5 to {dt_path.relative_to(REPO_ROOT)}")

    # RF gini importance
    rf_importance = compute_gini_importance(RandomForestClassifier, prepared.x_train, prepared.y_train)
    rf_top5 = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    rf_path = results_dir / f"feature-importance-{config.sample_size}-rf.txt"
    with open(rf_path, "w") as f:
        f.write("\n".join([str(feature) for feature in rf_top5]))
    print(f"Saved RF top-5 to {rf_path.relative_to(REPO_ROOT)}")

    # Permutation importance
    print(f"Computing permutation importance (5 models)...")
    importances_df = compute_permutation_importance(
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
        prepared.x_test.columns.tolist(),
    )
    perm_path = results_dir / f"feature-importance-{config.sample_size}-permutation.csv"
    importances_df.to_csv(perm_path)
    print(f"Saved permutation importance to {perm_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose CSV files are not present locally.",
    )
    parser.add_argument(
        "--export-rule-features-only",
        action="store_true",
        help="Only write the matched LLM rule-feature JSON used by standardized reports.",
    )
    args = parser.parse_args()

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    configs = [DATASETS[key] for key in keys]

    if args.export_rule_features_only:
        out_path = export_llm_rule_features(configs)
        print(f"Saved LLM rule features to {out_path.relative_to(REPO_ROOT)}")
        return

    processed_configs: list[DatasetConfig] = []
    for config in configs:
        if args.skip_missing:
            try:
                raw_a, raw_b, _ = load_raw_frames(config)
            except FileNotFoundError:
                print(f"\nSkipping {config.display_name}: data files not found")
                continue
        process_dataset(config)
        processed_configs.append(config)

    if processed_configs:
        out_path = export_llm_rule_features(processed_configs)
        print(f"Saved LLM rule features to {out_path.relative_to(REPO_ROOT)}")

    print(f"\n{'='*70}")
    print("Generating consolidated summary artifact...")
    print(f"{'='*70}")
    generate_summary_artifact(list(DATASETS.values()))

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
