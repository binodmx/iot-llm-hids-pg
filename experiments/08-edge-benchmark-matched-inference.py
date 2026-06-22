"""Matched DT/RF/LLM-rule inference benchmark.

This script replaces the ad hoc timing cells in the dataset notebooks with a
single reproducible benchmark.  It trains one Decision Tree and one Random
Forest on the same rows used by the binary notebooks, evaluates the final
hand-transcribed LLM rule policy on the same test rows, and repeats each timed
condition at least 30 times.

By default the timed region measures inference only.  Pass
``--include-preprocessing`` to include test-set preprocessing in the timed
region; model fitting is never included.
"""

from __future__ import annotations

import argparse
import json
import operator
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "results" / "timing"


OPS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
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
        # From the saved final generated-rule file.  The compound
        # dst_bytes/src_bytes rule is implemented in ton_rule_votes().
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
        # Mirrors the notebook preprocessing for this dataset: train and test
        # object columns were label-encoded independently before ML inference.
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
        # Saved final TON rule: attack if dst_bytes == 0 and src_bytes == 0.
        votes.append(((rows["dst_bytes"] == 0) & (rows["src_bytes"] == 0)).to_numpy(dtype=bool))

    attack_votes = np.vstack(votes).sum(axis=0)
    return np.where(attack_votes > (len(votes) / 2), config.attack_value, config.normal_value)


def train_models(prepared: PreparedDataset) -> tuple[DecisionTreeClassifier, RandomForestClassifier]:
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    dt.fit(prepared.x_train, prepared.y_train)
    rf.fit(prepared.x_train, prepared.y_train)
    return dt, rf


def summarize_timings(samples_ms_per_row: list[float]) -> dict[str, float]:
    q1, q3 = np.percentile(samples_ms_per_row, [25, 75])
    return {
        "median_ms_per_row": float(statistics.median(samples_ms_per_row)),
        "iqr_ms_per_row": float(q3 - q1),
        "q1_ms_per_row": float(q1),
        "q3_ms_per_row": float(q3),
    }


def benchmark_callable(
    fn: Callable[[], Any], rows: int, repeats: int, warmups: int
) -> tuple[dict[str, float], list[float]]:
    for _ in range(warmups):
        fn()

    samples = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        fn()
        elapsed_ns = time.perf_counter_ns() - start_ns
        samples.append((elapsed_ns / 1_000_000) / rows)
    return summarize_timings(samples), samples


def machine_metadata() -> dict[str, str]:
    return {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "node": platform.node(),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
    }


def benchmark_dataset(
    config: DatasetConfig, repeats: int, warmups: int, include_preprocessing: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_a, raw_b, source = load_raw_frames(config)
    prepared = prepare_dataset(config)
    dt, rf = train_models(prepared)
    rows = len(prepared.x_test)
    meta = machine_metadata()
    meta.update(
        {
            "dataset": config.display_name,
            "dataset_key": config.key,
            "source": source,
            "rows": rows,
            "repeats": repeats,
            "warmups": warmups,
            "timing_includes_preprocessing": include_preprocessing,
            "categorical_columns_encoded": ",".join(prepared.categorical_cols),
        }
    )

    def preprocess_for_timing() -> PreparedDataset:
        if raw_b is None:
            raw_train, raw_test = split_one_file(raw_a, config)
        else:
            raw_train, raw_test = split_train_test_files(raw_a, raw_b, config)

        train_for_ml = raw_train.copy()
        test_for_ml = raw_test.copy()
        categorical_cols: list[str] = []
        if config.encode_categorical:
            categorical_cols = train_for_ml.select_dtypes(include=["object"]).columns.tolist()
            train_for_ml, test_for_ml = encode_notebook_style(
                train_for_ml,
                test_for_ml,
                categorical_cols,
                independent_test=config.notebook_style_test_encoding,
            )
        x_test = test_for_ml.drop(columns=list(config.drop_cols))
        y_test = test_for_ml[config.label_col]
        return PreparedDataset(
            config=config,
            raw_train=raw_train,
            raw_test=raw_test,
            x_train=prepared.x_train,
            y_train=prepared.y_train,
            x_test=x_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
        )

    conditions: list[tuple[str, Callable[[], Any]]] = []
    if include_preprocessing:
        conditions = [
            ("DT", lambda: dt.predict(preprocess_for_timing().x_test)),
            ("RF", lambda: rf.predict(preprocess_for_timing().x_test)),
            ("LLM_rule_policy", lambda: llm_rule_predict(preprocess_for_timing().raw_test, config)),
        ]
    else:
        conditions = [
            ("DT", lambda: dt.predict(prepared.x_test)),
            ("RF", lambda: rf.predict(prepared.x_test)),
            ("LLM_rule_policy", lambda: llm_rule_predict(prepared.raw_test, config)),
        ]

    results = []
    raw_samples: dict[str, list[float]] = {}
    for condition, fn in conditions:
        summary, samples = benchmark_callable(fn, rows=rows, repeats=repeats, warmups=warmups)
        raw_samples[condition] = samples
        row = {**meta, "condition": condition, **summary}
        results.append(row)
        print(
            f"{config.display_name:12s} {condition:15s} "
            f"median={row['median_ms_per_row']:.6f} ms/row "
            f"IQR={row['iqr_ms_per_row']:.6f} rows={rows:,}"
        )

    return results, {"metadata": meta, "samples_ms_per_row": raw_samples}


def write_outputs(rows: list[dict[str, Any]], raw_runs: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"matched-inference-benchmark-{stamp}.csv"
    json_path = out_dir / f"matched-inference-benchmark-{stamp}.json"
    latest_csv = out_dir / "matched-inference-benchmark-latest.csv"
    latest_json = out_dir / "matched-inference-benchmark-latest.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(rows).to_csv(latest_csv, index=False)
    payload = {"summary": rows, "raw_runs": raw_runs}
    json_path.write_text(json.dumps(payload, indent=2))
    latest_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Timed repetitions per condition. Must be at least 30.",
    )
    parser.add_argument("--warmups", type=int, default=3, help="Untimed warmup repetitions.")
    parser.add_argument(
        "--include-preprocessing",
        action="store_true",
        help="Include test-set preprocessing in the timed region. Training remains excluded.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose CSV files are not present locally.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.repeats < 30:
        raise ValueError("--repeats must be at least 30 for the paper benchmark")

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_rows: list[dict[str, Any]] = []
    all_raw_runs: list[dict[str, Any]] = []

    for key in keys:
        config = DATASETS[key]
        try:
            rows, raw_runs = benchmark_dataset(
                config=config,
                repeats=args.repeats,
                warmups=args.warmups,
                include_preprocessing=args.include_preprocessing,
            )
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"Skipping {config.display_name}: {exc}")
                continue
            raise
        all_rows.extend(rows)
        all_raw_runs.append(raw_runs)

    if not all_rows:
        raise RuntimeError("No benchmark results were produced")
    write_outputs(all_rows, all_raw_runs, args.out_dir)


if __name__ == "__main__":
    main()
