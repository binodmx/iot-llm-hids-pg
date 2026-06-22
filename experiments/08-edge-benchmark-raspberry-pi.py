"""Raspberry Pi edge inference benchmark for deployed IDS policies.

This benchmark reuses the dataset preparation and DT/RF baseline semantics
from ``08-edge-benchmark-matched-inference.py`` while measuring the deployment path
needed for Raspberry Pi 4/5 devices.  The Pi workload is CSV replay of
precomputed flow features.  It does not run LLM generation, Chroma retrieval,
or notebook code on-device.

Telemetry is software-only.  The output reports latency, memory, CPU time,
frequency, temperature, and throttling/resource proxies; it does not report
true watts or joules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import operator
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

try:
    import psutil
except ImportError:  # pragma: no cover - exercised on minimal Pi installs.
    psutil = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = REPO_ROOT / "experiments" / "policies" / "llm-rule-policies.json"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "results" / "edge"
MATCHED_BENCHMARK_PATH = REPO_ROOT / "experiments" / "08-edge-benchmark-matched-inference.py"

OPS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def load_matched_benchmark_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "matched_inference_benchmark", MATCHED_BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {MATCHED_BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_matched_benchmark_module()
DATASETS = bench.DATASETS


@dataclass(frozen=True)
class RuleSpec:
    feature: str
    op: str
    value: Any
    value_type: str


@dataclass(frozen=True)
class CompoundRuleSpec:
    name: str
    logic: str
    conditions: tuple[RuleSpec, ...]


@dataclass(frozen=True)
class PolicySpec:
    dataset_key: str
    display_name: str
    label_col: str
    normal_value: Any
    attack_value: Any
    majority_vote_policy: str
    rules: tuple[RuleSpec, ...]
    compound_rules: tuple[CompoundRuleSpec, ...]


def parse_rule(raw: dict[str, Any]) -> RuleSpec:
    return RuleSpec(
        feature=str(raw["feature"]),
        op=str(raw["op"]),
        value=raw["value"],
        value_type=str(raw.get("value_type", "string")),
    )


def load_policies(path: Path) -> dict[str, PolicySpec]:
    payload = json.loads(path.read_text())
    policies: dict[str, PolicySpec] = {}
    for raw_policy in payload["policies"]:
        compounds = []
        for raw_compound in raw_policy.get("compound_rules", []):
            compounds.append(
                CompoundRuleSpec(
                    name=str(raw_compound["name"]),
                    logic=str(raw_compound.get("logic", "all")),
                    conditions=tuple(parse_rule(item) for item in raw_compound["conditions"]),
                )
            )
        policy = PolicySpec(
            dataset_key=str(raw_policy["dataset_key"]),
            display_name=str(raw_policy["display_name"]),
            label_col=str(raw_policy["label_col"]),
            normal_value=raw_policy["normal_value"],
            attack_value=raw_policy["attack_value"],
            majority_vote_policy=str(raw_policy["majority_vote_policy"]),
            rules=tuple(parse_rule(item) for item in raw_policy["rules"]),
            compound_rules=tuple(compounds),
        )
        if policy.majority_vote_policy != "attack_if_attack_votes_gt_half":
            raise ValueError(f"Unsupported majority vote policy: {policy.majority_vote_policy}")
        policies[policy.dataset_key] = policy
    return policies


def coerce_series(series: pd.Series, rule: RuleSpec) -> pd.Series:
    if rule.value_type == "number":
        return pd.to_numeric(series, errors="coerce")
    if rule.value_type == "string":
        return series.astype(str)
    raise ValueError(f"Unsupported value_type {rule.value_type!r} for {rule.feature}")


def rule_to_series(rule: RuleSpec, rows: pd.DataFrame) -> pd.Series:
    if rule.feature not in rows.columns:
        raise KeyError(f"Rule feature {rule.feature!r} not present in rows")
    series = coerce_series(rows[rule.feature], rule)
    return OPS[rule.op](series, rule.value)


def coerce_scalar(value: Any, rule: RuleSpec) -> Any:
    if rule.value_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan
    if rule.value_type == "string":
        return str(value)
    raise ValueError(f"Unsupported value_type {rule.value_type!r} for {rule.feature}")


def rule_to_bool(rule: RuleSpec, row: pd.Series) -> bool:
    if rule.feature not in row.index:
        raise KeyError(f"Rule feature {rule.feature!r} not present in row")
    return bool(OPS[rule.op](coerce_scalar(row[rule.feature], rule), rule.value))


def compound_to_series(compound: CompoundRuleSpec, rows: pd.DataFrame) -> pd.Series:
    parts = [rule_to_series(rule, rows).to_numpy(dtype=bool) for rule in compound.conditions]
    if not parts:
        raise ValueError(f"Compound rule {compound.name!r} has no conditions")
    stacked = np.vstack(parts)
    if compound.logic == "all":
        return pd.Series(stacked.all(axis=0), index=rows.index)
    if compound.logic == "any":
        return pd.Series(stacked.any(axis=0), index=rows.index)
    raise ValueError(f"Unsupported compound logic {compound.logic!r}")


def compound_to_bool(compound: CompoundRuleSpec, row: pd.Series) -> bool:
    values = [rule_to_bool(rule, row) for rule in compound.conditions]
    if compound.logic == "all":
        return all(values)
    if compound.logic == "any":
        return any(values)
    raise ValueError(f"Unsupported compound logic {compound.logic!r}")


def llm_rule_predict_batch(rows: pd.DataFrame, policy: PolicySpec) -> np.ndarray:
    votes = [rule_to_series(rule, rows).to_numpy(dtype=bool) for rule in policy.rules]
    votes.extend(
        compound_to_series(compound, rows).to_numpy(dtype=bool)
        for compound in policy.compound_rules
    )
    attack_votes = np.vstack(votes).sum(axis=0)
    return np.where(attack_votes > (len(votes) / 2), policy.attack_value, policy.normal_value)


def llm_rule_predict_streaming(rows: pd.DataFrame, policy: PolicySpec) -> np.ndarray:
    predictions = []
    total_votes = len(policy.rules) + len(policy.compound_rules)
    for _, row in rows.iterrows():
        attack_votes = sum(rule_to_bool(rule, row) for rule in policy.rules)
        attack_votes += sum(compound_to_bool(compound, row) for compound in policy.compound_rules)
        predictions.append(policy.attack_value if attack_votes > (total_votes / 2) else policy.normal_value)
    return np.asarray(predictions)


def batched_slices(rows: pd.DataFrame, batch_size: int | None) -> list[pd.DataFrame]:
    if batch_size is None or batch_size <= 0 or batch_size >= len(rows):
        return [rows]
    return [rows.iloc[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def predict_llm(rows: pd.DataFrame, policy: PolicySpec, mode: str, batch_size: int | None) -> np.ndarray:
    if mode == "streaming":
        return llm_rule_predict_streaming(rows, policy)
    return np.concatenate(
        [llm_rule_predict_batch(part, policy) for part in batched_slices(rows, batch_size)]
    )


def predict_model(
    model: DecisionTreeClassifier | RandomForestClassifier,
    rows: pd.DataFrame,
    mode: str,
    batch_size: int | None,
) -> np.ndarray:
    if mode == "streaming":
        return np.concatenate([model.predict(rows.iloc[[idx]]) for idx in range(len(rows))])
    return np.concatenate([model.predict(part) for part in batched_slices(rows, batch_size)])


def parse_batch_size(value: str) -> int | None:
    if value.lower() == "full":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be a positive integer or 'full'")
    return parsed


def current_rss_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def read_first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return path.read_text().strip()
    return None


def run_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=1.0)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def read_cpu_temp_c() -> float | None:
    raw = read_first_existing([Path("/sys/class/thermal/thermal_zone0/temp")])
    if raw:
        try:
            return float(raw) / 1000.0
        except ValueError:
            pass
    raw = run_command(["vcgencmd", "measure_temp"])
    if raw and "temp=" in raw:
        try:
            return float(raw.split("temp=")[1].split("'")[0])
        except (IndexError, ValueError):
            return None
    return None


def read_arm_freq_mhz() -> float | None:
    raw = run_command(["vcgencmd", "measure_clock", "arm"])
    if raw and "=" in raw:
        try:
            return float(raw.split("=")[1]) / 1_000_000.0
        except ValueError:
            pass
    raw = read_first_existing([Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")])
    if raw:
        try:
            return float(raw) / 1000.0
        except ValueError:
            return None
    return None


def read_throttled_raw() -> str | None:
    return run_command(["vcgencmd", "get_throttled"])


def throttled_value(raw: str | None) -> int | None:
    if not raw or "=" not in raw:
        return None
    try:
        return int(raw.split("=")[1], 16)
    except ValueError:
        return None


def read_governors() -> str | None:
    governors = sorted(
        {
            path.read_text().strip()
            for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")
            if path.exists()
        }
    )
    return ",".join(governors) if governors else None


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        preferred_keys = ("Model", "Hardware", "model name")
        lines = cpuinfo.read_text(errors="ignore").splitlines()
        for key in preferred_keys:
            for line in lines:
                if line.startswith(key) and ":" in line:
                    return line.split(":", 1)[1].strip()
    return platform.processor() or None


def total_ram_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def summarize_optional(values: list[float | int | None]) -> dict[str, float | None]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(min(numeric)),
        "median": float(statistics.median(numeric)),
        "max": float(max(numeric)),
    }


class TelemetrySampler:
    def __init__(self, hz: float) -> None:
        self.hz = hz
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rss_samples: list[int | None] = []
        self.temp_samples: list[float | None] = []
        self.freq_samples: list[float | None] = []
        self.process_cpu_samples: list[float | None] = []
        self.system_cpu_samples: list[float | None] = []
        self.throttled_samples: list[int | None] = []
        self.process_time_start = 0.0
        self.process_time_end = 0.0
        self.wall_start = 0.0
        self.wall_end = 0.0

    def __enter__(self) -> "TelemetrySampler":
        self.process_time_start = time.process_time()
        self.wall_start = time.perf_counter()
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(None)
            psutil.cpu_percent(None)
        self._sample()
        if self.hz > 0:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._sample()
        self.wall_end = time.perf_counter()
        self.process_time_end = time.process_time()

    def _run(self) -> None:
        interval = 1.0 / self.hz
        while not self.stop_event.wait(interval):
            self._sample()

    def _sample(self) -> None:
        self.rss_samples.append(current_rss_bytes())
        self.temp_samples.append(read_cpu_temp_c())
        self.freq_samples.append(read_arm_freq_mhz())
        throttled_raw = read_throttled_raw()
        self.throttled_samples.append(throttled_value(throttled_raw))
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            self.process_cpu_samples.append(float(proc.cpu_percent(None)))
            self.system_cpu_samples.append(float(psutil.cpu_percent(None)))

    def summary(self) -> dict[str, Any]:
        rss = [value for value in self.rss_samples if value is not None]
        temp = summarize_optional(self.temp_samples)
        freq = summarize_optional(self.freq_samples)
        throttled = [value for value in self.throttled_samples if value is not None]
        process_cpu = summarize_optional(self.process_cpu_samples)
        system_cpu = summarize_optional(self.system_cpu_samples)
        process_cpu_time_sec = self.process_time_end - self.process_time_start
        wall_time_sec = self.wall_end - self.wall_start
        return {
            "process_cpu_time_sec": float(process_cpu_time_sec),
            "process_cpu_time_per_1k_flows_sec": None,
            "telemetry_wall_time_sec": float(wall_time_sec),
            "rss_min_bytes": int(min(rss)) if rss else None,
            "rss_max_bytes": int(max(rss)) if rss else None,
            "rss_end_bytes": int(rss[-1]) if rss else None,
            "process_cpu_percent_median": process_cpu["median"],
            "process_cpu_percent_max": process_cpu["max"],
            "system_cpu_percent_median": system_cpu["median"],
            "system_cpu_percent_max": system_cpu["max"],
            "cpu_temp_c_min": temp["min"],
            "cpu_temp_c_median": temp["median"],
            "cpu_temp_c_max": temp["max"],
            "arm_freq_mhz_min": freq["min"],
            "arm_freq_mhz_median": freq["median"],
            "arm_freq_mhz_max": freq["max"],
            "throttled_start": throttled[0] if throttled else None,
            "throttled_end": throttled[-1] if throttled else None,
            "throttled_during_run": any(value != 0 for value in throttled) if throttled else None,
        }


def latency_summary(samples_ms_per_row: list[float]) -> dict[str, float]:
    q1, q3 = np.percentile(samples_ms_per_row, [25, 75])
    p95, p99 = np.percentile(samples_ms_per_row, [95, 99])
    median = statistics.median(samples_ms_per_row)
    return {
        "median_ms_per_row": float(median),
        "iqr_ms_per_row": float(q3 - q1),
        "q1_ms_per_row": float(q1),
        "q3_ms_per_row": float(q3),
        "p95_ms_per_row": float(p95),
        "p99_ms_per_row": float(p99),
        "min_ms_per_row": float(min(samples_ms_per_row)),
        "max_ms_per_row": float(max(samples_ms_per_row)),
        "throughput_flows_per_sec": float(1000.0 / median) if median > 0 else float("inf"),
    }


def machine_metadata() -> dict[str, Any]:
    return {
        "machine": platform.platform(),
        "node": platform.node(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "psutil_version": getattr(psutil, "__version__", None),
        "cpu_model": cpu_model(),
        "total_ram_bytes": total_ram_bytes(),
        "cpu_governor": read_governors(),
        "power_measurement_method": "software_telemetry_proxy_only",
    }


def trim_prepared(
    prepared: Any,
    rows: int | None,
    rows_sample_mode: str,
    rows_sample_seed: int,
) -> Any:
    if rows is None:
        return prepared
    limit = min(rows, len(prepared.x_test))
    if rows_sample_mode == "random":
        selected_index = prepared.x_test.sample(n=limit, random_state=rows_sample_seed).index
    elif rows_sample_mode == "head":
        selected_index = prepared.x_test.index[:limit]
    else:
        raise ValueError(f"Unsupported rows_sample_mode {rows_sample_mode!r}")
    return bench.PreparedDataset(
        config=prepared.config,
        raw_train=prepared.raw_train,
        raw_test=prepared.raw_test.loc[selected_index].copy(),
        x_train=prepared.x_train,
        y_train=prepared.y_train,
        x_test=prepared.x_test.loc[selected_index].copy(),
        y_test=prepared.y_test.loc[selected_index].copy(),
        categorical_cols=prepared.categorical_cols,
    )


def build_condition_assets(
    condition: str,
    prepared: Any,
    policy: PolicySpec,
) -> tuple[Callable[[str, int | None], np.ndarray], int | None]:
    rss_before_assets = current_rss_bytes()
    if condition == "llm":
        after_assets = current_rss_bytes()

        def run(mode: str, batch_size: int | None) -> np.ndarray:
            return predict_llm(prepared.raw_test, policy, mode, batch_size)

        return run, None if rss_before_assets is None or after_assets is None else after_assets - rss_before_assets

    if condition == "dt":
        model = DecisionTreeClassifier(random_state=42)
    elif condition == "rf":
        model = RandomForestClassifier(random_state=42, n_jobs=1)
    else:
        raise ValueError(f"Unsupported condition {condition!r}")

    model.fit(prepared.x_train, prepared.y_train)
    after_assets = current_rss_bytes()

    def run(mode: str, batch_size: int | None) -> np.ndarray:
        return predict_model(model, prepared.x_test, mode, batch_size)

    return run, None if rss_before_assets is None or after_assets is None else after_assets - rss_before_assets


def benchmark_callable(
    fn: Callable[[], np.ndarray],
    rows: int,
    repeats: int,
    warmups: int,
    telemetry_hz: float,
) -> tuple[dict[str, Any], list[float], list[dict[str, Any]]]:
    for _ in range(warmups):
        fn()

    samples_ms_per_row = []
    telemetry_runs = []
    for _ in range(repeats):
        with TelemetrySampler(telemetry_hz) as sampler:
            start_ns = time.perf_counter_ns()
            fn()
            elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_ms_per_row = (elapsed_ns / 1_000_000) / rows
        telemetry = sampler.summary()
        telemetry["process_cpu_time_per_1k_flows_sec"] = (
            telemetry["process_cpu_time_sec"] / rows * 1000.0
        )
        samples_ms_per_row.append(elapsed_ms_per_row)
        telemetry_runs.append(telemetry)

    summary = latency_summary(samples_ms_per_row)
    summary.update(summarize_telemetry_runs(telemetry_runs))
    return summary, samples_ms_per_row, telemetry_runs


def summarize_telemetry_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[Any]:
        return [run.get(key) for run in runs if run.get(key) is not None]

    summarized: dict[str, Any] = {}
    for key in (
        "process_cpu_time_sec",
        "process_cpu_time_per_1k_flows_sec",
        "process_cpu_percent_median",
        "process_cpu_percent_max",
        "system_cpu_percent_median",
        "system_cpu_percent_max",
        "cpu_temp_c_min",
        "cpu_temp_c_median",
        "cpu_temp_c_max",
        "arm_freq_mhz_min",
        "arm_freq_mhz_median",
        "arm_freq_mhz_max",
    ):
        numeric = [float(value) for value in values(key)]
        summarized[f"{key}_median"] = float(statistics.median(numeric)) if numeric else None
        summarized[f"{key}_max"] = float(max(numeric)) if numeric else None

    for key in ("rss_min_bytes", "rss_max_bytes", "rss_end_bytes"):
        numeric = [int(value) for value in values(key)]
        summarized[f"{key}_max"] = int(max(numeric)) if numeric else None

    throttled_values = values("throttled_during_run")
    summarized["throttled_during_any_repeat"] = (
        any(bool(value) for value in throttled_values) if throttled_values else None
    )
    return summarized


def condition_label(condition: str) -> str:
    return {"llm": "LLM_rule_policy", "dt": "DT", "rf": "RF"}[condition]


def mode_label(mode: str, batch_size: int | None) -> str:
    if mode == "streaming":
        return "streaming"
    return f"batch_{batch_size if batch_size is not None else 'full'}"


def benchmark_dataset(
    dataset_key: str,
    policies: dict[str, PolicySpec],
    conditions: list[str],
    modes: list[str],
    batch_size: int | None,
    rows_limit: int | None,
    rows_sample_mode: str,
    rows_sample_seed: int,
    repeats: int,
    warmups: int,
    telemetry_hz: float,
    cooling_note: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = DATASETS[dataset_key]
    policy = policies[dataset_key]
    _raw_a, _raw_b, source = bench.load_raw_frames(config)
    rss_before_prepare = current_rss_bytes()
    prepared = trim_prepared(
        bench.prepare_dataset(config),
        rows=rows_limit,
        rows_sample_mode=rows_sample_mode,
        rows_sample_seed=rows_sample_seed,
    )
    rss_after_prepare = current_rss_bytes()
    rows = len(prepared.x_test)
    if rows == 0:
        raise ValueError(f"No test rows available for {config.display_name}")

    base_meta = machine_metadata()
    base_meta.update(
        {
            "dataset": config.display_name,
            "dataset_key": dataset_key,
            "source": source,
            "rows": rows,
            "rows_limit": rows_limit,
            "rows_sample_mode": rows_sample_mode if rows_limit is not None else "all",
            "rows_sample_seed": rows_sample_seed if rows_limit is not None else None,
            "repeats": repeats,
            "warmups": warmups,
            "telemetry_hz": telemetry_hz,
            "cooling_note": cooling_note,
            "categorical_columns_encoded": ",".join(prepared.categorical_cols),
            "rss_before_prepare_bytes": rss_before_prepare,
            "rss_after_prepare_bytes": rss_after_prepare,
            "rss_prepare_delta_bytes": (
                None
                if rss_before_prepare is None or rss_after_prepare is None
                else rss_after_prepare - rss_before_prepare
            ),
        }
    )

    summary_rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    for condition in conditions:
        run_condition, asset_delta = build_condition_assets(condition, prepared, policy)
        for mode in modes:
            effective_batch_size = 1 if mode == "streaming" else batch_size

            def fn() -> np.ndarray:
                return run_condition(mode, effective_batch_size)

            predictions = fn()
            report = classification_report(
                prepared.y_test,
                predictions,
                digits=4,
                output_dict=True,
                zero_division=0,
            )
            summary, samples, telemetry_runs = benchmark_callable(
                fn=fn,
                rows=rows,
                repeats=repeats,
                warmups=warmups,
                telemetry_hz=telemetry_hz,
            )
            row = {
                **base_meta,
                "condition": condition_label(condition),
                "mode": mode,
                "batch_size": effective_batch_size if effective_batch_size is not None else "full",
                "memory_delta_after_policy_model_load_bytes": asset_delta,
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "accuracy": float(report["accuracy"]),
                **summary,
            }
            summary_rows.append(row)
            raw_runs.append(
                {
                    "metadata": row,
                    "samples_ms_per_row": samples,
                    "telemetry_runs": telemetry_runs,
                    "classification_report": report,
                }
            )
            print(
                f"{config.display_name:12s} {condition_label(condition):15s} "
                f"{mode_label(mode, effective_batch_size):14s} "
                f"median={row['median_ms_per_row']:.6f} ms/flow "
                f"p95={row['p95_ms_per_row']:.6f} rows={rows:,}"
            )
    return summary_rows, raw_runs


def write_outputs(rows: list[dict[str, Any]], raw_runs: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"pi-edge-benchmark-{stamp}.csv"
    json_path = out_dir / f"pi-edge-benchmark-{stamp}.json"
    latest_csv = out_dir / "pi-edge-benchmark-latest.csv"
    latest_json = out_dir / "pi-edge-benchmark-latest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(rows).to_csv(latest_csv, index=False)
    payload = {
        "summary": rows,
        "raw_runs": raw_runs,
        "power_note": (
            "Software telemetry only. CPU time, frequency, temperature, and "
            "throttling are resource proxies, not direct power or energy measurements."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    latest_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {display_path(csv_path)}")
    print(f"Wrote {display_path(json_path)}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="cic")
    parser.add_argument("--condition", choices=["all", "llm", "dt", "rf"], default="llm")
    parser.add_argument("--mode", choices=["all", "streaming", "batch"], default="batch")
    parser.add_argument("--batch-size", type=parse_batch_size, default=None)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--telemetry-hz", type=float, default=2.0)
    parser.add_argument("--rows", type=int, default=None, help="Limit test rows for smoke tests.")
    parser.add_argument(
        "--rows-sample-mode",
        choices=["random", "head"],
        default="random",
        help="How to choose rows when --rows is set. Default: random.",
    )
    parser.add_argument(
        "--rows-sample-seed",
        type=int,
        default=42,
        help="Random seed used when --rows-sample-mode=random.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--cooling-note", default=None)
    parser.add_argument(
        "--allow-few-repeats",
        action="store_true",
        help="Permit fewer than 30 repeats for smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 30 and not args.allow_few_repeats:
        raise ValueError("--repeats must be at least 30 unless --allow-few-repeats is set")
    if args.telemetry_hz < 0:
        raise ValueError("--telemetry-hz must be >= 0")

    policies = load_policies(args.policy_path)
    dataset_keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    conditions = ["llm", "dt", "rf"] if args.condition == "all" else [args.condition]
    modes = ["streaming", "batch"] if args.mode == "all" else [args.mode]
    all_rows: list[dict[str, Any]] = []
    all_raw_runs: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        try:
            rows, raw_runs = benchmark_dataset(
                dataset_key=dataset_key,
                policies=policies,
                conditions=conditions,
                modes=modes,
                batch_size=args.batch_size,
                rows_limit=args.rows,
                rows_sample_mode=args.rows_sample_mode,
                rows_sample_seed=args.rows_sample_seed,
                repeats=args.repeats,
                warmups=args.warmups,
                telemetry_hz=args.telemetry_hz,
                cooling_note=args.cooling_note,
            )
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"Skipping {dataset_key}: {exc}")
                continue
            raise
        all_rows.extend(rows)
        all_raw_runs.extend(raw_runs)
    if not all_rows:
        raise RuntimeError("No benchmark results were produced")
    write_outputs(all_rows, all_raw_runs, args.out_dir)


if __name__ == "__main__":
    main()
