from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EDGE_PATH = REPO_ROOT / "experiments" / "08-edge-benchmark-raspberry-pi.py"


def load_edge_module():
    spec = importlib.util.spec_from_file_location("edge_inference_benchmark", EDGE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {EDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


edge = load_edge_module()


class EdgeInferencePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policies = edge.load_policies(edge.DEFAULT_POLICY_PATH)

    def test_cic_policy_matches_existing_benchmark_predictor(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "flow_duration": 0.5,
                    "Header_Length": 50,
                    "Duration": 60,
                    "Srate": 60,
                    "ack_flag_number": 0,
                },
                {
                    "flow_duration": 5.0,
                    "Header_Length": 2000,
                    "Duration": 80,
                    "Srate": 10,
                    "ack_flag_number": 1,
                },
                {
                    "flow_duration": 2.0,
                    "Header_Length": 10,
                    "Duration": 80,
                    "Srate": 100,
                    "ack_flag_number": 1,
                },
            ]
        )
        expected = edge.bench.llm_rule_predict(rows, edge.DATASETS["cic"])
        policy = self.policies["cic"]
        np.testing.assert_array_equal(edge.llm_rule_predict_batch(rows, policy), expected)
        np.testing.assert_array_equal(edge.llm_rule_predict_streaming(rows, policy), expected)

    def test_ton_policy_matches_existing_benchmark_predictor_with_compound_rule(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "dst_port": 4444,
                    "conn_state": "OTH",
                    "duration": 0.0,
                    "weird_name": "-",
                    "dst_bytes": 0,
                    "src_bytes": 0,
                },
                {
                    "dst_port": 80,
                    "conn_state": "S0",
                    "duration": 1.5,
                    "weird_name": "notice",
                    "dst_bytes": 100,
                    "src_bytes": 25,
                },
                {
                    "dst_port": 80,
                    "conn_state": "OTH",
                    "duration": 0.0,
                    "weird_name": "notice",
                    "dst_bytes": 0,
                    "src_bytes": 0,
                },
            ]
        )
        expected = edge.bench.llm_rule_predict(rows, edge.DATASETS["ton"])
        policy = self.policies["ton"]
        np.testing.assert_array_equal(edge.llm_rule_predict_batch(rows, policy), expected)
        np.testing.assert_array_equal(edge.llm_rule_predict_streaming(rows, policy), expected)

    def test_batched_prediction_matches_full_batch_prediction(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "flow_duration": value,
                    "Header_Length": value * 100,
                    "Duration": 50 + value,
                    "Srate": 20 + value,
                    "ack_flag_number": value % 2,
                }
                for value in range(1, 10)
            ]
        )
        policy = self.policies["cic"]
        full = edge.predict_llm(rows, policy, mode="batch", batch_size=None)
        batched = edge.predict_llm(rows, policy, mode="batch", batch_size=3)
        streaming = edge.predict_llm(rows, policy, mode="streaming", batch_size=1)
        np.testing.assert_array_equal(batched, full)
        np.testing.assert_array_equal(streaming, full)


if __name__ == "__main__":
    unittest.main()
