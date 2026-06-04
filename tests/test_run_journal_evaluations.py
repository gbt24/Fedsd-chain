# -*- coding: UTF-8 -*-
import json
import os
import tempfile
import unittest

from run_journal_evaluations import target_clients, write_summary


class JournalEvaluationRunnerTest(unittest.TestCase):
    def test_target_clients_defaults_to_all_clients(self):
        self.assertEqual(target_clients(3, None), [0, 1, 2])

    def test_target_clients_parses_comma_separated_subset(self):
        self.assertEqual(target_clients(10, "0, 2,4"), [0, 2, 4])

    def test_write_summary_aggregates_trace_accuracy_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracing_dir = os.path.join(tmpdir, "eval", "tracing")
            os.makedirs(tracing_dir)
            reports = {
                "client_0.json": {"identified_client": 0, "confidence": 0.9},
                "client_1.json": {"identified_client": 2, "confidence": 0.7},
            }
            for name, payload in reports.items():
                with open(os.path.join(tracing_dir, name), "w") as f:
                    json.dump(payload, f)

            summary = write_summary(tmpdir, [0, 1])

            self.assertEqual(summary["num_trace_reports"], 2)
            self.assertEqual(summary["trace_accuracy"], 0.5)
            self.assertAlmostEqual(summary["mean_confidence"], 0.8)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "eval", "summary.json")))


if __name__ == "__main__":
    unittest.main()
